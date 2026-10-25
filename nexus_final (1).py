"""
================================================================================
 NEXUS  --  Systemic Architecture for High-Frequency Alpha Extraction
 Takneek 2026  |  Problem Statement: ZENITH / ALPHA
================================================================================

A seven-strategy long/short framework for a 20-stock universe, built on
non-extensive (Tsallis) statistical mechanics, market-microstructure features,
and Mean-CVaR capital allocation, with an event-driven backtester that enforces
the competition's capital and exposure constraints at execution time.

RUN
---
    python nexus_alpha.py <data_directory>
    python nexus_alpha.py <data_directory> --neutral

  The loader auto-detects either dataset convention:
    competition : date,datetime,open,high,low,close,volume        (assetN.csv)
    alpaca/IEX  : symbol,timestamp_utc,open,...,trade_count,vwap  (TICKER.csv)

STRUCTURE
---------
    SECTION 1   Configuration           every tunable constant, in one place
    SECTION 2   Data ingestion          bars, corporate-action ledger
    SECTION 3   Econophysics            Tsallis q-Gaussian fits, ATRE, Hurst
    SECTION 4   Feature engineering     volatility, liquidity, order flow
    SECTION 5   Strategy sub-engines    the seven sleeves
    SECTION 6   Mean-CVaR optimiser     Rockafellar-Uryasev LP, Ledoit-Wolf
    SECTION 7   Portfolio construction  blending, sizing, risk, constraints
    SECTION 8   Simulated exchange      share-level accounting with costs
    SECTION 9   Performance analytics   the reported metric set
    SECTION 10  Entry point

THE SEVEN STRATEGIES
--------------------
    S1 GAP      filtered gap-down mean reversion        event driven, long
    S2 MLRISK   LightGBM gap-risk exposure scaler       defensive overlay
    S3 XSREV    cross-sectional reversal + vol premium  dollar neutral
    S4 TSALLIS  asymmetric tail-risk premium (ATRE)     dollar neutral
    S5 PAIRS    GARCH cointegration pairs               market neutral
    S6 TURTLE   Donchian breakout, pyramided            trend / position sizing
    S7 MRW      multifractal scaling regime switch      two-sided

  plus the capital-deployment core in SECTION 7, which occupies the seventh
  slot when a net-long tilt is configured (S2 is an overlay, not a strategy).

NO LOOK-AHEAD
-------------
Signals are computed from data up to the close of bar t and executed at the
OPEN of bar t+1. The engine holds the pending target for exactly one bar, so
the delay is structural rather than an easily-broken `.shift(1)`. Every
statistic is a trailing window; every model refit trains only on data strictly
before the block it trades, with an embargo between them. Verified by
truncation invariance and an execution-timing oracle -- see the audit module
in the development package.

DETERMINISM
-----------
Fixed seeds throughout; no wall-clock or unseeded randomness. Re-running
reproduces results exactly.
"""
from __future__ import annotations

import argparse
import dataclasses
import glob
import itertools
import os
import sys
import time
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy import stats
from scipy.optimize import linprog, minimize
from sklearn.covariance import LedoitWolf

warnings.filterwarnings("ignore")

try:
    import lightgbm as lgb
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False
    from sklearn.ensemble import GradientBoostingClassifier

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from statsmodels.tsa.stattools import adfuller
@dataclass(frozen=True)
class DataConfig:
    frequency: str = "daily"          # "daily" session bars, or "hourly"
    min_universe_coverage: float = 0.60
    event_screen: float = 0.18
    split_ratios: tuple = (1.5, 2.0, 2.5, 3.0, 4.0, 5.0)
    split_tolerance: float = 0.10
    split_max_dollar_vol: float = 4.0
    competition_split_drop: float = 0.30
    mask_structural: bool = True
    quarantine_bars: int = 5
@dataclass(frozen=True)
class PhysicsConfig:
    q_window: int = 250               # rolling sample for the q fit
    q_min: float = 1.01               # q -> 1 recovers the Gaussian
    q_max: float = 1.66               # variance diverges at q = 5/3
    q_min_obs: int = 60               # minimum observations in a half-sample
    q_trim: float = 0.005             # winsorisation applied before the moment fit
    hurst_window: int = 120
    hurst_lags: tuple = (1, 2, 3, 4, 5, 7, 10, 14, 20)
    hurst_moments: tuple = (1.0, 2.0)
    hurst_trend_threshold: float = 0.55
@dataclass(frozen=True)
class FeatureConfig:
    vol_window: int = 20              # Parkinson / Garman-Klass window
    atr_window: int = 20              # Wilder ATR -- the Turtle sizing unit N
    amihud_window: int = 20
    gap_window: int = 60              # rolling mean/std of overnight gaps
    range_fast: int = 5               # range-expansion numerator
    range_slow: int = 40              # range-expansion baseline
    volume_window: int = 40
    momentum_window: int = 20
    beta_window: int = 120
    corr_window: int = 120
@dataclass(frozen=True)
class StrategyConfig:
    s1_gap_z: float = -1.357954  # [.-max]
    s1_range_expansion: float = 1.000000  # [.-max]
    s1_volume_z: float = 2.000000  # [.-max]
    s1_hold_bars: int = 10  # [.-max]
    s1_stop_loss: float = -0.027025  # [.-max]
    s2_gap_event_z: float = -2.0      # label: next-session gap below this
    s2_train_bars: int = 500
    s2_refit_every: int = 63
    s2_embargo: int = 5
    s2_min_scale: float = 0.50        # exposure retained on a flagged name
    s2_flag_quantile: float = 0.90    # cross-sectional cut for "high risk"
    s3_quantile: float = 0.148057  # [.-max]
    s3_amihud_exclude: float = 0.90   # drop the top decile of illiquidity
    s3_vol_premium_power: float = 0.467114  # [.-max]
    s4_smooth: float = 33.265165  # [.-max]
    s4_quantile: float = 0.179150  # [.-max]
    s4_coskew_window: int = 120
    s5_formation: int = 252
    s5_refit_every: int = 63
    s5_max_pairs: int = 5
    s5_adf_pvalue: float = 0.05
    s5_entry_z: float = 2.0
    s5_exit_z: float = 0.5
    s5_stop_z: float = 4.0
    s6_entry_window: int = 94  # [.-max]
    s6_exit_window: int = 48  # [.-max]
    s6_max_units: int = 7  # [.-max]
    s6_add_step: float = 0.5          # add a unit every 0.5 N in our favour
    s6_stop_mult: float = 3.857168  # [.-max]
    s6_risk_fraction: float = 0.004486  # [.-max]
    s6_short_scale: float = 0.812890  # [.-max]
    s7_vol_surge: float = 1.479815  # [.-max]
    s7_hurst_trend: float = 0.567318  # [.-max]
    s7_hurst_revert: float = 0.500000  # [.-max]
    s7_trail_stop: float = 0.212949  # [.-max]
    s7_min_abs_trend: float = 0.0
@dataclass(frozen=True)
class PortfolioConfig:
    initial_capital: float = 2_000_000.0
    fee: float = 0.001
    cvar_beta: float = 0.95           # tail probability
    cvar_window: int = 196  # [.-max]
    cvar_refit_every: int = 21
    cvar_risk_aversion: float = 3405.103857  # [.-max]
    cvar_turnover_penalty: float = 0.932029  # [.-max]
    sleeve_cost_scale: float = 0.185480  # [.-max]
    cvar_max_sleeve: float = 0.725836  # [.-max]
    cvar_min_sleeve: float = 0.006554  # [.-max]
    shrink_to_equal: float = 0.0
    sleeve_tilt: float = 3.459503  # [.-max]
    core_scheme: str = "inverse_vol"  # "inverse_vol" or "equal"
    sleeve_scale: tuple = (("S1_GAP", 1.866182), ("S3_XSREV", 1.883013),
                           ("S4_TSALLIS", 2.113595),
                           ("S5_PAIRS", 0.220778), ("S6_TURTLE", 2.380236),
                           ("S7_MRW", 0.499406))
    asset_sleeve_override: tuple = (("asset15", "S7_MRW"), ("BAX", "S7_MRW"))
    override_scale: float = 1.0
    core_trend_tilt: float = 0.113477  # [.-max]
    use_overlay: bool = True          # S2 ML gap-risk exposure scaling
    per_name_cap_mult: float = 14.427117  # [.-max]
    score_halflife: float = 3.953825  # [.-max]
    weight_halflife: float = 19.283803  # [.-max]
    max_gross: float = 1.513107  # [.-max]
    net_long_target: float = 1.010133  # [.-max]
    max_short: float = 0.180598  # [.-max]
    trailing_stop: float = 0.503541  # [.-max]
    stop_cooldown: int = 15  # [.-max]
    redeploy_idle: float = 0.519809  # [.-max]
    no_trade_band: float = 0.006172  # [.-max]
    exposure_margin: float = 0.019055  # [.-max]
    exposure_trigger: float = 0.99
    exposure_reduce_to: float = 0.97
    exposure_tolerance: float = 0.02
    allow_fractional: bool = False
@dataclass(frozen=True)
class WFOConfig:
    warmup: int = 300
    core_warmup: int = 40
    train_bars: int = 500
    test_bars: int = 63
    embargo: int = 5
    expanding: bool = False


@dataclass(frozen=True)
class NexusConfig:
    data: DataConfig = field(default_factory=DataConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    wfo: WFOConfig = field(default_factory=WFOConfig)
    seed: int = 20260902

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT = NexusConfig()
PERIODS_PER_YEAR = {"daily": 252, "hourly": 252 * 7}

SPLIT, STRUCTURAL, NEWS, NONE = "SPLIT", "STRUCTURAL", "NEWS", ""
def _read_competition(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["datetime", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["datetime"])
    return df[["ts", "open", "high", "low", "close", "volume"]]


def _read_alpaca(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["timestamp_utc", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["timestamp_utc"], utc=True).dt.tz_localize(None)
    return df[["ts", "open", "high", "low", "close", "volume"]]


def _detect_format(path: str) -> str:
    with open(path, "r") as fh:
        header = fh.readline()
    if "timestamp_utc" in header:
        return "alpaca"
    if "datetime" in header:
        return "competition"
    raise ValueError("Unrecognised OHLCV schema in " + path)


def discover_files(data_dir: str) -> dict:
    """Map asset name -> csv path, skipping metadata side-cars."""
    skip = {"audit_summary", "corporate_actions", "market_calendar", "README", "zenith_asset_metrics", "zenith_portfolio_equity"}
    out = {}
    for path in sorted(glob.glob(os.path.join(data_dir, "*.csv"))):
        name = os.path.splitext(os.path.basename(path))[0]
        if name in skip or name.startswith("supplement"):
            continue
        out[name] = path
    if all(k.startswith("asset") and k[5:].isdigit() for k in out):
        out = {k: out[k] for k in sorted(out, key=lambda s: int(s[5:]))}
    return out
def _aggregate(hourly: pd.DataFrame, frequency: str) -> pd.DataFrame:
    """
    Collapse intraday bars to the trading frequency, retaining the intraday
    shape statistics that the microstructure sleeves consume.

    `signed_volume` is the session's tick-rule order-flow proxy: each intraday
    bar contributes its volume with the sign of its own open-to-close move, so
    a session that closed up on rising volume registers positive net flow.
    This is the closest observable analogue of true Order Flow Imbalance that
    OHLCV permits, and it is genuinely intraday information rather than a
    restatement of the daily bar.
    """
    h = hourly.sort_values("ts").copy()
    if frequency == "hourly":
        out = h.set_index("ts")[["open", "high", "low", "close", "volume"]].copy()
        out["first_bar_close"] = out["close"]
        out["last_bar_open"] = out["open"]
        out["n_bars"] = 1
        sign = np.sign(out["close"] - out["open"]).replace(0.0, np.nan).ffill().fillna(0.0)
        out["signed_volume"] = sign * out["volume"]
        out.index.name = "date"
        return out

    bar_sign = np.sign(h["close"] - h["open"])
    bar_sign = bar_sign.replace(0.0, np.nan).ffill().fillna(0.0)
    h["signed_volume"] = bar_sign * h["volume"]

    day = h["ts"].dt.normalize()
    g = h.groupby(day)
    daily = pd.DataFrame({
        "open": g["open"].first(),
        "high": g["high"].max(),
        "low": g["low"].min(),
        "close": g["close"].last(),
        "volume": g["volume"].sum(),
        "signed_volume": g["signed_volume"].sum(),
        "first_bar_close": g["close"].first(),
        "last_bar_open": g["open"].last(),
        "n_bars": g["close"].size(),
    })
    daily.index.name = "date"
    return daily
def _classify_events(close: pd.Series, volume: pd.Series,
                     cfg: DataConfig) -> pd.DataFrame:
    """
    Build the per-bar corporate action ledger described in the module header.

    Returns a frame indexed like `close` with columns
        kind    -- "", "SPLIT", "STRUCTURAL" or "NEWS"
        factor  -- multiplicative correction for SPLIT bars (1.0 otherwise)
        ratio   -- the observed close ratio, for the audit report
    """
    ratio = close / close.shift(1)
    screened = ((ratio - 1.0).abs() > cfg.event_screen).fillna(False)

    kind = pd.Series(NONE, index=close.index, dtype=object)
    factor = pd.Series(1.0, index=close.index, dtype=float)

    dollar_vol = close * volume
    typical_dv = dollar_vol.rolling(60, min_periods=20).median().shift(1)
    targets = [1.0 / m for m in cfg.split_ratios] + [float(m) for m in cfg.split_ratios]

    for ts in close.index[screened]:
        rat = float(ratio.loc[ts])
        if not np.isfinite(rat) or rat <= 0:
            continue
        if rat < 1.0 - cfg.competition_split_drop:
            kind.loc[ts] = SPLIT
            factor.loc[ts] = 1.0 / rat
            continue

        dv = float(dollar_vol.loc[ts])
        typ = float(typical_dv.get(ts, np.nan))
        heavy = np.isfinite(typ) and typ > 0 and dv / typ >= cfg.split_max_dollar_vol

        err, best = min((abs(rat / t - 1.0), t) for t in targets)
        if err < cfg.split_tolerance and not heavy:
            kind.loc[ts] = SPLIT              # clean ratio + ordinary turnover
            factor.loc[ts] = 1.0 / best
        elif not heavy and cfg.mask_structural and rat < 1.0:
            kind.loc[ts] = STRUCTURAL
        else:
            kind.loc[ts] = NEWS

    return pd.DataFrame({"kind": kind, "factor": factor, "ratio": ratio})


def _analysis_close(close: pd.Series, ledger: pd.DataFrame,
                    cfg: DataConfig) -> tuple:
    """
    Reconstruct a continuous price path for statistical work.

    Returns (adj_close, masked) where `masked` marks bars whose return carries
    no economic information and must be excluded from rolling statistics.
    """
    r = close.pct_change().to_numpy(dtype=float, copy=True)
    kind = ledger["kind"].to_numpy()
    fac = ledger["factor"].to_numpy()
    masked = np.zeros(len(r), dtype=bool)

    for i in range(len(r)):
        if kind[i] == SPLIT:
            r[i] = (1.0 + r[i]) * fac[i] - 1.0
        elif kind[i] == STRUCTURAL:
            r[i] = 0.0
            masked[i] = True
        elif kind[i] == NEWS:
            r[i] = float(np.clip(r[i], -cfg.event_screen, cfg.event_screen))

    r[0] = 0.0
    adj = pd.Series(np.nan_to_num(r), index=close.index)
    return float(close.iloc[0]) * (1.0 + adj).cumprod(), pd.Series(masked, index=close.index)
@dataclass
class MarketData:
    """Wide panels, one column per asset, on a shared bar index."""
    assets: list
    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    volume: pd.DataFrame
    signed_volume: pd.DataFrame
    first_bar_close: pd.DataFrame
    last_bar_open: pd.DataFrame
    adj_close: pd.DataFrame          # corporate-action neutralised, features only
    event: pd.DataFrame              # ledger kind per bar (object)
    masked: pd.DataFrame             # bool, bar carries no economic return
    quarantine: pd.DataFrame         # bool, name barred from new risk
    frequency: str = "daily"
    meta: dict = field(default_factory=dict)

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.close.index

    @property
    def adj_returns(self) -> pd.DataFrame:
        """Returns for statistical work: masked bars are NaN, not zero, so
        rolling estimators drop them instead of counting a fake flat bar."""
        r = self.adj_close.pct_change()
        return r.mask(self.masked)

    @property
    def raw_returns(self) -> pd.DataFrame:
        return self.close.pct_change()

    def slice(self, start=None, end=None) -> "MarketData":
        idx = self.close.index
        m = np.ones(len(idx), dtype=bool)
        if start is not None:
            m &= idx >= pd.Timestamp(start)
        if end is not None:
            m &= idx <= pd.Timestamp(end)
        fields = ("open", "high", "low", "close", "volume", "signed_volume",
                  "first_bar_close", "last_bar_open", "adj_close", "event",
                  "masked", "quarantine")
        sub = {f: getattr(self, f).loc[m] for f in fields}
        return MarketData(assets=list(self.assets), frequency=self.frequency,
                          meta=dict(self.meta), **sub)


def load_market_data(data_dir: str, cfg: DataConfig | None = None,
                     verbose: bool = True) -> MarketData:
    """Load a directory of OHLCV CSVs into an aligned panel."""
    cfg = cfg or DataConfig()
    files = discover_files(data_dir)
    if not files:
        raise FileNotFoundError("No OHLCV CSVs found in " + data_dir)

    frames, ledgers, masks = {}, {}, {}
    for name, path in files.items():
        fmt = _detect_format(path)
        hourly = _read_competition(path) if fmt == "competition" else _read_alpaca(path)
        hourly = hourly.dropna(subset=["open", "high", "low", "close"])
        bars = _aggregate(hourly, cfg.frequency)
        if cfg.frequency == "daily":
            bars = bars[bars["n_bars"] >= 2]      # a session needs a real auction
        ledger = _classify_events(bars["close"], bars["volume"], cfg)
        bars["adj_close"], masks[name] = _analysis_close(bars["close"], ledger, cfg)
        frames[name] = bars
        ledgers[name] = ledger

    assets = list(frames)
    index = pd.DatetimeIndex(sorted(set().union(*(f.index for f in frames.values()))))
    has_bar = pd.DataFrame(
        {a: frames[a]["close"].reindex(index).notna() for a in assets}, index=index)
    coverage = has_bar.sum(axis=1) / max(len(assets), 1)
    keep = coverage >= cfg.min_universe_coverage
    dropped = 0
    if keep.any():
        first, last = index[keep][0], index[keep][-1]
        trimmed = index[(index >= first) & (index <= last)]
        trimmed = trimmed[keep.reindex(trimmed).to_numpy()]
        dropped = len(index) - len(trimmed)
        index = trimmed

    def panel(col: str, source=frames, default=np.nan) -> pd.DataFrame:
        return pd.DataFrame({a: source[a][col].reindex(index) for a in assets},
                            index=index)

    event = pd.DataFrame({a: ledgers[a]["kind"].reindex(index).fillna(NONE)
                          for a in assets}, index=index)
    masked = pd.DataFrame({a: masks[a].reindex(index).fillna(False).astype(bool)
                           for a in assets}, index=index)
    flagged = event.ne(NONE)
    quarantine = (flagged.astype(float)
                  .rolling(cfg.quarantine_bars, min_periods=1).max().fillna(0.0) > 0)

    md = MarketData(
        assets=assets,
        open=panel("open"), high=panel("high"), low=panel("low"),
        close=panel("close"), volume=panel("volume"),
        signed_volume=panel("signed_volume"),
        first_bar_close=panel("first_bar_close"),
        last_bar_open=panel("last_bar_open"),
        adj_close=panel("adj_close"),
        event=event, masked=masked, quarantine=quarantine,
        frequency=cfg.frequency,
        meta={"source_dir": data_dir, "files": files,
              "bars_dropped_low_coverage": int(dropped)},
    )
    for f in ("open", "high", "low", "close", "first_bar_close",
              "last_bar_open", "adj_close"):
        setattr(md, f, getattr(md, f).ffill())
    md.volume = md.volume.fillna(0.0)
    md.signed_volume = md.signed_volume.fillna(0.0)

    counts = event.apply(lambda c: c.value_counts()).fillna(0).sum(axis=1)
    md.meta["event_counts"] = {k: int(v) for k, v in counts.items() if k}

    if verbose:
        print("[data] %d assets | %d %s bars | %s -> %s"
              % (len(assets), len(index), cfg.frequency,
                 index[0].date(), index[-1].date()))
        print("[data] corporate action ledger: %s"
              % (md.meta["event_counts"] or "none flagged"))
        if dropped:
            print("[data] %d bars dropped: under %.0f%% of the universe trading"
                  % (dropped, cfg.min_universe_coverage * 100))
    return md


def event_report(md: MarketData) -> pd.DataFrame:
    """Flat table of every ledger entry, for the documentation appendix."""
    rows = []
    for a in md.assets:
        col = md.event[a]
        for ts in col.index[col.ne(NONE)]:
            prev = md.close[a].shift(1).loc[ts]
            rows.append({"asset": a, "date": ts, "kind": col.loc[ts],
                         "raw_return": md.close[a].loc[ts] / prev - 1.0
                         if np.isfinite(prev) and prev > 0 else np.nan,
                         "adj_return": md.adj_returns[a].loc[ts]})
    out = pd.DataFrame(rows)
    return out.sort_values(["date", "asset"]).reset_index(drop=True) if len(out) else out
_P_HI, _P_LO = 0.95, 0.75
_H_HI, _H_LO = (1.0 + _P_HI) / 2.0, (1.0 + _P_LO) / 2.0
_NU_GRID = np.concatenate([
    np.linspace(2.05, 8.0, 240),
    np.linspace(8.0, 30.0, 120)[1:],
    np.linspace(30.0, 400.0, 80)[1:],
])
_R_GRID = stats.t.ppf(_H_HI, _NU_GRID) / stats.t.ppf(_H_LO, _NU_GRID)
_R_ASC = _R_GRID[::-1]
_NU_ASC = _NU_GRID[::-1]


def nu_to_q(nu: np.ndarray) -> np.ndarray:
    """Degrees of freedom -> Tsallis non-extensivity parameter."""
    return (nu + 3.0) / (nu + 1.0)


def q_to_nu(q: np.ndarray) -> np.ndarray:
    """Tsallis non-extensivity parameter -> degrees of freedom."""
    return (3.0 - q) / (q - 1.0)


def _ratio_to_nu(ratio: np.ndarray) -> np.ndarray:
    """Invert R(nu) = t_nu^-1(0.95)/t_nu^-1(0.75) by grid interpolation."""
    out = np.interp(ratio, _R_ASC, _NU_ASC, left=_NU_ASC[0], right=_NU_ASC[-1])
    return np.where(np.isfinite(ratio), out, np.nan)
def _rolling_half_fit(x: np.ndarray, window: int, side: int,
                      min_obs: int) -> tuple:
    """
    Rolling q and scale for one tail of one asset.

    `side` is +1 for the gain spectrum and -1 for the loss spectrum. The
    half-sample is the magnitudes of the returns with that sign, which is the
    half q-Gaussian the ATRE construction calls for.

    Returns (q, scale) arrays aligned to `x`, NaN before the window fills.
    Every value at index t is computed from x[t-window+1 .. t] inclusive, so
    the estimate is available at the close of bar t and is strictly causal.
    """
    n = len(x)
    q = np.full(n, np.nan)
    scale = np.full(n, np.nan)
    if n < window:
        return q, scale
    half = np.where(np.sign(x) == side, np.abs(x), np.nan)

    win = sliding_window_view(half, window)              # (n-window+1, window)
    valid = np.sum(np.isfinite(win), axis=1)

    with np.errstate(invalid="ignore"):
        hi = np.nanquantile(win, _P_HI, axis=1)
        lo = np.nanquantile(win, _P_LO, axis=1)

    ratio = np.where(lo > 0, hi / lo, np.nan)
    nu = _ratio_to_nu(ratio)
    q_win = nu_to_q(nu)
    with np.errstate(invalid="ignore", divide="ignore"):
        s_win = lo / stats.t.ppf(_H_LO, np.clip(nu, 2.05, None))

    ok = valid >= min_obs
    q[window - 1:] = np.where(ok, q_win, np.nan)
    scale[window - 1:] = np.where(ok, s_win, np.nan)
    return q, scale


def tsallis_panel(returns: pd.DataFrame, cfg: PhysicsConfig) -> dict:
    """
    Fit asymmetric Tsallis parameters across the whole panel.

    Returns a dict of DataFrames aligned to `returns`:
        q_left, q_right   non-extensivity of the loss / gain spectrum
        s_left, s_right   fitted scale of each spectrum
        q_asym            q_left - q_right, positive when crash risk dominates
        beta_left         inverse temperature of the loss spectrum
        atre              asymmetric Tsallis relative entropy, loss vs gain
    """
    idx, cols = returns.index, list(returns.columns)
    out = {k: pd.DataFrame(np.nan, index=idx, columns=cols)
           for k in ("q_left", "q_right", "s_left", "s_right")}

    for a in cols:
        x = returns[a].to_numpy(dtype=float)
        ql, sl = _rolling_half_fit(x, cfg.q_window, -1, cfg.q_min_obs)
        qr, sr = _rolling_half_fit(x, cfg.q_window, +1, cfg.q_min_obs)
        out["q_left"][a] = ql
        out["q_right"][a] = qr
        out["s_left"][a] = sl
        out["s_right"][a] = sr

    for k in ("q_left", "q_right"):
        out[k] = out[k].clip(cfg.q_min, cfg.q_max)
    out["q_asym"] = out["q_left"] - out["q_right"]
    nu_l = q_to_nu(out["q_left"])
    out["beta_left"] = (nu_l + 1.0) / (2.0 * nu_l * out["s_left"] ** 2)

    out["atre"] = asymmetric_relative_entropy(
        out["q_left"], out["s_left"], out["q_right"], out["s_right"])
    return out
def asymmetric_relative_entropy(q_l, s_l, q_r, s_r, n_grid: int = 129):
    """
    Tsallis relative entropy between the fitted loss and gain half-densities,

        D_q(p||r) = 1/(q-1) * integral p(x) [ (p(x)/r(x))^(q-1) - 1 ] dx

    evaluated on a shared standardised grid with q taken from the loss side,
    since it is downside risk the measure is required to price. The result is
    zero for a symmetric distribution and grows as the loss spectrum becomes
    heavier or wider than the gain spectrum, which is exactly the ordering a
    risk score needs.

    Vectorised over the whole panel: inputs are DataFrames, output matches.
    """
    ql = np.asarray(q_l, dtype=float)
    sl = np.asarray(s_l, dtype=float)
    qr = np.asarray(q_r, dtype=float)
    sr = np.asarray(s_r, dtype=float)

    nu_l, nu_r = q_to_nu(ql), q_to_nu(qr)
    grid = np.linspace(1e-4, 8.0, n_grid)
    dx = grid[1] - grid[0]

    g = grid.reshape((1,) * ql.ndim + (n_grid,))
    p = stats.t.pdf(g, nu_l[..., None], scale=sl[..., None])
    r = stats.t.pdf(g, nu_r[..., None], scale=sr[..., None])

    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        ratio = np.where(r > 1e-300, p / np.maximum(r, 1e-300), np.nan)
        expo = ql[..., None] - 1.0
        integrand = p * (np.power(ratio, expo) - 1.0)
        d = np.nansum(integrand, axis=-1) * dx * sl / expo[..., 0]

    d = np.where(np.isfinite(d), d, np.nan)
    if isinstance(q_l, pd.DataFrame):
        return pd.DataFrame(d, index=q_l.index, columns=q_l.columns)
    return d


def tsallis_expected_shortfall(q, scale, beta: float = 0.95):
    """
    Expected shortfall of the fitted q-Gaussian loss spectrum.

    Under the Student-t equivalence with nu degrees of freedom and scale s,

        ES_b = s * pdf_nu(t_b) / (1 - b) * (nu + t_b^2) / (nu - 1)

    with t_b the standard-t quantile at b. This is the non-extensive tail
    measure the risk engine uses in place of a Gaussian sigma multiple: at
    q = 1.5 (nu = 3) it is roughly 1.5x the Gaussian figure, and the gap widens
    as q rises, which is the whole point of the exercise.
    """
    qv = np.asarray(q, dtype=float)
    sv = np.asarray(scale, dtype=float)
    nu = np.clip(q_to_nu(qv), 1.05, None)
    tb = stats.t.ppf(beta, nu)
    es = sv * stats.t.pdf(tb, nu) / (1.0 - beta) * (nu + tb ** 2) / (nu - 1.0)
    es = np.where(np.isfinite(es) & (nu > 1.0), es, np.nan)
    if isinstance(q, pd.DataFrame):
        return pd.DataFrame(es, index=q.index, columns=q.columns)
    return es
def generalised_hurst(log_price: pd.DataFrame, cfg: PhysicsConfig) -> dict:
    """
    Rolling generalised Hurst exponents by the structure-function method.

    For the m-th order structure function of a self-affine process,

        S_m(tau) = < | X(t+tau) - X(t) | ^ m >  ~  tau ^ ( m * H(m) )

    so H(m) is the slope of log S_m against log tau, divided by m. H(2) = 0.5
    is a random walk, above 0.5 is persistent (trending), below is
    anti-persistent (mean-reverting).

    A single Hurst exponent describes a *mono*fractal process. Real price
    series are multifractal: H(m) varies with m. The spread H(1) - H(2) is
    reported as the multifractal width, and a widening spread is the signature
    of an intermittent, burst-prone regime.

    Implementation note: the regression across lags has fixed design points
    log(tau), so the slope collapses to a fixed weighted sum of the log
    structure functions. That makes the whole panel computable with rolling
    means alone, with no per-bar loop.
    """
    lags = np.array(cfg.hurst_lags, dtype=int)
    logtau = np.log(lags.astype(float))
    w = (logtau - logtau.mean()) / np.sum((logtau - logtau.mean()) ** 2)

    out = {}
    for m in cfg.hurst_moments:
        acc = None
        for i, tau in enumerate(lags):
            d = (log_price - log_price.shift(int(tau))).abs() ** m
            s = d.rolling(cfg.hurst_window, min_periods=cfg.hurst_window // 2).mean()
            term = np.log(s.replace(0.0, np.nan)) * w[i]
            acc = term if acc is None else acc + term
        out["H%g" % m] = acc / m

    h1 = out.get("H1")
    h2 = out.get("H2")
    if h1 is not None and h2 is not None:
        out["multifractal_width"] = h1 - h2
    return out


def volatility_persistence(returns: pd.DataFrame, window: int = 120,
                           lags: int = 5) -> pd.DataFrame:
    """
    Mean autocorrelation of squared returns over the first `lags` lags.

    Directional autocorrelation of these series decays within a bar, but the
    autocorrelation of squared returns is still significant past twenty lags
    -- the ARCH signature. Strategy 7 tightens its trailing stop as this
    measure decays, because a multifractal trend regime dies when its
    volatility clustering dies.
    """
    r2 = returns ** 2
    acc = None
    for k in range(1, lags + 1):
        c = r2.rolling(window, min_periods=window // 2).corr(r2.shift(k))
        acc = c if acc is None else acc + c
    return acc / lags
ep = sys.modules[__name__]

_EPS = 1e-12


def _z(x: pd.DataFrame, window: int, min_periods: int | None = None) -> pd.DataFrame:
    """Trailing cross-time z-score. Closed at t: no centring, no future."""
    mp = min_periods or max(10, window // 3)
    mu = x.rolling(window, min_periods=mp).mean()
    sd = x.rolling(window, min_periods=mp).std()
    return (x - mu) / sd.replace(0.0, np.nan)


def _xs_z(x: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional z-score within each bar."""
    mu = x.mean(axis=1)
    sd = x.std(axis=1).replace(0.0, np.nan)
    return x.sub(mu, axis=0).div(sd, axis=0)


def _xs_rank(x: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional rank in [0, 1], NaN-safe."""
    return x.rank(axis=1, pct=True)
def parkinson(high: pd.DataFrame, low: pd.DataFrame, window: int,
              periods: int = 252) -> pd.DataFrame:
    """
    Parkinson (1980) range volatility, annualised.

        sigma^2 = 1/(4 N ln2) * sum ( ln(H/L) )^2

    Assumes a driftless geometric Brownian motion, so it understates when a
    session trends hard, but it uses the whole intraday range instead of one
    point of it.
    """
    hl = np.log((high / low.replace(0.0, np.nan)).clip(lower=1.0 + _EPS)) ** 2
    var = hl.rolling(window, min_periods=max(5, window // 3)).mean() / (4.0 * np.log(2.0))
    return np.sqrt(var.clip(lower=0.0) * periods)


def garman_klass(open_: pd.DataFrame, high: pd.DataFrame, low: pd.DataFrame,
                 close: pd.DataFrame, window: int, periods: int = 252) -> pd.DataFrame:
    """
    Garman-Klass (1980) volatility, annualised.

        sigma^2 = 0.5 (ln H/L)^2 - (2 ln2 - 1) (ln C/O)^2

    More efficient than Parkinson because it uses the open and close as well
    as the range. The second term can drive a single bar negative, so the
    per-bar estimate is floored at zero before averaging.
    """
    hl = np.log((high / low.replace(0.0, np.nan)).clip(lower=1.0 + _EPS)) ** 2
    co = np.log((close / open_.replace(0.0, np.nan)).clip(lower=_EPS)) ** 2
    daily = 0.5 * hl - (2.0 * np.log(2.0) - 1.0) * co
    var = daily.clip(lower=0.0).rolling(window, min_periods=max(5, window // 3)).mean()
    return np.sqrt(var.clip(lower=0.0) * periods)


def amihud(returns: pd.DataFrame, dollar_volume: pd.DataFrame,
           window: int) -> pd.DataFrame:
    """
    Amihud (2002) illiquidity: mean of |r| / dollar volume over the window.

    Scaled by 1e6 so the numbers are readable; only the cross-sectional
    ordering is ever used, so the scaling is cosmetic.
    """
    dv = dollar_volume.replace(0.0, np.nan)
    ratio = returns.abs() / dv
    return ratio.rolling(window, min_periods=max(5, window // 3)).mean() * 1e6
@dataclass
class FeatureSet:
    """Every panel a strategy is allowed to see, all aligned and all trailing."""
    panels: dict = field(default_factory=dict)

    def __getitem__(self, k: str) -> pd.DataFrame:
        return self.panels[k]

    def __contains__(self, k: str) -> bool:
        return k in self.panels

    def get(self, k: str, default=None):
        return self.panels.get(k, default)

    def keys(self):
        return self.panels.keys()


def build_features(md: MarketData, fcfg: FeatureConfig,
                   pcfg: PhysicsConfig, periods: int = 252,
                   verbose: bool = True) -> FeatureSet:
    """Compute the full feature panel from a loaded MarketData."""
    o, h, l, c = md.open, md.high, md.low, md.close
    adj = md.adj_close
    vol = md.volume
    r = md.adj_returns

    p: dict = {}
    p["returns"] = r
    p["log_price"] = np.log(adj.clip(lower=_EPS))
    p["dollar_volume"] = c * vol
    p["parkinson"] = parkinson(h, l, fcfg.vol_window, periods)
    p["garman_klass"] = garman_klass(o, h, l, c, fcfg.vol_window, periods)
    p["gk_slow"] = garman_klass(o, h, l, c, fcfg.vol_window * 3, periods)
    p["realised_vol"] = (r.rolling(fcfg.vol_window,
                                   min_periods=max(5, fcfg.vol_window // 3)).std()
                         * np.sqrt(periods))
    p["vol_surge"] = p["garman_klass"] / p["gk_slow"].replace(0.0, np.nan)
    scale = (adj / c.replace(0.0, np.nan))
    adj_h, adj_l = h * scale, l * scale
    prev_c = adj.shift(1)
    tr = pd.concat([(adj_h - adj_l).abs(),
                    (adj_h - prev_c).abs(),
                    (adj_l - prev_c).abs()]).groupby(level=0).max()
    tr = tr.reindex(index=adj.index, columns=adj.columns).mask(md.masked)
    p["true_range"] = tr
    p["atr"] = tr.ewm(alpha=1.0 / fcfg.atr_window, min_periods=fcfg.atr_window,
                      adjust=False).mean()
    p["atr_pct"] = p["atr"] / adj.replace(0.0, np.nan)
    p["amihud"] = amihud(r, p["dollar_volume"], fcfg.amihud_window)
    p["amihud_rank"] = _xs_rank(p["amihud"])
    p["volume_z"] = _z(vol, fcfg.volume_window)
    p["dollar_volume_z"] = _z(p["dollar_volume"], fcfg.volume_window)
    split_adj_open = o * (adj / c.replace(0.0, np.nan))
    gap = split_adj_open / adj.shift(1).replace(0.0, np.nan) - 1.0
    gap = gap.mask(md.masked)
    p["gap"] = gap
    p["gap_z"] = _z(gap, fcfg.gap_window)
    rng = (h - l) / c.replace(0.0, np.nan)
    p["range"] = rng
    fast = rng.rolling(fcfg.range_fast, min_periods=2).mean()
    slow = rng.rolling(fcfg.range_slow, min_periods=10).mean()
    p["range_expansion"] = fast / slow.replace(0.0, np.nan)
    p["mom_short"] = adj / adj.shift(fcfg.momentum_window) - 1.0
    p["mom_long"] = adj.shift(21) / adj.shift(252) - 1.0          # 12-1
    p["mom_scaled"] = p["mom_short"] / p["realised_vol"].replace(0.0, np.nan)
    p["mom_volume"] = p["mom_scaled"] * (1.0 + p["volume_z"].clip(-3, 3) / 3.0)
    p["reversal_1"] = -r
    p["reversal_5"] = -(adj / adj.shift(5) - 1.0)
    span = (h - l).replace(0.0, np.nan)
    p["upper_wick"] = (h - np.maximum(o, c)) / span
    p["lower_wick"] = (np.minimum(o, c) - l) / span
    p["body"] = (c - o).abs() / span
    p["wick_imbalance"] = p["lower_wick"] - p["upper_wick"]
    p["flow_ratio"] = md.signed_volume / vol.replace(0.0, np.nan)
    vpt = (r.fillna(0.0) * vol).cumsum()
    p["vpt"] = vpt
    p["vpt_slope"] = (vpt - vpt.shift(5)) / p["dollar_volume"].rolling(
        fcfg.volume_window, min_periods=10).mean().replace(0.0, np.nan)
    p["ofi"] = (p["flow_ratio"].clip(-1, 1)
                + 0.5 * p["wick_imbalance"].clip(-1, 1)
                + 0.5 * np.tanh(p["dollar_volume_z"].clip(-4, 4) / 2.0)
                * np.sign(p["flow_ratio"]).fillna(0.0))
    p["ofi_z"] = _z(p["ofi"], fcfg.volume_window)
    tsallis = ep.tsallis_panel(r, pcfg)
    p.update(tsallis)
    p["es_left"] = ep.tsallis_expected_shortfall(
        tsallis["q_left"], tsallis["s_left"], 0.95)

    hurst = ep.generalised_hurst(p["log_price"], pcfg)
    p.update(hurst)
    p["vol_persistence"] = ep.volatility_persistence(r, pcfg.hurst_window)
    p["dd_from_high"] = adj / adj.rolling(252, min_periods=126).max() - 1.0
    p["core_quality"] = 0.5 * (_xs_rank(p["mom_long"]) + _xs_rank(p["dd_from_high"]))
    mkt = r.mean(axis=1)
    p["market_return"] = mkt
    cov = r.rolling(fcfg.beta_window, min_periods=fcfg.beta_window // 2).cov(mkt)
    mvar = mkt.rolling(fcfg.beta_window, min_periods=fcfg.beta_window // 2).var()
    p["beta"] = cov.div(mvar.replace(0.0, np.nan), axis=0)
    p["idio_vol"] = (r.sub(p["beta"].mul(mkt, axis=0))
                     .rolling(fcfg.vol_window, min_periods=10).std() * np.sqrt(periods))

    if verbose:
        n = len(p)
        cov_pct = float(np.mean([p[k].notna().to_numpy().mean()
                                 for k in p if isinstance(p[k], pd.DataFrame)]))
        print("[features] %d panels built | mean non-null coverage %.1f%%"
              % (n, cov_pct * 100))
        print("[features] median q_left %.3f | q_right %.3f | H(2) %.3f"
              % (np.nanmedian(p["q_left"]), np.nanmedian(p["q_right"]),
                 np.nanmedian(p["H2"])))
    return FeatureSet(p)

class Strategy(ABC):
    """One alpha sub-engine."""

    key: str = "BASE"
    name: str = "base"
    directional: bool = False        # True if the sleeve may carry net exposure

    @abstractmethod
    def generate(self, md, fs, scfg) -> pd.DataFrame:
        """Return the target weight panel [bar x asset]."""
    @staticmethod
    def unit_book(w: pd.DataFrame, gross: float = 1.0) -> pd.DataFrame:
        """Scale each row so that sum|w| == gross, leaving all-zero rows flat."""
        g = w.abs().sum(axis=1)
        scale = (gross / g.replace(0.0, np.nan)).fillna(0.0)
        return w.mul(scale, axis=0).fillna(0.0)

    @staticmethod
    def dollar_neutral(w: pd.DataFrame) -> pd.DataFrame:
        """Remove any net exposure, then re-normalise to a unit book."""
        n = w.notna().sum(axis=1).replace(0, np.nan)
        centred = w.sub(w.sum(axis=1) / n, axis=0).where(w.notna(), 0.0)
        return Strategy.unit_book(centred.fillna(0.0))

    @staticmethod
    def long_short_from_rank(score: pd.DataFrame, q: float,
                             eligible: pd.DataFrame | None = None
                             ) -> pd.DataFrame:
        """
        Long the bottom `q` of `score`, short the top `q`, equally weighted
        within each leg and dollar-neutral across them.

        Ranking is done inside each bar, so it is invariant to any drift in
        the level of the score and needs no trailing normalisation.
        """
        s = score if eligible is None else score.where(eligible)
        valid = s.notna().sum(axis=1)
        r = s.rank(axis=1, pct=True)
        longs = (r <= q) & s.notna()
        shorts = (r >= 1.0 - q) & s.notna()
        nl, ns = longs.sum(axis=1), shorts.sum(axis=1)
        w = (longs.div(nl.replace(0, np.nan), axis=0).fillna(0.0)
             - shorts.div(ns.replace(0, np.nan), axis=0).fillna(0.0))
        w = w.where(valid >= 4, 0.0)     # too few names to rank meaningfully
        return Strategy.unit_book(w)


def event_book(entry: pd.DataFrame, price: pd.DataFrame, hold: int,
               stop_loss: float | None = None,
               trail_stop: float | None = None,
               direction: int = 1) -> pd.DataFrame:
    """
    Turn a sparse entry trigger into a held book with an exit policy.

    Shared by the event-driven sleeves (1, 6 and 7). A position opened on the
    signal at bar t is held from t+1 for at most `hold` bars, and is closed
    early if the loss from the entry mark breaches `stop_loss`, or if the
    give-back from the position's peak breaches `trail_stop`.

    `price` is the corporate-action-adjusted series, deliberately: on raw
    prices a 2-for-1 split is an instant -50% and would stop out every holder
    of a perfectly healthy company on a day the engine books no loss at all.

    Bar alignment
    -------------
    The trigger bar itself carries the position. A trigger at bar t is fully
    determined by information at or before t (the gap prints at t's open, the
    filters are t-1's), so out[t] = 1 means "hold from t+1", which is the
    contract every sleeve in this package obeys. Arming the position on t+1
    instead would delay entry by a full bar and forfeit the first bounce bar
    -- which is where most of the measured edge in a 3-bar reversion trade
    actually is.

    The entry mark is the trigger bar's close, the last price observable when
    the decision is made. The engine fills at the next open, so realised entry
    differs by one overnight gap; the stop is therefore approximate by that
    much, which is the honest cost of not being able to trade inside a bar.

    Returns a 0/1 (or 0/-1) occupancy panel. Sizing is the caller's business.
    """
    idx, cols = entry.index, list(entry.columns)
    trig = entry.fillna(False).to_numpy(dtype=bool)
    px = price.to_numpy(dtype=float)
    out = np.zeros_like(px, dtype=float)
    n, m = px.shape

    for j in range(m):
        held = 0
        entry_px = np.nan
        peak = np.nan
        for i in range(n):
            if trig[i, j]:
                held = hold
                entry_px = px[i, j]
                peak = px[i, j]

            if held <= 0:
                continue

            p = px[i, j]
            alive = True
            if np.isfinite(p) and np.isfinite(entry_px) and entry_px > 0:
                pnl = direction * (p / entry_px - 1.0)
                peak = p if not np.isfinite(peak) else (
                    max(peak, p) if direction > 0 else min(peak, p))
                if stop_loss is not None and pnl <= stop_loss:
                    alive = False
                if trail_stop is not None and np.isfinite(peak) and peak > 0:
                    give = direction * (p / peak - 1.0)
                    if give <= -abs(trail_stop):
                        alive = False
            if alive:
                out[i, j] = float(direction)
                held -= 1
            else:
                held = 0
            if held == 0:
                entry_px = np.nan
                peak = np.nan
    return pd.DataFrame(out, index=idx, columns=cols)


def sleeve_returns(weights: pd.DataFrame, returns: pd.DataFrame) -> pd.Series:
    """
    Notional return of a sleeve, marked one bar forward.

    Row t of `weights` is the book held from t+1, so it is paired with the
    return realised at t+1. This series is what the Mean-CVaR optimiser
    consumes as its scenario set, and its alignment is the difference between
    an honest allocator and one that has read tomorrow's paper.
    """
    aligned = weights.reindex(columns=returns.columns).fillna(0.0)
    return (aligned * returns.shift(-1).fillna(0.0)).sum(axis=1).shift(1).fillna(0.0)

class GapReversion(Strategy):
    key = "S1_GAP"
    name = "Intraday Gap-Down Mean Reversion"
    directional = True               # long-only by construction

    def generate(self, md, fs, scfg) -> pd.DataFrame:
        gap_z = fs["gap_z"]
        expansion = fs["range_expansion"]
        vol_z = fs["volume_z"]
        exhausted = (expansion.shift(1) >= scfg.s1_range_expansion)
        capitulating = (vol_z.shift(1) >= scfg.s1_volume_z)

        trigger = (gap_z <= scfg.s1_gap_z) & exhausted & capitulating
        trigger &= ~md.quarantine
        trigger = trigger.fillna(False)

        occupancy = event_book(
            trigger, md.adj_close,
            hold=scfg.s1_hold_bars,
            stop_loss=scfg.s1_stop_loss,
            direction=1,
        )
        inv_vol = 1.0 / fs["realised_vol"].clip(lower=0.05)
        w = occupancy * inv_vol.reindex_like(occupancy).fillna(0.0)
        return self.unit_book(w)

_FEATURES = [
    "range_expansion", "volume_z", "dollar_volume_z", "mom_scaled",
    "vol_surge", "parkinson", "garman_klass", "amihud_rank",
    "q_left", "q_asym", "upper_wick", "lower_wick", "ofi_z", "reversal_5",
]


class GapRiskOverlay(Strategy):
    key = "S2_MLRISK"
    name = "ML Gap-Risk Defensive Overlay"
    directional = False
    is_overlay = True                # consumed as a scale, not as a book

    def generate(self, md, fs, scfg) -> pd.DataFrame:
        """Return the exposure scale panel, one value per asset per bar."""
        prob = self.risk_probability(md, fs, scfg)
        cut = prob.quantile(scfg.s2_flag_quantile, axis=1)
        flagged = prob.gt(cut, axis=0) & prob.notna()
        scale = pd.DataFrame(1.0, index=prob.index, columns=prob.columns)
        scale[flagged] = scfg.s2_min_scale
        return scale
    def risk_probability(self, md, fs, scfg) -> pd.DataFrame:
        """Walk-forward out-of-sample P(severe gap down at t+1)."""
        cols = [c for c in _FEATURES if c in fs]
        idx = md.dates
        assets = md.assets
        label = (fs["gap_z"].shift(-1) <= scfg.s2_gap_event_z)

        stacked = pd.concat(
            [fs[c].stack(future_stack=True).rename(c) for c in cols], axis=1)
        y = label.stack(future_stack=True).rename("y")
        panel = stacked.join(y).dropna(subset=cols)
        if panel.empty:
            return pd.DataFrame(0.0, index=idx, columns=assets)

        bar_of = {d: i for i, d in enumerate(idx)}
        panel["bar"] = [bar_of.get(d, -1) for d in
                        panel.index.get_level_values(0)]
        panel = panel[panel["bar"] >= 0]

        out = pd.DataFrame(np.nan, index=idx, columns=assets)
        start = scfg.s2_train_bars
        X_all = panel[cols].to_numpy(dtype=float)
        y_all = panel["y"].to_numpy()
        bar_all = panel["bar"].to_numpy()

        for lo in range(start, len(idx), scfg.s2_refit_every):
            tr = (bar_all < lo - scfg.s2_embargo) & np.isfinite(y_all.astype(float))
            te = (bar_all >= lo) & (bar_all < lo + scfg.s2_refit_every)
            if tr.sum() < 200 or te.sum() == 0:
                continue
            ytr = y_all[tr].astype(int)
            if ytr.sum() < 20 or ytr.sum() == len(ytr):
                continue

            model = self._fit(X_all[tr], ytr)
            p = model.predict_proba(X_all[te])[:, 1]

            rows = panel.index[te]
            for (d, a), pv in zip(rows, p):
                out.at[d, a] = pv

        return out.ffill().fillna(0.0)

    @staticmethod
    def _fit(X, y):
        if _HAS_LGB:
            m = lgb.LGBMClassifier(
                n_estimators=180, learning_rate=0.05, num_leaves=15,
                min_child_samples=60, subsample=0.8, subsample_freq=1,
                colsample_bytree=0.8, reg_lambda=1.0,
                random_state=20260902, n_jobs=1, verbose=-1,
            )
        else:                                              # pragma: no cover
            m = GradientBoostingClassifier(random_state=20260902)
        m.fit(X, y)
        return m

class CrossSectionalReversal(Strategy):
    key = "S3_XSREV"
    name = "Cross-Sectional Volatility Premium and Reversal"
    directional = False

    def generate(self, md, fs, scfg) -> pd.DataFrame:
        score = fs["mom_volume"]

        liquid = fs["amihud_rank"] <= scfg.s3_amihud_exclude
        eligible = liquid & ~md.quarantine & score.notna()
        w = self.long_short_from_rank(score, scfg.s3_quantile, eligible)
        prem = fs["parkinson"].clip(lower=0.05) ** scfg.s3_vol_premium_power
        prem = prem.div(prem.mean(axis=1), axis=0).clip(0.5, 2.0)
        w = w * prem.reindex_like(w).fillna(1.0)

        return self.dollar_neutral(w)

class TsallisArbitrage(Strategy):
    key = "S4_TSALLIS"
    name = "Tsallis Asymmetric Tail-Risk Premium"
    directional = False

    def generate(self, md, fs, scfg) -> pd.DataFrame:
        atre = fs["atre"].ewm(halflife=scfg.s4_smooth, min_periods=21).mean()
        q_left = fs["q_left"].ewm(halflife=scfg.s4_smooth, min_periods=21).mean()
        score = (atre.rank(axis=1, pct=True) + q_left.rank(axis=1, pct=True)) / 2.0
        r = fs["returns"]
        mkt = fs["market_return"]
        win = scfg.s4_coskew_window
        m2 = (mkt ** 2).rolling(win, min_periods=win // 2).mean()
        num = r.mul(mkt ** 2, axis=0).rolling(win, min_periods=win // 2).mean()
        sd_r = r.rolling(win, min_periods=win // 2).std()
        coskew = num.div(sd_r * np.sqrt(m2.clip(lower=1e-16)), axis=0)
        coskew_rank = coskew.rank(axis=1, pct=True)

        eligible = score.notna() & ~md.quarantine
        w = self.long_short_from_rank(-score, scfg.s4_quantile, eligible)

        cr = coskew_rank.reindex_like(w)
        crowded = ((w > 0) & (cr < 0.2)) | ((w < 0) & (cr > 0.8))
        w = w.mask(crowded.fillna(False), w * 0.5)

        return self.dollar_neutral(w)

def _fit_garch11(e: np.ndarray) -> tuple:
    """
    Fit GARCH(1,1) to a zero-mean innovation series by Gaussian MLE.

    Implemented directly rather than pulled from `arch` so that the same
    recursion used for fitting is the one run forward in `_filter_garch`,
    with no ambiguity about scaling or mean models between the two. Falls
    back to RiskMetrics (0, 0.06, 0.94) if the optimiser misbehaves.
    """

    e = np.asarray(e, dtype=float)
    e = e[np.isfinite(e)]
    if len(e) < 60:
        return 0.0, 0.06, 0.94
    var0 = float(np.var(e))
    if not np.isfinite(var0) or var0 <= 0:
        return 0.0, 0.06, 0.94
    s = float(np.sqrt(var0))
    x = e / s                                    # scale-free fit, restored below

    def nll(theta):
        w, a, b = np.exp(theta)                  # positivity by construction
        if a + b >= 0.999:
            return 1e6
        h = np.empty(len(x))
        h[0] = 1.0
        for i in range(1, len(x)):
            h[i] = w + a * x[i - 1] ** 2 + b * h[i - 1]
        h = np.maximum(h, 1e-10)
        return 0.5 * np.sum(np.log(h) + x ** 2 / h)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = minimize(nll, np.log([0.05, 0.08, 0.88]), method="Nelder-Mead",
                         options={"maxiter": 400, "xatol": 1e-3, "fatol": 1e-3})
        w, a, b = np.exp(r.x)
        if not np.isfinite(w + a + b) or a + b >= 0.999 or a <= 0 or b <= 0:
            raise ValueError
        return float(w * var0), float(a), float(b)
    except Exception:
        return 0.0, 0.06, 0.94


def _filter_garch(e: np.ndarray, omega: float, alpha: float, beta: float,
                  h0: float) -> np.ndarray:
    """
    Run the frozen variance recursion forward.

    h[t] uses innovations only up to t-1, so the conditional volatility
    available at bar t for sizing a threshold never contains bar t's own
    surprise.
    """
    n = len(e)
    h = np.empty(n)
    h[0] = h0
    for i in range(1, n):
        prev = e[i - 1]
        prev = prev ** 2 if np.isfinite(prev) else h[i - 1]
        h[i] = omega + alpha * prev + beta * h[i - 1]
    return np.maximum(h, 1e-14)


class GarchPairs(Strategy):
    key = "S5_PAIRS"
    name = "GARCH Cointegration Pairs"
    directional = False

    def generate(self, md, fs, scfg) -> pd.DataFrame:
        logp = fs["log_price"]
        idx = logp.index
        assets = list(logp.columns)
        w = pd.DataFrame(0.0, index=idx, columns=assets)

        n = len(idx)
        start = scfg.s5_formation
        if n <= start + 2:
            return w

        for lo in range(start, n, scfg.s5_refit_every):
            form = logp.iloc[lo - scfg.s5_formation:lo]
            hi = min(lo + scfg.s5_refit_every, n)
            pairs = self._select_pairs(form, scfg)
            if not pairs:
                continue

            per_pair = 1.0 / len(pairs)
            for (a, b, hedge, mu) in pairs:
                spread_form = (form[a] - hedge * form[b] - mu).to_numpy(dtype=float)
                omega, alpha, beta = _fit_garch11(spread_form)
                h0 = float(np.var(spread_form[np.isfinite(spread_form)]) or 1e-8)
                seg = logp.iloc[lo - 60:hi]
                e = (seg[a] - hedge * seg[b] - mu).to_numpy(dtype=float)
                h = _filter_garch(e, omega, alpha, beta, h0)
                z = e / np.sqrt(h)

                pos = self._position_path(z, scfg)[60:]
                w.iloc[lo:hi, w.columns.get_loc(a)] += per_pair * pos * 0.5
                w.iloc[lo:hi, w.columns.get_loc(b)] -= per_pair * pos * 0.5
        w = w.where(~md.quarantine, 0.0)
        return self.dollar_neutral(w)
    @staticmethod
    def _select_pairs(form: pd.DataFrame, scfg) -> list:
        """Engle-Granger: OLS hedge ratio, then ADF on the residual."""
        cols = [c for c in form.columns if form[c].notna().sum() > len(form) * 0.9]
        found = []
        for a, b in itertools.combinations(cols, 2):
            x, y = form[b].to_numpy(dtype=float), form[a].to_numpy(dtype=float)
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() < 100:
                continue
            xv, yv = x[ok], y[ok]
            vx = xv.var()
            if vx <= 0:
                continue
            hedge = float(np.cov(yv, xv, bias=True)[0, 1] / vx)
            mu = float(yv.mean() - hedge * xv.mean())
            resid = yv - hedge * xv - mu
            if not np.isfinite(resid).all() or resid.std() <= 0:
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    p = adfuller(resid, maxlag=1, regression="c",
                                 autolag=None)[1]
            except Exception:
                continue
            if p < scfg.s5_adf_pvalue:
                found.append((p, a, b, hedge, mu))
        found.sort(key=lambda t: t[0])
        return [(a, b, h, m) for _, a, b, h, m in found[:scfg.s5_max_pairs]]

    @staticmethod
    def _position_path(z: np.ndarray, scfg) -> np.ndarray:
        """
        State machine on the conditional z-score.

        Enter against the divergence past `s5_entry_z`, scale out at
        `s5_exit_z`, and abandon past `s5_stop_z` -- a spread that keeps
        widening is evidence the cointegrating relationship has broken, and
        the correct response to a broken model is to stop trading it, not to
        add to the position.
        """
        pos = np.zeros(len(z))
        cur = 0.0
        for i, zi in enumerate(z):
            if not np.isfinite(zi):
                pos[i] = cur
                continue
            if cur == 0.0:
                if zi > scfg.s5_entry_z:
                    cur = -1.0                       # spread rich: short it
                elif zi < -scfg.s5_entry_z:
                    cur = 1.0                        # spread cheap: long it
            else:
                if abs(zi) > scfg.s5_stop_z:
                    cur = 0.0
                elif cur > 0 and zi > -scfg.s5_exit_z:
                    cur = 0.0
                elif cur < 0 and zi < scfg.s5_exit_z:
                    cur = 0.0
            pos[i] = cur
        return pos

class TurtleBreakout(Strategy):
    key = "S6_TURTLE"
    name = "Turtle Breakout with Pyramiding"
    directional = True

    def generate(self, md, fs, scfg) -> pd.DataFrame:
        px = md.adj_close
        atr = fs["atr"]

        entry_n = max(int(round(scfg.s6_entry_window)), 5)
        exit_n = max(int(round(scfg.s6_exit_window)), 3)
        hi = px.rolling(entry_n, min_periods=entry_n).max().shift(1)
        lo = px.rolling(entry_n, min_periods=entry_n).min().shift(1)
        exit_lo = px.rolling(exit_n, min_periods=exit_n).min().shift(1)
        exit_hi = px.rolling(exit_n, min_periods=exit_n).max().shift(1)

        long_break = (px > hi) & ~md.quarantine
        short_break = (px < lo) & ~md.quarantine

        units = self._pyramid(
            px.to_numpy(dtype=float), atr.to_numpy(dtype=float),
            long_break.fillna(False).to_numpy(),
            short_break.fillna(False).to_numpy(),
            exit_lo.to_numpy(dtype=float), exit_hi.to_numpy(dtype=float),
            max_units=max(int(round(scfg.s6_max_units)), 1),
            add_step=float(scfg.s6_add_step),
            stop_mult=float(scfg.s6_stop_mult),
            short_scale=float(scfg.s6_short_scale),
        )
        units = pd.DataFrame(units, index=px.index, columns=px.columns)
        n_pct = (atr / px.replace(0.0, np.nan)).clip(lower=0.002, upper=0.25)
        unit_w = scfg.s6_risk_fraction / (scfg.s6_stop_mult * n_pct)
        w = units * unit_w.reindex_like(units).fillna(0.0)

        return self.unit_book(w)
    @staticmethod
    def _pyramid(px, atr, long_break, short_break, exit_lo, exit_hi,
                 max_units, add_step, stop_mult, short_scale):
        """
        Signed unit count per bar per asset.

        The state machine is inherently sequential -- how many units are held
        depends on the entire path of adds and stops since the breakout -- so
        it is a genuine loop rather than a vectorisation that was not
        attempted. It runs once per sleeve build, over bars x assets.
        """
        n, m = px.shape
        out = np.zeros((n, m))

        for j in range(m):
            side = 0             # +1 long, -1 short, 0 flat
            units = 0
            last_entry = np.nan
            stop = np.nan

            for i in range(n):
                p = px[i, j]
                N = atr[i, j]
                if not np.isfinite(p) or not np.isfinite(N) or N <= 0:
                    out[i, j] = side * units
                    continue
                if side != 0:
                    stopped = (p <= stop) if side > 0 else (p >= stop)
                    channel_exit = ((np.isfinite(exit_lo[i, j]) and p < exit_lo[i, j])
                                    if side > 0 else
                                    (np.isfinite(exit_hi[i, j]) and p > exit_hi[i, j]))
                    if stopped or channel_exit:
                        side, units, last_entry, stop = 0, 0, np.nan, np.nan
                    elif units < max_units:
                        advance = (p - last_entry) * side
                        if advance >= add_step * N:
                            units += 1
                            last_entry = p
                            stop = p - side * stop_mult * N
                if side == 0:
                    if long_break[i, j]:
                        side, units, last_entry = 1, 1, p
                        stop = p - stop_mult * N
                    elif short_break[i, j] and short_scale > 0:
                        side, units, last_entry = -1, 1, p
                        stop = p + stop_mult * N

                out[i, j] = side * units * (short_scale if side < 0 else 1.0)
        return out

class MRWBreakout(Strategy):
    key = "S7_MRW"
    name = "Multifractal Random Walk Breakout"
    directional = True

    def generate(self, md, fs, scfg) -> pd.DataFrame:
        h2 = fs["H2"]
        surge = fs["vol_surge"]
        trend = fs["mom_scaled"]
        persist = fs["vol_persistence"]

        hi, lo = scfg.s7_hurst_trend, scfg.s7_hurst_revert
        surging = (surge >= scfg.s7_vol_surge)
        tradeable = (surging & (trend.abs() > scfg.s7_min_abs_trend)
                     & ~md.quarantine).fillna(False)

        persistent = (h2 >= hi) & tradeable          # follow the move
        antipersistent = (h2 <= lo) & tradeable      # fade the move
        regime = (persistent.astype(float) - antipersistent.astype(float))
        direction = (np.sign(trend) * regime).fillna(0.0)
        decay = persist.rank(axis=1, pct=True).fillna(0.5)
        width = scfg.s7_trail_stop * (0.5 + 0.5 * decay)
        occupancy = self._trail(direction, md.adj_close, width)

        inv_vol = 1.0 / fs["realised_vol"].clip(lower=0.05)
        w = occupancy * inv_vol.reindex_like(occupancy).fillna(0.0)
        return self.unit_book(w)
    @staticmethod
    def _trail(direction: pd.DataFrame, price: pd.DataFrame,
               width: pd.DataFrame) -> pd.DataFrame:
        """
        Hold while the signal persists, exit on the adaptive trailing stop.

        A stopped-out name is barred until the signal turns off and back on,
        so the sleeve cannot immediately re-enter the position it just closed.
        """
        d = direction.to_numpy(dtype=float)
        px = price.to_numpy(dtype=float)
        wd = width.reindex_like(direction).to_numpy(dtype=float)
        out = np.zeros_like(d)
        n, m = d.shape

        for j in range(m):
            side = 0.0
            peak = np.nan
            barred = False
            for i in range(n):
                sig = d[i, j]
                p = px[i, j]
                if side != 0.0:
                    if np.isfinite(p):
                        peak = p if not np.isfinite(peak) else (
                            max(peak, p) if side > 0 else min(peak, p))
                        give = side * (p / peak - 1.0) if peak > 0 else 0.0
                        w_ij = wd[i, j] if np.isfinite(wd[i, j]) else 0.12
                        if give <= -abs(w_ij) or sig == 0.0 or np.sign(sig) != side:
                            side, peak = 0.0, np.nan
                            barred = sig != 0.0    # bar only if signal still on
                if side == 0.0:
                    if sig == 0.0:
                        barred = False
                    elif not barred:
                        side, peak = float(np.sign(sig)), p
                out[i, j] = side
        return pd.DataFrame(out, index=direction.index, columns=direction.columns)
ALPHA_SLEEVES = [GapReversion, CrossSectionalReversal,
                 TsallisArbitrage, GarchPairs, TurtleBreakout,
                 MRWBreakout]
OVERLAY = GapRiskOverlay
def shrunk_covariance(returns: np.ndarray) -> np.ndarray:
    """
    Ledoit-Wolf shrinkage of a sample covariance matrix.

    Falls back to the sample covariance with a ridge if the estimator cannot
    run (too few observations, a constant column), because an allocator that
    raises in production is worse than one that is slightly less efficient.
    """
    X = np.asarray(returns, dtype=float)
    X = X[np.isfinite(X).all(axis=1)]
    k = X.shape[1] if X.ndim == 2 else 0
    if X.ndim != 2 or len(X) < max(20, k + 2):
        return np.eye(max(k, 1))
    try:
        lw = LedoitWolf(assume_centered=False).fit(X)
        S = lw.covariance_
        if not np.isfinite(S).all():
            raise ValueError
        return S
    except Exception:                                      # pragma: no cover
        S = np.cov(X, rowvar=False)
        return S + np.eye(k) * 1e-8


def risk_parity_prior(cov: np.ndarray, floor: float = 1e-10) -> np.ndarray:
    """Inverse-volatility weights from a covariance matrix, normalised."""
    v = np.sqrt(np.clip(np.diag(cov), floor, None))
    w = 1.0 / v
    s = w.sum()
    return w / s if s > 0 else np.full(len(v), 1.0 / max(len(v), 1))
def mean_cvar_allocate(scenarios: np.ndarray, prev: np.ndarray | None,
                       cfg: PortfolioConfig) -> np.ndarray:
    """
    Solve the Rockafellar-Uryasev Mean-CVaR program for sleeve weights.

    `scenarios` is [T x K] of realised sleeve returns; `prev` is the previous
    allocation, used for the L1 turnover penalty. Returns a length-K vector on
    the simplex.
    """
    R = np.asarray(scenarios, dtype=float)
    R = R[np.isfinite(R).all(axis=1)]
    T, K = R.shape if R.ndim == 2 else (0, 0)
    equal = np.full(max(K, 1), 1.0 / max(K, 1))
    if T < 30 or K == 0:
        return equal

    prev = equal if prev is None else np.asarray(prev, dtype=float)
    mu = R.mean(axis=0)
    tail = 1.0 / ((1.0 - cfg.cvar_beta) * T)
    n = 2 * K + 1 + T
    ia, iz, iu, idd = 0, K, K + 1, K + 1 + T

    c = np.zeros(n)
    c[ia:ia + K] = -cfg.cvar_risk_aversion * mu          # maximise mean
    c[iz] = 1.0                                          # CVaR: z term
    c[iu:iu + T] = tail                                  # CVaR: tail average
    c[idd:idd + K] = cfg.cvar_turnover_penalty           # L1 turnover
    A1 = np.zeros((T, n))
    A1[:, ia:ia + K] = -R
    A1[:, iz] = -1.0
    A1[np.arange(T), iu + np.arange(T)] = -1.0
    b1 = np.zeros(T)
    A2 = np.zeros((K, n))
    A2[np.arange(K), ia + np.arange(K)] = 1.0
    A2[np.arange(K), idd + np.arange(K)] = -1.0
    b2 = prev.copy()
    A3 = np.zeros((K, n))
    A3[np.arange(K), ia + np.arange(K)] = -1.0
    A3[np.arange(K), idd + np.arange(K)] = -1.0
    b3 = -prev.copy()

    A_ub = np.vstack([A1, A2, A3])
    b_ub = np.concatenate([b1, b2, b3])

    A_eq = np.zeros((1, n))
    A_eq[0, ia:ia + K] = 1.0
    b_eq = np.array([1.0])

    bounds = ([(0.0, cfg.cvar_max_sleeve)] * K
              + [(None, None)]
              + [(0.0, None)] * T
              + [(0.0, None)] * K)

    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                      bounds=bounds, method="highs")
        if not res.success:
            return equal
        a = np.clip(res.x[ia:ia + K], 0.0, None)
    except Exception:                                      # pragma: no cover
        return equal

    s = a.sum()
    if not np.isfinite(s) or s <= 0:
        return equal

    a = a / s
    prior = risk_parity_prior(shrunk_covariance(R))
    a = (1.0 - cfg.shrink_to_equal) * a + cfg.shrink_to_equal * prior
    a = np.clip(a, cfg.cvar_min_sleeve, cfg.cvar_max_sleeve)
    return a / a.sum()
def allocation_path(sleeve_returns: pd.DataFrame, cfg: PortfolioConfig,
                    warmup: int) -> pd.DataFrame:
    """
    Roll the Mean-CVaR program forward through history.

    At each refit the LP sees only the trailing `cvar_window` bars of REALISED
    sleeve returns, and the allocation it produces is used for the following
    `cvar_refit_every` bars and no earlier ones. The result is then lagged one
    further bar before use, so an allocation dated t is formed from returns
    through t-1 and traded at t+1.

    Before `warmup` the allocation is equal-weight: there is not yet enough
    realised history to prefer one sleeve over another, and pretending
    otherwise is the most common way a walk-forward backtest smuggles in
    hindsight.
    """
    idx = sleeve_returns.index
    keys = list(sleeve_returns.columns)
    K = len(keys)
    out = pd.DataFrame(np.nan, index=idx, columns=keys)
    out.iloc[:min(warmup, len(idx))] = 1.0 / K

    R = sleeve_returns.to_numpy(dtype=float)
    prev = np.full(K, 1.0 / K)

    for lo in range(warmup, len(idx), cfg.cvar_refit_every):
        window = R[max(0, lo - cfg.cvar_window):lo]        # strictly trailing
        a = mean_cvar_allocate(window, prev, cfg)
        hi = min(lo + cfg.cvar_refit_every, len(idx))
        out.iloc[lo:hi] = a
        prev = a
    return out.ffill().shift(1).fillna(1.0 / K)

class PortfolioBuilder:
    """Assembles the traded book from the sleeve layer."""

    def __init__(self, cfg: NexusConfig, verbose: bool = True,
                 sleeve_cache: dict | None = None,
                 exclude: set | None = None):
        """
        `sleeve_cache` maps a sleeve key to an already-computed book, letting a
        caller skip regeneration for sleeves whose parameters it is holding
        fixed. It is a pure memoisation hook -- passing a cache must produce
        exactly the book the sleeve would have produced -- and exists because
        the search in ..py evaluates thousands of portfolio configurations
        against a fixed set of expensive sleeves.
        """
        self.cfg = cfg
        self.verbose = verbose
        self.sleeve_cache = sleeve_cache or {}
        self.exclude = set(exclude or ())
        self.diagnostics: dict = {}
    def build(self, md, fs) -> pd.DataFrame:
        cfg = self.cfg
        pcfg = cfg.portfolio
        rets = md.adj_returns
        books: dict = {}
        raw: dict = {}
        scales = dict(pcfg.sleeve_scale)
        for cls in ALPHA_SLEEVES:
            s = cls()
            if s.key in self.exclude:
                continue
            cached = self.sleeve_cache.get(s.key)
            book = (cached if cached is not None
                    else s.generate(md, fs, cfg.strategy).fillna(0.0))
            raw[s.key] = book
            k = float(scales.get(s.key, 1.0))
            books[s.key] = book if k == 1.0 else book * k
        self.diagnostics["books"] = books
        self.diagnostics["raw_books"] = raw
        gross = pd.DataFrame({k: sleeve_returns(w, rets) for k, w in books.items()})
        turn = pd.DataFrame({k: w.diff().abs().sum(axis=1) for k, w in books.items()})
        net = (gross - turn * pcfg.fee * pcfg.sleeve_cost_scale).fillna(0.0)
        self.diagnostics["sleeve_gross"] = gross
        self.diagnostics["sleeve_net"] = net
        self.diagnostics["sleeve_turnover"] = turn
        alpha = (allocation_path(net, pcfg, cfg.wfo.warmup) if len(books)
                 else pd.DataFrame(index=md.dates))
        self.diagnostics["allocation"] = alpha
        combined = None
        for k, w in books.items():
            contrib = w.mul(alpha[k], axis=0)
            combined = contrib if combined is None else combined + contrib
        if combined is None:                       # every sleeve excluded
            combined = pd.DataFrame(0.0, index=md.dates, columns=md.assets)
        combined = combined.fillna(0.0) * pcfg.sleeve_tilt
        if pcfg.use_overlay and "S2_MLRISK" not in self.exclude:
            scale = self.sleeve_cache.get("S2_MLRISK")
            if scale is None:
                scale = OVERLAY().generate(md, fs, cfg.strategy)
            scale = scale.reindex_like(combined)
            combined = combined * scale.fillna(1.0)
            self.diagnostics["overlay_scale"] = scale
        routed = {a: k for a, k in pcfg.asset_sleeve_override
                  if a in combined.columns and k in books}
        for asset, key in routed.items():
            combined[asset] = books[key][asset] * pcfg.override_scale
        self.diagnostics["routed"] = routed
        if pcfg.net_long_target > 0:
            combined = self._deploy_core(combined, md, fs, pcfg,
                                         skip=set(routed))
        combined = combined.ewm(halflife=pcfg.score_halflife,
                                min_periods=1).mean().fillna(0.0)
        n_names = max(len(md.assets), 1)
        cap = pcfg.per_name_cap_mult / n_names
        combined = combined.clip(-cap, cap)
        combined = self._trailing_stop(combined, md, pcfg)
        combined = combined.ewm(halflife=pcfg.weight_halflife,
                                min_periods=1).mean().fillna(0.0)
        book = self._project(combined, pcfg)
        book = book.where(md.close.notna(), 0.0).fillna(0.0)
        self.diagnostics["book"] = book

        if self.verbose:
            self._report(alpha, book)
        return book
    def _deploy_core(self, combined, md, fs, pcfg, skip: set | None = None):
        """
        Inverse-volatility long core over the eligible universe.

        Sizing is 1/sigma using the Ledoit-Wolf shrunk covariance diagonal
        rather than raw realised volatility, so a name with a single noisy
        window is not handed an outsized weight. The core is added to the
        sleeve tilts rather than replacing them, so a name the sleeves like
        gets more than the core alone would give it and a name they dislike
        gets less -- the tilts fund themselves out of the core instead of
        paying a second spread.
        """
        vol = fs["realised_vol"].clip(lower=0.08, upper=2.0)
        eligible = md.close.notna() & ~md.quarantine & vol.notna()
        for asset in (skip or ()):                 # routed names get no core
            if asset in eligible.columns:
                eligible[asset] = False

        if pcfg.core_scheme == "equal":
            inv = eligible.astype(float)
        else:
            inv = (1.0 / vol).where(eligible, 0.0)
        t = float(pcfg.core_trend_tilt)
        if t > 0 and "core_quality" in fs:
            q = fs["core_quality"].reindex_like(inv).fillna(0.5)
            inv = inv * (1.0 + t * (2.0 * q - 1.0)).clip(lower=0.0)

        core = inv.div(inv.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
        core = core * pcfg.net_long_target
        shortfall = (pcfg.net_long_target - combined.sum(axis=1)).clip(lower=0.0)
        scale = (shortfall / pcfg.net_long_target).clip(0.0, 1.0)
        applied = core.mul(scale, axis=0)
        self.diagnostics["core"] = applied
        return combined + applied
    @staticmethod
    def _trailing_stop(w, md, pcfg):
        """
        Close a name that has given back `trailing_stop` from its peak since
        the position was opened, and bar it for `stop_cooldown` bars.

        Evaluated on the ADJUSTED series. On raw prices a 2-for-1 split is an
        instantaneous -50% and would stop out every holder of a perfectly
        healthy company on a bar where the engine books no loss whatsoever.
        """
        px = md.adj_close.to_numpy(dtype=float)
        wv = w.to_numpy(dtype=float).copy()
        n, m = wv.shape
        for j in range(m):
            peak = np.nan
            side = 0.0
            cooldown = 0
            for i in range(n):
                if cooldown > 0:
                    cooldown -= 1
                    wv[i, j] = 0.0
                    continue
                target = wv[i, j]
                p = px[i, j]
                if side != 0.0 and np.isfinite(p):
                    peak = p if not np.isfinite(peak) else (
                        max(peak, p) if side > 0 else min(peak, p))
                    if peak > 0:
                        give = side * (p / peak - 1.0)
                        if give <= -pcfg.trailing_stop:
                            wv[i, j] = 0.0
                            side, peak = 0.0, np.nan
                            cooldown = pcfg.stop_cooldown
                            continue
                new_side = float(np.sign(target))
                if new_side != side:
                    side, peak = new_side, p
        return pd.DataFrame(wv, index=w.index, columns=w.columns)
    @staticmethod
    def _project(w, pcfg):
        """
        Project onto the feasible set.

        The rulebook caps long and short exposure independently, each at 100%
        of portfolio value, so the two sides are scaled independently -- a
        single gross scaling would distort the intended long/short balance
        whenever only one side breached.

        The caps applied here are tighter than the rulebook's: the engine
        re-projects on live prices before every fill, and leaving headroom
        here means the engine's projection is rarely the binding one, which
        keeps realised exposure close to intent.
        """
        longs = w.clip(lower=0.0)
        shorts = -w.clip(upper=0.0)
        lsum = longs.sum(axis=1)
        ssum = shorts.sum(axis=1)

        max_long = 1.0 - pcfg.exposure_margin
        lscale = (max_long / lsum.replace(0.0, np.nan)).clip(upper=1.0).fillna(1.0)
        sscale = (pcfg.max_short / ssum.replace(0.0, np.nan)).clip(upper=1.0).fillna(1.0)
        out = longs.mul(lscale, axis=0) - shorts.mul(sscale, axis=0)
        if pcfg.redeploy_idle > 0.0:
            target = min(pcfg.net_long_target, max_long)
            lnow = out.clip(lower=0.0).sum(axis=1)
            want = (target / lnow.replace(0.0, np.nan)).clip(lower=1.0)
            want = (1.0 + pcfg.redeploy_idle * (want - 1.0)).fillna(1.0)
            want = np.minimum(want, (max_long / lnow.replace(0.0, np.nan))
                              .clip(lower=1.0).fillna(1.0))
            out = out.clip(lower=0.0).mul(want, axis=0) + out.clip(upper=0.0)
        gross = out.abs().sum(axis=1)
        gscale = (pcfg.max_gross / gross.replace(0.0, np.nan)).clip(upper=1.0).fillna(1.0)
        return out.mul(gscale, axis=0)
    def _report(self, alpha, book):
        mean_alloc = alpha.mean()
        print("[portfolio] mean Mean-CVaR allocation across sleeves:")
        for k, v in mean_alloc.sort_values(ascending=False).items():
            print("            %-12s %5.1f%%" % (k, v * 100))
        print("[portfolio] book: gross %.2f | net %+.2f | long %.2f | short %.2f"
              % (book.abs().sum(axis=1).mean(), book.sum(axis=1).mean(),
                 book.clip(lower=0).sum(axis=1).mean(),
                 -book.clip(upper=0).sum(axis=1).mean()))
        print("[portfolio] mean daily turnover %.3f"
              % book.diff().abs().sum(axis=1).mean())

@dataclass
class BacktestResult:
    equity: pd.Series
    cash: pd.Series
    weights: pd.DataFrame           # realised weights, marked at each close
    positions: pd.DataFrame         # shares held
    trades: pd.DataFrame
    gross_exposure: pd.Series
    net_exposure: pd.Series
    long_exposure: pd.Series
    short_exposure: pd.Series
    fees_paid: pd.Series
    splits: pd.DataFrame = field(default_factory=pd.DataFrame)
    meta: dict = field(default_factory=dict)

    @property
    def returns(self) -> pd.Series:
        return self.equity.pct_change().fillna(0.0)


class SimulatedExchange:
    """Target-weight portfolio simulator with share granularity and costs."""

    def __init__(self, prices_open: pd.DataFrame, prices_close: pd.DataFrame,
                 target_weights: pd.DataFrame, initial_capital: float = 2_000_000.0,
                 fee: float = 0.001, no_trade_band: float = 0.015,
                 exposure_margin: float = 0.03, exposure_tolerance: float = 0.02,
                 split_threshold: float = 0.30, allow_fractional: bool = False,
                 exposure_trigger: float = 0.99, exposure_reduce_to: float = 0.97,
                 verbose: bool = False):
        common = prices_close.index.intersection(target_weights.index)
        self.assets = list(prices_close.columns)
        self.open = prices_open.loc[common, self.assets]
        self.close = prices_close.loc[common, self.assets]
        self.targets = target_weights.loc[common, self.assets].fillna(0.0)
        self.dates = common
        self.initial_capital = float(initial_capital)
        self.fee = float(fee)
        self.no_trade_band = float(no_trade_band)
        self.exposure_margin = float(exposure_margin)
        self.exposure_tolerance = float(exposure_tolerance)
        self.split_threshold = float(split_threshold)
        self.allow_fractional = allow_fractional
        self.exposure_trigger = float(exposure_trigger)
        self.exposure_reduce_to = float(exposure_reduce_to)
        self.verbose = verbose
    def run(self) -> BacktestResult:
        n_dates = len(self.dates)
        assets = self.assets
        n_assets = len(assets)

        cash = self.initial_capital
        shares = np.zeros(n_assets)

        eq_hist = np.zeros(n_dates)
        cash_hist = np.zeros(n_dates)
        pos_hist = np.zeros((n_dates, n_assets))
        w_hist = np.zeros((n_dates, n_assets))
        fee_hist = np.zeros(n_dates)
        long_hist = np.zeros(n_dates)
        short_hist = np.zeros(n_dates)
        trades = []
        split_events = []
        guard_events = []

        open_mat = self.open.to_numpy(dtype=float)
        close_mat = self.close.to_numpy(dtype=float)
        tgt_mat = self.targets.to_numpy(dtype=float)

        pending = None                       # target set at t-1, executed at t

        for i in range(n_dates):
            px_open = open_mat[i]
            px_close = close_mat[i]
            prev_close = close_mat[i - 1] if i > 0 else None
            if prev_close is not None:
                shares, ev = self._apply_splits(shares, prev_close, px_open,
                                                self.dates[i], "open")
                split_events.extend(ev)
            if pending is not None:
                cash, shares, fees, fills = self._execute(
                    cash, shares, px_open, pending, self.dates[i])
                fee_hist[i] = fees
                trades.extend(fills)
            cash, shares, gfee, gfills, gev = self._enforce_exposure(
                cash, shares, px_open, self.dates[i], "open")
            fee_hist[i] += gfee
            trades.extend(gfills)
            guard_events.extend(gev)

            if prev_close is not None:
                shares, ev = self._apply_splits(shares, px_open, px_close,
                                                self.dates[i], "intraday")
                split_events.extend(ev)
            cash, shares, gfee, gfills, gev = self._enforce_exposure(
                cash, shares, px_close, self.dates[i], "close")
            fee_hist[i] += gfee
            trades.extend(gfills)
            guard_events.extend(gev)
            valid = np.isfinite(px_close)
            mark = np.where(valid, px_close, 0.0)
            holdings = shares * mark
            equity = cash + holdings.sum()

            eq_hist[i] = equity
            cash_hist[i] = cash
            pos_hist[i] = shares
            long_hist[i] = holdings[holdings > 0].sum()
            short_hist[i] = -holdings[holdings < 0].sum()
            w_hist[i] = holdings / equity if equity > 0 else 0.0

            if equity <= 0:                  # ruin: stop rather than report fiction
                eq_hist[i:] = equity
                if self.verbose:
                    print("[engine] equity non-positive at %s"
                          % self.dates[i].date())
                break
            pending = tgt_mat[i]

        res = BacktestResult(
            equity=pd.Series(eq_hist, index=self.dates, name="equity"),
            cash=pd.Series(cash_hist, index=self.dates, name="cash"),
            weights=pd.DataFrame(w_hist, index=self.dates, columns=assets),
            positions=pd.DataFrame(pos_hist, index=self.dates, columns=assets),
            trades=pd.DataFrame(trades) if trades else pd.DataFrame(
                columns=["date", "asset", "side", "shares", "price", "value", "fee"]),
            gross_exposure=pd.Series((long_hist + short_hist) / eq_hist,
                                     index=self.dates, name="gross"),
            net_exposure=pd.Series((long_hist - short_hist) / eq_hist,
                                   index=self.dates, name="net"),
            long_exposure=pd.Series(long_hist / eq_hist, index=self.dates, name="long"),
            short_exposure=pd.Series(short_hist / eq_hist, index=self.dates, name="short"),
            fees_paid=pd.Series(fee_hist, index=self.dates, name="fees"),
            splits=pd.DataFrame(split_events) if split_events else pd.DataFrame(),
            meta={"initial_capital": self.initial_capital, "fee": self.fee,
                  "guard_events": (pd.DataFrame(guard_events) if guard_events
                                   else pd.DataFrame()),
                  "guard_trigger": self.exposure_trigger,
                  "guard_reduce_to": self.exposure_reduce_to},
        )
        self._assert_constraints(res)
        return res
    def _apply_splits(self, shares, px_before, px_after, date, stage):
        events = []
        if px_before is None:
            return shares, events
        ratio = np.divide(px_after, px_before,
                          out=np.full_like(px_after, np.nan),
                          where=np.isfinite(px_before) & (px_before > 0))
        hit = (np.isfinite(ratio) & (ratio > 0)
               & (ratio < 1.0 - self.split_threshold) & (shares != 0.0))
        for j in np.flatnonzero(hit):
            old = shares[j]
            new = np.sign(old) * np.ceil(abs(old) / ratio[j])
            shares[j] = new
            events.append({"date": date, "asset": self.assets[j], "stage": stage,
                           "price_before": px_before[j], "price_after": px_after[j],
                           "ratio": ratio[j], "shares_before": old,
                           "shares_after": new})
        return shares, events
    def _enforce_exposure(self, cash, shares, px, date, stage):
        """
        Force the book back inside the rulebook's exposure limits.

        The rule is stated pointwise -- "at any point in time, the total
        absolute market value of all short positions cannot exceed the current
        portfolio value" -- so it has to be checked whenever the book is
        marked, not only when it is traded.

        Why a separate mechanism from the target projection: the projection
        constrains what we ASK for at the next open. It cannot constrain what
        the market does to a position we already hold. A short squeeze is the
        dangerous case because it moves both sides of the ratio the wrong way
        at once -- the shorts gain market value while the equity funding them
        falls -- so |short MV| / equity can cross 1.0 with no trade having
        taken place. The same applies to longs when the book is levered by a
        cash drawdown.

        The response is a pro-rata reduction of the offending side, sized to
        land at `exposure_reduce_to` rather than exactly at the trigger, so the
        guard does not re-fire on the next bar for a rounding error. Covering
        a short BUYS shares back (spending cash); trimming a long SELLS. Both
        leave equity unchanged apart from the fee, so the ratio falls.

        Causality: this uses only bar `date`'s own observed price, which is
        knowable at the moment it is acted on -- a market-on-close order at the
        close, or a market order at the open. It is a risk reduction and can
        never open or extend a position, so it cannot manufacture return.
        """
        valid = np.isfinite(px) & (px > 0)
        mv = np.where(valid, shares * px, 0.0)
        equity = cash + mv.sum()
        if equity <= 0:
            return cash, shares, 0.0, [], []

        long_mv = mv[mv > 0].sum()
        short_mv = -mv[mv < 0].sum()
        fills, events = [], []
        total_fee = 0.0

        for side in ("short", "long"):
            exposure = (short_mv if side == "short" else long_mv) / equity
            if exposure <= self.exposure_trigger:
                continue
            keep = self.exposure_reduce_to / exposure
            idx = np.flatnonzero((mv < 0) if side == "short" else (mv > 0))
            reduced = 0.0
            for j in idx:
                target_sh = shares[j] * keep
                if not self.allow_fractional:
                    target_sh = (np.ceil(target_sh) if shares[j] < 0
                                 else np.trunc(target_sh))
                q = target_sh - shares[j]
                if q == 0.0:
                    continue
                price = px[j]
                gross = abs(q) * price
                fee = gross * self.fee
                cash += (-gross - fee) if q > 0 else (gross - fee)
                shares[j] += q
                total_fee += fee
                reduced += gross
                fills.append({"date": date, "asset": self.assets[j],
                              "side": "BUY" if q > 0 else "SELL",
                              "shares": abs(q), "price": price,
                              "value": gross, "fee": fee})

            events.append({"date": date, "stage": stage, "side": side,
                           "exposure_before": exposure,
                           "target": self.exposure_reduce_to,
                           "value_reduced": reduced})

            mv = np.where(valid, shares * px, 0.0)
            equity = cash + mv.sum()
            long_mv = mv[mv > 0].sum()
            short_mv = -mv[mv < 0].sum()

        return cash, shares, total_fee, fills, events
    def _execute(self, cash, shares, px, target_w, date):
        """Trade toward `target_w` at `px`, honouring the exposure caps."""
        valid = np.isfinite(px) & (px > 0)
        equity = cash + float(np.nansum(np.where(valid, shares * px, 0.0)))
        if equity <= 0:
            return cash, shares, 0.0, []

        w = np.where(valid, target_w, 0.0)
        long_w = np.clip(w, 0.0, None).sum()
        short_w = -np.clip(w, None, 0.0).sum()
        safety = 1.0 - self.exposure_margin
        if long_w > safety:
            w = np.where(w > 0, w * safety / long_w, w)
        if short_w > safety:
            w = np.where(w < 0, w * safety / short_w, w)

        target_val = w * equity
        target_shares = np.where(valid, target_val / np.where(valid, px, 1.0), 0.0)
        if not self.allow_fractional:
            target_shares = np.trunc(target_shares)
        delta = target_shares - shares
        drift = np.abs(delta * px) / equity
        keep = drift >= self.no_trade_band
        reducing = np.abs(target_shares) < np.abs(shares)
        banded = np.where(keep | reducing, delta, 0.0)
        held_long = np.clip(shares + banded, 0.0, None) @ np.where(valid, px, 0.0)
        want_long = np.clip(target_shares, 0.0, None) @ np.where(valid, px, 0.0)
        if abs(held_long - want_long) > self.exposure_tolerance * equity:
            banded = delta

        delta = np.where(valid, banded, 0.0)

        fills = []
        total_fee = 0.0
        for j in np.argsort(delta):
            q = delta[j]
            if q == 0.0:
                continue
            price = px[j]
            if q > 0:
                affordable = cash / (price * (1.0 + self.fee))
                if not self.allow_fractional:
                    affordable = np.trunc(affordable)
                q = min(q, max(affordable, 0.0))
                if q <= 0:
                    continue
            gross = abs(q) * price
            fee = gross * self.fee
            cash += (-gross - fee) if q > 0 else (gross - fee)
            shares[j] += q
            total_fee += fee
            fills.append({"date": date, "asset": self.assets[j],
                          "side": "BUY" if q > 0 else "SELL",
                          "shares": abs(q), "price": price,
                          "value": gross, "fee": fee})
        return cash, shares, total_fee, fills
    @staticmethod
    def _assert_constraints(res: BacktestResult, tol: float = 1e-6):
        """Fail loudly on any breach; silence here costs 10% of the score."""
        bad_long = res.long_exposure[res.long_exposure > 1.0 + tol]
        bad_short = res.short_exposure[res.short_exposure > 1.0 + tol]
        if len(bad_long):
            raise AssertionError(
                "long exposure exceeded portfolio value on %d bars, worst %.4f at %s"
                % (len(bad_long), bad_long.max(), bad_long.idxmax().date()))
        if len(bad_short):
            raise AssertionError(
                "short exposure exceeded portfolio value on %d bars, worst %.4f at %s"
                % (len(bad_short), bad_short.max(), bad_short.idxmax().date()))
def split_adjusted_close(close: pd.DataFrame, threshold: float = 0.30) -> pd.DataFrame:
    """
    The price series implied by the competition's split rule, for BENCHMARKS.

    The engine keeps a position's value constant across a fall beyond the
    threshold by restating share count. Stated in price terms, the fall never
    happened. Any benchmark quoted against this strategy must be computed on
    THIS series: comparing a split-protected portfolio against a buy-and-hold
    that eats the raw -50% cliffs would flatter the strategy by events it did
    not trade. Only decreases are neutralised; a rise is a real gain.
    """
    r = close.pct_change()
    keep = r.where(~(r < -threshold), 0.0).fillna(0.0)
    keep.iloc[0] = 0.0
    return close.iloc[0] * (1.0 + keep).cumprod()

def performance_metrics(res: BacktestResult, periods: int = 252,
                        rf: float = 0.0) -> dict:
    """Headline metrics for the report."""
    eq = res.equity.dropna()
    r = eq.pct_change().dropna()
    if len(r) < 2:
        return {}

    years = len(r) / periods
    total = eq.iloc[-1] / eq.iloc[0] - 1.0
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1.0 if years > 0 else np.nan
    vol = r.std() * np.sqrt(periods)
    sharpe = ((r.mean() - rf / periods) / r.std() * np.sqrt(periods)
              if r.std() > 0 else np.nan)
    downside = r[r < 0].std()
    sortino = (r.mean() / downside * np.sqrt(periods)
               if downside and downside > 0 else np.nan)

    dd = eq / eq.cummax() - 1.0
    max_dd = dd.min()
    calmar = cagr / abs(max_dd) if max_dd < 0 else np.nan

    return {
        "Total Return": total,
        "CAGR": cagr,
        "Annualised Volatility": vol,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Max Drawdown": max_dd,
        "Calmar Ratio": calmar,
        "Daily Win Rate": float((r > 0).mean()),
        "Number of Trades": int(len(res.trades)),
        "Total Fees Paid": float(res.fees_paid.sum()),
        "Avg Daily Turnover": float(res.weights.diff().abs().sum(axis=1).mean()),
        "Avg Gross Exposure": float(res.gross_exposure.mean()),
        "Avg Net Exposure": float(res.net_exposure.mean()),
        "Max Long Exposure": float(res.long_exposure.max()),
        "Max Short Exposure": float(res.short_exposure.max()),
        "Final Equity": float(eq.iloc[-1]),
        "Net PnL": float(eq.iloc[-1] - eq.iloc[0]),
    }


def trade_statistics(res: BacktestResult) -> dict:
    """
    Round-trip win rate by average-cost matching, per asset.

    A position's cost basis is tracked as shares accumulate; when exposure is
    reduced, the realised P&L of the closed portion is booked against that
    basis. Works symmetrically for longs and shorts.
    """
    if res.trades.empty:
        return {"Round-trip Win Rate": np.nan, "Round Trips": 0}

    wins = losses = 0
    pnls = []
    for _, grp in res.trades.groupby("asset"):
        pos = 0.0
        basis = 0.0
        for _, tr in grp.iterrows():
            q = tr["shares"] if tr["side"] == "BUY" else -tr["shares"]
            price = tr["price"]
            if pos == 0 or np.sign(q) == np.sign(pos):
                new_pos = pos + q
                if new_pos != 0:
                    basis = (basis * pos + price * q) / new_pos
                pos = new_pos
            else:
                closed = min(abs(q), abs(pos)) * np.sign(pos)
                pnl = closed * (price - basis)
                pnls.append(pnl)
                wins += pnl > 0
                losses += pnl <= 0
                prev = pos
                pos += q
                if pos != 0 and np.sign(pos) != np.sign(prev):
                    basis = price
    total = wins + losses
    wins_l = [p for p in pnls if p > 0]
    loss_l = [p for p in pnls if p <= 0]
    return {
        "Round-trip Win Rate": wins / total if total else np.nan,
        "Round Trips": total,
        "Avg Win": float(np.mean(wins_l)) if wins_l else 0.0,
        "Avg Loss": float(np.mean(loss_l)) if loss_l else 0.0,
        "Profit Factor": (float(sum(wins_l) / abs(sum(loss_l)))
                          if loss_l and sum(loss_l) != 0 else np.nan),
    }


def benchmark_equity(close: pd.DataFrame, initial: float = 2_000_000.0,
                     fee: float = 0.001) -> pd.Series:
    """
    Equal-weight buy-and-hold: 1/N of capital into each name on the first bar,
    held untouched. This is the honest hurdle -- the rules hand every team
    100,000 per stock, so doing nothing at all produces this curve.

    Must be computed on the split-adjusted series (see engine.split_adjusted_
    close), or the benchmark eats -50% cliffs that the strategy, protected by
    the competition's own share-restatement rule, does not.
    """
    first = close.ffill().bfill().iloc[0]
    per_name = initial / close.shape[1]
    shares = (per_name * (1.0 - fee)) / first
    return (close.ffill() * shares).sum(axis=1)


def drawdown_series(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1.0


def annual_breakdown(equity: pd.Series, benchmark: pd.Series | None = None
                     ) -> pd.DataFrame:
    """Calendar-year returns, for the report's year-by-year table."""
    rows = []
    for year, seg in equity.groupby(equity.index.year):
        if len(seg) < 2:
            continue
        row = {"Year": int(year), "Return": seg.iloc[-1] / seg.iloc[0] - 1.0,
               "MaxDD": float(drawdown_series(seg).min())}
        if benchmark is not None:
            bseg = benchmark.loc[seg.index]
            row["Benchmark"] = bseg.iloc[-1] / bseg.iloc[0] - 1.0
            row["Excess"] = row["Return"] - row["Benchmark"]
        rows.append(row)
    return pd.DataFrame(rows).set_index("Year")


def format_metrics(m: dict) -> str:
    """Render the metric dict as an aligned block for the console."""
    pct = {"Total Return", "CAGR", "Annualised Volatility", "Max Drawdown",
           "Daily Win Rate", "Round-trip Win Rate", "Avg Net Exposure",
           "Avg Gross Exposure", "Max Long Exposure", "Max Short Exposure"}
    money = {"Final Equity", "Total Fees Paid", "Net PnL", "Avg Win", "Avg Loss"}
    lines = []
    for k, v in m.items():
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            lines.append("  %-26s %12s" % (k, "n/a"))
        elif k in pct:
            lines.append("  %-26s %11.2f%%" % (k, v * 100))
        elif k in money:
            lines.append("  %-26s %12s" % (k, "{:,.0f}".format(v)))
        elif isinstance(v, (int, np.integer)):
            lines.append("  %-26s %12d" % (k, v))
        else:
            lines.append("  %-26s %12.3f" % (k, v))
    return "\n".join(lines)


def run(data_dir: str, cfg: NexusConfig = None, verbose: bool = True):
    """Run the whole framework end to end and return the result bundle."""
    cfg = cfg or NexusConfig()
    t0 = time.time()
    periods = PERIODS_PER_YEAR.get(cfg.data.frequency, 252)

    md = load_market_data(data_dir, cfg.data, verbose=verbose)
    fs = build_features(md, cfg.features, cfg.physics, periods, verbose=verbose)

    builder = PortfolioBuilder(cfg, verbose=verbose)
    book = builder.build(md, fs)
    book.iloc[:cfg.wfo.core_warmup] = 0.0

    res = SimulatedExchange(
        prices_open=md.open, prices_close=md.close, target_weights=book,
        initial_capital=cfg.portfolio.initial_capital, fee=cfg.portfolio.fee,
        no_trade_band=cfg.portfolio.no_trade_band,
        exposure_margin=cfg.portfolio.exposure_margin,
        exposure_tolerance=cfg.portfolio.exposure_tolerance,
        split_threshold=cfg.data.competition_split_drop,
        allow_fractional=cfg.portfolio.allow_fractional,
        exposure_trigger=cfg.portfolio.exposure_trigger,
        exposure_reduce_to=cfg.portfolio.exposure_reduce_to,
        verbose=verbose).run()

    metrics = performance_metrics(res, periods)
    metrics.update(trade_statistics(res))

    bench_px = split_adjusted_close(md.close, cfg.data.competition_split_drop)
    bench = benchmark_equity(bench_px, cfg.portfolio.initial_capital,
                             cfg.portfolio.fee)
    br = bench.pct_change().dropna()
    years = len(br) / periods
    bench_metrics = {
        "Total Return": bench.iloc[-1] / bench.iloc[0] - 1.0,
        "CAGR": (bench.iloc[-1] / bench.iloc[0]) ** (1 / years) - 1.0,
        "Annualised Volatility": br.std() * np.sqrt(periods),
        "Sharpe Ratio": br.mean() / br.std() * np.sqrt(periods),
        "Max Drawdown": (bench / bench.cummax() - 1.0).min(),
        "Final Equity": float(bench.iloc[-1]),
        "Net PnL": float(bench.iloc[-1] - bench.iloc[0]),
    }

    if verbose:
        print("[engine] %d corporate action restatements applied"
              % (0 if res.splits.empty else len(res.splits)))
        print("[run] complete in %.1fs" % (time.time() - t0))
    return md, fs, book, res, metrics, bench, bench_metrics, builder


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="NEXUS -- seven-strategy long/short framework")
    ap.add_argument("data_dir", help="directory of OHLCV CSVs")
    ap.add_argument("--neutral", action="store_true",
                    help="run the dollar-neutral book (net_long_target = 0)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    cfg = NexusConfig()
    if args.neutral:
        cfg = dataclasses.replace(
            cfg, portfolio=dataclasses.replace(cfg.portfolio, net_long_target=0.0))

    md, fs, book, res, metrics, bench, bmet, builder = run(
        args.data_dir, cfg, verbose=not args.quiet)

    print("\n" + "=" * 66)
    print("NEXUS -- %s" % args.data_dir)
    print("=" * 66)
    print(format_metrics(metrics))
    print("\n  --- equal-weight buy & hold (split-restated) ---")
    print(format_metrics(bmet))
    print("\n" + annual_breakdown(res.equity, bench)
          .to_string(float_format=lambda v: "%7.2f%%" % (v * 100)))

    try:
        import matplotlib.pyplot as plt
        import pandas as pd
        cum_pnl = res.equity - cfg.portfolio.initial_capital
        bench_pnl = bench - cfg.portfolio.initial_capital
        
        plt.figure(figsize=(10, 6))
        plt.plot(cum_pnl.index, cum_pnl, label="Nexus Alpha", color="blue")
        plt.plot(bench_pnl.index, bench_pnl, label="Buy & Hold (Equal-Weight)", color="gray", linestyle="--")
        plt.title("Cumulative Profit and Loss over Time")
        plt.xlabel("Date")
        plt.ylabel("PnL ($)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig("cumulative_pnl.png", dpi=300)
        plt.close()
        fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
        axes[0].plot(res.long_exposure.index, res.long_exposure, label="Long Exposure", color="green")
        axes[0].plot(res.short_exposure.index, -res.short_exposure, label="Short Exposure (Negative)", color="red")
        axes[0].plot(res.net_exposure.index, res.net_exposure, label="Net Exposure", color="black", linestyle="--")
        axes[0].set_title("Portfolio Exposures over Time")
        axes[0].set_ylabel("Exposure (Fraction of Equity)")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc='upper left')
        pos_w = res.weights.clip(lower=0)
        neg_w = res.weights.clip(upper=0)
        axes[1].stackplot(pos_w.index, pos_w.T, labels=pos_w.columns, alpha=0.8)
        axes[1].stackplot(neg_w.index, neg_w.T, alpha=0.8)
        axes[1].set_title("Asset Allocations (Where Equity is Assigned)")
        axes[1].set_ylabel("Weight")
        axes[1].axhline(0, color='black', linewidth=1)
        axes[1].legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize='small', ncol=1)
        alpha_df = builder.diagnostics.get("allocation")
        if alpha_df is not None and not alpha_df.empty:
            axes[2].stackplot(alpha_df.index, alpha_df.T, labels=alpha_df.columns, alpha=0.8)
            axes[2].set_title("Strategy-wise Allocations (Mean-CVaR)")
            axes[2].set_ylabel("Weight")
            axes[2].legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize='small', ncol=1)

        axes[-1].set_xlabel("Date")

        plt.tight_layout()
        plt.savefig("position_sizing.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("\n[+] Position sizing graph saved to 'position_sizing.png'.")
        import os
        plot_dir = "Plots"
        os.makedirs(plot_dir, exist_ok=True)
        asset_pnl = (res.positions.shift(1) * md.close.diff()).sum()
        asset_pnl.to_csv("cumulative_metrics_assetwise.csv")
        print("[+] Asset-wise cumulative metrics saved to 'cumulative_metrics_assetwise.csv'")
        books = builder.diagnostics.get("books", {})
        alpha = builder.diagnostics.get("allocation")

        if alpha is not None and books:
            port_ret = (res.equity.diff() / cfg.portfolio.initial_capital).fillna(0)
            strat_pnl_df = pd.DataFrame(index=port_ret.index, columns=alpha.columns)
            for strat in alpha.columns:
                strat_pnl_df[strat] = port_ret * alpha[strat].shift(1).fillna(0)
            strat_cum_pnl = strat_pnl_df.sum() * cfg.portfolio.initial_capital
            strat_cum_pnl.to_csv("cumulative_metrics_strategywise.csv")
            print("[+] Strategy-wise cumulative metrics saved to 'cumulative_metrics_strategywise.csv'")
            for asset in res.weights.columns:
                asset_contrib = {}
                for strat_name, strat_book in books.items():
                    if asset in strat_book.columns and strat_name in alpha.columns:
                        asset_contrib[strat_name] = strat_book[asset] * alpha[strat_name]

                df_asset = pd.DataFrame(asset_contrib)
                if df_asset.empty: continue

                fig, ax = plt.subplots(figsize=(10, 6))
                pos_contrib = df_asset.clip(lower=0)
                neg_contrib = df_asset.clip(upper=0)
                if pos_contrib.sum().sum() > 0:
                    ax.stackplot(pos_contrib.index, pos_contrib.T, labels=pos_contrib.columns, alpha=0.8)
                if neg_contrib.sum().sum() < 0:
                    ax.stackplot(neg_contrib.index, neg_contrib.T, alpha=0.8)

                ax.set_title(f"Strategy Contributions for {asset}")
                ax.set_xlabel("Date")
                ax.set_ylabel("Weight in Portfolio")
                ax.axhline(0, color='black', linewidth=1)

                handles, labels = ax.get_legend_handles_labels()
                if handles:
                    ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize='small', ncol=1)

                plt.tight_layout()
                plt.savefig(os.path.join(plot_dir, f"{asset}_strategies.png"), dpi=200, bbox_inches='tight')
                plt.close(fig)
            print(f"[+] Saved {len(res.weights.columns)} asset-wise strategy plots to '{plot_dir}/'")
        
    except Exception as e:
        print(f"[-] Could not generate plots: {e}")

    return res


if __name__ == "__main__":
    main()
