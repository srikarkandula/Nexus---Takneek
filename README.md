# Nexus: Algorithmic Long/Short Portfolio Allocation
 
A multi-strategy quantitative trading framework built for the **Zenith Problem Statement (ALPHA)** at **Takneek 2026**, IIT Kanpur. It allocates capital across a 20-stock universe using six independent alpha sleeves, a cost-aware Mean-CVaR allocator, a machine-learning risk overlay and an event-driven backtester.
 
 
## Results
 
| Metric | Nexus | Buy & Hold | Excess |
|---|---|---|---|
| Total return | **283.48%** | 127.07% | +156.41% |
| Sharpe ratio | **1.076** | 0.676 | +0.400 |
| Max drawdown | **-31.06%** | -34.58% | +3.52% |
 
Other realised figures from the final run: Sortino 1.421, net PnL 5,669,532, fees paid 227,921, 20,568 trades, win rate 57.02%.
 
All numbers are **after** the 0.1% transaction fee, a one-bar execution delay, whole-share rounding and exposure checks. See [Limitations](#limitations) before reading too much into them.
 
| Year | Return | Max drawdown | Benchmark | Excess |
|---|---|---|---|---|
| 2014 | 3.15% | -12.00% | 15.71% | -12.56% |
| 2015 | 2.95% | -10.44% | -4.28% | 7.24% |
| 2016 | 23.84% | -12.22% | 12.65% | 11.19% |
| 2017 | 31.63% | -3.94% | 25.02% | 6.61% |
| 2018 | 2.61% | -15.95% | -9.90% | 12.51% |
| 2019 | 38.05% | -10.02% | 24.82% | 13.23% |
| 2020 | 34.65% | -31.06% | 15.06% | 19.58% |
| 2021 | 16.44% | -9.62% | 14.37% | 2.07% |
 
---
 
## The Problem
 
Trade a universe of **20 equities** starting with **INR 2,000,000** (20 x 100,000), subject to:
 
- a maximum of **7 deployed strategies**
- a **0.1% fee** on the gross value of every buy and sell
- long exposure and short exposure each capped at the current portfolio value
- a stock-split rule for corporate actions 
---
 
## Architecture

 
1. **Data integrity:** raw prices drive execution and PnL; a corporate-action-adjusted series drives all features.
2. **Features:** range-based volatility, liquidity, an OHLCV order-flow proxy, Tsallis tail statistics and Hurst exponents.
3. **Alpha sleeves:** six strategies, each producing its own target book.
4. **Allocation:** cost-aware Mean-CVaR across sleeves, plus an inverse-volatility long core.
5. **Risk:** LightGBM gap-risk overlay, per-name caps, smoothing, trailing-stop quarantine and exposure projection.
6. **Execution:** a target formed on bar `t` can only fill at the **open of bar `t+1`**.
### Data layer
 
- **Corporate actions:** splits are detected from price discontinuities and neutralised in the feature series. Distributions with no reliable divisor are masked from rolling statistics rather than guessed. The execution engine and the Buy & Hold benchmark apply the same split rule to share counts.
- **Range-based volatility:** Parkinson and Garman-Klass estimators. Average True Range is used as the unit of risk in the breakout sleeve.
- **Liquidity:** Amihud illiquidity, used both to exclude names and as risk context.
- **Order-flow proxy:** signed volume, wick imbalance and dollar-volume surprise. This is a proxy only, since true order flow imbalance needs limit order book data.
- **Causality:** every feature uses a trailing window ending at or before the current bar.
### Econophysics features
 
- **q-Gaussian / Tsallis tails:** gains and losses are fitted separately and compared with **Asymmetric Tsallis Relative Entropy (ATRE)** as a measure of asymmetric tail risk.
- **Generalised Hurst exponent:** classifies the market as persistent or anti-persistent, combined with volatility state before the regime sleeve trades.
### Alpha sleeves
 
| # | Sleeve | Idea |
|---|---|---|
| 1 | **Gap-Down Mean Reversion** | Buy exhaustion-driven overnight gaps (gap z-score <= -1.31 with a range-expansion and volume-surge filter), volatility-scaled, with a finite holding period and hard loss stop. |
| 2 | **Cross-Sectional Reversal** | Long recent losers, short recent winners using volume-conditioned, volatility-scaled momentum ranks. Dollar neutral, with a Parkinson-volatility tilt and illiquid names excluded. |
| 3 | **GARCH Cointegration Pairs** | Pairs that pass an ADF stationarity test on the log-price spread, hedge ratio frozen per block, entries and exits on a GARCH(1,1)-standardised spread (enter at \|z\| > 2, exit at < 0.5, stop at > 4). |
| 4 | **Tsallis Tail-Risk Premium** | Cross-sectional rank of smoothed ATRE and downside tail shape. Trades the *level* of asymmetric tail risk as a risk premium, with a co-skewness throttle. |
| 5 | **Donchian Breakout + Pyramiding** | ATR-sized breakout entries with up to six pyramid units added every 0.5 ATR, so winners are allowed to grow instead of being trimmed back. |
| 6 | **Multifractal Regime Switching** | Follows moves when the Hurst exponent signals persistence and fades them when it signals anti-persistence, but only during a volatility surge. Adaptive trailing stop. |
| - | **Capital Deployment Core** | Inverse-volatility long book so capital is not left idle. |
 
### Portfolio construction and risk
 
- **LightGBM gap-risk overlay:** forecasts the probability of a severe overnight gap and scales the riskiest names down to 50% of intended exposure. It is a *defensive* overlay, not a directional alpha. Walk-forward: 500-bar training window, refit every 63 bars, 5-bar embargo.
- **Mean-CVaR allocation:** sleeve returns are taken net of an estimate of their own turnover cost, then capital is allocated by minimising 95% CVaR with a return term and a turnover penalty, subject to sleeve concentration limits. Ledoit-Wolf shrinkage is used only for the risk-based prior or fallback.
- **Final controls:** per-name cap, signal and weight smoothing, trailing-stop quarantine with cooldown, a no-trade band, and independent long and short exposure projection.
### Backtester
 
## Repository Structure
 
```
nexus/
├── README.md
├── nexus.py                
└── report/
    └── Nexus_ALPHA_Report.pdf

