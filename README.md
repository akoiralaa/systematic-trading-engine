# Statistical Vector Zone Trading System

[![Tests](https://github.com/akoiralaa/systematic-trading-engine/actions/workflows/tests.yml/badge.svg)](https://github.com/akoiralaa/systematic-trading-engine/actions/workflows/tests.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A systematic intraday equity trading engine built for Alpaca Markets. Combines OLS-based regime detection, Bayesian-adjusted fractional Kelly sizing, power-law market impact modeling, and Monte Carlo tail-risk quantification — with a full statistical validation suite to guard against overfitting.

**Designed to show: mathematical rigor, production engineering, and honest empirical self-assessment.**

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Component Deep-Dive](#component-deep-dive)
3. [Signal Generation](#signal-generation)
4. [Regime Detection](#regime-detection)
5. [Position Sizing](#position-sizing)
6. [Market Friction Model](#market-friction-model)
7. [Risk Engine (Monte Carlo)](#risk-engine-monte-carlo)
8. [Statistical Validation](#statistical-validation)
9. [Exit Management](#exit-management)
10. [Backtester](#backtester)
11. [Production Architecture](#production-architecture)
12. [Setup & Execution](#setup--execution)
13. [Test Suite](#test-suite)
14. [Configuration](#configuration)
15. [Known Limitations](#known-limitations)
16. [References](#references)

---

## Architecture Overview

```
Signal Layer                    Decision Layer                  Execution Layer
──────────────────────         ──────────────────────          ──────────────────────
EMA Vector Price           →   RegimeDetector                  AlpacaTrader
ATR Dead-Band Filter       →   └─ OLS on log returns           └─ REST API wrapper
Normalized Strength        →   └─ p-value gate (α=0.05)        └─ Rate-limit retry
                                                                └─ Reconnect logic
                           →   BayesianKellyCriterion
                               └─ Calibrated win prob          PositionTracker
                               └─ f* = (p·b − q) / b          └─ JSON persistence
                               └─ Fractional (×0.5)            └─ Crash recovery
                               └─ Concentration cap (20%)       └─ Reconciliation

                           →   MarketFrictionModel              ExitManager
                               └─ Power-law impact              └─ Profit target
                               └─ Bid-ask half-spread           └─ Hard stop loss
                               └─ ADV participation cap         └─ Trailing stop
                                                                └─ Time limit (4hr)
                           →   EV Gate (reject if EV ≤ 0)      └─ EOD force-close

Risk Layer
──────────────────────
MonteCarloStressTest           StatisticalTests
└─ 10K bootstrap paths         └─ ADF (stationarity)
└─ Probability cone            └─ Ljung-Box (autocorr)
└─ VaR / CVaR (ES)             └─ Jarque-Bera (normality)
└─ Risk of Ruin                └─ t-test (strategy edge)
└─ Black Swan injection        └─ Sensitivity analysis

AdvancedBacktester
└─ Event-driven (no look-ahead bias)
└─ Walk-forward IS/OOS validation
└─ Regime-conditional breakdown
└─ Exit reason analysis
```

**124 tests, 0 failures.** 11 modules, ~2,600 lines of source.

---

## Component Deep-Dive

### Signal Generation

The primary signal is built from three derived series over 5-minute OHLCV bars:

**Vector Price** — 20-period EMA of close, used as the structural trend anchor:
```
V_t = EMA(Close, 20)
```

**ATR Dead-Band** — ±k·ATR(14) around the vector line. A signal is only valid if price *clears* this band, filtering stochastic noise:
```
Upper = V_t + k · ATR_t
Lower = V_t − k · ATR_t          (k = 2.0 by default, 1.5 in production)
```

**Normalized Strength** — Price deviation from vector, scaled to [0, 1]:
```
strength_t = clip( |Close_t − V_t| / (1.5 · ATR_t) , 0, 1 )
```

A long signal requires `Close_t > Upper` AND `strength_t ≥ 0.51`.

---

### Regime Detection

`src/regime_detector.py` classifies the market into three states using OLS regression on **cumulative log returns** (a stationary transformation — not raw prices):

```python
log_prices = log(Close[-30:])
returns    = diff(log_prices)
cum_ret    = cumsum(returns)
slope, _, r², p, _ = linregress(t, cum_ret)
```

State assignment:
| Condition | State |
|---|---|
| `p < 0.05` AND `|slope| > 0.5·σ` | `TRENDING` |
| `σ > vol_threshold` | `VOLATILE` |
| Otherwise | `SIDEWAYS` |

Trades are **blocked in SIDEWAYS** — the regime filter is the first gate. Trend strength is the signal-to-noise ratio `|slope| / σ`. R² is reported as the confidence measure in trending regimes.

**Why log returns, not price?** Price is I(1) (unit root). Running OLS on a random walk produces spurious R² near 1.0. Log returns are stationary; the OLS slope then captures genuine directional drift.

---

### Position Sizing

`src/bayesian_kelly.py` implements fractional Kelly with a conservative probability calibration layer.

**The Kelly Formula:**
```
f* = (p·b − q) / b

where:
  p  = calibrated win probability
  q  = 1 − p
  b  = reward/risk ratio (default 2.0)
```

**Win probability calibration** — `vector_strength` is *not* used directly as `p`. Raw signal strength is not probability; equating them over-bets. Instead:

- *Pre-calibration* (< 50 trades): conservative linear map `[0.51, 1.0] → [0.51, 0.65]`
- *Post-calibration* (≥ 50 trades): empirical binned win rates from the trade log, with per-bin minimum of 10 trades; falls back to conservative default if bin is underpopulated

**Constraints applied in order:**
1. Kelly fraction capped at 25% (`min(f*, 0.25)`)
2. Fractional Kelly multiplier applied (`× 0.5`) — trades half the optimal fraction to reduce variance at cost of sub-optimal growth
3. Concentration limit: `position_value ≤ 20%` of equity
4. Liquidity limit: `position_value ≤ buying_power`

The sizing formula allocates the Kelly risk-budget to *risk*, not *position value*:
```
shares = (equity × f*) / risk_per_share
```
where `risk_per_share = entry_price − stop_price`.

Calibration data persists to `state/kelly_calibration.json` across sessions.

---

### Market Friction Model

`src/market_friction_model.py` estimates implementation shortfall using a **power-law participation model** (Almgren & Chriss 2001):

**Market impact (slippage):**
```
I = α · (V_order / V_ADV)^1.5   [in basis points]

where:
  α        = 0.1 (market_impact_coeff)
  V_order  = order shares
  V_ADV    = average daily volume
  exponent = 1.5 (non-linear: disproportionate impact near liquidity ceiling)
```

**Total friction (per side):**
```
TotalFriction = I + (spread_bps / 2)
ExecutionPrice = ArrivalPrice × (1 + direction × TotalFriction / 10000)
```

**Liquidity constraint:**
```
MaxSize = floor(V_ADV × 0.05)   (5% ADV participation cap)
```

Stop-loss and urgent exits carry 2× slippage multiplier (adverse fills under stress).

---

### Risk Engine (Monte Carlo)

`src/monte_carlo_stress_test.py` runs 10,000-simulation bootstrap analysis on trade return distributions.

**Probability Cone** — Bootstrap resampling of trade returns to build equity path distributions:
```python
paths = bootstrap(trade_returns, n=10000)   # shape: (10000, N_trades)
equity = cumprod(1 + paths) × initial_equity
```
Percentile fan: P5, P25, P50, P75, P95 equity trajectories.

**Block Bootstrap** (Politis & Romano 1994) — For bar-level returns with serial correlation, a block size > 1 preserves temporal structure. The IID bootstrap is used for trade-level returns (where positions don't overlap).

**Risk of Ruin:**
```
P(E_final < E_0 × (1 − threshold))    threshold = 20% by default
```

**VaR and CVaR (Expected Shortfall):**
```
VaR_α    = percentile(final_equity_dist, 1 − α)
CVaR_α   = E[E_final | E_final ≤ VaR_α]     (mean of left tail)
```
CVaR is the coherent risk measure (Artzner et al. 1999) — superior to VaR for non-normal distributions.

**Black Swan Injection:**
```
With probability 0.10: inject shock of −10% into one bar of each path
Survival rate = P(E_final > 0.8 × E_0)
```

---

### Statistical Validation

`src/statistical_tests.py` runs four hypothesis tests on the trade return series before trusting backtest results:

| Test | H₀ | Failure Interpretation |
|---|---|---|
| **ADF** (Said & Dickey 1984) | Unit root (non-stationary) | Returns may be I(1); inference invalid |
| **Ljung-Box** (1978) | No autocorrelation up to lag 10 | IID assumption violated; use block bootstrap |
| **Jarque-Bera** (1987) | Returns are normally distributed | Fat tails present; parametric VaR understates risk |
| **t-test** | Mean return = 0 (no edge) | Strategy has no statistically significant positive edge |

The strategy t-test is the critical gate: `H₀: μ = 0`. Rejection requires `p < 0.05` AND `μ > 0`. A high Sharpe ratio on insufficient data (N < 30) should not be trusted — this is explicitly surfaced as a warning.

**Parameter sensitivity analysis** is also included: perturbs Kelly fraction and ATR multiplier ±10/20/30% and measures Sharpe range to identify which parameters the system is most sensitive to.

---

### Exit Management

`src/exit_manager.py` evaluates five exit conditions in priority order on every cycle:

1. **Profit Target** — Exit at `target_price = entry + risk_per_share × 2.0`
2. **Hard Stop Loss** — Exit at `stop_price = entry − k·ATR` (volatility-adjusted, floored at 5% below entry)
3. **Trailing Stop** — Ratchets stop up as price rises; locks in profits after ≥2% gain
4. **Time Limit** — Maximum 240 minutes (configurable); prevents overnight gaps
5. **EOD Force-Close** — All positions closed 5 minutes before market close (Eastern time)

Stop placement uses the ATR dead-band logic:
```
stop = min(entry − k·ATR, vector − ATR)
stop = max(stop, entry × 0.95)       # 5% floor
```

---

### Backtester

`src/backtester.py` is event-driven with explicit **look-ahead bias prevention**:

```
Signal generated at bar i (close price)
    ↓
Fill executes at bar i+1's OPEN price   ← key: not at signal close
    ↓
Stop/target use intrabar conditions from bar i+1 onward
```

Stop and target are recalculated from the actual fill price (not the signal price), so reported P&L is based on what you would have received, not what you saw when you decided to trade.

**Walk-forward validation** splits data into 5 segments with an expanding in-sample window (60% initial training). Per-split IS and OOS Sharpe are computed. The **Sharpe degradation ratio** (`OOS Sharpe / IS Sharpe`) is the primary overfitting diagnostic — values near 1.0 indicate robustness; values near 0 indicate curve-fitting.

**Regime-conditional P&L** — Each trade is tagged with the regime at entry (TRENDING/VOLATILE/SIDEWAYS). The results breakdown shows mean return, win rate, and trade count per regime, verifying that the regime filter adds value.

---

### Production Architecture

`production_trader_v2.py` — the live/paper trading entry point:

```
startup
├── AlpacaTrader.connect()         # API auth with reconnect logic
├── TradingPipeline(equity)        # Initialize with live account equity
├── PositionTracker.load()         # Restore state from state/positions.json
└── ExitManager(240, 5)            # 4hr hold limit, 5min EOD close

main loop (every 300s)
├── is_market_open()               # Alpaca clock check
├── ExitManager.should_close_all() # EOD gate
├── PositionTracker.update_all()   # Mark-to-market all open positions
├── for symbol in TRADING_SYMBOLS:
│   ├── fetch 5-min bars (5 days)
│   ├── compute features (ATR, EMA, strength)
│   ├── exit_manager.check_exit()  # 5-condition exit check
│   ├── engine.execute_trading_cycle()   # 8-step signal pipeline
│   └── alpaca_trader.place_order()      # Fire order if approved
├── update equity from account
└── circuit breakers:
    ├── MAX_CONSECUTIVE_LOSSES = 5       # Halt after 5 losses
    └── MAX_DAILY_LOSS_PERCENT = 10%     # Halt if down >10% today
```

**PositionTracker** (`src/position_tracker.py`) writes state atomically to `state/positions.json` (write-to-temp then rename) and reconciles on startup against live Alpaca positions, handling discrepancies from crashes or interrupted sessions.

**Rotating log files** — 5MB max, 5 backups. Separate console (INFO) and file (DEBUG) handlers.

---

## Setup & Execution

### 1. Clone and Install

```bash
git clone https://github.com/akoiralaa/trading-bot.git
cd trading-bot
pip install -r requirements.txt
```

Dependencies: `numpy`, `pandas`, `scipy`, `statsmodels`, `alpaca-trade-api`, `pytz`, `matplotlib`, `pytest`

On Raspberry Pi / PEP 668 environments:
```bash
pip install -r requirements.txt --break-system-packages
```

### 2. API Credentials

Create a `.env` file (never commit this):
```bash
ALPACA_API_KEY=your_paper_api_key
ALPACA_SECRET_KEY=your_paper_secret_key
```

Get paper trading keys from [alpaca.markets](https://alpaca.markets) — free account, no live capital needed.

The system reads credentials via environment variables (`os.environ.get`). You can also export them in your shell:
```bash
export ALPACA_API_KEY="..."
export ALPACA_SECRET_KEY="..."
```

### 3. Fetch Historical Data

```bash
# Default: SPY, QQQ, AAPL, NVDA, TSLA, AMD, MSFT, GOOGL, AMZN, META (10 years daily)
python fetch_data.py

# Specific symbols
python fetch_data.py SPY QQQ AAPL

# Custom timeframe
python fetch_data.py --years 5
```

Data is saved as CSV to `data/{SYMBOL}.csv` with columns: `Date, Open, High, Low, Close, Volume`.

To use your own data: place any CSV with those columns in `data/` — the backtest runner discovers them automatically.

### 4. Run Backtests

```bash
# All symbols in data/
python run_backtest.py

# Specific symbols
python run_backtest.py SPY QQQ AAPL

# Output: results/backtest_results.json
```

Each symbol produces:
- Sharpe, Sortino, Calmar ratios
- Max drawdown, win rate, expectancy
- Walk-forward IS/OOS comparison (5 splits)
- Statistical tests (ADF, Ljung-Box, Jarque-Bera, t-test)
- Monte Carlo: probability cone, VaR, CVaR, risk of ruin
- Regime-conditional performance breakdown
- Exit reason analysis

### 5. Run Tests

```bash
python -m pytest tests/ -v          # All 124 tests
python -m pytest tests/ -q          # Quiet mode
python -m pytest tests/test_bayesian_kelly.py -v   # Single module
```

### 6. Paper Trading

```bash
python production_trader_v2.py
```

The system will:
1. Connect to Alpaca paper account
2. Load any persisted positions from `state/positions.json`
3. Scan `TRADING_SYMBOLS` every 5 minutes during market hours
4. Apply the full 8-step decision pipeline
5. Execute approved trades via Alpaca REST API
6. Monitor and exit positions via 5-condition exit logic

Stop with `Ctrl+C` — positions persist to `state/` for the next session.

### 7. Research Notebook

```bash
jupyter lab notebooks/research.ipynb
```

Covers: equity curves, regime classification plots, Monte Carlo fan charts, parameter sensitivity, walk-forward split comparisons.

---

## The 8-Step Trading Cycle

Each scan runs `TradingPipeline.execute_trading_cycle()`:

```
1. Regime Detection     OLS on cum log returns → TRENDING / VOLATILE / SIDEWAYS
                        Block if SIDEWAYS; require p < 0.05

2. Breakout Confirmation  price > EMA + k·ATR  AND  strength ≥ 0.51
                          Reject if dead-band not cleared

3. Dynamic Stop           stop = min(entry − k·ATR, vector − ATR)
                          Floored at 5% below entry

4. Kelly Sizing           f* = (p·b − q) / b
                          Apply fractional (×0.5), cap at 25%
                          shares = (equity × f*) / risk_per_share

5. Friction Adjustment    I = 0.1 × (qty / ADV)^1.5  [bps]
                          execution_price = arrival × (1 + (I + spread/2) / 10000)

6. Liquidity Gate         qty ≤ floor(ADV × 0.05)

7. EV Gate                EV = p·win_amt − q·loss_amt
                          Reject if EV ≤ 0

8. Assemble Parameters    symbol, qty, entry, stop, target, friction, kelly_fraction, EV
```

---

## Test Suite

124 tests across 11 test files:

| File | Coverage |
|---|---|
| `test_trading_pipeline.py` | 8-step cycle, rejection paths, equity updates |
| `test_regime_detector.py` | OLS state classification, dead-band math, stop placement |
| `test_bayesian_kelly.py` | Kelly formula, calibration, concentration limits, persistence |
| `test_market_friction.py` | Power-law impact, bid-ask spread, liquidity cap |
| `test_monte_carlo.py` | Bootstrap, block bootstrap, VaR, CVaR, risk of ruin, shock injection |
| `test_statistical_tests.py` | ADF, Ljung-Box, Jarque-Bera, t-test, sensitivity analysis |
| `test_backtester.py` | Look-ahead prevention, walk-forward splits, performance metrics |
| `test_exit_manager.py` | All 5 exit conditions, EOD timing, Eastern timezone |
| `test_position_tracker.py` | CRUD, persistence, reconciliation, crash recovery |
| `test_alpaca_trader.py` | Connection, rate-limit retry, order types |
| `test_integration.py` | Full pipeline end-to-end with mock Alpaca API |

---

## Configuration

All parameters centralized in `config/trading_config.py`:

```python
# Position sizing
DEFAULT_KELLY_FRACTION   = 0.03   # 3% full Kelly; start conservative
MAX_RISK_PERCENT         = 0.05   # 5% absolute max per trade

# Signal parameters
DEFAULT_ATR_PERIOD       = 14     # ATR lookback
ZONE_WIDTH_MIN           = 0.5    # Dead-band width (ATR units)

# Regime detection
DEFAULT_ALPHA            = 0.05   # p-value significance threshold

# Execution
SCAN_INTERVAL_SECONDS    = 300    # 5-min scan cycle
MAX_HOLD_MINUTES         = 240    # 4-hour max hold
EOD_CLOSE_MINUTES        = 5      # Force-close before close

# Risk controls
MAX_CONSECUTIVE_LOSSES   = 5      # Halt after 5 consecutive losses
MAX_DAILY_LOSS_PERCENT   = 0.10   # Halt if down >10% in a day
MAX_CONCURRENT_POSITIONS = 5      # Maximum open positions
MAX_POSITION_CAPITAL_PCT = 0.10   # Max 10% of buying power per trade

# Monte Carlo
MONTE_CARLO_SAMPLES      = 10000  # Bootstrap simulations

# Symbols
TRADING_SYMBOLS = ['SPY', 'QQQ', 'GOOGL', 'AMZN', 'NVDA',
                   'TSLA', 'META', 'NFLX', 'AMD', 'PLTR']
```

---

## Known Limitations

These are documented explicitly because honest self-assessment is part of the engineering discipline:

1. **Small sample size on daily data** — The strategy is designed for 5-minute bars. Daily OHLCV produces very few signals (N < 30); statistical tests have low power and confidence intervals are wide. Proper evaluation requires intraday data and a live paper-trading track record.

2. **Win probability uncalibrated pre-50 trades** — The system uses a conservative linear mapping `[0.51, 0.65]` until 50 completed trades are logged. Empirical calibration replaces this automatically. Pre-calibration Kelly sizing is deliberately conservative.

3. **No live order book data** — Market impact uses a simplified power-law model calibrated to ADV, not real L2 depth. For small-cap or thinly-traded names, actual slippage may exceed the model significantly.

4. **Long-only** — No short selling support. Strategy is directionally biased and will underperform in sustained downtrends.

5. **IID bootstrap assumption** — The default Monte Carlo bootstrap resamples trades as IID. If trade returns show serial correlation (Ljung-Box rejects), the block bootstrap should be used — it is implemented but not the default.

6. **Constant R:R ratio** — Kelly criterion assumes a fixed 2:1 reward/risk ratio. Actual trade outcomes vary; this is a simplification.

7. **Synchronous REST polling** — No WebSocket price feeds. The 5-minute scan cycle is appropriate for intraday momentum but not latency-sensitive strategies. This is a deliberate design choice for the Raspberry Pi deployment target.

8. **Alpaca paper only** — Tested against Alpaca paper trading API. Live trading requires additional validation and a funded account with margin permissions if shorting is ever added.

---

## Project Structure

```
trading-bot/
├── src/
│   ├── trading_pipeline.py          8-step orchestrator
│   ├── regime_detector.py           OLS regime classification + ATR dead-bands
│   ├── bayesian_kelly.py            Fractional Kelly with empirical calibration
│   ├── market_friction_model.py     Power-law market impact + spread model
│   ├── monte_carlo_stress_test.py   10K bootstrap: VaR, CVaR, RoR, shock test
│   ├── statistical_tests.py         ADF, Ljung-Box, Jarque-Bera, t-test, sensitivity
│   ├── backtester.py                Event-driven backtest + walk-forward validation
│   ├── exit_manager.py              5-condition exit system (target/stop/trail/time/EOD)
│   ├── position_tracker.py          State persistence + crash recovery + reconciliation
│   └── alpaca_trader.py             API wrapper with retry/reconnect
├── config/
│   ├── trading_config.py            All tunable parameters (centralized)
│   └── logging_config.py            Rotating log handler setup
├── tests/                           124 unit + integration tests
│   ├── test_trading_pipeline.py
│   ├── test_regime_detector.py
│   ├── test_bayesian_kelly.py
│   ├── test_market_friction.py
│   ├── test_monte_carlo.py
│   ├── test_statistical_tests.py
│   ├── test_backtester.py
│   ├── test_exit_manager.py
│   ├── test_position_tracker.py
│   ├── test_alpaca_trader.py
│   └── test_integration.py
├── notebooks/
│   └── research.ipynb               Equity curves, regime plots, MC fan charts
├── docs/
│   └── METHODOLOGY.md               Formal mathematical specification
├── data/                            OHLCV CSVs (Date,Open,High,Low,Close,Volume)
├── results/                         Backtest output (JSON)
├── state/                           Live position state (gitignored)
├── logs/                            Rotating log files (gitignored)
├── fetch_data.py                    Historical data downloader (Alpaca API)
├── run_backtest.py                  Batch backtest runner + statistical report
├── production_trader_v2.py          Live/paper trading entry point (v2 - full exit logic)
├── production_trader.py             V1 signal scanner (entry only, deprecated)
└── requirements.txt
```

---

## References

- Almgren, R., & Chriss, N. (2001). Optimal execution of portfolio transactions. *Journal of Risk*, 3(2), 5–39.
- Artzner, P., Delbaen, F., Eber, J.M., & Heath, D. (1999). Coherent measures of risk. *Mathematical Finance*, 9(3), 203–228.
- Kelly, J.L. (1956). A new interpretation of information rate. *Bell System Technical Journal*, 35(4), 917–926.
- Ljung, G.M., & Box, G.E.P. (1978). On a measure of lack of fit in time series models. *Biometrika*, 65(2), 297–303.
- Politis, D.N., & Romano, J.P. (1994). The stationary bootstrap. *Journal of the American Statistical Association*, 89(428), 1303–1313.
- Said, S.E., & Dickey, D.A. (1984). Testing for unit roots in autoregressive-moving average models of unknown order. *Biometrika*, 71(3), 599–607.
- Wilder, J.W. (1978). *New Concepts in Technical Trading Systems*. Trend Research.

---

## Disclaimer

This software is for educational and research purposes. All trading involves substantial risk of loss. Paper trade for a minimum of 30 days before any live deployment. Past backtest performance does not guarantee future results.
