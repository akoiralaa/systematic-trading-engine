# Methodology: Statistical Vector Zone Trading System

## 1. Signal Generation

### 1.1 Vector Price (Exponential Moving Average)

The "vector price" is an exponentially weighted moving average (EMA) of close prices:

```
V_t = alpha * P_t + (1 - alpha) * V_{t-1}
```

where `alpha = 2 / (span + 1)`, default `span = 20`.

### 1.2 Signal Strength

Signal strength measures price deviation from the vector, normalized by ATR:

```
strength_t = min(1, |P_t - V_t| / (ATR_t * 1.5))
```

This produces a value in [0, 1] representing how far price has moved relative to recent volatility.

### 1.3 Average True Range (ATR)

ATR follows Wilder (1978):

```
TR_t = max(H_t - L_t, |H_t - C_{t-1}|, |L_t - C_{t-1}|)
ATR_t = SMA(TR, period=14)
```

Reference: Wilder, J.W. (1978). *New Concepts in Technical Trading Systems*.

## 2. Regime Detection

Market state is classified via OLS linear regression on the most recent `lookback` (default 30) bars:

```
P_t = beta_0 + beta_1 * t + epsilon_t
```

Using `scipy.stats.linregress`, we obtain:
- `slope` (beta_1): direction and magnitude of trend
- `r_value`: correlation coefficient; `R^2 = r_value^2` measures proportion of variance explained
- `p_value`: significance of the slope (H0: slope = 0)

### State Classification

| State     | Condition                              |
|-----------|----------------------------------------|
| TRENDING  | p < 0.05 AND normalized drift > 0.001 |
| VOLATILE  | log-return volatility > 0.02           |
| SIDEWAYS  | otherwise                              |

The `confidence` field reports `R^2` for trending regimes (proportion of variance explained by the linear trend), which is a more appropriate measure than `1 - p_value`.

## 3. Position Sizing: Kelly Criterion

### 3.1 Full Kelly

The Kelly criterion (Kelly, 1956) for the optimal fraction of capital to risk:

```
f* = (p * b - q) / b
```

where:
- `p` = probability of winning (calibrated, not raw signal strength)
- `q = 1 - p`
- `b` = reward-to-risk ratio (default 2.0)

### 3.2 Fractional Kelly

We use fractional Kelly (default 50%) to reduce variance:

```
f_safe = f* * fractional_kelly
```

Capped at 25% maximum concentration per position.

### 3.3 Win Probability Calibration

**The system does NOT equate signal strength with win probability.**

Default calibration maps `[0.51, 1.0]` signal strength to `[0.51, 0.65]` win probability via linear interpolation. This conservative mapping prevents over-betting on uncalibrated signals.

Once 50+ trades are collected, empirical binned calibration replaces the default curve.

Reference: Kelly, J.L. (1956). A new interpretation of information rate. *Bell System Technical Journal*, 35(4), 917-926.

## 4. Market Impact Model

Transaction costs are modeled using a power-law participation model inspired by Almgren & Chriss (2001):

```
Impact_bps = alpha * (V_order / V_ADV * 100)^1.5
```

where:
- `alpha` = market impact coefficient (default 0.1)
- `V_order` = order quantity
- `V_ADV` = average daily volume

Total friction includes:
- Market impact (power-law above)
- Half bid-ask spread (default 1 bps per side)

Maximum position size is constrained to 5% of ADV to limit market impact.

Reference: Almgren, R., & Chriss, N. (2001). Optimal execution of portfolio transactions. *Journal of Risk*, 3(2), 5-39.

## 5. Risk Metrics

### 5.1 Value at Risk (VaR)

VaR at confidence level alpha is the (1-alpha) percentile of the simulated terminal equity distribution:

```
VaR_alpha = Percentile(E_final, (1-alpha)*100)
```

### 5.2 Conditional VaR (Expected Shortfall)

CVaR is the expected loss conditional on exceeding VaR:

```
CVaR = E[E_final | E_final <= VaR_alpha]
```

CVaR is a coherent risk measure (Artzner et al., 1999), unlike VaR.

### 5.3 Risk of Ruin

Probability that terminal equity falls below a threshold:

```
P(E_final < E_0 * (1 - theta))
```

Default threshold theta = 0.20 (20% drawdown).

## 6. Monte Carlo Simulation

### 6.1 Bootstrap Method

Default: IID bootstrap resampling of trade returns (10,000 simulations).

For bar-level returns where serial correlation may exist, block bootstrap (Politis & Romano, 1994) is available with configurable block size.

### 6.2 Stress Testing

Black swan injection: with probability `shock_prob` (default 10%), inject a return of `shock_mag` (default -10%) into a random position in each simulated path.

Reference: Politis, D.N., & Romano, J.P. (1994). The stationary bootstrap. *Journal of the American Statistical Association*, 89(428), 1303-1313.

## 7. Statistical Validation

The system includes formal hypothesis tests (see `src/statistical_tests.py`):

| Test           | H0                     | Purpose                                    |
|----------------|------------------------|--------------------------------------------|
| ADF            | Unit root (non-stat.)  | Verify returns are stationary              |
| Ljung-Box      | No autocorrelation     | Validate IID assumption for bootstrap      |
| Jarque-Bera    | Normality              | Quantify non-normality (expected to fail)  |
| t-test         | Mean return = 0        | Is the strategy edge statistically real?   |

### Walk-Forward Validation

Out-of-sample testing via expanding-window walk-forward:
- Split data into N segments (default 5)
- For each split: train on expanding in-sample, test on next segment
- Report both IS and OOS Sharpe ratios
- Sharpe degradation ratio = OOS Sharpe / IS Sharpe

A degradation ratio close to 1.0 suggests the strategy is not overfit.

## 8. Known Assumptions & Limitations

1. **Long-only**: The system only takes long positions. Short selling is not implemented.
2. **Single position per symbol**: No pyramiding or scaling into positions.
3. **IID trade returns**: Bootstrap assumes trades are independent. This is approximately valid for non-overlapping positions but breaks down for overlapping or correlated positions.
4. **Constant reward/risk ratio**: The Kelly criterion assumes a fixed 2:1 R:R, which may not reflect actual market conditions.
5. **No order book modeling**: Market impact uses a simplified power-law, not actual order book data.
6. **Sample size**: With N=20 trades per symbol, statistical tests have low power. Results should be interpreted cautiously.
7. **No regime-dependent sizing**: Position size does not adapt to the confidence of regime classification.
8. **Linear calibration**: Win probability calibration uses a linear map, which may not reflect the true relationship.
9. **No transaction cost optimization**: Entry/exit timing does not minimize transaction costs.
10. **Backtests are in-sample**: Until walk-forward validation is run on live data, all reported metrics are subject to overfitting.

## References

- Almgren, R., & Chriss, N. (2001). Optimal execution of portfolio transactions. *Journal of Risk*, 3(2), 5-39.
- Artzner, P., Delbaen, F., Eber, J.M., & Heath, D. (1999). Coherent measures of risk. *Mathematical Finance*, 9(3), 203-228.
- Kelly, J.L. (1956). A new interpretation of information rate. *Bell System Technical Journal*, 35(4), 917-926.
- Politis, D.N., & Romano, J.P. (1994). The stationary bootstrap. *JASA*, 89(428), 1303-1313.
- Said, S.E., & Dickey, D.A. (1984). Testing for unit roots in autoregressive-moving average models. *Biometrika*, 71(3), 599-607.
- Wilder, J.W. (1978). *New Concepts in Technical Trading Systems*. Trend Research.
