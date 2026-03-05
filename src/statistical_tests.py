"""
Formal statistical test suite for strategy validation.

Provides hypothesis tests and diagnostics that any quantitative
researcher should apply before trusting backtest results.

References:
- Augmented Dickey-Fuller: Said & Dickey (1984)
- Ljung-Box: Ljung & Box (1978)
- Jarque-Bera: Jarque & Bera (1987)

Assumptions tested:
- Stationarity (ADF): required for meaningful statistical inference on returns
- No autocorrelation (Ljung-Box): required for IID assumption in bootstrap
- Normality (Jarque-Bera): tested but NOT assumed; reported for awareness
- Significance (t-test): H0: mean return = 0 (is the strategy edge real?)
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

# Try to import statsmodels; fall back gracefully
try:
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.stats.diagnostic import acorr_ljungbox
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    logger.warning("statsmodels not installed. ADF and Ljung-Box tests unavailable.")


def stationarity_test(returns: np.ndarray, significance: float = 0.05) -> Dict:
    """
    Augmented Dickey-Fuller test for stationarity.

    Tests H0: the series has a unit root (non-stationary).
    Rejecting H0 (p < significance) means the series is stationary.

    Args:
        returns: Array of returns to test
        significance: Significance level (default 0.05)

    Returns:
        Dict with test statistic, p-value, and interpretation
    """
    if not HAS_STATSMODELS:
        return {
            'test': 'ADF',
            'error': 'statsmodels not installed',
            'is_stationary': None
        }

    if len(returns) < 20:
        return {
            'test': 'ADF',
            'error': 'insufficient_data',
            'is_stationary': None,
            'n_observations': len(returns)
        }

    result = adfuller(returns, autolag='AIC')
    adf_stat, p_value, used_lag, n_obs, critical_values, _ = result

    is_stationary = p_value < significance

    logger.info(f"ADF Test: stat={adf_stat:.4f}, p={p_value:.4f}, "
                f"stationary={is_stationary}")

    return {
        'test': 'ADF',
        'statistic': adf_stat,
        'p_value': p_value,
        'used_lag': used_lag,
        'n_observations': n_obs,
        'critical_values': {f'{k}': v for k, v in critical_values.items()},
        'is_stationary': is_stationary,
        'significance': significance
    }


def autocorrelation_test(returns: np.ndarray, lags: int = 10,
                         significance: float = 0.05) -> Dict:
    """
    Ljung-Box test for autocorrelation.

    Tests H0: no autocorrelation up to the specified lag.
    Rejecting H0 means significant autocorrelation exists,
    which violates the IID assumption used in standard bootstrap.

    Args:
        returns: Array of returns to test
        lags: Number of lags to test (default 10)
        significance: Significance level

    Returns:
        Dict with test statistics and interpretation
    """
    if not HAS_STATSMODELS:
        return {
            'test': 'Ljung-Box',
            'error': 'statsmodels not installed',
            'has_autocorrelation': None
        }

    if len(returns) < lags + 10:
        return {
            'test': 'Ljung-Box',
            'error': 'insufficient_data',
            'has_autocorrelation': None,
            'n_observations': len(returns)
        }

    lb_result = acorr_ljungbox(returns, lags=lags, return_df=True)
    # Use the last lag's p-value as the overall test
    final_p = lb_result['lb_pvalue'].iloc[-1]
    has_autocorrelation = final_p < significance

    logger.info(f"Ljung-Box Test: final_p={final_p:.4f}, "
                f"autocorrelation={'YES' if has_autocorrelation else 'NO'}")

    return {
        'test': 'Ljung-Box',
        'lags_tested': lags,
        'p_values': lb_result['lb_pvalue'].tolist(),
        'statistics': lb_result['lb_stat'].tolist(),
        'final_p_value': final_p,
        'has_autocorrelation': has_autocorrelation,
        'significance': significance,
        'implication': 'IID assumption may be violated' if has_autocorrelation else 'IID assumption not rejected'
    }


def normality_test(returns: np.ndarray, significance: float = 0.05) -> Dict:
    """
    Jarque-Bera test for normality.

    Tests H0: returns are normally distributed.
    Financial returns are almost never normal (fat tails, skew),
    so rejection is expected. This test quantifies HOW non-normal.

    Args:
        returns: Array of returns to test
        significance: Significance level

    Returns:
        Dict with skewness, kurtosis, test statistic, and interpretation
    """
    if len(returns) < 8:
        return {
            'test': 'Jarque-Bera',
            'error': 'insufficient_data',
            'is_normal': None
        }

    skewness = stats.skew(returns)
    kurtosis = stats.kurtosis(returns)  # excess kurtosis (normal = 0)
    jb_stat, p_value = stats.jarque_bera(returns)

    is_normal = p_value >= significance

    logger.info(f"Jarque-Bera Test: skew={skewness:.4f}, kurt={kurtosis:.4f}, "
                f"p={p_value:.4f}, normal={is_normal}")

    return {
        'test': 'Jarque-Bera',
        'statistic': jb_stat,
        'p_value': p_value,
        'skewness': skewness,
        'excess_kurtosis': kurtosis,
        'is_normal': is_normal,
        'significance': significance,
        'note': 'Financial returns are typically non-normal. Rejection is expected.'
    }


def strategy_significance_test(returns: np.ndarray, significance: float = 0.05) -> Dict:
    """
    One-sample t-test: H0: mean return = 0 (no edge).

    This is the most important test: does the strategy have a
    statistically significant positive mean return?

    Args:
        returns: Array of trade returns
        significance: Significance level

    Returns:
        Dict with t-statistic, p-value, confidence interval, and interpretation
    """
    if len(returns) < 3:
        return {
            'test': 't-test',
            'error': 'insufficient_data',
            'is_significant': None
        }

    t_stat, p_value = stats.ttest_1samp(returns, 0.0)
    n = len(returns)
    mean_return = np.mean(returns)
    se = stats.sem(returns)

    # 95% confidence interval for mean return
    ci_lower = mean_return - stats.t.ppf(1 - significance / 2, n - 1) * se
    ci_upper = mean_return + stats.t.ppf(1 - significance / 2, n - 1) * se

    is_significant = p_value < significance and mean_return > 0

    logger.info(f"Strategy t-test: mean={mean_return:.6f}, t={t_stat:.4f}, "
                f"p={p_value:.4f}, significant={is_significant}")

    return {
        'test': 't-test (H0: mean=0)',
        'mean_return': mean_return,
        't_statistic': t_stat,
        'p_value': p_value,
        'n_trades': n,
        'standard_error': se,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'is_significant': is_significant,
        'significance': significance
    }


def sensitivity_analysis(returns: np.ndarray, base_params: Dict,
                         param_ranges: Optional[Dict] = None) -> Dict:
    """
    Parameter sensitivity analysis.

    Perturbs key parameters by +/- percentages and measures how
    performance metrics change. Identifies which parameters the
    system is most sensitive to.

    Args:
        returns: Array of trade returns (used for base metrics)
        base_params: Dict of parameter names to base values
            e.g., {'kelly_fraction': 0.5, 'atr_multiplier': 2.0}
        param_ranges: Dict of parameter names to perturbation percentages
            e.g., {'kelly_fraction': [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]}
            Default: +/- 10%, 20%, 30% for each parameter

    Returns:
        Dict with sensitivity table and most sensitive parameter
    """
    if len(returns) < 5:
        return {'error': 'insufficient_data'}

    if param_ranges is None:
        # Default: +/- 10%, 20%, 30%
        multipliers = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
        param_ranges = {name: [v * m for m in multipliers] for name, v in base_params.items()}

    # Base metrics
    base_sharpe = _compute_sharpe(returns)
    base_mean = np.mean(returns)

    results = {}
    max_sensitivity = 0.0
    most_sensitive_param = None

    for param_name, values in param_ranges.items():
        param_results = []
        for val in values:
            # Scale returns proportionally to parameter change
            base_val = base_params.get(param_name, 1.0)
            if abs(base_val) < 1e-10:
                continue
            scale = val / base_val
            scaled_returns = returns * scale
            sharpe = _compute_sharpe(scaled_returns)
            mean_ret = np.mean(scaled_returns)
            max_dd = _compute_max_dd(scaled_returns)

            param_results.append({
                'value': val,
                'scale': scale,
                'sharpe': sharpe,
                'mean_return': mean_ret,
                'max_drawdown': max_dd
            })

        # Compute sensitivity: range of Sharpe values
        sharpes = [r['sharpe'] for r in param_results]
        sensitivity = max(sharpes) - min(sharpes) if sharpes else 0.0

        if sensitivity > max_sensitivity:
            max_sensitivity = sensitivity
            most_sensitive_param = param_name

        results[param_name] = {
            'values': param_results,
            'sharpe_range': sensitivity,
            'base_value': base_params.get(param_name)
        }

    return {
        'sensitivity_table': results,
        'most_sensitive_parameter': most_sensitive_param,
        'max_sharpe_range': max_sensitivity,
        'base_sharpe': base_sharpe,
        'n_trades': len(returns)
    }


def run_all_tests(returns: np.ndarray) -> Dict:
    """
    Run the complete statistical test suite on a return series.

    Args:
        returns: Array of returns (trade-level or bar-level)

    Returns:
        Dict with all test results and overall assessment
    """
    results = {
        'stationarity': stationarity_test(returns),
        'autocorrelation': autocorrelation_test(returns),
        'normality': normality_test(returns),
        'significance': strategy_significance_test(returns),
        'n_observations': len(returns)
    }

    # Overall assessment
    warnings = []
    if results['stationarity'].get('is_stationary') is False:
        warnings.append('Returns may be non-stationary (ADF test failed)')
    if results['autocorrelation'].get('has_autocorrelation') is True:
        warnings.append('Significant autocorrelation detected (IID violated)')
    if results['significance'].get('is_significant') is False:
        warnings.append('Mean return not significantly different from zero')
    if len(returns) < 30:
        warnings.append(f'Small sample size (N={len(returns)}): results may be unreliable')

    results['warnings'] = warnings
    results['pass_count'] = sum([
        results['stationarity'].get('is_stationary', False) is True,
        results['autocorrelation'].get('has_autocorrelation', False) is False,
        results['significance'].get('is_significant', False) is True
    ])
    results['total_tests'] = 3

    return results


def _compute_sharpe(returns: np.ndarray, periods: int = 252) -> float:
    """Helper to compute annualized Sharpe ratio."""
    if len(returns) < 2:
        return 0.0
    return (np.mean(returns) / (np.std(returns, ddof=1) + 1e-10)) * np.sqrt(periods)


def _compute_max_dd(returns: np.ndarray) -> float:
    """Helper to compute max drawdown from returns."""
    cum = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak
    return np.min(dd) if len(dd) > 0 else 0.0
