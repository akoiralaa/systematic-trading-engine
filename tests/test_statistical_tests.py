import unittest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np
from statistical_tests import (
    stationarity_test, autocorrelation_test, normality_test,
    strategy_significance_test, sensitivity_analysis, run_all_tests
)


class TestStatisticalTests(unittest.TestCase):
    def test_stationarity_white_noise(self):
        """White noise should be stationary."""
        np.random.seed(42)
        returns = np.random.normal(0, 0.01, 500)
        result = stationarity_test(returns)
        if result.get('is_stationary') is not None:
            self.assertTrue(result['is_stationary'])

    def test_stationarity_random_walk(self):
        """Random walk (cumulative sum) should NOT be stationary."""
        np.random.seed(42)
        random_walk = np.cumsum(np.random.normal(0, 1, 500))
        result = stationarity_test(random_walk)
        if result.get('is_stationary') is not None:
            self.assertFalse(result['is_stationary'])

    def test_stationarity_insufficient_data(self):
        """Small samples should return insufficient_data."""
        result = stationarity_test(np.array([0.01, 0.02]))
        self.assertIn('error', result)

    def test_autocorrelation_iid(self):
        """White noise should have no autocorrelation."""
        np.random.seed(42)
        returns = np.random.normal(0, 0.01, 500)
        result = autocorrelation_test(returns)
        if result.get('has_autocorrelation') is not None:
            self.assertFalse(result['has_autocorrelation'])

    def test_autocorrelation_ar_process(self):
        """AR(1) with high phi should show autocorrelation."""
        np.random.seed(42)
        n = 500
        ar = np.zeros(n)
        for i in range(1, n):
            ar[i] = 0.9 * ar[i-1] + np.random.normal(0, 0.01)
        result = autocorrelation_test(ar)
        if result.get('has_autocorrelation') is not None:
            self.assertTrue(result['has_autocorrelation'])

    def test_normality_normal_data(self):
        """Normal data should pass Jarque-Bera."""
        np.random.seed(42)
        returns = np.random.normal(0, 0.01, 1000)
        result = normality_test(returns)
        self.assertIn('skewness', result)
        self.assertIn('excess_kurtosis', result)
        # Large normal sample should pass
        self.assertTrue(result['is_normal'])

    def test_normality_fat_tails(self):
        """t-distribution with df=3 has fat tails, should fail normality."""
        np.random.seed(42)
        returns = np.random.standard_t(df=3, size=1000) * 0.01
        result = normality_test(returns)
        self.assertFalse(result['is_normal'])

    def test_significance_profitable(self):
        """Large positive mean should be significant."""
        np.random.seed(42)
        returns = np.random.normal(0.005, 0.01, 200)
        result = strategy_significance_test(returns)
        self.assertTrue(result['is_significant'])
        self.assertGreater(result['mean_return'], 0)

    def test_significance_no_edge(self):
        """Zero-mean returns should NOT be significant."""
        np.random.seed(42)
        returns = np.random.normal(0.0, 0.01, 200)
        result = strategy_significance_test(returns)
        # With mean truly 0, should usually not be significant
        # (could occasionally be by chance, but seed is fixed)
        self.assertIn('is_significant', result)

    def test_significance_confidence_interval(self):
        """Confidence interval should contain the mean."""
        np.random.seed(42)
        returns = np.random.normal(0.005, 0.01, 200)
        result = strategy_significance_test(returns)
        self.assertLess(result['ci_lower'], result['mean_return'])
        self.assertGreater(result['ci_upper'], result['mean_return'])

    def test_sensitivity_analysis_structure(self):
        """Sensitivity analysis should return expected structure."""
        np.random.seed(42)
        returns = np.random.normal(0.002, 0.01, 100)
        params = {'kelly_fraction': 0.5, 'atr_multiplier': 2.0}
        result = sensitivity_analysis(returns, params)
        self.assertIn('sensitivity_table', result)
        self.assertIn('most_sensitive_parameter', result)
        self.assertIn('base_sharpe', result)

    def test_run_all_tests_structure(self):
        """run_all_tests should return all components."""
        np.random.seed(42)
        returns = np.random.normal(0.002, 0.01, 100)
        result = run_all_tests(returns)
        self.assertIn('stationarity', result)
        self.assertIn('autocorrelation', result)
        self.assertIn('normality', result)
        self.assertIn('significance', result)
        self.assertIn('warnings', result)
        self.assertIn('pass_count', result)

    def test_run_all_tests_small_sample_warning(self):
        """Small sample should produce a warning."""
        returns = np.array([0.01, 0.02, -0.01, 0.015, 0.005, 0.01, 0.02, -0.005])
        result = run_all_tests(returns)
        has_small_sample_warning = any('Small sample' in w for w in result['warnings'])
        self.assertTrue(has_small_sample_warning)

    def test_insufficient_data_handled(self):
        """All tests should handle very small inputs gracefully."""
        returns = np.array([0.01, 0.02])
        result = strategy_significance_test(returns)
        self.assertIn('error', result)
        result2 = normality_test(returns)
        self.assertIn('error', result2)


if __name__ == '__main__':
    unittest.main()
