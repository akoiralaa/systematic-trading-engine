import unittest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from unittest.mock import MagicMock
import numpy as np
from trading_pipeline import TradingPipeline


class TestTradingPipeline(unittest.TestCase):
    """Integration tests for TradingPipeline (formerly QuantumFractalEngine)."""

    def setUp(self):
        self.mock_api = MagicMock()
        mock_account = MagicMock()
        mock_account.buying_power = '100000'
        self.mock_api.get_account.return_value = mock_account
        self.engine = TradingPipeline(
            api=self.mock_api, account_equity=100000, fractional_kelly=0.5
        )

    def test_stress_test_strategy_structure(self):
        """stress_test_strategy should return all 4 components."""
        returns = np.random.normal(0.001, 0.02, 50)
        result = self.engine.stress_test_strategy(returns)
        self.assertIn('probability_cone', result)
        self.assertIn('risk_of_ruin', result)
        self.assertIn('crash_stress_test', result)
        self.assertIn('var_cvar', result)

    def test_stress_test_small_returns(self):
        """Should handle small returns array."""
        returns = np.array([0.01, -0.005, 0.02])
        result = self.engine.stress_test_strategy(returns)
        self.assertIn('probability_cone', result)

    def test_institutional_report_structure(self):
        """Report should contain all required sections."""
        returns = np.random.normal(0.001, 0.02, 50)
        report = self.engine.get_institutional_report(returns)
        self.assertIn('risk_metrics', report)
        self.assertIn('stress_tests', report)
        self.assertIn('probability_cone', report)
        self.assertIn('kelly_usage', report)

    def test_institutional_report_risk_metrics(self):
        """Risk metrics should be numeric."""
        returns = np.random.normal(0.001, 0.02, 50)
        report = self.engine.get_institutional_report(returns)
        rm = report['risk_metrics']
        self.assertIsInstance(rm['var_95'], (int, float, np.floating))
        self.assertIsInstance(rm['cvar_95'], (int, float, np.floating))
        self.assertIsInstance(rm['risk_of_ruin_20pct'], (int, float, np.floating))

    def test_institutional_report_cone_percentiles(self):
        """Probability cone p5 < p50 < p95."""
        returns = np.random.normal(0.001, 0.02, 50)
        report = self.engine.get_institutional_report(returns)
        cone = report['probability_cone']
        self.assertLess(cone['p5'], cone['p50'])
        self.assertLess(cone['p50'], cone['p95'])

    def test_risk_of_ruin_in_range(self):
        """Risk of ruin should be between 0 and 100."""
        returns = np.random.normal(0.001, 0.02, 50)
        result = self.engine.stress_test_strategy(returns)
        ror = result['risk_of_ruin']['risk_of_ruin_pct']
        self.assertGreaterEqual(ror, 0)
        self.assertLessEqual(ror, 100)

    def test_all_positive_returns(self):
        """All positive returns should have low risk of ruin."""
        returns = np.full(50, 0.01)
        result = self.engine.stress_test_strategy(returns)
        self.assertLess(result['risk_of_ruin']['risk_of_ruin_pct'], 5)

    def test_mixed_returns(self):
        """Mixed returns should produce valid stress test."""
        returns = np.array([0.02, -0.01, 0.015, -0.005, 0.01] * 10)
        result = self.engine.stress_test_strategy(returns)
        self.assertIn('var_cvar', result)
        self.assertIn('var_alpha', result['var_cvar'])

    def test_execute_cycle_rejects_weak_signal(self):
        """Weak signal strength should be rejected."""
        n = 60
        prices = np.linspace(100, 100.5, n)
        vectors = np.full(n, 100.0)
        strengths = np.full(n, 0.3)  # Below 0.51 threshold
        atr = np.full(n, 1.0)
        result = self.engine.execute_trading_cycle(
            'TEST', prices, vectors, strengths, atr, 100000
        )
        self.assertIsNone(result.get('trade'))

    def test_execute_cycle_sideways_rejection(self):
        """Sideways regime should reject signals."""
        n = 60
        prices = np.full(n, 100.0)  # Flat = sideways
        vectors = np.full(n, 100.0)
        strengths = np.full(n, 0.8)
        atr = np.full(n, 1.0)
        result = self.engine.execute_trading_cycle(
            'TEST', prices, vectors, strengths, atr, 100000
        )
        self.assertIsNone(result.get('trade'))

    def test_kelly_usage_in_report(self):
        """Report should contain Kelly parameters."""
        returns = np.random.normal(0.001, 0.02, 50)
        report = self.engine.get_institutional_report(returns)
        kelly = report['kelly_usage']
        self.assertEqual(kelly['fractional_kelly'], 0.5)
        self.assertEqual(kelly['reward_risk_ratio'], 2.0)
        self.assertEqual(kelly['min_vector_strength'], 0.51)


if __name__ == '__main__':
    unittest.main()
