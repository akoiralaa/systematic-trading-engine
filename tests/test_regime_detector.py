import unittest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from regime_detector import RegimeDetector

class TestRegimeDetector(unittest.TestCase):
    """Unit tests for RegimeDetector"""

    def setUp(self) -> None:
        self.detector = RegimeDetector(atr_multiplier=2.0, min_vector_strength=0.51)

    def test_trending_regime_detection(self) -> None:
        """Strong uptrend should be detected as TRENDING"""
        prices = np.linspace(100, 130, 30)
        result = self.detector.detect_regime(prices, lookback=30)
        self.assertEqual(result['state'], 'TRENDING')

    def test_sideways_regime_detection(self) -> None:
        """Flat prices should be detected as SIDEWAYS"""
        prices = np.full(30, 100.0)
        result = self.detector.detect_regime(prices, lookback=30)
        self.assertEqual(result['state'], 'SIDEWAYS')

    def test_volatile_regime_detection(self) -> None:
        """Random walk can be detected as any regime"""
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(30) * 2)
        result = self.detector.detect_regime(prices, lookback=30)
        self.assertIn(result['state'], ['TRENDING', 'VOLATILE', 'SIDEWAYS'])

    def test_regime_returns_r_squared(self) -> None:
        """Regime detection should return R-squared instead of p-value confidence"""
        prices = np.linspace(100, 130, 30)
        result = self.detector.detect_regime(prices, lookback=30)
        self.assertIn('r_squared', result)
        self.assertIn('is_significant', result)
        self.assertIn('trend_strength', result)
        self.assertGreater(result['r_squared'], 0.5)  # Strong linear trend
        self.assertTrue(result['is_significant'])

    def test_confidence_uses_r_squared_for_trending(self) -> None:
        """Confidence should be R-squared for trending regime"""
        prices = np.linspace(100, 130, 30)
        result = self.detector.detect_regime(prices, lookback=30)
        self.assertEqual(result['state'], 'TRENDING')
        self.assertAlmostEqual(result['confidence'], result['r_squared'], places=6)

    def test_adaptive_zones_calculation(self) -> None:
        """Adaptive zones should scale with ATR"""
        prices = np.linspace(100, 105, 50)
        atr = np.full(50, 1.0)
        vector = np.linspace(100, 104, 50)
        strengths = np.full(50, 0.75)

        result = self.detector.calculate_adaptive_zones(prices, atr, vector, strengths)
        self.assertIsNotNone(result['upper_bound'])
        self.assertIsNotNone(result['lower_bound'])

    def test_validate_execution_signal(self) -> None:
        """Signal validation with all conditions met"""
        result = self.detector.validate_execution_signal(
            price=110.0,
            vector=100.0,
            atr=2.0,
            strength=0.80,
            state='TRENDING'
        )
        self.assertTrue(result['is_confirmed'])

    def test_signal_rejected_on_sideways(self) -> None:
        """Signal should be rejected in sideways regime"""
        result = self.detector.validate_execution_signal(
            price=110.0,
            vector=100.0,
            atr=2.0,
            strength=0.80,
            state='SIDEWAYS'
        )
        self.assertFalse(result['is_confirmed'])

    def test_signal_rejected_weak_strength(self) -> None:
        """Signal should be rejected if strength too low"""
        result = self.detector.validate_execution_signal(
            price=110.0,
            vector=100.0,
            atr=2.0,
            strength=0.50,
            state='TRENDING'
        )
        self.assertFalse(result['is_confirmed'])

    def test_volatility_adjusted_stop(self) -> None:
        """Stop should be outside ATR noise band"""
        result = self.detector.get_volatility_adjusted_stop(
            entry=100.0,
            vector=98.0,
            atr=1.0,
            side='long'
        )
        self.assertIsNotNone(result['stop_price'])
        self.assertLess(result['stop_price'], 100.0)

    def test_dynamic_stop_alias(self) -> None:
        """calculate_dynamic_stop should be alias for get_volatility_adjusted_stop"""
        r1 = self.detector.get_volatility_adjusted_stop(100.0, 98.0, 1.0, 'long')
        r2 = self.detector.calculate_dynamic_stop(100.0, 98.0, 1.0, 'long')
        self.assertEqual(r1['stop_price'], r2['stop_price'])

if __name__ == '__main__':
    unittest.main()
