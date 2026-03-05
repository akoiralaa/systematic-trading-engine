import unittest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from bayesian_kelly import BayesianKellyCriterion

class TestBayesianKellyCriterion(unittest.TestCase):
    """Unit tests for Bayesian Kelly Criterion"""

    def setUp(self) -> None:
        self.kelly = BayesianKellyCriterion(account_equity=100000)

    def test_at_threshold_returns_value(self) -> None:
        """At minimum threshold, should return non-zero kelly"""
        frac = self.kelly.calculate_kelly_fraction(0.51)
        self.assertGreater(frac, 0)

    def test_below_threshold_returns_zero(self) -> None:
        """Below minimum strength should return zero"""
        frac = self.kelly.calculate_kelly_fraction(0.50)
        self.assertEqual(frac, 0.0)

    def test_high_confidence_higher_kelly(self) -> None:
        """Higher confidence should produce larger Kelly fraction"""
        frac_low = self.kelly.calculate_kelly_fraction(0.60)
        frac_high = self.kelly.calculate_kelly_fraction(0.90)
        self.assertLess(frac_low, frac_high)

    def test_kelly_capped_at_25_percent(self) -> None:
        """Kelly fraction should be capped at 25%"""
        frac = self.kelly.calculate_kelly_fraction(0.99)
        self.assertLessEqual(frac, 0.25)

    def test_position_size_scales_with_confidence(self) -> None:
        """Position size should increase with vector strength"""
        # current_price=50 means buying power of 100000 can buy 2000 shares max
        qty_low = self.kelly.calculate_position_size(0.55, 2.0, 100000, current_price=50.0)
        qty_high = self.kelly.calculate_position_size(0.90, 2.0, 100000, current_price=50.0)
        self.assertLessEqual(qty_low, qty_high)

    def test_position_size_respects_buying_power(self) -> None:
        """Position size should not exceed what buying power can afford"""
        # risk_per_share=1.0, buying_power=1000, current_price=10 -> can afford 100 shares
        qty = self.kelly.calculate_position_size(0.80, 1.0, 1000, current_price=10.0)
        self.assertLessEqual(qty * 10.0, 1000)

    def test_position_size_respects_concentration_limit(self) -> None:
        """Position should respect 20% concentration limit"""
        # With current_price=100, 20% of 100000 equity = 20000, so max 200 shares
        qty = self.kelly.calculate_position_size(0.90, 1.0, 100000, current_price=100.0, max_concentration=0.20)
        self.assertLessEqual(qty * 100.0, 100000 * 0.20)

    def test_below_minimum_strength_returns_unfavorable_ev(self) -> None:
        """Below minimum strength should return unfavorable EV"""
        ev = self.kelly.get_expected_value(0.50, 100.0, 95.0, 110.0)
        self.assertFalse(ev['is_favorable'])

    def test_get_expected_value_positive_for_good_trade(self) -> None:
        """Good trade with high strength and wide target should have positive EV"""
        ev = self.kelly.get_expected_value(0.90, 100.0, 95.0, 115.0)
        self.assertGreater(ev['ev'], 0)
        self.assertTrue(ev['is_favorable'])

    def test_conservative_calibration(self) -> None:
        """Win probability should be conservatively calibrated, not raw strength"""
        p = self.kelly._estimate_win_probability(0.90)
        # With conservative mapping [0.51,1.0] -> [0.51,0.65], 0.90 should map to ~0.61
        self.assertLess(p, 0.70)
        self.assertGreater(p, 0.50)

    def test_calibration_bounds(self) -> None:
        """Calibrated probability should be between 0.51 and 0.65"""
        for strength in [0.51, 0.6, 0.7, 0.8, 0.9, 1.0]:
            p = self.kelly._estimate_win_probability(strength)
            self.assertGreaterEqual(p, 0.50)
            self.assertLessEqual(p, 0.66)

    def test_ev_returns_calibrated_probability(self) -> None:
        """EV result should include calibrated_p_win field"""
        ev = self.kelly.get_expected_value(0.70, 100.0, 95.0, 110.0)
        self.assertIn('calibrated_p_win', ev)
        self.assertIn('raw_strength', ev)
        self.assertEqual(ev['raw_strength'], 0.70)

    def test_add_calibration_trade(self) -> None:
        """Adding calibration trades should be recorded"""
        initial_count = len(self.kelly._calibration_data)
        self.kelly.add_calibration_trade(0.75, 10.0)
        self.assertEqual(len(self.kelly._calibration_data), initial_count + 1)

if __name__ == '__main__':
    unittest.main()
