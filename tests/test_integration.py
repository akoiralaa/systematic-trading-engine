import unittest
import sys
import os
import tempfile
import shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from unittest.mock import MagicMock
import numpy as np
import pandas as pd
from trading_pipeline import TradingPipeline
from regime_detector import RegimeDetector
from bayesian_kelly import BayesianKellyCriterion
from market_friction_model import MarketFrictionModel
from monte_carlo_stress_test import MonteCarloStressTest
from exit_manager import ExitManager
import position_tracker as pt_module
from position_tracker import PositionTracker


class TestFullPipelineIntegration(unittest.TestCase):
    """End-to-end integration tests with synthetic data."""

    def setUp(self):
        self.mock_api = MagicMock()
        mock_account = MagicMock()
        mock_account.buying_power = '100000'
        self.mock_api.get_account.return_value = mock_account
        self.engine = TradingPipeline(
            api=self.mock_api, account_equity=100000, fractional_kelly=0.5
        )

    def test_trending_market_signal(self):
        """Strong uptrend should produce a regime detection of TRENDING."""
        detector = RegimeDetector(atr_multiplier=2.0, min_vector_strength=0.51)
        prices = np.linspace(100, 130, 60)
        result = detector.detect_regime(prices, lookback=30)
        self.assertEqual(result['state'], 'TRENDING')
        self.assertIn('r_squared', result)
        self.assertTrue(result['is_significant'])

    def test_sideways_market_rejection(self):
        """Flat market should reject signals."""
        n = 60
        prices = np.full(n, 100.0)
        vectors = np.full(n, 100.0)
        strengths = np.full(n, 0.8)
        atr = np.full(n, 1.0)

        result = self.engine.execute_trading_cycle(
            'TEST', prices, vectors, strengths, atr, 100000
        )
        self.assertIsNone(result.get('trade'))

    def test_regime_to_kelly_flow(self):
        """Regime detection feeds into Kelly sizing correctly."""
        detector = RegimeDetector(atr_multiplier=2.0, min_vector_strength=0.51)
        kelly = BayesianKellyCriterion(account_equity=100000)

        prices = np.linspace(100, 130, 60)
        regime = detector.detect_regime(prices, lookback=30)
        self.assertEqual(regime['state'], 'TRENDING')

        # Kelly should produce a position for valid strength
        frac = kelly.calculate_kelly_fraction(0.75)
        self.assertGreater(frac, 0)

    def test_friction_reduces_buy_price(self):
        """Buy-side friction should increase execution price."""
        friction = MarketFrictionModel(market_impact_coeff=0.1, bid_ask_spread_bps=2.0)
        result = friction.calculate_total_friction(100, 100000, 100.0, side='buy')
        self.assertGreater(result['execution_price'], 100.0)

    def test_friction_sell_side(self):
        """Sell-side friction should decrease execution price."""
        friction = MarketFrictionModel(market_impact_coeff=0.1, bid_ask_spread_bps=2.0)
        result = friction.calculate_total_friction(100, 100000, 100.0, side='sell')
        self.assertLess(result['execution_price'], 100.0)

    def test_monte_carlo_with_returns(self):
        """Monte Carlo should produce valid stress test from synthetic returns."""
        mc = MonteCarloStressTest(initial_equity=100000, simulations=500)
        returns = np.random.normal(0.001, 0.02, 50)
        cone = mc.run_probability_cone(returns)
        self.assertLess(cone['p5_worst_case'], cone['p95_best_case'])

        var = mc.get_tail_risk_metrics(returns, alpha=0.95)
        self.assertIn('var_alpha', var)
        self.assertLessEqual(var['cvar_expected_shortfall'], var['var_alpha'])

    def test_exit_manager_integration(self):
        """ExitManager should detect profit target."""
        em = ExitManager(max_hold_minutes=240)
        position = {
            'symbol': 'AAPL',
            'entry_price': 100.0,
            'execution_price': 100.05,
            'stop_price': 95.0,
            'target_price': 110.0,
            'trailing_stop_activated': False,
            'trailing_stop_price': None,
            'entry_time': pd.Timestamp.now()
        }
        # Price at target
        should_exit, reason, price = em.check_exit(position, 110.5)
        self.assertTrue(should_exit)
        self.assertEqual(reason, 'profit_target')

    def test_statistical_tests_on_synthetic(self):
        """Statistical test suite should run on synthetic data."""
        try:
            from statistical_tests import run_all_tests
            np.random.seed(42)
            returns = np.random.normal(0.002, 0.01, 200)
            result = run_all_tests(returns)
            self.assertIn('stationarity', result)
            self.assertIn('normality', result)
            self.assertIn('significance', result)
            self.assertIn('pass_count', result)
        except ImportError:
            self.skipTest("statsmodels not available")


class TestPositionTrackerLifecycle(unittest.TestCase):
    """End-to-end position lifecycle test."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self._orig_state_dir = pt_module.STATE_DIR
        self._orig_positions_file = pt_module.POSITIONS_FILE
        self._orig_history_file = pt_module.TRADE_HISTORY_FILE
        pt_module.STATE_DIR = self.temp_dir
        pt_module.POSITIONS_FILE = os.path.join(self.temp_dir, 'positions.json')
        pt_module.TRADE_HISTORY_FILE = os.path.join(self.temp_dir, 'trade_history.csv')

    def tearDown(self):
        pt_module.STATE_DIR = self._orig_state_dir
        pt_module.POSITIONS_FILE = self._orig_positions_file
        pt_module.TRADE_HISTORY_FILE = self._orig_history_file
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_full_lifecycle(self):
        """Open, update, close position and verify P&L."""
        tracker = PositionTracker()

        trade = {
            'symbol': 'AAPL',
            'qty': 10,
            'entry_price': 100.0,
            'execution_price': 100.05,
            'timestamp': pd.Timestamp.now(),
            'stop_price': 95.0,
            'target_price': 110.0,
            'risk_per_share': 5.0,
            'vector_strength': 0.7,
            'regime': 'TRENDING',
            'kelly_fraction': 0.05
        }
        tracker.add_position(trade)
        self.assertTrue(tracker.has_position('AAPL'))

        # Update with rising price
        tracker.update_position('AAPL', 105.0, 2.0)
        pos = tracker.get_position('AAPL')
        self.assertEqual(pos['highest_price_seen'], 105.0)

        # Close at profit
        closed = tracker.close_position('AAPL', 108.0, 'profit_target')
        self.assertFalse(tracker.has_position('AAPL'))
        self.assertGreater(closed['total_pnl'], 0)
        self.assertEqual(closed['exit_reason'], 'profit_target')

        # Performance summary
        summary = tracker.get_performance_summary()
        self.assertEqual(summary['total_trades'], 1)
        self.assertEqual(summary['winners'], 1)
        self.assertEqual(summary['win_rate'], 100.0)


if __name__ == '__main__':
    unittest.main()
