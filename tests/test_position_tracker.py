import unittest
import sys
import os
import tempfile
import json
import shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import pandas as pd
import position_tracker as pt_module
from position_tracker import PositionTracker


def _make_trade(symbol='AAPL', entry_price=100.0, execution_price=100.05):
    return {
        'symbol': symbol,
        'qty': 10,
        'entry_price': entry_price,
        'execution_price': execution_price,
        'timestamp': pd.Timestamp.now(),
        'stop_price': 95.0,
        'target_price': 110.0,
        'risk_per_share': 5.0,
        'vector_strength': 0.75,
        'regime': 'TRENDING',
        'kelly_fraction': 0.05
    }


class TestPositionTracker(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self._orig_state_dir = pt_module.STATE_DIR
        self._orig_positions_file = pt_module.POSITIONS_FILE
        self._orig_history_file = pt_module.TRADE_HISTORY_FILE
        pt_module.STATE_DIR = self.temp_dir
        pt_module.POSITIONS_FILE = os.path.join(self.temp_dir, 'positions.json')
        pt_module.TRADE_HISTORY_FILE = os.path.join(self.temp_dir, 'trade_history.csv')
        self.tracker = PositionTracker()

    def tearDown(self):
        pt_module.STATE_DIR = self._orig_state_dir
        pt_module.POSITIONS_FILE = self._orig_positions_file
        pt_module.TRADE_HISTORY_FILE = self._orig_history_file
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_add_position(self):
        trade = _make_trade()
        self.tracker.add_position(trade)
        self.assertTrue(self.tracker.has_position('AAPL'))
        pos = self.tracker.get_position('AAPL')
        self.assertEqual(pos['qty'], 10)

    def test_add_position_stores_execution_price(self):
        trade = _make_trade(execution_price=100.10)
        self.tracker.add_position(trade)
        pos = self.tracker.get_position('AAPL')
        self.assertAlmostEqual(pos['execution_price'], 100.10)

    def test_close_position_uses_execution_price_for_pnl(self):
        trade = _make_trade(entry_price=100.0, execution_price=100.05)
        self.tracker.add_position(trade)
        closed = self.tracker.close_position('AAPL', 105.0, 'profit_target')
        # P&L should be from execution_price (100.05), not entry_price (100.0)
        expected_pnl_per_share = 105.0 - 100.05
        self.assertAlmostEqual(closed['pnl_per_share'], expected_pnl_per_share, places=2)

    def test_close_position_profit(self):
        trade = _make_trade(execution_price=100.0)
        self.tracker.add_position(trade)
        closed = self.tracker.close_position('AAPL', 110.0, 'profit_target')
        self.assertGreater(closed['total_pnl'], 0)
        self.assertEqual(closed['exit_reason'], 'profit_target')

    def test_close_position_loss(self):
        trade = _make_trade(execution_price=100.0)
        self.tracker.add_position(trade)
        closed = self.tracker.close_position('AAPL', 95.0, 'stop_loss')
        self.assertLess(closed['total_pnl'], 0)

    def test_has_position_false(self):
        self.assertFalse(self.tracker.has_position('MSFT'))

    def test_get_all_positions(self):
        self.tracker.add_position(_make_trade('AAPL'))
        self.tracker.add_position(_make_trade('MSFT'))
        positions = self.tracker.get_all_positions()
        self.assertEqual(len(positions), 2)
        self.assertIn('AAPL', positions)
        self.assertIn('MSFT', positions)

    def test_trailing_stop_activation(self):
        trade = _make_trade(entry_price=100.0, execution_price=100.0)
        self.tracker.add_position(trade)
        # Price goes up by 1x ATR (atr=2.0), should activate trailing stop
        self.tracker.update_position('AAPL', 102.0, 2.0)
        pos = self.tracker.get_position('AAPL')
        self.assertTrue(pos['trailing_stop_activated'])
        self.assertIsNotNone(pos['trailing_stop_price'])

    def test_trailing_stop_raises(self):
        trade = _make_trade(entry_price=100.0, execution_price=100.0)
        self.tracker.add_position(trade)
        self.tracker.update_position('AAPL', 103.0, 2.0)  # Activate
        pos = self.tracker.get_position('AAPL')
        first_trail = pos['trailing_stop_price']
        self.tracker.update_position('AAPL', 106.0, 2.0)  # Price higher
        pos = self.tracker.get_position('AAPL')
        self.assertGreater(pos['trailing_stop_price'], first_trail)

    def test_performance_summary_empty(self):
        summary = self.tracker.get_performance_summary()
        self.assertEqual(summary['total_trades'], 0)
        self.assertEqual(summary['win_rate'], 0.0)

    def test_performance_summary_with_trades(self):
        self.tracker.add_position(_make_trade('AAPL', execution_price=100.0))
        self.tracker.close_position('AAPL', 110.0, 'profit_target')
        self.tracker.add_position(_make_trade('MSFT', execution_price=100.0))
        self.tracker.close_position('MSFT', 95.0, 'stop_loss')
        summary = self.tracker.get_performance_summary()
        self.assertEqual(summary['total_trades'], 2)
        self.assertEqual(summary['winners'], 1)
        self.assertEqual(summary['losers'], 1)
        self.assertAlmostEqual(summary['win_rate'], 50.0)

    def test_state_persistence(self):
        trade = _make_trade()
        self.tracker.add_position(trade)
        # Create new tracker - should load saved state
        tracker2 = PositionTracker()
        self.assertTrue(tracker2.has_position('AAPL'))
        pos = tracker2.get_position('AAPL')
        self.assertEqual(pos['qty'], 10)

    def test_close_nonexistent_returns_empty(self):
        result = self.tracker.close_position('ZZZZ', 100.0, 'test')
        self.assertEqual(result, {})


if __name__ == '__main__':
    unittest.main()
