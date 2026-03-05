import unittest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from datetime import datetime, timedelta
from exit_manager import ExitManager


def _make_position(entry_price=100.0, stop=95.0, target=110.0,
                   minutes_ago=30, trailing=False, trail_price=None):
    return {
        'symbol': 'AAPL',
        'entry_price': entry_price,
        'execution_price': entry_price + 0.05,
        'stop_price': stop,
        'target_price': target,
        'trailing_stop_activated': trailing,
        'trailing_stop_price': trail_price,
        'entry_time': datetime.now() - timedelta(minutes=minutes_ago)
    }


class TestExitManager(unittest.TestCase):
    def setUp(self):
        self.em = ExitManager(max_hold_minutes=240, eod_close_minutes=5)

    def test_profit_target_hit(self):
        pos = _make_position(target=110.0)
        should_exit, reason, price = self.em.check_exit(pos, 110.5)
        self.assertTrue(should_exit)
        self.assertEqual(reason, 'profit_target')

    def test_stop_loss_hit(self):
        pos = _make_position(stop=95.0)
        should_exit, reason, price = self.em.check_exit(pos, 94.0)
        self.assertTrue(should_exit)
        self.assertEqual(reason, 'stop_loss')

    def test_trailing_stop_hit(self):
        pos = _make_position(trailing=True, trail_price=105.0)
        should_exit, reason, price = self.em.check_exit(pos, 104.0)
        self.assertTrue(should_exit)
        self.assertEqual(reason, 'trailing_stop')

    def test_trailing_stop_not_activated(self):
        pos = _make_position(trailing=False, trail_price=None)
        should_exit, reason, price = self.em.check_exit(pos, 96.0)
        # Should hit stop loss, not trailing stop
        if should_exit:
            self.assertNotEqual(reason, 'trailing_stop')

    def test_time_limit_exceeded(self):
        pos = _make_position(minutes_ago=250)
        should_exit, reason, price = self.em.check_exit(pos, 100.0)
        self.assertTrue(should_exit)
        self.assertEqual(reason, 'time_limit')

    def test_no_exit_conditions_met(self):
        pos = _make_position(stop=90.0, target=120.0, minutes_ago=30)
        should_exit, reason, price = self.em.check_exit(pos, 100.0)
        # Only fails if we're not near EOD
        if datetime.now().hour < 15 or (datetime.now().hour == 15 and datetime.now().minute < 55):
            self.assertFalse(should_exit)
            self.assertIsNone(reason)

    def test_profit_target_priority_over_time(self):
        """When both profit target and time limit are met, profit target wins (checked first)."""
        pos = _make_position(target=99.0, minutes_ago=250)
        should_exit, reason, price = self.em.check_exit(pos, 100.0)
        self.assertTrue(should_exit)
        self.assertEqual(reason, 'profit_target')

    def test_stop_loss_priority_over_trailing(self):
        """Stop loss is checked before trailing stop."""
        pos = _make_position(stop=95.0, trailing=True, trail_price=97.0)
        should_exit, reason, price = self.em.check_exit(pos, 94.0)
        self.assertTrue(should_exit)
        self.assertEqual(reason, 'stop_loss')

    def test_get_exit_summary_structure(self):
        pos = _make_position()
        summary = self.em.get_exit_summary(pos, 102.0)
        self.assertIn('current_pnl_pct', summary)
        self.assertIn('distance_to_target_pct', summary)
        self.assertIn('distance_to_stop_pct', summary)
        self.assertIn('hold_time_minutes', summary)
        self.assertIn('time_remaining_minutes', summary)
        self.assertIn('trailing_stop_active', summary)

    def test_format_exit_status_returns_string(self):
        pos = _make_position()
        status = self.em.format_exit_status(pos, 102.0)
        self.assertIsInstance(status, str)
        self.assertIn('P&L', status)
        self.assertIn('Target', status)

    def test_different_max_hold(self):
        em_short = ExitManager(max_hold_minutes=60)
        pos = _make_position(stop=80.0, target=130.0, minutes_ago=65)
        should_exit, reason, _ = em_short.check_exit(pos, 100.0)
        self.assertTrue(should_exit)
        self.assertEqual(reason, 'time_limit')

    def test_trailing_stop_distance_in_summary(self):
        pos = _make_position(trailing=True, trail_price=98.0)
        summary = self.em.get_exit_summary(pos, 102.0)
        self.assertIn('distance_to_trailing_pct', summary)


if __name__ == '__main__':
    unittest.main()
