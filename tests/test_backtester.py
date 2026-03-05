import unittest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from unittest.mock import MagicMock
import numpy as np
import pandas as pd
from backtester import AdvancedBacktester


def _make_mock_engine(trade_at_bars=None, entry_price=100.0, stop=95.0, target=110.0):
    """Create a mock engine that returns trades at specified bar indices."""
    if trade_at_bars is None:
        trade_at_bars = set()
    call_count = [0]

    def mock_cycle(symbol, prices, vector_prices, vector_strengths, atr_values, avg_volume):
        bar_idx = len(prices) - 1
        call_count[0] += 1
        if bar_idx in trade_at_bars:
            return {
                'trade': {
                    'symbol': symbol,
                    'qty': 10,
                    'entry_price': prices[-1],
                    'execution_price': prices[-1] * 1.001,
                    'stop_price': prices[-1] * 0.95,
                    'target_price': prices[-1] * 1.02,
                    'risk_per_share': prices[-1] * 0.05,
                    'vector_strength': 0.7,
                    'kelly_fraction': 0.05,
                    'timestamp': pd.Timestamp.now()
                }
            }
        return {'trade': None}

    engine = MagicMock()
    engine.execute_trading_cycle.side_effect = mock_cycle
    # Mock kelly criterion with reward_risk_ratio for look-ahead bias fix
    engine.kelly = MagicMock()
    engine.kelly.reward_risk_ratio = 2.0
    return engine


class TestAdvancedBacktester(unittest.TestCase):
    def test_no_trades_returns_zero_status(self):
        engine = _make_mock_engine(trade_at_bars=set())
        bt = AdvancedBacktester(engine, initial_equity=100000)
        prices = np.linspace(100, 100.5, 100)
        result = bt.run_backtest(prices, prices, np.full(100, 1.0), np.full(100, 10000))
        self.assertEqual(result.get('status'), 'ZERO_TRADES_EXECUTED')

    def test_backtest_with_trades(self):
        # Entry at bar 55, price trends up so target hit
        engine = _make_mock_engine(trade_at_bars={55})
        bt = AdvancedBacktester(engine, initial_equity=100000)
        prices = np.linspace(100, 115, 120)  # Strong uptrend - target (2%) hit
        atr = np.full(120, 1.0)
        volume = np.full(120, 10000)
        result = bt.run_backtest(prices, prices, atr, volume)
        self.assertIn('trade_count', result)
        self.assertGreaterEqual(result['trade_count'], 1)

    def test_sharpe_uses_ddof1(self):
        bt = AdvancedBacktester(MagicMock(), initial_equity=100000)
        returns = np.array([0.01, 0.02, -0.005, 0.015, 0.01])
        sharpe = bt._calculate_sharpe_ratio(returns)
        # Manually compute with ddof=1
        expected = (np.mean(returns) / (np.std(returns, ddof=1) + 1e-10)) * np.sqrt(252)
        self.assertAlmostEqual(sharpe, expected, places=4)

    def test_sortino_ratio(self):
        bt = AdvancedBacktester(MagicMock(), initial_equity=100000)
        returns = np.array([0.01, 0.02, -0.005, -0.01, 0.015])
        sortino = bt._calculate_sortino_ratio(returns)
        self.assertIsInstance(sortino, float)

    def test_max_drawdown(self):
        bt = AdvancedBacktester(MagicMock(), initial_equity=100000)
        cum_returns = np.array([1.0, 1.1, 1.05, 0.9, 1.0])
        mdd = bt._calculate_max_drawdown(cum_returns)
        # Max drawdown from 1.1 to 0.9 = (0.9-1.1)/1.1 = -0.1818
        self.assertLess(mdd, 0)
        self.assertAlmostEqual(mdd, (0.9 - 1.1) / 1.1, places=3)

    def test_force_close_at_end_of_data(self):
        # Trade at bar 55, then price stays flat - no stop/target hit
        engine = _make_mock_engine(trade_at_bars={55})
        bt = AdvancedBacktester(engine, initial_equity=100000)
        prices = np.full(70, 100.0)  # Flat prices, no stop/target
        atr = np.full(70, 1.0)
        volume = np.full(70, 10000)
        result = bt.run_backtest(prices, prices, atr, volume, max_hold_bars=100)
        # Trade should be force-closed at end of data
        if result.get('trade_count', 0) > 0:
            last_trade = bt.trades[-1]
            self.assertEqual(last_trade['exit_reason'], 'end_of_data')

    def test_stop_loss_exit(self):
        engine = _make_mock_engine(trade_at_bars={55})
        bt = AdvancedBacktester(engine, initial_equity=100000)
        # Price drops sharply after entry
        prices = np.concatenate([np.full(56, 100.0), np.linspace(100, 85, 44)])
        atr = np.full(100, 1.0)
        volume = np.full(100, 10000)
        result = bt.run_backtest(prices, prices, atr, volume)
        if result.get('trade_count', 0) > 0:
            has_stop = any(t.get('exit_reason') == 'stop_loss' for t in bt.trades)
            self.assertTrue(has_stop)

    def test_time_limit_exit(self):
        engine = _make_mock_engine(trade_at_bars={55})
        bt = AdvancedBacktester(engine, initial_equity=100000)
        prices = np.full(120, 100.0)  # Flat, no stop/target
        atr = np.full(120, 1.0)
        volume = np.full(120, 10000)
        result = bt.run_backtest(prices, prices, atr, volume, max_hold_bars=10)
        if result.get('trade_count', 0) > 0:
            has_time = any(t.get('exit_reason') == 'time_limit' for t in bt.trades)
            self.assertTrue(has_time)

    def test_performance_report_structure(self):
        engine = _make_mock_engine(trade_at_bars={55})
        bt = AdvancedBacktester(engine, initial_equity=100000)
        prices = np.linspace(100, 115, 120)
        result = bt.run_backtest(prices, prices, np.full(120, 1.0), np.full(120, 10000))
        if result.get('trade_count', 0) > 0:
            self.assertIn('sharpe_ratio', result)
            self.assertIn('sortino_ratio', result)
            self.assertIn('max_drawdown', result)
            self.assertIn('exit_reasons', result)
            self.assertIn('avg_bars_held', result)

    def test_calmar_ratio(self):
        bt = AdvancedBacktester(MagicMock(), initial_equity=100000)
        returns = np.array([0.01, 0.02, -0.005, 0.015])
        calmar = bt._calculate_calmar_ratio(returns, -0.1)
        self.assertIsInstance(calmar, float)
        self.assertGreater(calmar, 0)


if __name__ == '__main__':
    unittest.main()
