import unittest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from unittest.mock import patch, MagicMock
from alpaca_trader import AlpacaTrader, retry_on_rate_limit


class TestRetryDecorator(unittest.TestCase):
    def test_retry_on_429(self):
        """Should retry on 429 rate limit errors."""
        call_count = [0]

        @retry_on_rate_limit(max_retries=2, base_delay=0.01)
        def failing_func():
            call_count[0] += 1
            if call_count[0] < 3:
                raise Exception("429 Too Many Requests")
            return "success"

        result = failing_func()
        self.assertEqual(result, "success")
        self.assertEqual(call_count[0], 3)

    def test_no_retry_on_other_errors(self):
        """Should NOT retry on non-retryable errors."""
        @retry_on_rate_limit(max_retries=2, base_delay=0.01)
        def failing_func():
            raise ValueError("Bad input")

        with self.assertRaises(ValueError):
            failing_func()

    def test_retry_on_connection_error(self):
        """Should retry on connection errors."""
        call_count = [0]

        @retry_on_rate_limit(max_retries=2, base_delay=0.01)
        def failing_func():
            call_count[0] += 1
            if call_count[0] < 2:
                raise Exception("Connection reset by peer")
            return "ok"

        result = failing_func()
        self.assertEqual(result, "ok")

    def test_exhausted_retries_raises(self):
        """Should raise after exhausting retries."""
        @retry_on_rate_limit(max_retries=1, base_delay=0.01)
        def always_fails():
            raise Exception("429 Rate Limited")

        with self.assertRaises(Exception):
            always_fails()


class TestAlpacaTrader(unittest.TestCase):
    @patch('alpaca_trader.os.getenv')
    @patch('alpaca_trader.REST')
    def test_connect_success(self, mock_rest_cls, mock_getenv):
        mock_getenv.side_effect = lambda key, default=None: {
            'ALPACA_API_KEY': 'test_key',
            'ALPACA_SECRET_KEY': 'test_secret',
            'ALPACA_BASE_URL': 'https://paper-api.alpaca.markets'
        }.get(key, default)
        mock_rest_cls.return_value.get_account.return_value = MagicMock()

        trader = AlpacaTrader()
        result = trader.connect()
        self.assertTrue(result)
        self.assertTrue(trader._connected)

    @patch('alpaca_trader.os.getenv')
    def test_connect_missing_credentials(self, mock_getenv):
        mock_getenv.return_value = None
        trader = AlpacaTrader()
        result = trader.connect()
        self.assertFalse(result)

    @patch('alpaca_trader.os.getenv')
    @patch('alpaca_trader.REST')
    def test_connect_api_failure(self, mock_rest_cls, mock_getenv):
        mock_getenv.side_effect = lambda key, default=None: {
            'ALPACA_API_KEY': 'test_key',
            'ALPACA_SECRET_KEY': 'test_secret',
        }.get(key, default)
        mock_rest_cls.return_value.get_account.side_effect = Exception("Auth failed")

        trader = AlpacaTrader()
        result = trader.connect()
        self.assertFalse(result)

    def test_get_account_info_not_connected(self):
        trader = AlpacaTrader()
        with self.assertRaises(ConnectionError):
            trader.get_account_info()

    @patch('alpaca_trader.os.getenv')
    @patch('alpaca_trader.REST')
    def test_get_account_info_success(self, mock_rest_cls, mock_getenv):
        mock_getenv.side_effect = lambda key, default=None: {
            'ALPACA_API_KEY': 'key', 'ALPACA_SECRET_KEY': 'secret'
        }.get(key, default)
        mock_account = MagicMock()
        mock_account.cash = '50000'
        mock_account.buying_power = '100000'
        mock_account.portfolio_value = '100000'
        mock_account.equity = '100000'
        mock_account.status = 'ACTIVE'
        mock_account.pattern_day_trader = False
        mock_account.trading_blocked = False
        mock_account.account_blocked = False
        mock_rest_cls.return_value.get_account.return_value = mock_account

        trader = AlpacaTrader()
        trader.connect()
        info = trader.get_account_info()
        self.assertEqual(info['cash'], '50000')
        self.assertEqual(info['equity'], '100000')

    @patch('alpaca_trader.os.getenv')
    @patch('alpaca_trader.REST')
    def test_place_order_success(self, mock_rest_cls, mock_getenv):
        mock_getenv.side_effect = lambda key, default=None: {
            'ALPACA_API_KEY': 'key', 'ALPACA_SECRET_KEY': 'secret'
        }.get(key, default)
        mock_order = MagicMock()
        mock_order.id = 'order123'
        mock_order.status = 'accepted'
        mock_order.symbol = 'AAPL'
        mock_order.qty = '10'
        mock_order.side = 'buy'
        mock_order.type = 'market'
        mock_rest_cls.return_value.get_account.return_value = MagicMock()
        mock_rest_cls.return_value.submit_order.return_value = mock_order

        trader = AlpacaTrader()
        trader.connect()
        result = trader.place_order('AAPL', 10, 'buy')
        self.assertIsNotNone(result)
        self.assertEqual(result['id'], 'order123')

    @patch('alpaca_trader.os.getenv')
    @patch('alpaca_trader.REST')
    def test_place_order_failure(self, mock_rest_cls, mock_getenv):
        mock_getenv.side_effect = lambda key, default=None: {
            'ALPACA_API_KEY': 'key', 'ALPACA_SECRET_KEY': 'secret'
        }.get(key, default)
        mock_rest_cls.return_value.get_account.return_value = MagicMock()
        mock_rest_cls.return_value.submit_order.side_effect = Exception("Order failed")

        trader = AlpacaTrader()
        trader.connect()
        result = trader.place_order('AAPL', 10, 'buy')
        self.assertIsNone(result)

    def test_get_positions_not_connected(self):
        trader = AlpacaTrader()
        result = trader.get_positions()
        self.assertEqual(result, [])

    @patch('alpaca_trader.os.getenv')
    @patch('alpaca_trader.REST')
    def test_ensure_connected_healthy(self, mock_rest_cls, mock_getenv):
        mock_getenv.side_effect = lambda key, default=None: {
            'ALPACA_API_KEY': 'key', 'ALPACA_SECRET_KEY': 'secret'
        }.get(key, default)
        mock_rest_cls.return_value.get_account.return_value = MagicMock()

        trader = AlpacaTrader()
        trader.connect()
        result = trader.ensure_connected()
        self.assertTrue(result)


if __name__ == '__main__':
    unittest.main()
