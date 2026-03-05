"""
Alpaca Trading Client Wrapper
Provides a unified interface for Alpaca API operations with
reconnection logic and rate limit handling.
"""

import os
import time
import logging
import traceback
import functools
from typing import Dict, Any, Optional

from dotenv import load_dotenv
from alpaca_trade_api import REST

logger = logging.getLogger("AlpacaTrader")

# Retry decorator for rate-limited API calls
def retry_on_rate_limit(max_retries: int = 3, base_delay: float = 1.0):
    """
    Decorator that retries API calls on 429 (rate limit) or transient errors.
    Uses exponential backoff: delay = base_delay * 2^attempt.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    error_str = str(e)
                    # Retry on rate limit (429) or connection errors
                    is_rate_limit = '429' in error_str or 'rate limit' in error_str.lower()
                    is_connection = any(x in error_str.lower() for x in [
                        'connection', 'timeout', 'reset', 'broken pipe'
                    ])
                    if (is_rate_limit or is_connection) and attempt < max_retries:
                        delay = base_delay * (2 ** attempt)
                        logger.warning(
                            f"API retry {attempt+1}/{max_retries} for {func.__name__}: "
                            f"{error_str}. Waiting {delay:.1f}s"
                        )
                        time.sleep(delay)
                    else:
                        raise
            raise last_exception
        return wrapper
    return decorator


class AlpacaTrader:
    """
    Production-grade wrapper for Alpaca Markets API.

    Handles authentication, connection management, automatic reconnection,
    and provides standardized access to account data and market operations.
    """

    def __init__(self) -> None:
        load_dotenv()
        self.api: Optional[REST] = None
        self._connected: bool = False
        self._api_key: Optional[str] = None
        self._secret_key: Optional[str] = None
        self._base_url: Optional[str] = None

    def connect(self) -> bool:
        """
        Establishes authenticated connection to Alpaca API.

        Returns:
            bool: True if connection successful, False otherwise.
        """
        try:
            self._api_key = os.getenv('ALPACA_API_KEY')
            self._secret_key = os.getenv('ALPACA_SECRET_KEY')
            self._base_url = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')

            if not self._api_key or not self._secret_key:
                logger.error("AuthError | Missing API credentials in environment.")
                return False

            self.api = REST(
                key_id=self._api_key,
                secret_key=self._secret_key,
                base_url=self._base_url
            )

            # Validate connection by fetching account
            self.api.get_account()
            self._connected = True
            logger.info("ConnectionEstablished | Alpaca API handshake successful.")
            return True

        except Exception as e:
            logger.error(f"ConnectionError | Failed to authenticate: {e}")
            logger.debug(traceback.format_exc())
            self._connected = False
            return False

    def ensure_connected(self) -> bool:
        """
        Health check with auto-reconnect on failure.
        Uses exponential backoff with 3 retries.

        Returns:
            bool: True if connected (possibly after reconnect), False if all retries failed.
        """
        if self._connected and self.api:
            try:
                self.api.get_account()
                return True
            except Exception:
                logger.warning("Connection health check failed. Attempting reconnect...")
                self._connected = False

        # Attempt reconnection with exponential backoff
        for attempt in range(3):
            delay = 2 ** attempt
            logger.info(f"Reconnection attempt {attempt+1}/3 (delay={delay}s)")
            time.sleep(delay)
            if self.connect():
                logger.info("Reconnection successful.")
                return True

        logger.error("All reconnection attempts failed.")
        return False

    @retry_on_rate_limit(max_retries=3, base_delay=1.0)
    def get_account_info(self) -> Dict[str, Any]:
        """
        Retrieves current account status and capital metrics.

        Returns:
            dict: Account information including cash, buying_power, portfolio_value.
        """
        if not self._connected or not self.api:
            raise ConnectionError("Not connected to Alpaca API.")

        account = self.api.get_account()
        return {
            'cash': account.cash,
            'buying_power': account.buying_power,
            'portfolio_value': account.portfolio_value,
            'equity': account.equity,
            'status': account.status,
            'pattern_day_trader': account.pattern_day_trader,
            'trading_blocked': account.trading_blocked,
            'account_blocked': account.account_blocked,
        }

    def is_market_open(self) -> bool:
        """Checks if the market is currently open for trading."""
        if not self._connected or not self.api:
            return False
        try:
            clock = self.api.get_clock()
            return clock.is_open
        except Exception as e:
            logger.error(f"Failed to check market status: {e}")
            return False

    @retry_on_rate_limit(max_retries=3, base_delay=1.0)
    def place_order(
        self,
        symbol: str,
        qty: int,
        side: str,
        order_type: str = 'market',
        time_in_force: str = 'day',
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Submits an order to the Alpaca API.

        Args:
            symbol: Ticker symbol (e.g., 'AAPL')
            qty: Number of shares
            side: 'buy' or 'sell'
            order_type: 'market', 'limit', 'stop', 'stop_limit'
            time_in_force: 'day', 'gtc', 'ioc', 'fok'
            limit_price: Limit price for limit orders
            stop_price: Stop price for stop orders

        Returns:
            dict: Order confirmation or None if failed.
        """
        if not self._connected or not self.api:
            raise ConnectionError("Not connected to Alpaca API.")

        try:
            order = self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type=order_type,
                time_in_force=time_in_force,
                limit_price=limit_price,
                stop_price=stop_price
            )
            logger.info(f"OrderSubmitted | {side.upper()} {qty} {symbol} @ {order_type}")
            return {
                'id': order.id,
                'status': order.status,
                'symbol': order.symbol,
                'qty': order.qty,
                'side': order.side,
                'type': order.type,
            }
        except Exception as e:
            logger.error(f"OrderError | Failed to submit order: {e}")
            logger.debug(traceback.format_exc())
            return None

    @retry_on_rate_limit(max_retries=3, base_delay=1.0)
    def get_positions(self) -> list:
        """Returns all current open positions."""
        if not self._connected or not self.api:
            return []
        return self.api.list_positions()

    @retry_on_rate_limit(max_retries=3, base_delay=1.0)
    def get_orders(self, status: str = 'open') -> list:
        """Returns orders filtered by status."""
        if not self._connected or not self.api:
            return []
        return self.api.list_orders(status=status)
