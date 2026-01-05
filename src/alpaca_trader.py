"""
Alpaca Trading Client Wrapper
Provides a unified interface for Alpaca API operations.
"""

import os
import logging
from typing import Dict, Any, Optional

from dotenv import load_dotenv
from alpaca_trade_api import REST

logger = logging.getLogger("AlpacaTrader")


class AlpacaTrader:
    """
    Production-grade wrapper for Alpaca Markets API.

    Handles authentication, connection management, and provides
    standardized access to account data and market operations.
    """

    def __init__(self) -> None:
        load_dotenv()
        self.api: Optional[REST] = None
        self._connected: bool = False

    def connect(self) -> bool:
        """
        Establishes authenticated connection to Alpaca API.

        Returns:
            bool: True if connection successful, False otherwise.
        """
        try:
            api_key = os.getenv('ALPACA_API_KEY')
            secret_key = os.getenv('ALPACA_SECRET_KEY')
            base_url = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')

            if not api_key or not secret_key:
                logger.error("AuthError | Missing API credentials in environment.")
                return False

            self.api = REST(
                key_id=api_key,
                secret_key=secret_key,
                base_url=base_url
            )

            # Validate connection by fetching account
            self.api.get_account()
            self._connected = True
            logger.info("ConnectionEstablished | Alpaca API handshake successful.")
            return True

        except Exception as e:
            logger.error(f"ConnectionError | Failed to authenticate: {e}")
            self._connected = False
            return False

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
        clock = self.api.get_clock()
        return clock.is_open

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
            return None

    def get_positions(self) -> list:
        """Returns all current open positions."""
        if not self._connected or not self.api:
            return []
        return self.api.list_positions()

    def get_orders(self, status: str = 'open') -> list:
        """Returns orders filtered by status."""
        if not self._connected or not self.api:
            return []
        return self.api.list_orders(status=status)
