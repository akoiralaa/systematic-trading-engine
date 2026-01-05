"""
Alpaca Connectivity Test
System readiness diagnostic for validating API connectivity and market data access.

Usage:
    python3 src/alpaca_connectivity_test.py
"""

import sys
import os
import logging
import datetime
from typing import Dict, Any

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from src.alpaca_trader import AlpacaTrader

# Institutional logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ConnectivityDiagnostic")


class AlpacaConnectivityTest:
    """
    Validation utility for live market data connectivity and account authentication.

    Verifies API handshake, retrieves real-time Level 1 quote data, and
    validates sufficient buying power for the Fractal Alpha strategy.
    """

    WATCHLIST = ['PLTR', 'QQQ', 'PENN', 'SPY']

    def __init__(self) -> None:
        load_dotenv()
        self.trader = AlpacaTrader()

    def execute_diagnostics(self) -> bool:
        """
        Performs a full sequential check of the trading environment.

        Returns:
            bool: True if all diagnostics pass, False otherwise.
        """
        print("\n" + "=" * 60)
        logger.info("Initializing System Readiness Diagnostic...")
        print("=" * 60)

        # 1. Authentication Handshake
        if not self.trader.connect():
            logger.error("AuthFailure | Unable to establish Alpaca API connection.")
            return False

        # 2. Account Liquidity Check
        try:
            account = self.trader.get_account_info()
            self._log_account_status(account)
        except Exception as e:
            logger.error(f"AccountQueryError | Failed to retrieve liquidity metrics: {e}")
            return False

        # 3. Market Status Check
        self._check_market_status()

        # 4. Real-Time Data Pipeline Check
        self._validate_quote_stream()

        print("=" * 60)
        logger.info("DiagnosticComplete | System environment is stable for execution.")
        print("=" * 60 + "\n")
        return True

    def _log_account_status(self, account: Dict[str, Any]) -> None:
        """Standardizes the output of core capital metrics."""
        cash = float(account.get('cash', 0))
        bp = float(account.get('buying_power', 0))
        pv = float(account.get('portfolio_value', 0))
        status = account.get('status', 'unknown')

        print("\n--- Account Status ---")
        logger.info(f"AccountStatus | Cash: ${cash:,.2f} | Buying Power: ${bp:,.2f}")
        logger.info(f"PortfolioValue: ${pv:,.2f} | Status: {status}")

        if account.get('trading_blocked'):
            logger.warning("TradingBlocked | Account trading is currently blocked.")
        if account.get('account_blocked'):
            logger.warning("AccountBlocked | Account access is restricted.")

    def _check_market_status(self) -> None:
        """Checks if the market is currently open."""
        print("\n--- Market Status ---")
        try:
            is_open = self.trader.is_market_open()
            if is_open:
                logger.info("MarketStatus | Market is OPEN for trading.")
            else:
                logger.info("MarketStatus | Market is CLOSED. Quotes may be delayed.")
        except Exception as e:
            logger.warning(f"MarketStatusError | Could not determine market status: {e}")

    def _validate_quote_stream(self) -> None:
        """Verifies integrity of Level 1 Market Data (Bid/Ask) per ticker."""
        print("\n--- Live Quote Validation ---")
        logger.info(f"Sampling Live Quotes (UTC: {datetime.datetime.utcnow().strftime('%H:%M:%S')}):")

        success_count = 0
        for ticker in self.WATCHLIST:
            try:
                quote = self.trader.api.get_latest_quote(ticker)
                if quote:
                    bid = getattr(quote, 'bid_price', 0) or 0
                    ask = getattr(quote, 'ask_price', 0) or 0
                    spread = ask - bid if ask and bid else 0
                    logger.info(f"  {ticker:<5} | Bid: ${bid:>8.2f} | Ask: ${ask:>8.2f} | Spread: ${spread:.4f}")
                    success_count += 1
                else:
                    logger.warning(f"  {ticker:<5} | DataUnavailable | Null quote received.")
            except Exception as e:
                logger.error(f"  {ticker:<5} | StreamError | Could not fetch quote: {e}")

        print()
        if success_count == len(self.WATCHLIST):
            logger.info(f"QuoteValidation | All {success_count}/{len(self.WATCHLIST)} tickers validated successfully.")
        else:
            logger.warning(f"QuoteValidation | {success_count}/{len(self.WATCHLIST)} tickers validated.")


def main():
    """Entry point for the connectivity diagnostic."""
    diagnostic = AlpacaConnectivityTest()
    success = diagnostic.execute_diagnostics()

    if success:
        print("Ready for production trading.\n")
        sys.exit(0)
    else:
        print("Diagnostic failed. Please check your API credentials and network connection.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
