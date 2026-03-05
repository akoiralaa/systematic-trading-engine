"""
Exit management system with 5 exit conditions.
Handles profit targets, stop losses, trailing stops, time limits, and EOD exits.

Assumptions:
- All market-time comparisons use US/Eastern timezone
- Position entry_time is a timezone-aware or naive datetime (treated as UTC if naive)
- EOD close time is relative to market hours (Eastern)
"""

from datetime import datetime, time
from typing import Dict, Optional, Tuple
import logging

try:
    import pytz
    EASTERN = pytz.timezone('US/Eastern')
except ImportError:
    EASTERN = None

logger = logging.getLogger(__name__)


def _now_eastern() -> datetime:
    """Get current time in US/Eastern timezone."""
    if EASTERN is not None:
        return datetime.now(EASTERN)
    return datetime.now()


class ExitManager:
    """
    Manages all exit conditions for open positions.

    Exit Conditions (checked in order):
    1. Profit Target - hit the target_price from entry signal
    2. Stop Loss - hit the stop_price from entry signal
    3. Trailing Stop - dynamic stop that locks in profits
    4. Time Limit - maximum hold time (default 4 hours)
    5. End of Day - close all positions before market close
    """

    def __init__(self, max_hold_minutes: int = 240, eod_close_minutes: int = 5):
        """
        Args:
            max_hold_minutes: Maximum time to hold a position (default 4 hours)
            eod_close_minutes: Minutes before market close to exit all (default 5)
        """
        self.max_hold_minutes = max_hold_minutes
        self.eod_close_minutes = max(1, eod_close_minutes)  # At least 1 minute buffer

        # Market closes at 4:00 PM ET (16:00)
        # Calculate exit time as (16:00 - eod_close_minutes)
        total_minutes = 16 * 60 - self.eod_close_minutes
        close_hour = total_minutes // 60
        close_minute = total_minutes % 60
        self.market_close_time = time(close_hour, close_minute)

    def check_exit(self, position: Dict, current_price: float) -> Tuple[bool, Optional[str], Optional[float]]:
        """
        Check all exit conditions for a position.

        Args:
            position: Position dict from PositionTracker
            current_price: Current market price

        Returns:
            Tuple of (should_exit, exit_reason, exit_price)
        """

        # 1. PROFIT TARGET
        if current_price >= position['target_price']:
            logger.info(f"Profit target hit: {position['symbol']} @ ${current_price:.2f}")
            return (True, 'profit_target', current_price)

        # 2. STOP LOSS
        if current_price <= position['stop_price']:
            logger.warning(f"Stop loss hit: {position['symbol']} @ ${current_price:.2f}")
            return (True, 'stop_loss', current_price)

        # 3. TRAILING STOP
        if position.get('trailing_stop_activated') and current_price <= position.get('trailing_stop_price', 0):
            logger.info(f"Trailing stop hit: {position['symbol']} @ ${current_price:.2f}")
            return (True, 'trailing_stop', current_price)

        # 4. TIME LIMIT
        now = _now_eastern()
        entry_time = position['entry_time']
        if hasattr(entry_time, 'to_pydatetime'):
            entry_time = entry_time.to_pydatetime()
        if entry_time.tzinfo is None and EASTERN is not None:
            entry_time = EASTERN.localize(entry_time)
        hold_time_minutes = (now - entry_time).total_seconds() / 60
        if hold_time_minutes >= self.max_hold_minutes:
            logger.warning(f"Time limit exceeded: {position['symbol']} (held {hold_time_minutes:.0f} min)")
            return (True, 'time_limit', current_price)

        # 5. END OF DAY
        current_time = now.time()
        if current_time >= self.market_close_time:
            logger.warning(f"End of day exit: {position['symbol']}")
            return (True, 'end_of_day', current_price)

        # No exit conditions met
        return (False, None, None)

    def should_close_all_positions(self) -> bool:
        """
        Check if we should close all positions (market closing soon).

        Returns:
            True if within EOD closing window
        """
        current_time = _now_eastern().time()
        return current_time >= self.market_close_time

    def get_exit_summary(self, position: Dict, current_price: float) -> Dict:
        """
        Get summary of how close position is to each exit condition.

        Args:
            position: Position dict
            current_price: Current market price

        Returns:
            Dict with distances to each exit level
        """
        entry_price = position.get('execution_price', position['entry_price'])

        pnl_pct = ((current_price - entry_price) / entry_price) * 100
        target_distance_pct = ((position['target_price'] - current_price) / current_price) * 100
        stop_distance_pct = ((current_price - position['stop_price']) / current_price) * 100

        now = _now_eastern()
        entry_time = position['entry_time']
        if hasattr(entry_time, 'to_pydatetime'):
            entry_time = entry_time.to_pydatetime()
        if entry_time.tzinfo is None and EASTERN is not None:
            entry_time = EASTERN.localize(entry_time)
        hold_time_minutes = (now - entry_time).total_seconds() / 60
        time_remaining = self.max_hold_minutes - hold_time_minutes

        summary = {
            'current_pnl_pct': pnl_pct,
            'distance_to_target_pct': target_distance_pct,
            'distance_to_stop_pct': stop_distance_pct,
            'hold_time_minutes': hold_time_minutes,
            'time_remaining_minutes': max(0, time_remaining),
            'trailing_stop_active': position.get('trailing_stop_activated', False)
        }

        if position.get('trailing_stop_activated') and position.get('trailing_stop_price'):
            trailing_distance_pct = ((current_price - position['trailing_stop_price']) / current_price) * 100
            summary['distance_to_trailing_pct'] = trailing_distance_pct

        return summary

    def format_exit_status(self, position: Dict, current_price: float) -> str:
        """
        Format a human-readable status string for position monitoring.

        Args:
            position: Position dict
            current_price: Current market price

        Returns:
            Formatted status string
        """
        summary = self.get_exit_summary(position, current_price)

        status_parts = [
            f"P&L: {summary['current_pnl_pct']:+.2f}%",
            f"Target: {summary['distance_to_target_pct']:+.2f}%",
            f"Stop: {summary['distance_to_stop_pct']:+.2f}%",
            f"Time: {summary['hold_time_minutes']:.0f}/{self.max_hold_minutes}min"
        ]

        if position.get('trailing_stop_activated') and 'distance_to_trailing_pct' in summary:
            status_parts.append(f"Trail: {summary['distance_to_trailing_pct']:+.2f}%")

        return " | ".join(status_parts)
