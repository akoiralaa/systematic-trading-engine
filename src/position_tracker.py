"""
Position tracking for live trading bot.
Monitors all open positions and their exit conditions.
Persists state to disk for crash recovery.
"""

import json
import csv
import os
import tempfile
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

STATE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'state')
POSITIONS_FILE = os.path.join(STATE_DIR, 'positions.json')
TRADE_HISTORY_FILE = os.path.join(STATE_DIR, 'trade_history.csv')
MAX_CLOSED_POSITIONS = 500


class PositionTracker:
    """
    Tracks all open positions with entry details and exit monitoring.
    Persists positions to disk via atomic writes for crash recovery.
    """

    def __init__(self):
        self.positions: Dict[str, Dict] = {}
        self.closed_positions: List[Dict] = []
        os.makedirs(STATE_DIR, exist_ok=True)
        self._load_state()

    def _load_state(self) -> None:
        """Load persisted positions from disk on startup."""
        if os.path.exists(POSITIONS_FILE):
            try:
                with open(POSITIONS_FILE, 'r') as f:
                    data = json.load(f)
                # Restore positions, converting timestamp strings back to datetime
                for symbol, pos in data.get('positions', {}).items():
                    if 'entry_time' in pos and isinstance(pos['entry_time'], str):
                        pos['entry_time'] = pd.Timestamp(pos['entry_time'])
                    self.positions[symbol] = pos
                # Restore closed positions
                for cp in data.get('closed_positions', []):
                    if 'entry_time' in cp and isinstance(cp['entry_time'], str):
                        cp['entry_time'] = pd.Timestamp(cp['entry_time'])
                    if 'exit_time' in cp and isinstance(cp['exit_time'], str):
                        cp['exit_time'] = pd.Timestamp(cp['exit_time'])
                    self.closed_positions.append(cp)
                logger.info(f"State loaded: {len(self.positions)} open, {len(self.closed_positions)} closed")
            except (json.JSONDecodeError, KeyError) as e:
                logger.error(f"Failed to load state file: {e}. Starting fresh.")
                self.positions = {}
                self.closed_positions = []

    def _save_state(self) -> None:
        """Persist positions to disk via atomic write (temp file + rename)."""
        data = {
            'positions': {},
            'closed_positions': []
        }
        for symbol, pos in self.positions.items():
            serializable_pos = {}
            for k, v in pos.items():
                if isinstance(v, (pd.Timestamp, datetime)):
                    serializable_pos[k] = str(v)
                else:
                    serializable_pos[k] = v
            data['positions'][symbol] = serializable_pos

        for cp in self.closed_positions:
            serializable_cp = {}
            for k, v in cp.items():
                if isinstance(v, (pd.Timestamp, datetime)):
                    serializable_cp[k] = str(v)
                else:
                    serializable_cp[k] = v
            data['closed_positions'].append(serializable_cp)

        try:
            fd, tmp_path = tempfile.mkstemp(dir=STATE_DIR, suffix='.json.tmp')
            with os.fdopen(fd, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            os.replace(tmp_path, POSITIONS_FILE)
        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    def _archive_if_needed(self) -> None:
        """Archive older closed trades to CSV if list exceeds MAX_CLOSED_POSITIONS."""
        if len(self.closed_positions) <= MAX_CLOSED_POSITIONS:
            return

        to_archive = self.closed_positions[:-MAX_CLOSED_POSITIONS]
        self.closed_positions = self.closed_positions[-MAX_CLOSED_POSITIONS:]

        file_exists = os.path.exists(TRADE_HISTORY_FILE)
        try:
            with open(TRADE_HISTORY_FILE, 'a', newline='') as f:
                if to_archive:
                    writer = csv.DictWriter(f, fieldnames=to_archive[0].keys())
                    if not file_exists:
                        writer.writeheader()
                    for trade in to_archive:
                        writer.writerow({k: str(v) for k, v in trade.items()})
            logger.info(f"Archived {len(to_archive)} trades to {TRADE_HISTORY_FILE}")
        except Exception as e:
            logger.error(f"Failed to archive trades: {e}")

    def add_position(self, trade: Dict) -> None:
        """
        Add a new position to tracking.

        Args:
            trade: Trade dict from TradingPipeline with all entry details
        """
        symbol = trade['symbol']

        position = {
            'symbol': symbol,
            'qty': trade['qty'],
            'entry_price': trade['entry_price'],
            'execution_price': trade.get('execution_price', trade['entry_price']),
            'entry_time': trade['timestamp'],
            'stop_price': trade['stop_price'],
            'target_price': trade['target_price'],
            'risk_per_share': trade['risk_per_share'],
            'vector_strength': trade['vector_strength'],
            'regime': trade['regime'],
            'kelly_fraction': trade['kelly_fraction'],
            'trailing_stop_activated': False,
            'trailing_stop_price': None,
            'highest_price_seen': trade['entry_price']
        }

        self.positions[symbol] = position
        self._save_state()
        logger.info(f"Position added: {symbol} - {trade['qty']} shares @ ${trade['entry_price']:.2f}")
        logger.info(f"   Stop: ${trade['stop_price']:.2f} | Target: ${trade['target_price']:.2f}")

    def update_position(self, symbol: str, current_price: float, current_atr: float) -> None:
        """
        Update position with current market price and trailing stop logic.

        Args:
            symbol: Stock symbol
            current_price: Current market price
            current_atr: Current ATR value
        """
        if symbol not in self.positions:
            return

        position = self.positions[symbol]

        # Track highest price for trailing stop
        if current_price > position['highest_price_seen']:
            position['highest_price_seen'] = current_price

        # Activate trailing stop once we're up 1x ATR
        profit_per_share = current_price - position['entry_price']
        if profit_per_share >= current_atr and not position['trailing_stop_activated']:
            position['trailing_stop_activated'] = True
            position['trailing_stop_price'] = current_price - (0.5 * current_atr)
            logger.info(f"Trailing stop activated for {symbol} @ ${position['trailing_stop_price']:.2f}")

        # Update trailing stop as price moves up
        elif position['trailing_stop_activated']:
            new_trailing = position['highest_price_seen'] - (0.5 * current_atr)
            if new_trailing > position['trailing_stop_price']:
                position['trailing_stop_price'] = new_trailing
                logger.info(f"Trailing stop raised for {symbol} to ${new_trailing:.2f}")

        self._save_state()

    def get_position(self, symbol: str) -> Optional[Dict]:
        """Get position details for a symbol."""
        return self.positions.get(symbol)

    def close_position(self, symbol: str, exit_price: float, exit_reason: str,
                       exit_slippage_bps: float = 0.0) -> Dict:
        """
        Close a position and record the result.

        Args:
            symbol: Stock symbol
            exit_price: Price at which position was closed (before slippage)
            exit_reason: Reason for exit (profit_target, stop_loss, etc.)
            exit_slippage_bps: Exit slippage in basis points (applied as cost)

        Returns:
            Dict with closed position details
        """
        if symbol not in self.positions:
            logger.error(f"Attempted to close non-existent position: {symbol}")
            return {}

        position = self.positions[symbol]

        # Apply exit slippage (seller receives less)
        adjusted_exit_price = exit_price * (1 - exit_slippage_bps / 10000)

        # Use execution_price (friction-adjusted) for P&L if available
        cost_basis = position.get('execution_price', position['entry_price'])
        pnl_per_share = adjusted_exit_price - cost_basis
        total_pnl = pnl_per_share * position['qty']
        pnl_pct = (pnl_per_share / cost_basis) * 100 if cost_basis > 0 else 0.0

        entry_time = position['entry_time']
        if isinstance(entry_time, str):
            entry_time = pd.Timestamp(entry_time)
        hold_time = datetime.now() - entry_time

        closed_position = {
            **position,
            'exit_price': exit_price,
            'adjusted_exit_price': adjusted_exit_price,
            'exit_slippage_bps': exit_slippage_bps,
            'exit_time': datetime.now(),
            'exit_reason': exit_reason,
            'pnl_per_share': pnl_per_share,
            'total_pnl': total_pnl,
            'pnl_pct': pnl_pct,
            'hold_time_minutes': hold_time.total_seconds() / 60
        }

        self.closed_positions.append(closed_position)
        del self.positions[symbol]
        self._archive_if_needed()
        self._save_state()

        slippage_note = f" (slip: {exit_slippage_bps:.1f}bps)" if exit_slippage_bps > 0 else ""
        logger.info(f"Position closed: {symbol}")
        logger.info(f"   Entry: ${cost_basis:.2f} -> Exit: ${adjusted_exit_price:.2f}{slippage_note}")
        logger.info(f"   P&L: ${total_pnl:+.2f} ({pnl_pct:+.2f}%) | Reason: {exit_reason}")
        logger.info(f"   Hold time: {hold_time.total_seconds() / 60:.1f} minutes")

        return closed_position

    def get_all_positions(self) -> Dict[str, Dict]:
        """Get all currently open positions."""
        return self.positions

    def has_position(self, symbol: str) -> bool:
        """Check if we have an open position in this symbol."""
        return symbol in self.positions

    def get_performance_summary(self) -> Dict:
        """
        Get overall performance statistics.

        Returns:
            Dict with win rate, avg P&L, total P&L, etc.
        """
        if not self.closed_positions:
            return {
                'total_trades': 0,
                'winners': 0,
                'losers': 0,
                'win_rate': 0.0,
                'total_pnl': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0
            }

        winners = [p for p in self.closed_positions if p['total_pnl'] > 0]
        losers = [p for p in self.closed_positions if p['total_pnl'] <= 0]

        total_pnl = sum(p['total_pnl'] for p in self.closed_positions)
        avg_win = sum(p['total_pnl'] for p in winners) / len(winners) if winners else 0
        avg_loss = sum(p['total_pnl'] for p in losers) / len(losers) if losers else 0

        gross_profit = sum(p['total_pnl'] for p in winners)
        gross_loss = abs(sum(p['total_pnl'] for p in losers))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        return {
            'total_trades': len(self.closed_positions),
            'winners': len(winners),
            'losers': len(losers),
            'win_rate': len(winners) / len(self.closed_positions) * 100,
            'total_pnl': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor
        }
