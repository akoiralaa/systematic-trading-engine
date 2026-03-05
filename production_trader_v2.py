import sys
import os
import traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
from src.trading_pipeline import TradingPipeline
from src.alpaca_trader import AlpacaTrader
from src.position_tracker import PositionTracker
from src.exit_manager import ExitManager
from config.trading_config import (
    TRADING_SYMBOLS, SCAN_INTERVAL_SECONDS, FILL_TIMEOUT_SECONDS,
    MAX_HOLD_MINUTES, EOD_CLOSE_MINUTES, PRODUCTION_TIMEFRAME,
    PRODUCTION_LOOKBACK_DAYS, MAX_CONSECUTIVE_LOSSES,
    MAX_DAILY_LOSS_PERCENT, MAX_CONCURRENT_POSITIONS, MAX_POSITION_CAPITAL_PCT
)
from datetime import datetime, timedelta
from typing import Dict
import time
import logging
from logging.handlers import RotatingFileHandler
import numpy as np
import pandas as pd

# --- Log Rotation Setup (2.7) ---
os.makedirs('logs', exist_ok=True)
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')

file_handler = RotatingFileHandler(
    'logs/production_trader.log',
    maxBytes=5 * 1024 * 1024,  # 5 MB
    backupCount=5
)
file_handler.setFormatter(log_formatter)
file_handler.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
console_handler.setLevel(logging.INFO)

root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)
root_logger.addHandler(file_handler)
root_logger.addHandler(console_handler)

logger = logging.getLogger(__name__)

print("\n" + "=" * 80)
print("PRODUCTION SYSTEM v2 - Entry & Exit Management")
print("=" * 80 + "\n")

# Initialize trading components
trader = AlpacaTrader()
if not trader.connect():
    sys.exit(1)

account_info = trader.get_account_info()
equity = float(account_info.get('equity', 100000))
starting_equity = equity
session_start = datetime.now()
engine = TradingPipeline(api=trader.api, account_equity=equity, fractional_kelly=0.5)

# Initialize position tracking and exit management
position_tracker = PositionTracker()
exit_manager = ExitManager(max_hold_minutes=MAX_HOLD_MINUTES, eod_close_minutes=EOD_CLOSE_MINUTES)

# --- Position Reconciliation on Startup (1.4) ---
def reconcile_positions():
    """Sync local position tracker with Alpaca's actual positions."""
    try:
        alpaca_positions = trader.get_positions()
        alpaca_symbols = set()

        for ap in alpaca_positions:
            symbol = ap.symbol
            alpaca_symbols.add(symbol)
            if not position_tracker.has_position(symbol):
                # Untracked Alpaca position - add with conservative defaults
                logger.warning(f"Reconciliation: Found untracked Alpaca position {symbol}, syncing with conservative defaults")
                position_tracker.positions[symbol] = {
                    'symbol': symbol,
                    'qty': int(ap.qty),
                    'entry_price': float(ap.avg_entry_price),
                    'execution_price': float(ap.avg_entry_price),
                    'entry_time': pd.Timestamp.now(),
                    'stop_price': float(ap.avg_entry_price) * 0.95,  # Conservative 5% stop
                    'target_price': float(ap.avg_entry_price) * 1.03,  # Conservative 3% target
                    'risk_per_share': float(ap.avg_entry_price) * 0.05,
                    'vector_strength': 0.51,
                    'regime': 'UNKNOWN',
                    'kelly_fraction': 0.01,
                    'trailing_stop_activated': False,
                    'trailing_stop_price': None,
                    'highest_price_seen': float(ap.current_price)
                }
                position_tracker._save_state()

        # Check for phantom positions (tracked but not in Alpaca)
        tracked_symbols = set(position_tracker.get_all_positions().keys())
        phantom_symbols = tracked_symbols - alpaca_symbols
        for symbol in phantom_symbols:
            logger.warning(f"Reconciliation: Phantom position {symbol} (tracked but not in Alpaca). Cleaning up.")
            pos = position_tracker.get_position(symbol)
            position_tracker.close_position(symbol, pos.get('entry_price', 0), 'reconciliation_cleanup')

        logger.info(f"Reconciliation complete: {len(alpaca_symbols)} Alpaca positions, "
                     f"{len(phantom_symbols)} phantoms cleaned")
    except Exception as e:
        logger.exception(f"Reconciliation failed: {e}")

reconcile_positions()

# --- Risk Limit State (2.6) ---
consecutive_losses = 0
daily_pnl = 0.0
trading_halted = False
last_reset_date = datetime.now().date()

def check_risk_limits() -> bool:
    """
    Check if risk limits have been breached.
    Returns True if trading should continue, False if halted.
    """
    global trading_halted, consecutive_losses, daily_pnl, last_reset_date

    # Reset daily counters at start of new day
    today = datetime.now().date()
    if today != last_reset_date:
        daily_pnl = 0.0
        consecutive_losses = 0
        trading_halted = False
        last_reset_date = today
        logger.info("Daily risk counters reset.")

    if trading_halted:
        logger.warning("Trading halted due to risk limits. Monitoring only.")
        return False

    if consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
        trading_halted = True
        logger.error(f"RISK HALT: {consecutive_losses} consecutive losses (limit: {MAX_CONSECUTIVE_LOSSES})")
        return False

    if starting_equity > 0 and abs(daily_pnl / starting_equity) >= MAX_DAILY_LOSS_PERCENT:
        if daily_pnl < 0:
            trading_halted = True
            logger.error(f"RISK HALT: Daily loss ${daily_pnl:,.2f} exceeds "
                         f"{MAX_DAILY_LOSS_PERCENT*100:.1f}% limit")
            return False

    # Max concurrent positions check
    open_count = len(position_tracker.get_all_positions())
    if open_count >= MAX_CONCURRENT_POSITIONS:
        logger.info(f"Max concurrent positions reached ({open_count}/{MAX_CONCURRENT_POSITIONS}). "
                     "Entry scanning paused.")
        return False

    return True


logger.info(f"System initialized")
logger.info(f"  Account equity: ${equity:,.2f}")
logger.info(f"  Max hold time: {MAX_HOLD_MINUTES} minutes")
logger.info(f"  Max concurrent positions: {MAX_CONCURRENT_POSITIONS}")
logger.info(f"  Risk limits: {MAX_CONSECUTIVE_LOSSES} consecutive losses, "
            f"{MAX_DAILY_LOSS_PERCENT*100:.0f}% daily loss")
logger.info(f"  Monitoring {len(TRADING_SYMBOLS)} symbols")


def calculate_atr(bars, period=14):
    """Calculate Average True Range with proper handling of first bars."""
    high = bars['high'].values
    low = bars['low'].values
    close = bars['close'].values

    n = len(close)
    if n < 2:
        # Not enough data, return price-based estimate
        return np.array([close[0] * 0.02]) if n == 1 else np.array([])

    tr = np.zeros(n)
    tr[0] = high[0] - low[0]  # First bar: just high-low

    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )

    # Rolling mean with min_periods=1 to avoid NaN
    atr = pd.Series(tr).rolling(window=period, min_periods=1).mean().values

    # Sanity check: ATR should be at least 0.1% of price
    min_atr = close * 0.001
    atr = np.maximum(atr, min_atr)

    return atr


def is_market_open():
    """Check if market is currently open."""
    try:
        clock = trader.api.get_clock()
        return clock.is_open
    except Exception as e:
        logger.exception(f"Failed to check market status: {e}")
        return False


def execute_order(trade: Dict, action: str = 'buy') -> Dict:
    """
    Execute a market order (buy or sell).

    Args:
        trade: Trade dict with symbol, qty, etc.
        action: 'buy' or 'sell'

    Returns:
        Dict with 'success', 'filled_qty', 'fill_price' keys
    """
    result = {'success': False, 'filled_qty': 0, 'fill_price': 0.0}
    try:
        symbol = trade['symbol']
        qty = trade['qty']

        logger.info(f"{'BUY' if action == 'buy' else 'SELL'} ORDER: {qty} {symbol}")

        order = trader.api.submit_order(
            symbol=symbol,
            qty=qty,
            side=action,
            type='market',
            time_in_force='day'
        )

        logger.info(f"  Order placed: {order.id}")

        # Wait for fill
        for _ in range(FILL_TIMEOUT_SECONDS):
            order = trader.api.get_order(order.id)
            if order.status == 'filled':
                fill_price = float(order.filled_avg_price)
                filled_qty = int(order.filled_qty)
                logger.info(f"  Order filled: {filled_qty} shares @ ${fill_price:.2f}")
                result['success'] = True
                result['filled_qty'] = filled_qty
                result['fill_price'] = fill_price
                return result
            elif order.status == 'partially_filled':
                # Partial fill detected - continue waiting but track
                filled_so_far = int(order.filled_qty) if order.filled_qty else 0
                logger.info(f"  Partial fill: {filled_so_far}/{qty} shares")
            elif order.status in ['canceled', 'rejected']:
                logger.error(f"  Order {order.status}")
                return result
            time.sleep(1)

        # Timeout - check for partial fill (2.4)
        order = trader.api.get_order(order.id)
        if order.status == 'partially_filled' and order.filled_qty:
            filled_qty = int(order.filled_qty)
            fill_price = float(order.filled_avg_price)
            logger.warning(f"  Order timeout with partial fill: {filled_qty}/{qty}. Canceling remainder.")
            try:
                trader.api.cancel_order(order.id)
            except Exception:
                pass
            result['success'] = True
            result['filled_qty'] = filled_qty
            result['fill_price'] = fill_price
            return result

        logger.warning(f"  Order timeout - no fill")
        try:
            trader.api.cancel_order(order.id)
        except Exception:
            pass
        return result

    except Exception as e:
        logger.exception(f"  Order failed: {e}")
        return result


def check_and_exit_positions():
    """
    Check all open positions for exit conditions.
    Returns number of positions exited.
    """
    global consecutive_losses, daily_pnl
    positions_exited = 0

    for symbol in list(position_tracker.get_all_positions().keys()):
        try:
            position = position_tracker.get_position(symbol)

            # Get current price
            bars = trader.api.get_bars(
                symbol, '1Min',
                start=(datetime.now() - timedelta(minutes=5)).strftime('%Y-%m-%d'),
                limit=5
            ).df

            if bars is None or len(bars) == 0:
                logger.warning(f"  {symbol}: No price data")
                continue

            current_price = bars['close'].values[-1]
            current_atr = calculate_atr(bars)[-1] if len(bars) >= 14 else 1.0

            # Update position (for trailing stop logic)
            position_tracker.update_position(symbol, current_price, current_atr)

            # Re-fetch position after update (trailing stop may have changed)
            position = position_tracker.get_position(symbol)

            # Check exit conditions
            should_exit, exit_reason, exit_price = exit_manager.check_exit(position, current_price)

            if should_exit:
                sell_trade = {
                    'symbol': symbol,
                    'qty': position['qty']
                }

                order_result = execute_order(sell_trade, action='sell')
                if order_result['success']:
                    actual_exit_price = order_result['fill_price'] if order_result['fill_price'] > 0 else exit_price

                    # Calculate exit slippage (worse for stop losses)
                    exit_type = 'stop_loss' if exit_reason == 'stop_loss' else 'normal'
                    slippage_bps = 2.0 if exit_type == 'stop_loss' else 1.0

                    closed = position_tracker.close_position(
                        symbol, actual_exit_price, exit_reason,
                        exit_slippage_bps=slippage_bps
                    )
                    positions_exited += 1

                    # Update risk tracking
                    if closed.get('total_pnl', 0) <= 0:
                        consecutive_losses += 1
                    else:
                        consecutive_losses = 0
                    daily_pnl += closed.get('total_pnl', 0)

                    # Feed closed trade to Kelly calibration
                    if closed.get('vector_strength') and closed.get('total_pnl') is not None:
                        engine.kelly.add_calibration_trade(
                            closed['vector_strength'],
                            closed['total_pnl']
                        )

                    perf = position_tracker.get_performance_summary()
                    logger.info(f"Overall: {perf['total_trades']} trades, "
                                f"{perf['win_rate']:.1f}% win rate, "
                                f"${perf['total_pnl']:+,.2f} P&L")
            else:
                status = exit_manager.format_exit_status(position, current_price)
                logger.info(f"  {symbol}: {status}")

        except Exception as e:
            logger.exception(f"  Error checking {symbol}: {e}")

    return positions_exited


try:
    cycle_count = 0

    while True:
        cycle_count += 1

        # Ensure API connection is healthy (2.2)
        if not trader.ensure_connected():
            logger.error("API connection lost. Waiting before retry...")
            time.sleep(60)
            continue

        if not is_market_open():
            logger.info(f"Market closed - waiting {SCAN_INTERVAL_SECONDS}s...")
            time.sleep(SCAN_INTERVAL_SECONDS)
            continue

        logger.info(f"\n{'=' * 80}")
        logger.info(f"Production Scan #{cycle_count} - {datetime.now().strftime('%H:%M:%S')}")
        logger.info(f"{'=' * 80}")

        # STEP 1: CHECK EXITS FOR EXISTING POSITIONS
        if position_tracker.get_all_positions():
            logger.info(f"Checking {len(position_tracker.get_all_positions())} open positions...")
            exits = check_and_exit_positions()
            if exits > 0:
                logger.info(f"Exited {exits} position(s)")

        # STEP 2: CHECK RISK LIMITS BEFORE SCANNING (2.6)
        can_trade = check_risk_limits()

        # STEP 3: SCAN FOR NEW ENTRY SIGNALS
        signals_found = 0

        if can_trade:
            for symbol in TRADING_SYMBOLS:
                if position_tracker.has_position(symbol):
                    continue

                try:
                    logger.info(f"\n{symbol}...")

                    bars = trader.api.get_bars(
                        symbol, PRODUCTION_TIMEFRAME,
                        start=(datetime.now() - timedelta(days=PRODUCTION_LOOKBACK_DAYS)).strftime('%Y-%m-%d'),
                        limit=500
                    ).df

                    if bars is None or len(bars) < 100:
                        logger.warning(f"  Insufficient data")
                        continue

                    logger.info(f"  {len(bars)} bars loaded")

                    close_prices = bars['close'].values
                    atr_values = calculate_atr(bars)
                    vector_prices = pd.Series(close_prices).ewm(span=20, adjust=False).mean().values
                    price_deviation = np.abs(close_prices - vector_prices) / (atr_values + 1e-10)
                    vector_strengths = np.clip(price_deviation / 1.5, 0, 1)
                    avg_volume = bars['volume'].mean()

                    # Use try/finally to ensure ATR multiplier is always restored
                    original_atr_mult = engine.regime_detector.atr_multiplier
                    try:
                        engine.regime_detector.atr_multiplier = 1.5

                        result = engine.execute_trading_cycle(
                            symbol=symbol, prices=close_prices, vector_prices=vector_prices,
                            vector_strengths=vector_strengths, atr_values=atr_values,
                            avg_volume=float(avg_volume)
                        )
                    finally:
                        engine.regime_detector.atr_multiplier = original_atr_mult

                    if result and result.get("trade") is not None:
                        trade = result["trade"]
                        signals_found += 1

                        logger.info(f"  SIGNAL #{signals_found}: BUY {trade['qty']} {symbol} @ ${trade['entry_price']:.2f}")
                        logger.info(f"     Stop: ${trade['stop_price']:.2f} | Target: ${trade['target_price']:.2f}")
                        logger.info(f"     EV: {trade['expected_value']['ev']:.2f} | Kelly: {trade['kelly_fraction']:.4f}")

                        # Pre-order buying power validation with 10% capital cap
                        buying_power = float(trader.api.get_account().buying_power)
                        capital_for_trade = buying_power * MAX_POSITION_CAPITAL_PCT
                        max_affordable = int(capital_for_trade / trade['entry_price'])
                        logger.info(f"  Capital cap: ${capital_for_trade:,.0f} "
                                    f"({MAX_POSITION_CAPITAL_PCT*100:.0f}% of ${buying_power:,.0f} available)")
                        if trade['qty'] > max_affordable:
                            logger.warning(f"Qty capped {trade['qty']} -> {max_affordable} (10% cap)")
                            trade['qty'] = max_affordable
                        if trade['qty'] <= 0:
                            logger.error(f"Cannot afford any shares of {symbol}")
                            continue

                        order_result = execute_order(trade, action='buy')
                        if order_result['success']:
                            # Update trade qty to actual filled qty (2.4)
                            if order_result['filled_qty'] != trade['qty']:
                                logger.info(f"  Partial fill: adjusted qty from {trade['qty']} to {order_result['filled_qty']}")
                                trade['qty'] = order_result['filled_qty']
                            if order_result['fill_price'] > 0:
                                trade['execution_price'] = order_result['fill_price']
                            position_tracker.add_position(trade)
                            logger.info(f"  Position opened and tracked")
                        else:
                            logger.error(f"  Order execution failed")
                    else:
                        logger.info(f"  -> No signal")

                except Exception as e:
                    logger.exception(f"  Error scanning {symbol}: {e}")

                # Rate limiting: small delay between symbols to avoid API limits
                time.sleep(0.2)

        # STEP 4: UPDATE ACCOUNT EQUITY
        try:
            account_info = trader.get_account_info()
            new_equity = float(account_info.get('equity', equity))
            if new_equity != equity:
                pnl = new_equity - equity
                logger.info(f"Equity updated: ${equity:,.2f} -> ${new_equity:,.2f} (${pnl:+,.2f})")
                equity = new_equity
                engine.update_equity(equity)
        except Exception as e:
            logger.exception(f"Error updating account: {e}")

        # STEP 5: SUMMARY (sourced from Alpaca account - authoritative)
        open_positions = len(position_tracker.get_all_positions())
        if open_positions > 0:
            logger.info(f"Open positions: {open_positions}")
        if signals_found == 0 and can_trade:
            logger.info(f"  No new signals from {len(TRADING_SYMBOLS)} stocks")

        try:
            acct = trader.api.get_account()
            current_equity = float(acct.equity)
            buying_power = float(acct.buying_power)
            session_pnl = current_equity - starting_equity
            session_pnl_pct = (session_pnl / starting_equity) * 100

            # Win/loss stats from local tracker (accurate entry vs exit tracking)
            perf = position_tracker.get_performance_summary()

            logger.info(f"Account Summary (Alpaca):")
            logger.info(f"   Equity:       ${current_equity:>12,.2f}")
            logger.info(f"   Session P&L:  ${session_pnl:>+12,.2f}  ({session_pnl_pct:+.2f}%)")
            logger.info(f"   Buying Power: ${buying_power:>12,.2f}")
            if perf['total_trades'] > 0:
                logger.info(f"   Trades: {perf['total_trades']} "
                            f"({perf['winners']}W / {perf['losers']}L) | "
                            f"Win Rate: {perf['win_rate']:.1f}% | "
                            f"Profit Factor: {perf['profit_factor']:.2f}")
        except Exception as e:
            logger.exception(f"Error fetching Alpaca summary: {e}")

        if trading_halted:
            logger.warning("TRADING HALTED - monitoring only until risk limits reset")

        logger.info(f"Next scan in {SCAN_INTERVAL_SECONDS}s...")
        time.sleep(SCAN_INTERVAL_SECONDS)

except KeyboardInterrupt:
    logger.info("Shutdown signal received")

    # Close all open positions before shutdown
    if position_tracker.get_all_positions():
        logger.info("Closing all open positions...")
        for symbol in list(position_tracker.get_all_positions().keys()):
            position = position_tracker.get_position(symbol)
            sell_trade = {'symbol': symbol, 'qty': position['qty']}
            order_result = execute_order(sell_trade, action='sell')
            exit_price = order_result['fill_price'] if order_result['success'] else position['entry_price']
            position_tracker.close_position(symbol, exit_price, 'manual_shutdown')

print("\n" + "=" * 80)
print("Production session ended")
print("=" * 80 + "\n")
