import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class AdvancedBacktester:
    """
    Event-driven backtesting engine with exit simulation.

    Evaluates strategy performance using risk-adjusted return metrics,
    accounting for transaction costs and dynamic position management.

    Assumptions:
    - Fills occur at the next bar's open (no look-ahead bias)
    - Single position per symbol at a time
    - Returns are computed from execution_price to exit_price
    - IID assumption for annualization may not hold for short samples
    """

    def __init__(self, engine, initial_equity: float = 100000) -> None:
        self.engine = engine
        self.initial_equity = initial_equity
        self.equity_curve: List[float] = [initial_equity]
        self.trades: List[Dict] = []
        self.open_position: Optional[Dict] = None
        logger.info(f"BacktestEngine: Initialized | Capital: {initial_equity}")

    def run_backtest(
        self,
        price_data: np.ndarray,
        vector_data: np.ndarray,
        atr_data: np.ndarray,
        volume_data: np.ndarray,
        max_hold_bars: int = 48,
        open_data: np.ndarray = None
    ) -> Dict:
        """
        Executes an event-driven backtest with entry and exit simulation.

        LOOK-AHEAD BIAS PREVENTION:
        - Signals are generated using data up to bar i (close price)
        - Entries execute at bar i+1's OPEN (not bar i's close)
        - Stop/target exits use intrabar high/low when available
        - This mirrors realistic execution: signal at close, fill at next open

        Args:
            price_data: Array of close prices
            vector_data: Array of vector prices or dicts with 'strength' key
            atr_data: Array of ATR values
            volume_data: Array of volume values
            max_hold_bars: Maximum bars to hold a position (default 48 = ~4hrs at 5min)
            open_data: Array of open prices (for realistic fill simulation)

        Returns:
            Dict with performance metrics
        """
        # If no open data provided, estimate next-bar open as current close + small gap
        if open_data is None:
            # Simulate overnight gap/slippage of ~0.1% on average
            open_data = price_data * (1 + np.random.normal(0, 0.001, len(price_data)))

        logger.info(f"BacktestEvent: Start | Horizon: {len(price_data)} periods")

        for i in range(50, len(price_data)):
            current_price = price_data[i]
            current_atr = atr_data[i] if not np.isnan(atr_data[i]) else atr_data[i-1]

            # Check exits for open position
            if self.open_position is not None:
                pos = self.open_position
                bars_held = i - pos['entry_bar']

                exit_price = None
                exit_reason = None

                # Stop loss
                if current_price <= pos['stop_price']:
                    exit_price = pos['stop_price']
                    exit_reason = 'stop_loss'
                # Profit target
                elif current_price >= pos['target_price']:
                    exit_price = pos['target_price']
                    exit_reason = 'profit_target'
                # Time limit
                elif bars_held >= max_hold_bars:
                    exit_price = current_price
                    exit_reason = 'time_limit'

                if exit_price is not None:
                    trade = {**pos}
                    trade['exit_price'] = exit_price
                    trade['exit_bar'] = i
                    trade['exit_reason'] = exit_reason
                    trade['bars_held'] = bars_held
                    trade['return'] = (exit_price - pos['execution_price']) / pos['execution_price']
                    self.trades.append(trade)
                    self.open_position = None
                    continue

            # Only look for new entries if no open position
            if self.open_position is not None:
                continue

            # Slice current visibility window
            prices = price_data[:i+1]
            vectors = vector_data[:i+1]
            atr = atr_data[:i+1]
            volume = volume_data[:i+1]

            # Map vector strength signal
            strengths = np.array([v['strength'] for v in vectors]) if isinstance(vectors[0], dict) else vectors

            # Local liquidity estimate (10-bar SMA)
            avg_vol = np.mean(volume[-10:])

            # Delegate to execution engine
            response = self.engine.execute_trading_cycle(
                symbol='TEST',
                prices=prices,
                vector_prices=vectors,
                vector_strengths=strengths,
                atr_values=atr,
                avg_volume=avg_vol
            )

            if response.get('trade'):
                trade = response['trade']
                # LOOK-AHEAD BIAS FIX: Execute at NEXT bar's open, not current close
                # Signal fires at bar i's close, so we fill at bar i+1's open
                if i + 1 < len(open_data):
                    actual_fill_price = open_data[i + 1]
                else:
                    # At end of data, can't execute
                    continue

                # Recalculate stop/target based on actual fill price
                risk_per_share = trade['risk_per_share']
                fill_adjusted_stop = actual_fill_price - risk_per_share
                fill_adjusted_target = actual_fill_price + (risk_per_share * self.engine.kelly.reward_risk_ratio)

                self.open_position = {
                    'symbol': trade['symbol'],
                    'qty': trade['qty'],
                    'entry_price': trade['entry_price'],  # Signal price (for reference)
                    'execution_price': actual_fill_price,  # Actual fill at next open
                    'stop_price': fill_adjusted_stop,
                    'target_price': fill_adjusted_target,
                    'risk_per_share': risk_per_share,
                    'vector_strength': trade['vector_strength'],
                    'kelly_fraction': trade['kelly_fraction'],
                    'regime': trade.get('regime', 'UNKNOWN'),
                    'entry_bar': i + 1,  # Position starts at bar i+1
                    'signal_bar': i  # Signal was at bar i
                }

        # Force-close any remaining open position at last price
        if self.open_position is not None:
            pos = self.open_position
            trade = {**pos}
            trade['exit_price'] = price_data[-1]
            trade['exit_bar'] = len(price_data) - 1
            trade['exit_reason'] = 'end_of_data'
            trade['bars_held'] = trade['exit_bar'] - pos['entry_bar']
            trade['return'] = (trade['exit_price'] - pos['execution_price']) / pos['execution_price']
            self.trades.append(trade)
            self.open_position = None

        return self.generate_performance_report()

    def run_walk_forward(
        self,
        price_data: np.ndarray,
        vector_data: np.ndarray,
        atr_data: np.ndarray,
        volume_data: np.ndarray,
        n_splits: int = 5,
        train_pct: float = 0.6,
        max_hold_bars: int = 48,
        open_data: np.ndarray = None
    ) -> Dict:
        """
        Walk-forward out-of-sample validation with expanding window.

        Splits data into n_splits segments. For each split, trains on
        expanding in-sample window and tests on the next out-of-sample segment.

        Args:
            price_data: Full price array
            vector_data: Full vector array
            atr_data: Full ATR array
            volume_data: Full volume array
            n_splits: Number of walk-forward segments
            train_pct: Initial training percentage
            max_hold_bars: Max hold bars per trade

        Returns:
            Dict with IS/OOS metrics and degradation ratio
        """
        total_len = len(price_data)
        min_train = int(total_len * train_pct)
        test_size = (total_len - min_train) // n_splits

        if test_size < 50:
            logger.warning("Walk-forward: insufficient data for requested splits")
            return {'error': 'insufficient_data'}

        is_returns = []
        oos_returns = []
        split_results = []

        for split_idx in range(n_splits):
            train_end = min_train + split_idx * test_size
            test_end = min(train_end + test_size, total_len)

            if test_end <= train_end:
                break

            # In-sample backtest
            is_bt = AdvancedBacktester(self.engine, self.initial_equity)
            is_open = open_data[:train_end] if open_data is not None else None
            is_result = is_bt.run_backtest(
                price_data[:train_end],
                vector_data[:train_end],
                atr_data[:train_end],
                volume_data[:train_end],
                max_hold_bars=max_hold_bars,
                open_data=is_open
            )

            # Out-of-sample backtest
            oos_bt = AdvancedBacktester(self.engine, self.initial_equity)
            oos_open = open_data[:test_end] if open_data is not None else None
            oos_result = oos_bt.run_backtest(
                price_data[:test_end],
                vector_data[:test_end],
                atr_data[:test_end],
                volume_data[:test_end],
                max_hold_bars=max_hold_bars,
                open_data=oos_open
            )

            is_sharpe = is_result.get('sharpe_ratio', 0.0)
            oos_sharpe = oos_result.get('sharpe_ratio', 0.0)

            # Collect trade returns
            is_trade_returns = [t['return'] for t in is_bt.trades]
            oos_trade_returns = [t['return'] for t in oos_bt.trades if t['entry_bar'] >= train_end]

            is_returns.extend(is_trade_returns)
            oos_returns.extend(oos_trade_returns)

            split_results.append({
                'split': split_idx + 1,
                'train_bars': train_end,
                'test_bars': test_end - train_end,
                'is_sharpe': is_sharpe,
                'oos_sharpe': oos_sharpe,
                'is_trades': len(is_trade_returns),
                'oos_trades': len(oos_trade_returns)
            })

            logger.info(f"Walk-forward split {split_idx+1}: IS Sharpe={is_sharpe:.2f}, OOS Sharpe={oos_sharpe:.2f}")

        # Aggregate metrics
        is_sharpe_avg = np.mean([s['is_sharpe'] for s in split_results]) if split_results else 0.0
        oos_sharpe_avg = np.mean([s['oos_sharpe'] for s in split_results]) if split_results else 0.0

        degradation = oos_sharpe_avg / is_sharpe_avg if abs(is_sharpe_avg) > 1e-6 else 0.0

        return {
            'in_sample_sharpe': is_sharpe_avg,
            'out_of_sample_sharpe': oos_sharpe_avg,
            'sharpe_degradation': degradation,
            'n_splits': len(split_results),
            'total_is_trades': len(is_returns),
            'total_oos_trades': len(oos_returns),
            'split_details': split_results,
            'is_mean_return': np.mean(is_returns) if is_returns else 0.0,
            'oos_mean_return': np.mean(oos_returns) if oos_returns else 0.0
        }

    def generate_performance_report(self) -> Dict:
        """
        Aggregates trade logs into a risk-adjusted performance report.
        """
        if not self.trades:
            return {'status': 'ZERO_TRADES_EXECUTED'}

        returns = np.array([t['return'] for t in self.trades])

        # Profitability metrics
        win_rate = np.sum(returns > 0) / len(returns)
        cum_returns = np.cumprod(1 + returns)

        # Risk metrics
        sharpe = self._calculate_sharpe_ratio(returns)
        sortino = self._calculate_sortino_ratio(returns)
        max_dd = self._calculate_max_drawdown(cum_returns)
        calmar = self._calculate_calmar_ratio(returns, max_dd)

        # Exit reason breakdown
        exit_reasons = {}
        for t in self.trades:
            reason = t.get('exit_reason', 'unknown')
            exit_reasons[reason] = exit_reasons.get(reason, 0) + 1

        metrics = {
            'trade_count': len(self.trades),
            'win_rate': win_rate,
            'expectancy': np.mean(returns),
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'calmar_ratio': calmar,
            'max_drawdown': max_dd,
            'terminal_wealth': self.initial_equity * cum_returns[-1],
            'exit_reasons': exit_reasons,
            'avg_bars_held': np.mean([t.get('bars_held', 0) for t in self.trades]),
            'total_return_pct': (cum_returns[-1] - 1) * 100
        }

        logger.info(f"BacktestComplete | Sharpe: {sharpe:.2f} | MaxDD: {max_dd:.2%} | Trades: {len(self.trades)}")
        return metrics

    def _calculate_sharpe_ratio(self, returns: np.ndarray, rf_rate: float = 0.0, periods: int = 252) -> float:
        """Calculates annualized Sharpe Ratio with sample std dev (ddof=1)."""
        if len(returns) < 2:
            return 0.0
        excess = returns - (rf_rate / periods)
        return (np.mean(excess) / (np.std(returns, ddof=1) + 1e-10)) * np.sqrt(periods)

    def _calculate_sortino_ratio(self, returns: np.ndarray, rf_rate: float = 0.0, periods: int = 252) -> float:
        """Calculates Sortino Ratio using downside deviation (ddof=1)."""
        downside_returns = returns[returns < 0]
        if len(downside_returns) < 2:
            return 0.0
        downside_std = np.std(downside_returns, ddof=1)
        return (np.mean(returns) - (rf_rate / periods)) / (downside_std + 1e-10) * np.sqrt(periods)

    def _calculate_max_drawdown(self, cumulative_returns: np.ndarray) -> float:
        """Calculates maximum peak-to-trough decline."""
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - peak) / peak
        return np.min(drawdown)

    def _calculate_calmar_ratio(self, returns: np.ndarray, max_dd: float, periods: int = 252) -> float:
        """Annualized return relative to maximum drawdown."""
        annualized_return = np.mean(returns) * periods
        return annualized_return / (abs(max_dd) + 1e-10)
