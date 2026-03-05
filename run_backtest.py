"""
Backtest runner: loads OHLCV CSVs, computes features, runs walk-forward
validation and statistical tests, saves results for the research notebook.

Usage:
    python run_backtest.py                     # All symbols
    python run_backtest.py SPY QQQ AAPL        # Specific symbols
"""

import sys
import os
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from trading_pipeline import TradingPipeline
from backtester import AdvancedBacktester
from statistical_tests import run_all_tests, sensitivity_analysis
from monte_carlo_stress_test import MonteCarloStressTest

logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s %(levelname)s %(message)s'
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Suppress verbose internal logging during backtesting
for mod in ['trading_pipeline', 'regime_detector', 'bayesian_kelly',
            'market_friction_model', 'monte_carlo_stress_test',
            'backtester', 'statistical_tests']:
    logging.getLogger(mod).setLevel(logging.ERROR)

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')


class MockAccount:
    """Fake Alpaca account for backtesting."""
    def __init__(self, equity=100000):
        self.buying_power = str(equity)
        self.equity = str(equity)


class MockAPI:
    """Fake Alpaca API that returns a mock account."""
    def __init__(self, equity=100000):
        self._account = MockAccount(equity)

    def get_account(self):
        return self._account


def calculate_atr(high, low, close, period=14):
    """Compute Average True Range from OHLC arrays."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar has no previous close
    atr = pd.Series(tr).rolling(window=period).mean().values
    return atr


def compute_features(df):
    """
    Derive vector prices, signal strengths, and ATR from OHLCV DataFrame.

    Mirrors the feature computation in production_trader_v2.py lines 387-391.
    """
    open_prices = df['Open'].values.astype(float)
    close = df['Close'].values.astype(float)
    high = df['High'].values.astype(float)
    low = df['Low'].values.astype(float)
    volume = df['Volume'].values.astype(float)

    atr = calculate_atr(high, low, close, period=14)

    # EMA-20 as vector price
    vector_prices = pd.Series(close).ewm(span=20, adjust=False).mean().values

    # Signal strength: normalized deviation from vector
    price_deviation = np.abs(close - vector_prices) / (atr + 1e-10)
    vector_strengths = np.clip(price_deviation / 1.5, 0, 1)

    return open_prices, close, vector_prices, vector_strengths, atr, volume


def load_symbol(symbol):
    """Load OHLCV CSV for a symbol and return a DataFrame."""
    path = os.path.join(DATA_DIR, f'{symbol}.csv')
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, parse_dates=['Date'])
    df.sort_values('Date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def run_single_backtest(symbol, df, initial_equity=100000):
    """Run backtest + walk-forward + statistical tests for one symbol."""
    logger.info(f"--- {symbol}: {len(df)} bars ---")

    open_prices, close, vector_prices, strengths, atr, volume = compute_features(df)

    # Need enough data for the backtester (starts at bar 50)
    if len(close) < 100:
        logger.warning(f"{symbol}: insufficient data ({len(close)} bars), skipping")
        return None

    api = MockAPI(initial_equity)
    engine = TradingPipeline(api=api, account_equity=initial_equity, fractional_kelly=0.5)
    # Use a tighter ATR multiplier like production
    engine.regime_detector.atr_multiplier = 1.5

    # --- Standard backtest (with look-ahead bias prevention) ---
    bt = AdvancedBacktester(engine, initial_equity)
    result = bt.run_backtest(close, vector_prices, atr, volume, max_hold_bars=48, open_data=open_prices)

    if result.get('status') == 'ZERO_TRADES_EXECUTED':
        logger.warning(f"{symbol}: zero trades generated")
        return {'symbol': symbol, 'status': 'no_trades'}

    trades = bt.trades
    trade_returns = np.array([t['return'] for t in trades])

    # --- Walk-forward validation (with look-ahead bias prevention) ---
    wf_bt = AdvancedBacktester(engine, initial_equity)
    wf_result = wf_bt.run_walk_forward(
        close, vector_prices, atr, volume,
        n_splits=5, train_pct=0.6, max_hold_bars=48, open_data=open_prices
    )

    # --- Statistical tests ---
    stat_tests = run_all_tests(trade_returns) if len(trade_returns) >= 3 else {'error': 'too_few_trades'}

    # --- Monte Carlo ---
    mc = MonteCarloStressTest(initial_equity=initial_equity, simulations=10000)
    mc_results = {}
    if len(trade_returns) >= 5:
        cone = mc.run_probability_cone(trade_returns)
        ror = mc.calculate_risk_of_ruin(trade_returns)
        tail = mc.get_tail_risk_metrics(trade_returns)
        stress = mc.stress_test_shocks(trade_returns)
        mc_results = {
            'p5_worst_case': float(cone['p5_worst_case']),
            'p50_median': float(cone['p50_median']),
            'p95_best_case': float(cone['p95_best_case']),
            'risk_of_ruin_pct': float(ror['risk_of_ruin_pct']),
            'var_95': float(tail['var_alpha']),
            'cvar_95': float(tail['cvar_expected_shortfall']),
            'shock_survival_rate': float(stress['shock_survival_rate']),
            'median_drawdown': float(stress['median_drawdown']),
        }

    # --- Regime-conditional analysis ---
    regime_breakdown = {}
    for trade in trades:
        regime = trade.get('regime', 'UNKNOWN')
        if regime not in regime_breakdown:
            regime_breakdown[regime] = {'returns': [], 'count': 0}
        regime_breakdown[regime]['returns'].append(trade['return'])
        regime_breakdown[regime]['count'] += 1

    for regime, data in regime_breakdown.items():
        rets = np.array(data['returns'])
        regime_breakdown[regime] = {
            'count': data['count'],
            'mean_return': float(np.mean(rets)),
            'win_rate': float(np.sum(rets > 0) / len(rets)),
            'total_return_pct': float((np.prod(1 + rets) - 1) * 100),
        }

    # --- Exit reason breakdown ---
    exit_reasons = {}
    for t in trades:
        reason = t.get('exit_reason', 'unknown')
        exit_reasons[reason] = exit_reasons.get(reason, 0) + 1

    # --- Build equity curve ---
    equity_curve = [initial_equity]
    for r in trade_returns:
        equity_curve.append(equity_curve[-1] * (1 + r))

    # --- Assemble result ---
    output = {
        'symbol': symbol,
        'data_bars': len(close),
        'date_range': {
            'start': str(df['Date'].iloc[0].date()),
            'end': str(df['Date'].iloc[-1].date()),
        },
        'backtest': {
            'trade_count': result['trade_count'],
            'win_rate': float(result['win_rate']),
            'expectancy': float(result['expectancy']),
            'sharpe_ratio': float(result['sharpe_ratio']),
            'sortino_ratio': float(result['sortino_ratio']),
            'calmar_ratio': float(result['calmar_ratio']),
            'max_drawdown': float(result['max_drawdown']),
            'total_return_pct': float(result['total_return_pct']),
            'terminal_wealth': float(result['terminal_wealth']),
            'avg_bars_held': float(result['avg_bars_held']),
            'exit_reasons': exit_reasons,
        },
        'walk_forward': _serialize_wf(wf_result),
        'statistical_tests': _serialize_stat_tests(stat_tests),
        'monte_carlo': mc_results,
        'regime_breakdown': regime_breakdown,
        'equity_curve': [float(e) for e in equity_curve],
        'trade_returns': [float(r) for r in trade_returns],
        'trades': [_serialize_trade(t) for t in trades],
    }

    _print_summary(output)
    return output


def _serialize_wf(wf):
    """Convert walk-forward result to JSON-safe dict."""
    if not wf or 'error' in wf:
        return wf or {}
    out = {}
    for k, v in wf.items():
        if k == 'split_details':
            out[k] = [{kk: float(vv) if isinstance(vv, (np.floating, float)) else vv
                        for kk, vv in s.items()} for s in v]
        elif isinstance(v, (np.floating, float, np.integer, int)):
            out[k] = float(v)
        else:
            out[k] = v
    return out


def _serialize_stat_tests(st):
    """Convert statistical test results to JSON-safe dict."""
    if not st or 'error' in st:
        return st or {}
    out = {}
    for key, val in st.items():
        if isinstance(val, dict):
            out[key] = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                        for k, v in val.items()}
        elif isinstance(val, list):
            out[key] = val
        elif isinstance(val, (np.floating, np.integer)):
            out[key] = float(val)
        else:
            out[key] = val
    return out


def _serialize_trade(t):
    """Convert a single trade dict to JSON-safe dict."""
    out = {}
    for k, v in t.items():
        if isinstance(v, (np.floating, np.integer)):
            out[k] = float(v)
        elif isinstance(v, pd.Timestamp):
            out[k] = str(v)
        elif isinstance(v, dict):
            out[k] = {kk: float(vv) if isinstance(vv, (np.floating, np.integer)) else str(vv)
                       for kk, vv in v.items()}
        else:
            out[k] = v
    return out


def _print_summary(output):
    """Print a concise summary of backtest results."""
    bt = output['backtest']
    wf = output['walk_forward']
    st = output['statistical_tests']
    mc = output['monte_carlo']

    print(f"\n{'='*60}")
    print(f"  {output['symbol']}  |  {output['data_bars']} bars  |  "
          f"{output['date_range']['start']} to {output['date_range']['end']}")
    print(f"{'='*60}")

    print(f"  Trades: {bt['trade_count']}  |  Win Rate: {bt['win_rate']:.1%}  |  "
          f"Expectancy: {bt['expectancy']:.4f}")
    print(f"  Sharpe: {bt['sharpe_ratio']:.2f}  |  Sortino: {bt['sortino_ratio']:.2f}  |  "
          f"Calmar: {bt['calmar_ratio']:.2f}")
    print(f"  Max DD: {bt['max_drawdown']:.2%}  |  Total Return: {bt['total_return_pct']:.1f}%  |  "
          f"Avg Hold: {bt['avg_bars_held']:.0f} bars")

    if bt.get('exit_reasons'):
        reasons = ', '.join(f"{k}: {v}" for k, v in bt['exit_reasons'].items())
        print(f"  Exits: {reasons}")

    if wf and 'in_sample_sharpe' in wf:
        print(f"\n  Walk-Forward ({wf.get('n_splits', '?')} splits):")
        print(f"    IS Sharpe: {wf['in_sample_sharpe']:.2f}  |  "
              f"OOS Sharpe: {wf['out_of_sample_sharpe']:.2f}  |  "
              f"Degradation: {wf['sharpe_degradation']:.2f}")
        print(f"    IS Trades: {wf['total_is_trades']}  |  OOS Trades: {wf['total_oos_trades']}")

    if st and 'significance' in st:
        sig = st['significance']
        if isinstance(sig, dict) and 'p_value' in sig:
            print(f"\n  Statistical Tests:")
            print(f"    Strategy t-test: p={sig['p_value']:.4f}  "
                  f"({'SIGNIFICANT' if sig.get('is_significant') else 'NOT significant'})")
            if 'ci_lower' in sig:
                print(f"    95% CI for mean return: [{sig['ci_lower']:.6f}, {sig['ci_upper']:.6f}]")

        adf = st.get('stationarity', {})
        if isinstance(adf, dict) and 'p_value' in adf:
            print(f"    ADF stationarity: p={adf['p_value']:.4f}  "
                  f"({'stationary' if adf.get('is_stationary') else 'NON-stationary'})")

        lb = st.get('autocorrelation', {})
        if isinstance(lb, dict) and 'final_p_value' in lb:
            print(f"    Ljung-Box autocorr: p={lb['final_p_value']:.4f}  "
                  f"({'autocorrelated' if lb.get('has_autocorrelation') else 'no autocorrelation'})")

    if mc:
        print(f"\n  Monte Carlo (10K sims):")
        print(f"    Median equity: ${mc.get('p50_median', 0):,.0f}  |  "
              f"VaR95: ${mc.get('var_95', 0):,.0f}  |  "
              f"CVaR95: ${mc.get('cvar_95', 0):,.0f}")
        print(f"    Risk of Ruin (20%): {mc.get('risk_of_ruin_pct', 0):.1f}%  |  "
              f"Shock survival: {mc.get('shock_survival_rate', 0):.1%}")

    if output.get('regime_breakdown'):
        print(f"\n  Regime Breakdown:")
        for regime, data in output['regime_breakdown'].items():
            print(f"    {regime}: {data['count']} trades, "
                  f"win rate {data['win_rate']:.1%}, "
                  f"mean return {data['mean_return']:.4f}")

    if st and 'warnings' in st:
        warnings = st['warnings']
        if warnings:
            print(f"\n  Warnings:")
            for w in warnings:
                print(f"    - {w}")


def main():
    symbols = sys.argv[1:] if len(sys.argv) > 1 else None

    # Discover available symbols from data directory
    if symbols is None:
        if not os.path.exists(DATA_DIR):
            print(f"No data directory found at {DATA_DIR}")
            print("Place OHLCV CSV files (with Date,Open,High,Low,Close,Volume columns) in data/")
            sys.exit(1)
        symbols = sorted([f.replace('.csv', '') for f in os.listdir(DATA_DIR) if f.endswith('.csv')])

    if not symbols:
        print("No data files found.")
        sys.exit(1)

    print(f"Running backtest on {len(symbols)} symbols: {', '.join(symbols)}")
    print(f"Initial equity: $100,000 | Kelly fraction: 50% | ATR multiplier: 1.5")

    all_results = {}
    all_returns = []

    for symbol in symbols:
        df = load_symbol(symbol)
        if df is None:
            logger.warning(f"{symbol}: no data file found, skipping")
            continue
        result = run_single_backtest(symbol, df)
        if result:
            all_results[symbol] = result
            if result.get('trade_returns'):
                all_returns.extend(result['trade_returns'])

    # --- Aggregate portfolio analysis ---
    if all_returns:
        all_returns = np.array(all_returns)
        print(f"\n{'='*60}")
        print(f"  PORTFOLIO AGGREGATE  |  {len(all_returns)} total trades across {len(all_results)} symbols")
        print(f"{'='*60}")
        print(f"  Win Rate: {np.sum(all_returns > 0) / len(all_returns):.1%}")
        print(f"  Mean Return: {np.mean(all_returns):.4f}")
        print(f"  Sharpe (annualized): {np.mean(all_returns) / (np.std(all_returns, ddof=1) + 1e-10) * np.sqrt(252):.2f}")
        cum = np.cumprod(1 + all_returns)
        peak = np.maximum.accumulate(cum)
        max_dd = np.min((cum - peak) / peak)
        print(f"  Max Drawdown: {max_dd:.2%}")
        print(f"  Total Return: {(cum[-1] - 1) * 100:.1f}%")

        # Portfolio statistical tests
        port_stats = run_all_tests(all_returns)
        if port_stats.get('significance', {}).get('p_value') is not None:
            print(f"  Strategy significance (t-test): p={port_stats['significance']['p_value']:.4f}")
        if port_stats.get('warnings'):
            print(f"  Warnings:")
            for w in port_stats['warnings']:
                print(f"    - {w}")

        all_results['_portfolio'] = {
            'total_trades': len(all_returns),
            'symbols_traded': len(all_results),
            'win_rate': float(np.sum(all_returns > 0) / len(all_returns)),
            'mean_return': float(np.mean(all_returns)),
            'sharpe': float(np.mean(all_returns) / (np.std(all_returns, ddof=1) + 1e-10) * np.sqrt(252)),
            'max_drawdown': float(max_dd),
            'total_return_pct': float((cum[-1] - 1) * 100),
            'statistical_tests': _serialize_stat_tests(port_stats),
            'all_returns': [float(r) for r in all_returns],
        }

    # --- Save results ---
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_path = os.path.join(RESULTS_DIR, 'backtest_results.json')
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
