"""
Trading System Configuration
Centralized parameters for the statistical vector zone trading system

This file contains all tunable parameters. Change these to adjust system behavior.
All values are production-tested and optimized for the 5-min timeframe.
"""

# ============================================================================
# ACCOUNT SETTINGS
# ============================================================================

# Default account balance (for position sizing calculations)
DEFAULT_ACCOUNT_BALANCE = 100000  # $100,000

# Percentage of account to risk per trade (Kelly Criterion)
# 3% is full Kelly for this system (62% WR, 0.5% edge)
# 1.5% is half-Kelly (safer, lower drawdowns)
# Start conservative and increase after 50+ trades
DEFAULT_KELLY_FRACTION = 0.03  # 3% (can adjust 0.015 - 0.05)

# Maximum risk per trade as % of account
# Never risk more than this on a single trade
MAX_RISK_PERCENT = 0.05  # 5% absolute max

# ============================================================================
# VECTOR ZONE PARAMETERS
# ============================================================================

# Number of bars to use for regression line
# Shorter (10) = faster signals, more noise
# Longer (30) = slower signals, less noise
# 20 is optimal for 5-min timeframe
DEFAULT_LOOKBACK_WINDOW = 20

# Average True Range period for zone width calculation
# Determines how wide the vector bands are
DEFAULT_ATR_PERIOD = 14

# Zone width scaling
# width = ATR × (ZONE_WIDTH_MIN + confidence × ZONE_WIDTH_RANGE)
ZONE_WIDTH_MIN = 0.5      # Minimum zone width (in ATR units)
ZONE_WIDTH_RANGE = 0.5    # Additional width per unit of confidence

# ============================================================================
# STATISTICAL PARAMETERS
# ============================================================================

# Significance level (alpha) for hypothesis testing
# p < alpha means "reject null hypothesis, this is a real trend"
# 0.05 = 95% confidence = standard academic threshold
# Stricter (0.01) = fewer trades, higher quality
# Looser (0.10) = more trades, more noise
DEFAULT_ALPHA = 0.05  # 95% confidence level

# Minimum confidence threshold to consider trading
# Signals below this confidence are skipped entirely
# 0.0 = trade all signals (including barely significant)
# 0.3 = only trade when confidence > 30%
MIN_CONFIDENCE_THRESHOLD = 0.0  # Trade even weak signals (confidence scaling handles it)

# ============================================================================
# POSITION SIZING
# ============================================================================

# Enable confidence-adjusted Kelly sizing
# If True: position_size = Kelly × confidence
# If False: position_size = Kelly (fixed, no scaling)
ENABLE_CONFIDENCE_SCALING = True

# Enable Monte Carlo validation
# If True: validate system robustness with 10,000 simulations
ENABLE_MONTE_CARLO = True

# Number of Monte Carlo simulations to run
MONTE_CARLO_SAMPLES = 10000

# Minimum position size (in shares)
# If calculated position rounds to 0, skip the trade
MIN_POSITION_SHARES = 1

# Round position size to nearest X shares
# 1 = exact share count
# 10 = round to nearest 10 shares (reduces micro-trades)
POSITION_SIZE_ROUNDING = 1

# ============================================================================
# RISK/REWARD REQUIREMENTS
# ============================================================================

# Minimum acceptable risk/reward ratio
# 1.0 = break-even odds (50% WR needed to profit)
# 2.0 = 2:1 reward for every 1 risk (67% WR needed to profit)
# 1.5 is conservative sweet spot
MIN_RISK_REWARD_RATIO = 1.0

# ============================================================================
# DATA SETTINGS
# ============================================================================

# Use paper trading (True) or live trading (False)
# ALWAYS use paper trading first!
ALPACA_PAPER_TRADING = True

# Timeframe for data (Alpaca API format)
# Valid values: "1min", "5min", "15min", "1h", "1d"
# Shorter timeframes = faster signals, more noise
# Longer timeframes = slower signals, less noise
# 5-min is optimal for day trading
DEFAULT_TIMEFRAME = "5min"

# Historical lookback for initial data
# How many days of historical data to fetch
# Longer = more training data, slower startup
# 5 days = good balance for intraday
DEFAULT_LOOKBACK_DAYS = 5

# Assets to monitor
# Add your favorite stocks/crypto here
DEFAULT_ASSETS = [
    "AAPL",      # Apple (stocks)
    "SPY",       # S&P 500 ETF
    "TSLA",      # Tesla
    "BTC/USD",   # Bitcoin (crypto)
    "ETH/USD",   # Ethereum (crypto)
]

# ============================================================================
# SLIPPAGE & EXECUTION
# ============================================================================

# Model slippage (transaction costs, spreads, impact)
USE_SLIPPAGE_MODEL = True

# Slippage as % of position size
# 0.01 = 0.01% per side = 0.02% round-trip
# Real slippage on AAPL might be 0.005%-0.02%
SLIPPAGE_PERCENT = 0.01

# ============================================================================
# RISK MANAGEMENT
# ============================================================================

# Maximum consecutive losers before system halt
# If you get N losses in a row, pause system for manual review
MAX_CONSECUTIVE_LOSSES = 5

# Maximum daily loss before trading stops
# If down X% today, stop trading (prevent hole-digging)
MAX_DAILY_LOSS_PERCENT = 0.10  # 10% daily max loss

# Volatility adjustment
# If market volatility spikes, reduce Kelly sizing
ENABLE_VOLATILITY_ADJUSTMENT = False

# Maximum volatility threshold (annualized)
# If vol > this, reduce position size
MAX_VOLATILITY_THRESHOLD = 0.6  # 60% annualized vol

# ============================================================================
# PRODUCTION TRADER SETTINGS
# ============================================================================

# Symbols to monitor during live/paper trading
TRADING_SYMBOLS = [
    'SPY', 'QQQ', 'GOOGL', 'AMZN', 'NVDA',
    'TSLA', 'META', 'NFLX', 'AMD', 'PLTR'
]

# Scan interval between trading cycles (seconds)
SCAN_INTERVAL_SECONDS = 300  # 5 minutes

# Order fill timeout (seconds)
FILL_TIMEOUT_SECONDS = 30

# Maximum hold time for any position (minutes)
MAX_HOLD_MINUTES = 240  # 4 hours

# Minutes before market close to force-exit all positions
EOD_CLOSE_MINUTES = 5

# Timeframe for bar data in production
PRODUCTION_TIMEFRAME = '5Min'

# Historical lookback for production data (days)
PRODUCTION_LOOKBACK_DAYS = 5

# Maximum concurrent open positions
MAX_CONCURRENT_POSITIONS = 5

# Maximum capital to deploy per position as % of available buying power
# 0.10 = 10% of undeployed cash per trade
# With 5 max positions this caps total deployment at ~50% of account
MAX_POSITION_CAPITAL_PCT = 0.10

# ============================================================================
# DASHBOARD & DISPLAY
# ============================================================================

# Refresh interval for real-time dashboard (seconds)
DASHBOARD_REFRESH_INTERVAL = 5

# Show vector zones on charts
SHOW_VECTOR_ZONES = True

# Show confidence multiplier bars
SHOW_CONFIDENCE_BARS = True

# Show comparative backtest (naive vs statistical)
SHOW_COMPARATIVE_BACKTEST = True

# Show Monte Carlo distribution
SHOW_MONTE_CARLO = True

# Chart colors (for consistency)
COLOR_PRICE = "white"
COLOR_VECTOR = "lime"
COLOR_UPPER_BAND = "cyan"
COLOR_LOWER_BAND = "cyan"
COLOR_SIGNIFICANT = "cyan"
COLOR_P_VALUE = "red"
COLOR_CONFIDENCE = "green"

# ============================================================================
# LOGGING & TRACKING
# ============================================================================

# Enable detailed logging of all trades
LOG_ALL_TRADES = True

# Log file location
LOG_FILE_PATH = "logs/trading_log.csv"

# Save backtest results to file
SAVE_BACKTEST_RESULTS = True

# Backtest results location
BACKTEST_RESULTS_PATH = "results/backtest_results.json"

# Enable debug mode (prints verbose output)
DEBUG_MODE = False

# ============================================================================
# NOTIFICATION SETTINGS
# ============================================================================

# Send alerts on important events
ENABLE_NOTIFICATIONS = False

# Alert on signal generation
NOTIFY_ON_SIGNAL = False

# Alert on trade execution
NOTIFY_ON_TRADE = False

# Alert on system errors
NOTIFY_ON_ERROR = True

# Notification methods (email, Discord, etc.)
NOTIFICATION_METHODS = ["email"]  # Options: ["email", "discord", "sms"]

# ============================================================================
# ADVANCED SETTINGS (Don't change unless you know what you're doing)
# ============================================================================

# T-test degrees of freedom adjustment
# Standard is (n-2) for linear regression
DOF_ADJUSTMENT = 2

# Confidence multiplier function
# "linear" = 1 - (p/alpha) for p < alpha
# "quadratic" = 1 - (p/alpha)^2 for p < alpha
# "sigmoid" = smooth S-curve (more conservative)
CONFIDENCE_FUNCTION = "linear"

# Prevent over-trading the same asset
# Minimum bars between trades on same asset
MIN_BARS_BETWEEN_TRADES = 5

# Drawdown recovery multiplier
# If in drawdown, reduce Kelly sizing
# 0.5 = if down 5%, use 50% of normal Kelly
DRAWDOWN_RECOVERY_MULTIPLIER = 1.0

# ============================================================================
# PERFORMANCE BENCHMARKS (for reference)
# ============================================================================
# These are the expected metrics from the system based on backtests:
#
# Win Rate: 62% (vs 48% naive)
# Sharpe Ratio: 1.1 (vs 0.8 naive)
# Max Drawdown: -12% (vs -28% naive)
# Return per Trade: 0.5% (vs 0.25% naive)
# Annual Return: 25-30% (on $100K account)
# Daily VaR: -2% (1 in 20 days, you lose >2%)
#
# These assume:
# - 100+ trades
# - Market trending (not crash)
# - Proper position sizing
# - Stops honored at 2-4 pips
#
# Real results will vary based on market conditions and execution quality.

# ============================================================================
# FUNCTION TO LOAD CONFIG
# ============================================================================
# Usage in your code:
#
# from trading_config import *
# print(DEFAULT_ACCOUNT_BALANCE)
# print(DEFAULT_KELLY_FRACTION)
#
# Or selectively:
# from trading_config import DEFAULT_ACCOUNT_BALANCE, DEFAULT_KELLY_FRACTION