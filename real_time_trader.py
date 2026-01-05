import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.quantum_fractal_engine import QuantumFractalEngine
from src.alpaca_trader import AlpacaTrader
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("\n" + "="*60)
print("QUANTUM FRACTAL ENGINE - LIVE PAPER TRADING")
print("="*60 + "\n")

# Initialize and connect
trader = AlpacaTrader()
if not trader.connect():
    print("✗ Failed to connect to Alpaca")
    sys.exit(1)

account_info = trader.get_account_info()
equity = float(account_info.get('equity', 100000))

engine = QuantumFractalEngine(
    api=trader.api,
    account_equity=equity,
    fractional_kelly=0.5
)

logger.info(f"✓ Live paper trading initialized")
logger.info(f"  Account equity: ${equity:,.2f}")
logger.info(f"  Buying power: ${account_info.get('buying_power', 'N/A')}")
logger.info("\n✓ System monitoring markets...")

# Keep running
try:
    while True:
        time.sleep(60)
except KeyboardInterrupt:
    logger.info("Shutdown signal received")

print("\n" + "="*60)
print("✓ Trading session ended")
print("="*60 + "\n")
