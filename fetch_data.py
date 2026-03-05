"""
Fetch real historical OHLCV data from Alpaca Markets.

Usage:
    python fetch_data.py                     # Default symbols, 10 years
    python fetch_data.py SPY QQQ AAPL        # Specific symbols
    python fetch_data.py --years 5           # Custom timeframe
"""

import sys
import os
import argparse
from datetime import datetime, timedelta
import urllib.request
import json
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

DEFAULT_SYMBOLS = ['SPY', 'QQQ', 'AAPL', 'NVDA', 'TSLA', 'AMD', 'MSFT', 'GOOGL', 'AMZN', 'META']

# Alpaca API credentials from environment
ALPACA_API_KEY = os.environ.get('ALPACA_API_KEY', '')
ALPACA_SECRET_KEY = os.environ.get('ALPACA_SECRET_KEY', '')


def fetch_alpaca(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch data from Alpaca Markets API."""
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        print("  Error: Alpaca API credentials not found in environment")
        return None

    # Alpaca data API endpoint for historical bars
    # Using IEX feed which is free
    base_url = "https://data.alpaca.markets/v2/stocks"
    url = f"{base_url}/{symbol}/bars?start={start_date}&end={end_date}&timeframe=1Day&limit=10000&feed=iex"

    headers = {
        'APCA-API-KEY-ID': ALPACA_API_KEY,
        'APCA-API-SECRET-KEY': ALPACA_SECRET_KEY,
    }

    all_bars = []
    next_page_token = None

    try:
        while True:
            page_url = url
            if next_page_token:
                page_url += f"&page_token={next_page_token}"

            req = urllib.request.Request(page_url, headers=headers)
            with urllib.request.urlopen(req, timeout=60) as response:
                data = json.loads(response.read().decode('utf-8'))

            bars = data.get('bars', [])
            if not bars:
                break

            all_bars.extend(bars)
            next_page_token = data.get('next_page_token')
            if not next_page_token:
                break

        if not all_bars:
            return None

        # Convert to DataFrame
        rows = []
        for bar in all_bars:
            rows.append({
                'Date': bar['t'][:10],  # Extract YYYY-MM-DD from timestamp
                'Open': bar['o'],
                'High': bar['h'],
                'Low': bar['l'],
                'Close': bar['c'],
                'Volume': bar['v']
            })

        df = pd.DataFrame(rows)
        df = df.drop_duplicates(subset=['Date']).sort_values('Date').reset_index(drop=True)
        return df

    except Exception as e:
        print(f"  Error fetching {symbol}: {e}")
        return None


def fetch_symbol(symbol: str, years: int = 10) -> bool:
    """Fetch and save data for a single symbol."""
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=years * 365)).strftime('%Y-%m-%d')

    print(f"Fetching {symbol}: {start_date} to {end_date}...")

    df = fetch_alpaca(symbol, start_date, end_date)

    if df is None or len(df) == 0:
        print(f"  Failed to fetch {symbol}")
        return False

    # Save to CSV
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, f'{symbol}.csv')
    df.to_csv(path, index=False)

    print(f"  Saved {len(df)} bars to {path}")
    return True


def main():
    parser = argparse.ArgumentParser(description='Fetch historical OHLCV data')
    parser.add_argument('symbols', nargs='*', default=DEFAULT_SYMBOLS,
                        help='Stock symbols to fetch')
    parser.add_argument('--years', type=int, default=10,
                        help='Number of years of history (default: 10)')
    args = parser.parse_args()

    print(f"Fetching {len(args.symbols)} symbols with {args.years} years of history")
    print(f"Using {'yfinance' if HAS_YFINANCE else 'manual download'}")
    print()

    success = 0
    for symbol in args.symbols:
        if fetch_symbol(symbol, args.years):
            success += 1

    print(f"\nFetched {success}/{len(args.symbols)} symbols successfully")

    if not HAS_YFINANCE:
        print("\nTip: Install yfinance for more reliable data fetching:")
        print("  pip install yfinance")


if __name__ == '__main__':
    main()
