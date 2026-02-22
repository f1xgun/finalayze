"""Seed historical data for US market symbols.

Downloads 2 years of daily OHLCV candles via YFinanceFetcher and validates
them through DataNormalizer.  Results are printed to stdout; database
persistence will be added in a later phase once Alembic/PG setup is complete.

Usage:
    uv run python scripts/seed_historical_data.py
    uv run python scripts/seed_historical_data.py --symbols AAPL MSFT --years 1
"""

from __future__ import annotations

import argparse
import logging
from datetime import UTC, datetime, timedelta

from finalayze.data.fetchers.yfinance import YFinanceFetcher
from finalayze.data.normalizer import DataNormalizer

DEFAULT_SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "SPY"]
DEFAULT_YEARS = 2
LOG = logging.getLogger(__name__)


def main() -> None:
    """Seed historical market data from Yahoo Finance."""
    parser = argparse.ArgumentParser(description="Seed historical market data")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=DEFAULT_SYMBOLS,
        help="Ticker symbols to download (default: AAPL MSFT GOOGL AMZN SPY)",
    )
    parser.add_argument(
        "--years",
        type=int,
        default=DEFAULT_YEARS,
        help="Number of years of history to fetch (default: 2)",
    )
    args = parser.parse_args()

    end = datetime.now(tz=UTC)
    start = end - timedelta(days=args.years * 365)

    fetcher = YFinanceFetcher(market_id="us")
    normalizer = DataNormalizer(market_id="us", source="yfinance")

    for symbol in args.symbols:
        LOG.info("Fetching %s from %s to %s", symbol, start.date(), end.date())
        candles = fetcher.fetch_candles(symbol, start, end)
        normalized = normalizer.normalize_batch(candles)
        print(f"{symbol}: fetched {len(normalized)} candles from {start.date()} to {end.date()}")

    print("Done.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
