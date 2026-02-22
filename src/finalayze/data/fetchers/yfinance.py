"""Yahoo Finance data fetcher (Layer 2)."""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

import pandas as pd
import yfinance as yf

from finalayze.core.schemas import Candle
from finalayze.data.fetchers.base import BaseFetcher

if TYPE_CHECKING:
    from datetime import datetime


class YFinanceFetcher(BaseFetcher):
    """Fetches OHLCV candles via the yfinance library."""

    def __init__(self, market_id: str = "us") -> None:
        self._market_id = market_id

    def fetch_candles(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str = "1d",
    ) -> list[Candle]:
        """Fetch OHLCV candles from Yahoo Finance.

        Args:
            symbol: Ticker symbol (e.g. "AAPL").
            start: Start of the date range (inclusive).
            end: End of the date range (exclusive).
            timeframe: Bar size passed as yfinance interval (e.g. "1d", "1h").

        Returns:
            List of Candle objects sorted by timestamp ascending,
            or an empty list if no data is available.
        """
        df = yf.download(
            symbol, start=start, end=end, interval=timeframe, progress=False, auto_adjust=True
        )
        # yfinance >= 0.2 may return multi-level columns when downloading a single ticker;
        # flatten to single-level so column access by name works correctly.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        if df.empty:
            return []

        candles: list[Candle] = []
        for timestamp, row in df.iterrows():
            candles.append(
                Candle(
                    symbol=symbol,
                    market_id=self._market_id,
                    timeframe=timeframe,
                    timestamp=timestamp,
                    open=Decimal(str(row["Open"])),
                    high=Decimal(str(row["High"])),
                    low=Decimal(str(row["Low"])),
                    close=Decimal(str(row["Close"])),
                    volume=int(row["Volume"]),
                )
            )

        return candles
