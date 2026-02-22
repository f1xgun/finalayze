"""Abstract base for data fetchers (Layer 2)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime  # noqa: TC003

from finalayze.core.schemas import Candle  # noqa: TC001


class BaseFetcher(ABC):
    """Abstract base class for market data fetchers."""

    @abstractmethod
    def fetch_candles(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str = "1d",
    ) -> list[Candle]:
        """Fetch OHLCV candles for a symbol.

        Args:
            symbol: Ticker symbol (e.g. "AAPL").
            start: Start of the date range (inclusive).
            end: End of the date range (exclusive).
            timeframe: Bar size, e.g. "1d", "1h".

        Returns:
            List of Candle objects sorted by timestamp ascending.
        """
        ...
