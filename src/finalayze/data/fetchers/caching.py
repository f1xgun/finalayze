"""Caching wrapper for any BaseFetcher (Layer 2).

Stores fetched candle data as JSON in `.cache/candles/` to avoid
re-downloading on repeated backtest runs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from finalayze.core.schemas import Candle
from finalayze.data.fetchers.base import BaseFetcher

if TYPE_CHECKING:
    from datetime import datetime

_DEFAULT_CACHE_DIR = Path(".cache/candles")


class CachingFetcher(BaseFetcher):
    """Transparent caching wrapper around any BaseFetcher delegate.

    Cache key format: ``{symbol}__{start:%Y%m%d}__{end:%Y%m%d}__{timeframe}.json``
    """

    def __init__(
        self,
        delegate: BaseFetcher,
        cache_dir: Path = _DEFAULT_CACHE_DIR,
    ) -> None:
        self._delegate = delegate
        self._cache_dir = cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_key(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str,
    ) -> Path:
        """Build the cache file path for a given request."""
        name = f"{symbol}__{start:%Y%m%d}__{end:%Y%m%d}__{timeframe}.json"
        return self._cache_dir / name

    def fetch_candles(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str = "1d",
    ) -> list[Candle]:
        """Return cached candles on hit, or delegate and cache on miss."""
        path = self._cache_key(symbol, start, end, timeframe)

        # Cache hit
        if path.exists():
            raw = json.loads(path.read_text())
            return [Candle.model_validate(item) for item in raw]

        # Cache miss — delegate
        candles = self._delegate.fetch_candles(symbol, start, end, timeframe)

        # Persist to cache
        if candles:
            path.write_text(json.dumps([c.model_dump(mode="json") for c in candles], default=str))

        return candles

    def invalidate(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str = "1d",
    ) -> bool:
        """Remove a cached entry. Returns True if a file was deleted."""
        path = self._cache_key(symbol, start, end, timeframe)
        if path.exists():
            path.unlink()
            return True
        return False
