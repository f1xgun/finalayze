"""Unit tests for CachingFetcher -- cache hit, miss, and invalidation."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from finalayze.core.schemas import Candle
from finalayze.data.fetchers.caching import CachingFetcher

_START = datetime(2023, 1, 1, tzinfo=UTC)
_END = datetime(2024, 1, 1, tzinfo=UTC)
_SYMBOL = "AAPL"


def _make_candle(idx: int = 0) -> Candle:
    from datetime import timedelta

    return Candle(
        symbol=_SYMBOL,
        market_id="us",
        timeframe="1d",
        timestamp=_START + timedelta(days=idx),
        open=Decimal("100.00"),
        high=Decimal("105.00"),
        low=Decimal("99.00"),
        close=Decimal("103.00"),
        volume=1_000_000,
    )


def _make_candles(n: int = 5) -> list[Candle]:
    return [_make_candle(idx=i) for i in range(n)]


class TestCachingFetcherMiss:
    """Cache miss: delegate is called and result is cached."""

    def test_miss_calls_delegate_and_caches(self, tmp_path: Path) -> None:
        delegate = MagicMock()
        candles = _make_candles(3)
        delegate.fetch_candles.return_value = candles

        fetcher = CachingFetcher(delegate, cache_dir=tmp_path)
        result = fetcher.fetch_candles(_SYMBOL, _START, _END)

        assert result == candles
        delegate.fetch_candles.assert_called_once_with(_SYMBOL, _START, _END, "1d")

        # Verify file was written
        cache_files = list(tmp_path.glob("*.json"))
        assert len(cache_files) == 1

    def test_miss_empty_result_not_cached(self, tmp_path: Path) -> None:
        delegate = MagicMock()
        delegate.fetch_candles.return_value = []

        fetcher = CachingFetcher(delegate, cache_dir=tmp_path)
        result = fetcher.fetch_candles(_SYMBOL, _START, _END)

        assert result == []
        cache_files = list(tmp_path.glob("*.json"))
        assert len(cache_files) == 0


class TestCachingFetcherHit:
    """Cache hit: delegate is not called."""

    def test_hit_returns_cached_data(self, tmp_path: Path) -> None:
        delegate = MagicMock()
        candles = _make_candles(3)
        delegate.fetch_candles.return_value = candles

        fetcher = CachingFetcher(delegate, cache_dir=tmp_path)

        # First call: miss
        result1 = fetcher.fetch_candles(_SYMBOL, _START, _END)
        assert len(result1) == 3

        # Reset mock to verify no second call
        delegate.reset_mock()

        # Second call: hit
        result2 = fetcher.fetch_candles(_SYMBOL, _START, _END)
        assert len(result2) == 3
        delegate.fetch_candles.assert_not_called()

        # Verify data is equivalent
        assert result1[0].symbol == result2[0].symbol
        assert result1[0].close == result2[0].close


class TestCachingFetcherInvalidation:
    """Invalidation removes cached files."""

    def test_invalidate_removes_cache(self, tmp_path: Path) -> None:
        delegate = MagicMock()
        delegate.fetch_candles.return_value = _make_candles(2)

        fetcher = CachingFetcher(delegate, cache_dir=tmp_path)
        fetcher.fetch_candles(_SYMBOL, _START, _END)

        assert fetcher.invalidate(_SYMBOL, _START, _END) is True
        assert list(tmp_path.glob("*.json")) == []

    def test_invalidate_nonexistent_returns_false(self, tmp_path: Path) -> None:
        delegate = MagicMock()
        fetcher = CachingFetcher(delegate, cache_dir=tmp_path)
        assert fetcher.invalidate(_SYMBOL, _START, _END) is False

    def test_after_invalidate_calls_delegate_again(self, tmp_path: Path) -> None:
        delegate = MagicMock()
        delegate.fetch_candles.return_value = _make_candles(2)

        fetcher = CachingFetcher(delegate, cache_dir=tmp_path)
        fetcher.fetch_candles(_SYMBOL, _START, _END)
        fetcher.invalidate(_SYMBOL, _START, _END)

        delegate.reset_mock()
        delegate.fetch_candles.return_value = _make_candles(3)

        result = fetcher.fetch_candles(_SYMBOL, _START, _END)
        delegate.fetch_candles.assert_called_once()
        assert len(result) == 3
