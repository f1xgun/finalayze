"""Tests for RedisCache (data/cache.py)."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import AsyncMock, patch

import pytest

from finalayze.core.schemas import Candle
from finalayze.data.cache import RedisCache


@pytest.fixture
def mock_redis() -> AsyncMock:
    """Create a mock redis.asyncio.Redis instance."""
    return AsyncMock()


@pytest.fixture
def cache(mock_redis: AsyncMock) -> RedisCache:
    """Create a RedisCache with mocked Redis connection."""
    c = RedisCache.__new__(RedisCache)
    c._redis = mock_redis
    return c


def _make_candle(price: str = "123.456") -> Candle:
    return Candle(
        symbol="AAPL",
        market_id="us",
        timeframe="1d",
        timestamp=datetime(2026, 1, 15, 12, 0, tzinfo=UTC),
        open=Decimal(price),
        high=Decimal(price),
        low=Decimal(price),
        close=Decimal(price),
        volume=1000,
        source="test",
    )


class TestCandleCache:
    """Tests for candle caching."""

    @pytest.mark.asyncio
    async def test_cache_miss_returns_none(self, cache: RedisCache, mock_redis: AsyncMock) -> None:
        mock_redis.get.return_value = None
        result = await cache.get_candles("us", "AAPL", "1d")
        assert result is None
        mock_redis.get.assert_called_once_with("candles:us:AAPL:1d")

    @pytest.mark.asyncio
    async def test_roundtrip(self, cache: RedisCache, mock_redis: AsyncMock) -> None:
        candle = _make_candle()
        await cache.set_candles("us", "AAPL", "1d", [candle])
        mock_redis.set.assert_called_once()

        call_args = mock_redis.set.call_args
        key = call_args.args[0]
        stored_json = call_args.args[1]
        assert key == "candles:us:AAPL:1d"

        # Simulate get returning what was stored
        mock_redis.get.return_value = stored_json
        result = await cache.get_candles("us", "AAPL", "1d")
        assert result is not None
        assert len(result) == 1
        assert result[0].symbol == "AAPL"

    @pytest.mark.asyncio
    async def test_decimal_roundtrip(self, cache: RedisCache, mock_redis: AsyncMock) -> None:
        """Verify Decimal fields survive JSON serialization."""
        candle = _make_candle("999.123456789")
        await cache.set_candles("us", "TEST", "1h", [candle])

        stored_json = mock_redis.set.call_args.args[1]
        mock_redis.get.return_value = stored_json

        result = await cache.get_candles("us", "TEST", "1h")
        assert result is not None
        assert result[0].close == Decimal("999.123456789")

    @pytest.mark.asyncio
    async def test_ttl_passed_to_redis(self, cache: RedisCache, mock_redis: AsyncMock) -> None:
        candle = _make_candle()
        custom_ttl = 60
        await cache.set_candles("us", "AAPL", "1d", [candle], ttl=custom_ttl)
        call_kwargs = mock_redis.set.call_args.kwargs
        assert call_kwargs["ex"] == custom_ttl

    @pytest.mark.asyncio
    async def test_default_ttl(self, cache: RedisCache, mock_redis: AsyncMock) -> None:
        candle = _make_candle()
        await cache.set_candles("us", "AAPL", "1d", [candle])
        call_kwargs = mock_redis.set.call_args.kwargs
        expected_ttl = 300  # _CANDLE_TTL_SECONDS
        assert call_kwargs["ex"] == expected_ttl


class TestSentimentCache:
    """Tests for sentiment caching."""

    @pytest.mark.asyncio
    async def test_cache_miss_returns_none(self, cache: RedisCache, mock_redis: AsyncMock) -> None:
        mock_redis.get.return_value = None
        result = await cache.get_sentiment("us_tech")
        assert result is None

    @pytest.mark.asyncio
    async def test_roundtrip(self, cache: RedisCache, mock_redis: AsyncMock) -> None:
        await cache.set_sentiment("us_tech", 0.75)
        mock_redis.set.assert_called_once()

        stored = mock_redis.set.call_args.args[1]
        mock_redis.get.return_value = stored

        result = await cache.get_sentiment("us_tech")
        assert result is not None
        assert abs(result - 0.75) < 1e-9

    @pytest.mark.asyncio
    async def test_sentiment_ttl(self, cache: RedisCache, mock_redis: AsyncMock) -> None:
        await cache.set_sentiment("us_tech", 0.5)
        call_kwargs = mock_redis.set.call_args.kwargs
        expected_ttl = 1800  # _SENTIMENT_TTL_SECONDS
        assert call_kwargs["ex"] == expected_ttl


class TestClose:
    """Tests for connection cleanup."""

    @pytest.mark.asyncio
    async def test_close(self, cache: RedisCache, mock_redis: AsyncMock) -> None:
        await cache.close()
        mock_redis.aclose.assert_called_once()
