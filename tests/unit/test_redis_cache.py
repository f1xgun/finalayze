"""Tests for RedisCache (data/cache.py)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

from finalayze.core.schemas import Candle
from finalayze.data.cache import RedisCache, _compute_sentiment_ttl

_SENTIMENT_TTL_DEFAULT = 1800  # must match cache._SENTIMENT_TTL_SECONDS
_SENTIMENT_TTL_BUFFER = 1800  # must match cache._SENTIMENT_TTL_BUFFER_SECONDS


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


class TestSentimentTTLFreeze:
    """Tests for _compute_sentiment_ttl() dynamic TTL based on MOEX market hours."""

    def test_sentiment_ttl_extended_when_market_closed(self) -> None:
        """When MOEX is closed (Saturday 14:00 MSK), TTL = seconds_to_next_open + 1800."""
        # Saturday 14:00 MSK = Saturday 11:00 UTC
        saturday_msk = datetime(2026, 4, 18, 11, 0, 0, tzinfo=UTC)  # Saturday
        with (
            patch("finalayze.data.cache.MOEX_MARKET_SCHEDULE.is_market_open", return_value=False),
            patch(
                "finalayze.data.cache.MOEX_MARKET_SCHEDULE.next_open",
                return_value=datetime(2026, 4, 20, 7, 0, 0, tzinfo=UTC),  # Monday 10:00 MSK
            ),
        ):
            ttl = _compute_sentiment_ttl(saturday_msk)
            seconds_to_open = int(
                (datetime(2026, 4, 20, 7, 0, 0, tzinfo=UTC) - saturday_msk).total_seconds()
            )
            expected = seconds_to_open + _SENTIMENT_TTL_BUFFER
            assert ttl == expected

    def test_sentiment_ttl_normal_when_market_open(self) -> None:
        """When MOEX is open (Wednesday 12:00 MSK), TTL = 1800."""
        wednesday_utc = datetime(2026, 4, 15, 9, 0, 0, tzinfo=UTC)  # Wed 12:00 MSK
        with patch("finalayze.data.cache.MOEX_MARKET_SCHEDULE.is_market_open", return_value=True):
            ttl = _compute_sentiment_ttl(wednesday_utc)
            assert ttl == _SENTIMENT_TTL_DEFAULT

    def test_sentiment_ttl_minimum_is_1800(self) -> None:
        """TTL never goes below 1800s even if next_open is very close."""
        # Market closed but next open is only 60 seconds away
        now = datetime(2026, 4, 15, 6, 59, 0, tzinfo=UTC)
        with (
            patch("finalayze.data.cache.MOEX_MARKET_SCHEDULE.is_market_open", return_value=False),
            patch(
                "finalayze.data.cache.MOEX_MARKET_SCHEDULE.next_open",
                return_value=now + timedelta(seconds=60),
            ),
        ):
            ttl = _compute_sentiment_ttl(now)
            # 60 + 1800 = 1860, which is > 1800, but the max() guard ensures >= 1800
            assert ttl >= _SENTIMENT_TTL_DEFAULT


class TestEventTypeCache:
    """Tests for set_event_type() / get_event_type() on RedisCache."""

    @pytest.mark.asyncio
    async def test_set_event_type_stores_in_redis(
        self, cache: RedisCache, mock_redis: AsyncMock
    ) -> None:
        """set_event_type(segment, code, ttl) stores float under event_type:{segment} key."""
        await cache.set_event_type("ru_blue_chips:SBER", 1.0, ttl=3600)
        mock_redis.set.assert_called_once_with("event_type:ru_blue_chips:SBER", "1.0", ex=3600)

    @pytest.mark.asyncio
    async def test_get_event_type_returns_float(
        self, cache: RedisCache, mock_redis: AsyncMock
    ) -> None:
        """get_event_type(segment) returns cached float."""
        mock_redis.get.return_value = "2.0"
        result = await cache.get_event_type("ru_blue_chips:SBER")
        assert result == 2.0
        mock_redis.get.assert_called_once_with("event_type:ru_blue_chips:SBER")

    @pytest.mark.asyncio
    async def test_get_event_type_returns_none_on_miss(
        self, cache: RedisCache, mock_redis: AsyncMock
    ) -> None:
        """Returns None when key not in Redis."""
        mock_redis.get.return_value = None
        result = await cache.get_event_type("ru_tech:YDEX")
        assert result is None


class TestClose:
    """Tests for connection cleanup."""

    @pytest.mark.asyncio
    async def test_close(self, cache: RedisCache, mock_redis: AsyncMock) -> None:
        await cache.close()
        mock_redis.aclose.assert_called_once()
