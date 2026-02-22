"""Unit tests for the Redis Streams event bus (Layer 0)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from finalayze.core.events import (
    EventBus,
    MarketDataEvent,
    SignalEvent,
)

# ── Constants (ruff PLR2004: no magic numbers) ──────────────────────────

REDIS_URL = "redis://localhost:6379/0"
TEST_STREAM = "test_stream"
TEST_MSG_ID = "1234567890-0"
TEST_SYMBOL = "AAPL"
TEST_MARKET_ID = "us"
TEST_SEGMENT_ID = "large_cap"
TEST_STRATEGY = "momentum"
TEST_DIRECTION = "BUY"
TEST_CONFIDENCE = 0.85
READ_COUNT = 5
READ_LAST_ID = "0"


# ── MarketDataEvent ──────────────────────────────────────────────────────


class TestMarketDataEvent:
    def test_create_market_data_event(self) -> None:
        candle = {"open": "100.0", "close": "101.0", "high": "102.0", "low": "99.0", "volume": 1000}
        event = MarketDataEvent(symbol=TEST_SYMBOL, market_id=TEST_MARKET_ID, candle=candle)
        assert event.symbol == TEST_SYMBOL
        assert event.market_id == TEST_MARKET_ID
        assert event.candle == candle

    def test_market_data_event_serializes_to_json(self) -> None:
        candle = {"open": "150.0", "close": "151.0"}
        event = MarketDataEvent(symbol=TEST_SYMBOL, market_id=TEST_MARKET_ID, candle=candle)
        json_str = event.model_dump_json()
        assert TEST_SYMBOL in json_str
        assert TEST_MARKET_ID in json_str


# ── SignalEvent ──────────────────────────────────────────────────────────


class TestSignalEvent:
    def test_create_signal_event(self) -> None:
        event = SignalEvent(
            strategy_name=TEST_STRATEGY,
            symbol=TEST_SYMBOL,
            market_id=TEST_MARKET_ID,
            segment_id=TEST_SEGMENT_ID,
            direction=TEST_DIRECTION,
            confidence=TEST_CONFIDENCE,
        )
        assert event.strategy_name == TEST_STRATEGY
        assert event.symbol == TEST_SYMBOL
        assert event.market_id == TEST_MARKET_ID
        assert event.segment_id == TEST_SEGMENT_ID
        assert event.direction == TEST_DIRECTION
        assert event.confidence == TEST_CONFIDENCE

    def test_signal_event_serializes_to_json(self) -> None:
        event = SignalEvent(
            strategy_name=TEST_STRATEGY,
            symbol=TEST_SYMBOL,
            market_id=TEST_MARKET_ID,
            segment_id=TEST_SEGMENT_ID,
            direction=TEST_DIRECTION,
            confidence=TEST_CONFIDENCE,
        )
        json_str = event.model_dump_json()
        assert TEST_STRATEGY in json_str
        assert TEST_SYMBOL in json_str


# ── EventBus ─────────────────────────────────────────────────────────────


class TestEventBus:
    @pytest.fixture
    def mock_redis(self) -> AsyncMock:
        mock = AsyncMock()
        mock.xadd = AsyncMock(return_value=TEST_MSG_ID)
        mock.xread = AsyncMock(return_value=[])
        mock.aclose = AsyncMock()
        return mock

    @pytest.fixture
    def event_bus(self, mock_redis: AsyncMock) -> EventBus:
        with patch("finalayze.core.events.redis.asyncio.from_url", return_value=mock_redis):
            return EventBus(redis_url=REDIS_URL)

    @pytest.mark.asyncio
    async def test_publish_calls_xadd_with_correct_stream(
        self, event_bus: EventBus, mock_redis: AsyncMock
    ) -> None:
        event = MarketDataEvent(
            symbol=TEST_SYMBOL,
            market_id=TEST_MARKET_ID,
            candle={"close": "100.0"},
        )
        await event_bus.publish(TEST_STREAM, event)
        mock_redis.xadd.assert_called_once()
        call_args = mock_redis.xadd.call_args
        assert call_args[0][0] == TEST_STREAM

    @pytest.mark.asyncio
    async def test_publish_includes_event_type_in_payload(
        self, event_bus: EventBus, mock_redis: AsyncMock
    ) -> None:
        event = MarketDataEvent(
            symbol=TEST_SYMBOL,
            market_id=TEST_MARKET_ID,
            candle={"close": "100.0"},
        )
        await event_bus.publish(TEST_STREAM, event)
        call_args = mock_redis.xadd.call_args
        data = call_args[0][1]
        assert "type" in data
        assert data["type"] == "MarketDataEvent"

    @pytest.mark.asyncio
    async def test_publish_includes_json_payload(
        self, event_bus: EventBus, mock_redis: AsyncMock
    ) -> None:
        event = SignalEvent(
            strategy_name=TEST_STRATEGY,
            symbol=TEST_SYMBOL,
            market_id=TEST_MARKET_ID,
            segment_id=TEST_SEGMENT_ID,
            direction=TEST_DIRECTION,
            confidence=TEST_CONFIDENCE,
        )
        await event_bus.publish(TEST_STREAM, event)
        call_args = mock_redis.xadd.call_args
        data = call_args[0][1]
        assert "payload" in data
        assert TEST_SYMBOL in data["payload"]
        assert TEST_STRATEGY in data["payload"]

    @pytest.mark.asyncio
    async def test_publish_returns_message_id(
        self, event_bus: EventBus, mock_redis: AsyncMock
    ) -> None:
        event = MarketDataEvent(
            symbol=TEST_SYMBOL,
            market_id=TEST_MARKET_ID,
            candle={},
        )
        msg_id = await event_bus.publish(TEST_STREAM, event)
        assert msg_id == TEST_MSG_ID

    @pytest.mark.asyncio
    async def test_read_calls_xread_with_stream_and_last_id(
        self, event_bus: EventBus, mock_redis: AsyncMock
    ) -> None:
        mock_redis.xread.return_value = []
        await event_bus.read(TEST_STREAM, count=READ_COUNT, last_id=READ_LAST_ID)
        mock_redis.xread.assert_called_once()
        call_kwargs = mock_redis.xread.call_args[1]
        assert call_kwargs.get("count") == READ_COUNT

    @pytest.mark.asyncio
    async def test_read_returns_empty_list_when_no_messages(
        self, event_bus: EventBus, mock_redis: AsyncMock
    ) -> None:
        mock_redis.xread.return_value = []
        result = await event_bus.read(TEST_STREAM)
        assert result == []

    @pytest.mark.asyncio
    async def test_read_returns_messages_from_xread(
        self, event_bus: EventBus, mock_redis: AsyncMock
    ) -> None:
        raw_messages = [
            (TEST_STREAM, [(TEST_MSG_ID, {"type": "MarketDataEvent", "payload": "{}"})])
        ]
        mock_redis.xread.return_value = raw_messages
        result = await event_bus.read(TEST_STREAM, count=READ_COUNT)
        assert len(result) == 1
        msg_id, fields = result[0]
        assert msg_id == TEST_MSG_ID
        assert fields["type"] == "MarketDataEvent"

    @pytest.mark.asyncio
    async def test_close_calls_aclose(self, event_bus: EventBus, mock_redis: AsyncMock) -> None:
        await event_bus.close()
        mock_redis.aclose.assert_called_once()

    def test_stream_constants_are_defined(self) -> None:
        assert EventBus.STREAM_MARKET_DATA == "market_data"
        assert EventBus.STREAM_SIGNALS == "signals"
        assert EventBus.STREAM_EXECUTION == "execution"

    def test_default_redis_url(self) -> None:
        with patch("finalayze.core.events.redis.asyncio.from_url") as mock_from_url:
            mock_from_url.return_value = MagicMock()
            bus = EventBus()
        mock_from_url.assert_called_once_with(REDIS_URL, decode_responses=True)
        assert bus is not None
