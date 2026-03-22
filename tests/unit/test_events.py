"""Unit tests for the Redis Streams event bus (Layer 0)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pydantic import BaseModel

from finalayze.core.events import EventBus

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


class _TestEvent(BaseModel):
    """Lightweight event model used only in tests."""

    symbol: str
    label: str


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
        event = _TestEvent(symbol=TEST_SYMBOL, label="candle")
        await event_bus.publish(TEST_STREAM, event)
        mock_redis.xadd.assert_called_once()
        call_args = mock_redis.xadd.call_args
        assert call_args[0][0] == TEST_STREAM

    @pytest.mark.asyncio
    async def test_publish_includes_event_type_in_payload(
        self, event_bus: EventBus, mock_redis: AsyncMock
    ) -> None:
        event = _TestEvent(symbol=TEST_SYMBOL, label="candle")
        await event_bus.publish(TEST_STREAM, event)
        call_args = mock_redis.xadd.call_args
        data = call_args[0][1]
        assert "type" in data
        assert data["type"] == "_TestEvent"

    @pytest.mark.asyncio
    async def test_publish_includes_json_payload(
        self, event_bus: EventBus, mock_redis: AsyncMock
    ) -> None:
        event = _TestEvent(symbol=TEST_SYMBOL, label=TEST_STRATEGY)
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
        event = _TestEvent(symbol=TEST_SYMBOL, label="test")
        msg_id = await event_bus.publish(TEST_STREAM, event)
        assert msg_id == TEST_MSG_ID

    @pytest.mark.asyncio
    async def test_read_calls_xread_with_stream_and_last_id(
        self, event_bus: EventBus, mock_redis: AsyncMock
    ) -> None:
        mock_redis.xread.return_value = []
        await event_bus.read(TEST_STREAM, count=READ_COUNT, last_id=READ_LAST_ID)
        mock_redis.xread.assert_called_once()
        call_kwargs = mock_redis.xread.call_args
        # xread is called as xread({stream: last_id}, count=count)
        streams_arg = call_kwargs[0][0]  # first positional arg
        assert streams_arg == {TEST_STREAM: READ_LAST_ID}
        assert call_kwargs[1].get("count") == READ_COUNT

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
            (TEST_STREAM, [(TEST_MSG_ID, {"type": "_TestEvent", "payload": "{}"})])
        ]
        mock_redis.xread.return_value = raw_messages
        result = await event_bus.read(TEST_STREAM, count=READ_COUNT)
        assert len(result) == 1
        msg_id, fields = result[0]
        assert msg_id == TEST_MSG_ID
        assert fields["type"] == "_TestEvent"

    @pytest.mark.asyncio
    async def test_close_calls_aclose(self, event_bus: EventBus, mock_redis: AsyncMock) -> None:
        await event_bus.close()
        mock_redis.aclose.assert_called_once()

    def test_stream_coupons_constant_is_defined(self) -> None:
        assert EventBus.STREAM_COUPONS == "coupons"

    def test_dead_stream_constants_removed(self) -> None:
        assert not hasattr(EventBus, "STREAM_MARKET_DATA")
        assert not hasattr(EventBus, "STREAM_SIGNALS")
        assert not hasattr(EventBus, "STREAM_EXECUTION")

    def test_default_redis_url(self) -> None:
        with patch("finalayze.core.events.redis.asyncio.from_url") as mock_from_url:
            mock_from_url.return_value = MagicMock()
            bus = EventBus()
        mock_from_url.assert_called_once_with(REDIS_URL, decode_responses=True)
        assert bus is not None

    @pytest.mark.asyncio
    async def test_create_group_suppresses_response_error(
        self, event_bus: EventBus, mock_redis: AsyncMock
    ) -> None:
        """create_group silently ignores redis.ResponseError (group already exists)."""
        import redis as redis_lib

        mock_redis.xgroup_create = AsyncMock(
            side_effect=redis_lib.ResponseError("BUSYGROUP Consumer Group already exists")
        )
        # Should not raise
        await event_bus.create_group(TEST_STREAM, "test_group")

    @pytest.mark.asyncio
    async def test_create_group_reraises_non_response_error(
        self, event_bus: EventBus, mock_redis: AsyncMock
    ) -> None:
        """create_group re-raises non-ResponseError exceptions (e.g., ConnectionError)."""
        mock_redis.xgroup_create = AsyncMock(side_effect=ConnectionError("connection lost"))
        with pytest.raises(ConnectionError, match="connection lost"):
            await event_bus.create_group(TEST_STREAM, "test_group")

    @pytest.mark.asyncio
    async def test_create_group_reraises_runtime_error(
        self, event_bus: EventBus, mock_redis: AsyncMock
    ) -> None:
        """create_group re-raises RuntimeError (not a ResponseError)."""
        mock_redis.xgroup_create = AsyncMock(side_effect=RuntimeError("unexpected"))
        with pytest.raises(RuntimeError, match="unexpected"):
            await event_bus.create_group(TEST_STREAM, "test_group")
