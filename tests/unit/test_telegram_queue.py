"""Unit tests for AlertQueue: priority ordering, rate limiting, batching, retry."""

from __future__ import annotations

import asyncio
import time
from collections import deque
from unittest.mock import AsyncMock, patch

import pytest

from finalayze.api.alerts import AlertPriority, AlertQueue, TelegramMessageQueue
from finalayze.api.telegram_transport import TelegramTransport

# ── Constants (ruff PLR2004) ─────────────────────────────────────────────────
VALID_TOKEN = "1234567890:AABBccDDeEFfGgHhIiJj"  # noqa: S105
VALID_CHAT_ID = "-1001234567890"
RATE_LIMIT = 20
BATCH_THRESHOLD = 5


def _make_queue(loop: asyncio.AbstractEventLoop) -> tuple[AlertQueue, AsyncMock]:
    """Return (queue, mock_transport_send) for testing."""
    transport = TelegramTransport(bot_token=VALID_TOKEN, chat_id=VALID_CHAT_ID)
    mock_send = AsyncMock(return_value=(True, None))
    transport.send = mock_send  # type: ignore[method-assign]
    queue = AlertQueue(loop=loop, transport=transport)
    return queue, mock_send


class TestAlertQueueAlias:
    """TelegramMessageQueue is an alias for AlertQueue."""

    def test_alias_is_same_class(self) -> None:
        assert TelegramMessageQueue is AlertQueue


class TestPriorityOrdering:
    """IMPORTANT messages are dequeued before INFO messages."""

    @pytest.mark.asyncio
    async def test_important_before_info(self) -> None:
        loop = asyncio.get_running_loop()
        queue, _ = _make_queue(loop)
        queue.post("info msg", AlertPriority.INFO)
        queue.post("important msg", AlertPriority.IMPORTANT)
        await asyncio.sleep(0.01)  # let call_soon_threadsafe deliver
        msg = await queue._queue.get()
        assert msg.priority == AlertPriority.IMPORTANT


class TestRateLimiting:
    """Rate limiter blocks when 20 messages sent in last 60 seconds."""

    @pytest.mark.asyncio
    async def test_rate_limit_blocks_at_threshold(self) -> None:
        loop = asyncio.get_running_loop()
        queue, _ = _make_queue(loop)
        now = time.monotonic()
        queue._sent_timestamps = deque([now - i for i in range(RATE_LIMIT)], maxlen=RATE_LIMIT * 2)
        assert queue._is_rate_limited() is True

    @pytest.mark.asyncio
    async def test_rate_limit_not_blocked_below_threshold(self) -> None:
        loop = asyncio.get_running_loop()
        queue, _ = _make_queue(loop)
        now = time.monotonic()
        limit_minus = 15
        queue._sent_timestamps = deque(
            [now - i for i in range(RATE_LIMIT - limit_minus)], maxlen=RATE_LIMIT * 2
        )
        assert queue._is_rate_limited() is False


class TestBatching:
    """5+ pending fill messages are batched into single digest."""

    @pytest.mark.asyncio
    async def test_batch_five_important_messages(self) -> None:
        loop = asyncio.get_running_loop()
        queue, _ = _make_queue(loop)
        batch_count = 6
        for i in range(batch_count):
            queue.post(f"Fill {i}", AlertPriority.IMPORTANT)
        await asyncio.sleep(0.01)
        messages = queue._collect_batch(AlertPriority.IMPORTANT)
        assert len(messages) >= BATCH_THRESHOLD


class TestRetryLogic:
    """Failed transport.send gets one retry after 5s delay, then dropped."""

    @pytest.mark.asyncio
    async def test_retry_on_failure(self) -> None:
        loop = asyncio.get_running_loop()
        queue, mock_send = _make_queue(loop)
        mock_send.side_effect = [(False, None), (True, None)]
        with patch("finalayze.api.alerts.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await queue._send_with_retry("test msg")
        assert result is True
        retry_delay = 5
        mock_sleep.assert_called_once_with(retry_delay)

    @pytest.mark.asyncio
    async def test_drop_after_second_failure(self) -> None:
        loop = asyncio.get_running_loop()
        queue, mock_send = _make_queue(loop)
        mock_send.return_value = (False, None)
        with patch("finalayze.api.alerts.asyncio.sleep", new_callable=AsyncMock):
            result = await queue._send_with_retry("test msg")
        assert result is False
        call_count = 2
        assert mock_send.call_count == call_count


class TestFIFOWithinTier:
    """Queue drain loop processes messages FIFO within same priority tier."""

    @pytest.mark.asyncio
    async def test_fifo_within_same_priority(self) -> None:
        loop = asyncio.get_running_loop()
        queue, _ = _make_queue(loop)
        queue.post("first", AlertPriority.INFO)
        queue.post("second", AlertPriority.INFO)
        await asyncio.sleep(0.01)
        msg1 = await queue._queue.get()
        msg2 = await queue._queue.get()
        assert msg1.timestamp <= msg2.timestamp
        assert msg1.text == "first"
        assert msg2.text == "second"


class TestStartStop:
    """start() creates drain task, stop() cancels it gracefully."""

    @pytest.mark.asyncio
    async def test_start_creates_drain_task(self) -> None:
        loop = asyncio.get_running_loop()
        queue, _ = _make_queue(loop)
        await queue.start()
        assert queue._drain_task is not None
        assert not queue._drain_task.done()
        await queue.stop()

    @pytest.mark.asyncio
    async def test_stop_cancels_drain_task(self) -> None:
        loop = asyncio.get_running_loop()
        queue, _ = _make_queue(loop)
        await queue.start()
        task = queue._drain_task
        await queue.stop()
        assert task is not None
        assert task.done() or task.cancelled()
