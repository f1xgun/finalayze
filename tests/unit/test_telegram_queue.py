"""Unit tests for TelegramMessageQueue with priority, rate limiting, batching, retry."""

from __future__ import annotations

import asyncio
import time
from collections import deque
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from finalayze.core.alerts import AlertPriority, TelegramAlerter, TelegramMessageQueue

# ── Constants (ruff PLR2004) ─────────────────────────────────────────────────
VALID_TOKEN = "1234567890:AABBccDDeEFfGgHhIiJj"  # noqa: S105
VALID_CHAT_ID = "-1001234567890"
RATE_LIMIT = 20
BATCH_THRESHOLD = 5


def _make_alerter() -> TelegramAlerter:
    alerter = TelegramAlerter(bot_token=VALID_TOKEN, chat_id=VALID_CHAT_ID)
    alerter._send = AsyncMock(return_value=True)  # type: ignore[assignment]
    return alerter


class TestCriticalBypass:
    """CRITICAL messages bypass queue and call _send immediately."""

    @pytest.mark.asyncio
    async def test_critical_bypass_sends_immediately(self) -> None:
        alerter = _make_alerter()
        queue = TelegramMessageQueue(alerter)
        await queue.enqueue("CRITICAL alert", AlertPriority.CRITICAL)
        alerter._send.assert_called_once()
        assert "CRITICAL alert" in alerter._send.call_args[0][0]


class TestPriorityOrdering:
    """IMPORTANT messages are dequeued before INFO messages."""

    @pytest.mark.asyncio
    async def test_important_before_info(self) -> None:
        alerter = _make_alerter()
        queue = TelegramMessageQueue(alerter)
        # Enqueue INFO first, then IMPORTANT
        await queue.enqueue("info msg", AlertPriority.INFO)
        await queue.enqueue("important msg", AlertPriority.IMPORTANT)
        # Drain one message
        msg = await queue._queue.get()
        assert msg.priority == AlertPriority.IMPORTANT


class TestRateLimiting:
    """Rate limiter blocks when 20 messages sent in last 60 seconds."""

    @pytest.mark.asyncio
    async def test_rate_limit_blocks_at_threshold(self) -> None:
        alerter = _make_alerter()
        queue = TelegramMessageQueue(alerter)
        now = time.monotonic()
        # Simulate 20 messages sent in the last 60s
        queue._sent_timestamps = deque([now - i for i in range(RATE_LIMIT)], maxlen=RATE_LIMIT * 2)
        # _is_rate_limited should return True
        assert queue._is_rate_limited() is True

    @pytest.mark.asyncio
    async def test_rate_limit_not_blocked_below_threshold(self) -> None:
        alerter = _make_alerter()
        queue = TelegramMessageQueue(alerter)
        now = time.monotonic()
        # Only 5 messages in last 60s
        limit_minus = 15
        queue._sent_timestamps = deque(
            [now - i for i in range(RATE_LIMIT - limit_minus)], maxlen=RATE_LIMIT * 2
        )
        assert queue._is_rate_limited() is False


class TestBatching:
    """5+ pending fill messages are batched into single digest."""

    @pytest.mark.asyncio
    async def test_batch_five_important_messages(self) -> None:
        alerter = _make_alerter()
        queue = TelegramMessageQueue(alerter)
        # Enqueue 6 IMPORTANT messages
        batch_count = 6
        for i in range(batch_count):
            await queue.enqueue(f"Fill {i}", AlertPriority.IMPORTANT)
        # Collect batched messages
        messages = queue._collect_batch(AlertPriority.IMPORTANT)
        assert len(messages) >= BATCH_THRESHOLD


class TestRetryLogic:
    """Failed _send gets one retry after 5s delay, then dropped."""

    @pytest.mark.asyncio
    async def test_retry_on_failure(self) -> None:
        alerter = _make_alerter()
        alerter._send = AsyncMock(side_effect=[False, True])  # type: ignore[assignment]
        queue = TelegramMessageQueue(alerter)
        with patch("finalayze.core.alerts.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await queue._send_with_retry("test msg")
            assert result is True
            retry_delay = 5
            mock_sleep.assert_called_once_with(retry_delay)

    @pytest.mark.asyncio
    async def test_drop_after_second_failure(self) -> None:
        alerter = _make_alerter()
        alerter._send = AsyncMock(return_value=False)  # type: ignore[assignment]
        queue = TelegramMessageQueue(alerter)
        with patch("finalayze.core.alerts.asyncio.sleep", new_callable=AsyncMock):
            result = await queue._send_with_retry("test msg")
            assert result is False
            call_count = 2
            assert alerter._send.call_count == call_count


class TestFIFOWithinTier:
    """Queue drain loop processes messages FIFO within same priority tier."""

    @pytest.mark.asyncio
    async def test_fifo_within_same_priority(self) -> None:
        alerter = _make_alerter()
        queue = TelegramMessageQueue(alerter)
        await queue.enqueue("first", AlertPriority.INFO)
        await queue.enqueue("second", AlertPriority.INFO)
        msg1 = await queue._queue.get()
        msg2 = await queue._queue.get()
        assert msg1.timestamp <= msg2.timestamp
        assert msg1.text == "first"
        assert msg2.text == "second"


class TestStartStop:
    """start() creates drain task, stop() cancels it gracefully."""

    @pytest.mark.asyncio
    async def test_start_creates_drain_task(self) -> None:
        alerter = _make_alerter()
        queue = TelegramMessageQueue(alerter)
        await queue.start()
        assert queue._drain_task is not None
        assert not queue._drain_task.done()
        await queue.stop()

    @pytest.mark.asyncio
    async def test_stop_cancels_drain_task(self) -> None:
        alerter = _make_alerter()
        queue = TelegramMessageQueue(alerter)
        await queue.start()
        task = queue._drain_task
        await queue.stop()
        assert task is not None
        assert task.done() or task.cancelled()
