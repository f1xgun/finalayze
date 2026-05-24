"""Tests for TelegramAlerter lifecycle (close / shutdown).

After the AlertQueue refactor:
- close() stops the queue if attached (queue owns drain loop)
- close() is idempotent via _closed flag
- TelegramAlerter no longer owns an httpx client (_client moved to TelegramTransport)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from finalayze.core.alerts import AlertQueue, TelegramAlerter, TelegramMessageQueue


@pytest.fixture
def alerter() -> TelegramAlerter:
    return TelegramAlerter(bot_token="", chat_id="")


class TestTelegramAlerterClose:
    @pytest.mark.asyncio
    async def test_close_stops_queue_if_set(self, alerter: TelegramAlerter) -> None:
        mock_queue = AsyncMock(spec=AlertQueue)
        mock_queue.stop = AsyncMock()
        alerter._queue = mock_queue

        await alerter.close()

        mock_queue.stop.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_close_is_idempotent(self, alerter: TelegramAlerter) -> None:
        mock_queue = AsyncMock(spec=AlertQueue)
        mock_queue.stop = AsyncMock()
        alerter._queue = mock_queue

        await alerter.close()
        await alerter.close()

        # stop only called once — idempotent via _closed
        mock_queue.stop.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_close_without_queue(self, alerter: TelegramAlerter) -> None:
        assert alerter._queue is None
        # Must not raise
        await alerter.close()

    @pytest.mark.asyncio
    async def test_closed_flag_set_after_close(self, alerter: TelegramAlerter) -> None:
        await alerter.close()
        assert alerter._closed is True

    def test_telegram_message_queue_is_alias(self) -> None:
        """TelegramMessageQueue must remain a valid alias for AlertQueue."""
        assert TelegramMessageQueue is AlertQueue
