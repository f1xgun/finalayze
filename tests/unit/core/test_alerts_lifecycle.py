"""Tests for TelegramAlerter lifecycle (close / shutdown).

Verifies:
- close() calls httpx.AsyncClient.aclose()
- close() stops the message queue if attached
- close() is idempotent (safe to call twice)
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from finalayze.core.alerts import TelegramAlerter, TelegramMessageQueue


@pytest.fixture
def alerter() -> TelegramAlerter:
    """Create a TelegramAlerter with empty token (no-op sends)."""
    return TelegramAlerter(bot_token="", chat_id="")


class TestTelegramAlerterClose:
    """Tests for TelegramAlerter.close() lifecycle."""

    @pytest.mark.asyncio
    async def test_close_calls_aclose_on_httpx_client(self, alerter: TelegramAlerter) -> None:
        """close() must call self._client.aclose() to release httpx resources."""
        alerter._client = AsyncMock()
        alerter._client.aclose = AsyncMock()

        await alerter.close()

        alerter._client.aclose.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_close_stops_queue_if_set(self, alerter: TelegramAlerter) -> None:
        """close() must stop the message queue before closing the client."""
        mock_queue = AsyncMock(spec=TelegramMessageQueue)
        mock_queue.stop = AsyncMock()
        alerter._queue = mock_queue
        alerter._client = AsyncMock()
        alerter._client.aclose = AsyncMock()

        await alerter.close()

        mock_queue.stop.assert_awaited_once()
        alerter._client.aclose.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_close_is_idempotent(self, alerter: TelegramAlerter) -> None:
        """Calling close() twice must not raise and must only close resources once."""
        alerter._client = AsyncMock()
        alerter._client.aclose = AsyncMock()

        await alerter.close()
        await alerter.close()

        # aclose should only be called once (idempotent via _closed flag)
        alerter._client.aclose.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_close_without_queue(self, alerter: TelegramAlerter) -> None:
        """close() works when no queue is attached (queue is None)."""
        assert alerter._queue is None
        alerter._client = AsyncMock()
        alerter._client.aclose = AsyncMock()

        await alerter.close()

        alerter._client.aclose.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_closed_flag_set_after_close(self, alerter: TelegramAlerter) -> None:
        """After close(), the _closed flag must be True."""
        alerter._client = AsyncMock()
        alerter._client.aclose = AsyncMock()

        await alerter.close()

        assert alerter._closed is True
