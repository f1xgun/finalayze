"""Failing tests for the AlertQueue + TelegramTransport seam refactor.

Design goal: one thread-safe post() path replaces the three-way
async/sync/fallback send_alert() logic. TelegramTransport owns HTTP
and DB persistence; AlertQueue owns rate-limiting, batching, and retry;
TelegramAlerter is a pure message formatter.
"""

from __future__ import annotations

import asyncio
import threading
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from finalayze.api.alerts import AlertPriority, AlertQueue, TelegramAlerter
from finalayze.api.telegram_transport import TelegramTransport

# ── Constants ────────────────────────────────────────────────────────────────

VALID_TOKEN = "1234567890:AABBccDDeEFfGgHhIiJj"  # noqa: S105
VALID_CHAT_ID = "-1001234567890"


# ── Fixtures ─────────────────────────────────────────────────────────────────


def _make_transport(*, token: str = VALID_TOKEN) -> TelegramTransport:
    return TelegramTransport(bot_token=token, chat_id=VALID_CHAT_ID)


# ── TelegramTransport ─────────────────────────────────────────────────────────


class TestTelegramTransport:
    """Transport owns HTTP and persistence — nothing else."""

    @pytest.mark.asyncio
    async def test_send_posts_to_telegram(self) -> None:
        transport = _make_transport()
        mock_resp = MagicMock(status_code=200)
        with patch.object(transport._client, "post", new=AsyncMock(return_value=mock_resp)):
            ok, _ = await transport.send("hello")
        assert ok is True

    @pytest.mark.asyncio
    async def test_send_returns_false_on_rate_limit(self) -> None:
        transport = _make_transport()
        mock_resp = MagicMock(status_code=429)
        mock_resp.json.return_value = {"parameters": {"retry_after": 1}}
        with patch.object(transport._client, "post", new=AsyncMock(return_value=mock_resp)):
            ok, _ = await transport.send("hello")
        assert ok is False

    @pytest.mark.asyncio
    async def test_send_noop_when_token_empty(self) -> None:
        transport = _make_transport(token="")
        with patch.object(transport._client, "post", new=AsyncMock()) as mock_post:
            ok, _ = await transport.send("hello")
        mock_post.assert_not_called()
        assert ok is True

    @pytest.mark.asyncio
    async def test_close_shuts_down_client(self) -> None:
        transport = _make_transport()
        with patch.object(transport._client, "aclose", new=AsyncMock()) as mock_close:
            await transport.close()
        mock_close.assert_called_once()

    def test_no_circular_dependency_to_alerter(self) -> None:
        """Transport must not import TelegramAlerter."""
        import finalayze.api.telegram_transport as mod

        assert not hasattr(mod, "TelegramAlerter"), (
            "TelegramTransport module must not reference TelegramAlerter"
        )


# ── AlertQueue ────────────────────────────────────────────────────────────────


class TestAlertQueue:
    """AlertQueue bridges sync callers to async delivery via call_soon_threadsafe."""

    @pytest.mark.asyncio
    async def test_post_from_sync_thread_delivers_message(self) -> None:
        loop = asyncio.get_running_loop()
        transport = _make_transport()
        transport.send = AsyncMock(return_value=(True, None))
        queue = AlertQueue(loop=loop, transport=transport)
        await queue.start()

        posted = threading.Event()

        def _thread_post() -> None:
            queue.post("hello from thread", AlertPriority.INFO)
            posted.set()

        t = threading.Thread(target=_thread_post)
        t.start()
        t.join(timeout=1)
        assert posted.is_set()

        # Give drain loop a moment to consume the message
        await asyncio.sleep(0.05)
        transport.send.assert_called_once()
        await queue.stop()

    @pytest.mark.asyncio
    async def test_post_from_async_context_delivers_message(self) -> None:
        loop = asyncio.get_running_loop()
        transport = _make_transport()
        transport.send = AsyncMock(return_value=(True, None))
        queue = AlertQueue(loop=loop, transport=transport)
        await queue.start()

        queue.post("hello from async", AlertPriority.INFO)
        await asyncio.sleep(0.05)

        transport.send.assert_called_once()
        await queue.stop()

    @pytest.mark.asyncio
    async def test_critical_message_delivered(self) -> None:
        loop = asyncio.get_running_loop()
        transport = _make_transport()
        transport.send = AsyncMock(return_value=(True, None))
        queue = AlertQueue(loop=loop, transport=transport)
        await queue.start()

        queue.post("CRITICAL alert", AlertPriority.CRITICAL)
        await asyncio.sleep(0.05)

        transport.send.assert_called_once()
        await queue.stop()

    def test_post_does_not_require_running_loop_in_caller(self) -> None:
        """post() must not call asyncio.get_running_loop() — it uses the stored loop."""
        loop = asyncio.new_event_loop()
        try:
            transport = _make_transport()
            queue = AlertQueue(loop=loop, transport=transport)
            # We're NOT in an async context here — this must not raise
            queue.post("test", AlertPriority.INFO)
        finally:
            loop.close()

    @pytest.mark.asyncio
    async def test_no_send_sync_on_queue(self) -> None:
        """AlertQueue must not expose _send_sync — that path is deleted."""
        loop = asyncio.get_running_loop()
        transport = _make_transport()
        queue = AlertQueue(loop=loop, transport=transport)
        assert not hasattr(queue, "_send_sync"), (
            "AlertQueue must not have _send_sync — transport owns HTTP"
        )


# ── TelegramAlerter ───────────────────────────────────────────────────────────


class TestTelegramAlerterSimplified:
    """Alerter is now a pure formatter — no HTTP, no loop management."""

    def test_send_alert_calls_queue_post(self) -> None:
        alerter = TelegramAlerter(bot_token=VALID_TOKEN, chat_id=VALID_CHAT_ID)
        mock_queue = MagicMock()
        alerter.set_queue(mock_queue)

        alerter.send_alert("test message", priority=AlertPriority.INFO)

        mock_queue.post.assert_called_once_with("test message", AlertPriority.INFO)

    def test_send_alert_noop_when_no_queue(self) -> None:
        alerter = TelegramAlerter(bot_token=VALID_TOKEN, chat_id=VALID_CHAT_ID)
        # No queue attached — must be a silent no-op, not raise
        alerter.send_alert("test message", priority=AlertPriority.INFO)

    def test_send_alert_noop_when_token_empty(self) -> None:
        alerter = TelegramAlerter(bot_token="", chat_id=VALID_CHAT_ID)
        mock_queue = MagicMock()
        alerter.set_queue(mock_queue)

        alerter.send_alert("test message", priority=AlertPriority.INFO)

        mock_queue.post.assert_not_called()

    def test_no_set_event_loop_method(self) -> None:
        """set_event_loop() must be gone — loop is owned by AlertQueue."""
        alerter = TelegramAlerter(bot_token=VALID_TOKEN, chat_id=VALID_CHAT_ID)
        assert not hasattr(alerter, "set_event_loop"), (
            "TelegramAlerter must not have set_event_loop — AlertQueue owns the loop"
        )

    def test_no_main_loop_attribute(self) -> None:
        """_main_loop must be gone — loop belongs to AlertQueue."""
        alerter = TelegramAlerter(bot_token=VALID_TOKEN, chat_id=VALID_CHAT_ID)
        assert not hasattr(alerter, "_main_loop"), "TelegramAlerter must not store _main_loop"

    def test_no_send_sync_method(self) -> None:
        """_send_sync() must be deleted — sync HTTP is replaced by queue bridge."""
        alerter = TelegramAlerter(bot_token=VALID_TOKEN, chat_id=VALID_CHAT_ID)
        assert not hasattr(alerter, "_send_sync"), "TelegramAlerter must not have _send_sync"

    def test_on_trade_filled_routes_through_queue(self) -> None:
        from decimal import Decimal

        from finalayze.execution.broker_base import OrderResult

        alerter = TelegramAlerter(bot_token=VALID_TOKEN, chat_id=VALID_CHAT_ID)
        mock_queue = MagicMock()
        alerter.set_queue(mock_queue)

        result = OrderResult(
            filled=True,
            fill_price=Decimal("280.50"),
            symbol="SBER",
            side="BUY",
            quantity=Decimal(10),
        )
        alerter.on_trade_filled(result, "moex", "tinkoff")

        mock_queue.post.assert_called_once()
        call_priority = mock_queue.post.call_args[0][1]
        assert call_priority == AlertPriority.IMPORTANT
