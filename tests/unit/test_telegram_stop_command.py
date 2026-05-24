"""Tests for /stop Telegram command in TelegramBotHandler.

Validates that /stop triggers trading loop shutdown and sends appropriate
alert messages via Telegram.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from finalayze.core.telegram_bot import TelegramBotHandler


def _make_handler(
    *,
    trading_loop: object | None = MagicMock(),
    allowed_chat_ids: list[str] | None = None,
) -> TelegramBotHandler:
    """Create TelegramBotHandler with mocked dependencies."""
    alerter = MagicMock()
    alerter.send_async = AsyncMock(return_value=(True, None))

    settings = MagicMock()
    settings.telegram_allowed_chat_ids = allowed_chat_ids or ["123"]

    return TelegramBotHandler(
        alerter=alerter,
        broker_router=MagicMock(),
        circuit_breakers={},
        settings=settings,
        trading_loop=trading_loop,
    )


class TestStopCommand:
    """Tests for /stop command handler."""

    @pytest.mark.asyncio
    async def test_stop_triggers_trading_loop_stop(self) -> None:
        """Sending /stop calls trading_loop.stop()."""
        loop = MagicMock()
        handler = _make_handler(trading_loop=loop)

        await handler.handle_stop("123")

        loop.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_sends_halted_message(self) -> None:
        """Sending /stop sends TRADING HALTED alert."""
        loop = MagicMock()
        handler = _make_handler(trading_loop=loop)

        await handler.handle_stop("123")

        handler._alerter.send_async.assert_called_once()
        msg = handler._alerter.send_async.call_args[0][0]
        assert "TRADING HALTED" in msg

    @pytest.mark.asyncio
    async def test_stop_without_loop_sends_api_only(self) -> None:
        """Sending /stop with no trading loop sends API-only message."""
        handler = _make_handler(trading_loop=None)

        await handler.handle_stop("123")

        msg = handler._alerter.send_async.call_args[0][0]
        assert "API-only mode" in msg

    @pytest.mark.asyncio
    async def test_stop_requires_whitelisted_chat_id(self) -> None:
        """Unauthorized chat_id is rejected for /stop command."""
        handler = _make_handler(allowed_chat_ids=["999"])

        update = {
            "message": {
                "chat": {"id": "123"},
                "text": "/stop",
            }
        }
        result = await handler.handle_update(update)
        assert result == {"ok": "ignored"}

    @pytest.mark.asyncio
    async def test_stop_command_registered(self) -> None:
        """/stop is registered in the commands dict."""
        handler = _make_handler()
        assert "/stop" in handler._commands
