"""Tests for TelegramBotHandler.handle_approve (Phase 58-04 Task 08).

SPEC AC #12 — extends the Phase 57 webhook (api/telegram_bot.py) with
``/approve <decision_id_short8>`` command. The command:
  - Parses the short8 (8 hex chars) via the locked regex pattern.
  - Dispatches MetaAgentApprover.handle_approve via asyncio.create_task,
    storing the task on self._pending_approve_tasks (RUF006).
  - Returns 200 OK regardless of approver outcome (D-15 fire-and-forget).
  - When meta_agent_approver is not configured, logs gracefully.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

# Module-level constants (PLR2004).
_ALLOWED_CHAT_ID = "12345"
_FAKE_SHORT8 = "ab12cd34"


def _make_handler(
    *,
    meta_agent_approver: Any | None = None,
    chat_id: str = _ALLOWED_CHAT_ID,
) -> Any:
    """Construct a TelegramBotHandler with minimal mocked dependencies."""
    from finalayze.api.telegram_bot import TelegramBotHandler

    settings = MagicMock()
    settings.telegram_allowed_chat_ids = [chat_id]
    settings.telegram_admin_chat_id = chat_id

    return TelegramBotHandler(
        alerter=MagicMock(),
        broker_router=MagicMock(),
        circuit_breakers={},
        settings=settings,
        meta_agent_approver=meta_agent_approver,
    )


@pytest.mark.asyncio
async def test_approve_command_dispatches_to_meta_agent_approver() -> None:
    """SPEC AC #12: /approve <short8> with a valid 8-hex-char id MUST
    dispatch MetaAgentApprover.handle_approve via asyncio.create_task,
    store the task on _pending_approve_tasks, and return processed.
    """
    mock_approver = MagicMock()
    mock_approver.handle_approve = AsyncMock()

    handler = _make_handler(meta_agent_approver=mock_approver)

    update = {
        "message": {
            "chat": {"id": int(_ALLOWED_CHAT_ID)},
            "text": f"/approve {_FAKE_SHORT8}",
        },
    }
    result = await handler.handle_update(update)

    assert result == {"ok": "processed"}, f"unexpected result: {result!r}"

    # Pending task tracked on handler set (RUF006).
    assert hasattr(handler, "_pending_approve_tasks"), (
        "handler must expose _pending_approve_tasks set (RUF006)"
    )

    # Wait for the dispatched task to complete.
    pending = list(handler._pending_approve_tasks)
    for task in pending:
        await task

    # Approver dispatched with (short8, chat_id=...).
    mock_approver.handle_approve.assert_awaited_once_with(
        _FAKE_SHORT8,
        chat_id=_ALLOWED_CHAT_ID,
    )


@pytest.mark.asyncio
async def test_approve_command_invalid_syntax_returns_no_approver_call() -> None:
    """SPEC AC #12: /approve without a valid short8 (no arg, wrong length,
    non-hex) MUST NOT call the approver. Webhook still returns 200.
    Logs meta_agent_approve_invalid_syntax.
    """
    import structlog

    mock_approver = MagicMock()
    mock_approver.handle_approve = AsyncMock()

    handler = _make_handler(meta_agent_approver=mock_approver)

    invalid_inputs = [
        "/approve",  # no arg
        "/approve abc",  # too short
        "/approve XYZ12345",  # non-hex
        "/approve abc12345 extra-arg",  # extra args (regex anchored)
    ]

    for raw_text in invalid_inputs:
        update = {
            "message": {
                "chat": {"id": int(_ALLOWED_CHAT_ID)},
                "text": raw_text,
            },
        }
        with structlog.testing.capture_logs() as logs:
            result = await handler.handle_update(update)

        # Webhook still returns processed (200 OK).
        assert result == {"ok": "processed"}, (
            f"unexpected result for {raw_text!r}: {result!r}"
        )

        # Approver NOT called for any invalid input.
        # (Reset between iterations to detect the per-input behaviour.)
        invalid_events = [
            log for log in logs if log.get("event") == "meta_agent_approve_invalid_syntax"
        ]
        assert len(invalid_events) >= 1, (
            f"expected meta_agent_approve_invalid_syntax event for {raw_text!r}, got {logs!r}"
        )

    mock_approver.handle_approve.assert_not_called()


@pytest.mark.asyncio
async def test_approve_command_chat_id_not_allowed_ignored() -> None:
    """The existing chat_id whitelist guard (telegram_bot.py:94-97) blocks
    /approve from non-allowed chats — auth surface unchanged (D-12).
    """
    mock_approver = MagicMock()
    mock_approver.handle_approve = AsyncMock()

    handler = _make_handler(meta_agent_approver=mock_approver)

    update = {
        "message": {
            "chat": {"id": 99999},  # NOT in the whitelist
            "text": f"/approve {_FAKE_SHORT8}",
        },
    }
    result = await handler.handle_update(update)

    assert result == {"ok": "ignored"}
    mock_approver.handle_approve.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
# Task 58-04-10: handler with no approver logs warning, doesn't raise
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_handler_with_no_approver_logs_warning() -> None:
    """SPEC AC #12 (graceful degradation): if TelegramBotHandler is
    constructed without meta_agent_approver, /approve must NOT raise —
    log meta_agent_approve_not_configured and return processed.
    """
    import structlog

    handler = _make_handler(meta_agent_approver=None)

    update = {
        "message": {
            "chat": {"id": int(_ALLOWED_CHAT_ID)},
            "text": f"/approve {_FAKE_SHORT8}",
        },
    }
    with structlog.testing.capture_logs() as logs:
        result = await handler.handle_update(update)

    assert result == {"ok": "processed"}

    not_configured_events = [
        log for log in logs if log.get("event") == "meta_agent_approve_not_configured"
    ]
    assert len(not_configured_events) == 1, (
        f"expected 1 meta_agent_approve_not_configured event, got {not_configured_events!r}"
    )
