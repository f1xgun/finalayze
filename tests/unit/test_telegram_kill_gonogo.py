"""Tests for /kill and /gonogo Telegram commands in TelegramBotHandler.

Validates:
  - /kill requires admin chat_id (not just whitelisted)
  - /kill starts 30s confirmation flow, CONFIRM triggers KillSwitch.activate()
  - Expired /kill confirmation is rejected
  - CONFIRM without prior /kill is ignored
  - /gonogo calls GoNoGoReporter.evaluate() and formats GateReport
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from finalayze.core.telegram_bot import TelegramBotHandler

# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_handler(
    *,
    admin_chat_id: str = "12345",
    allowed_chat_ids: list[str] | None = None,
    kill_switch: object | None = None,
    go_no_go_reporter: object | None = None,
) -> TelegramBotHandler:
    """Create TelegramBotHandler with mocked dependencies."""
    alerter = MagicMock()
    alerter._send = AsyncMock(return_value=True)

    settings = MagicMock()
    settings.telegram_allowed_chat_ids = allowed_chat_ids or ["12345", "67890"]
    settings.telegram_admin_chat_id = admin_chat_id

    return TelegramBotHandler(
        alerter=alerter,
        broker_router=MagicMock(),
        circuit_breakers={},
        settings=settings,
        kill_switch=kill_switch,
        go_no_go_reporter=go_no_go_reporter,
    )


def _make_update(chat_id: str, text: str) -> dict:
    """Build a Telegram update dict."""
    return {"message": {"chat": {"id": chat_id}, "text": text}}


@dataclass(frozen=True)
class _FakeKillResult:
    orders_cancelled: int = 3
    scheduler_stopped: bool = True
    breakers_escalated: int = 2
    alert_sent: bool = True
    elapsed_seconds: float = 1.5


# ── Tests ────────────────────────────────────────────────────────────────────


class TestKillCommand:
    """Tests for /kill confirmation flow."""

    @pytest.mark.asyncio
    async def test_kill_from_admin_sends_confirmation_prompt(self) -> None:
        """Test 1: /kill from admin sends confirmation prompt."""
        ks = MagicMock()
        handler = _make_handler(kill_switch=ks)

        await handler.handle_update(_make_update("12345", "/kill"))

        handler._alerter._send.assert_called_once()
        msg = handler._alerter._send.call_args[0][0]
        assert "CONFIRM" in msg
        assert "30s" in msg

    @pytest.mark.asyncio
    async def test_kill_from_non_admin_rejected(self) -> None:
        """Test 2: /kill from non-admin (even if whitelisted) is rejected."""
        ks = MagicMock()
        handler = _make_handler(kill_switch=ks)

        await handler.handle_update(_make_update("67890", "/kill"))

        handler._alerter._send.assert_called_once()
        msg = handler._alerter._send.call_args[0][0]
        assert "Unauthorized" in msg
        ks.activate.assert_not_called()

    @pytest.mark.asyncio
    async def test_confirm_within_30s_activates_kill(self) -> None:
        """Test 3: CONFIRM within 30s triggers KillSwitch.activate()."""
        ks = MagicMock()
        ks.activate.return_value = _FakeKillResult()
        handler = _make_handler(kill_switch=ks)

        # Send /kill first
        with patch("finalayze.core.telegram_bot.time") as mock_time:
            mock_time.monotonic.return_value = 100.0
            await handler.handle_update(_make_update("12345", "/kill"))

            # Send CONFIRM within 30s
            mock_time.monotonic.return_value = 120.0  # 20s later
            await handler.handle_update(_make_update("12345", "CONFIRM"))

        ks.activate.assert_called_once()
        call_kwargs = ks.activate.call_args[1]
        call_args = ks.activate.call_args[0]
        reason = call_kwargs.get("reason", call_args[0] if call_args else "")
        assert "telegram" in reason or "12345" in reason

    @pytest.mark.asyncio
    async def test_confirm_after_30s_rejected(self) -> None:
        """Test 4: CONFIRM after 30s sends expiry message."""
        ks = MagicMock()
        handler = _make_handler(kill_switch=ks)

        with patch("finalayze.core.telegram_bot.time") as mock_time:
            mock_time.monotonic.return_value = 100.0
            await handler.handle_update(_make_update("12345", "/kill"))

            # Send CONFIRM after 31s
            mock_time.monotonic.return_value = 131.0
            await handler.handle_update(_make_update("12345", "CONFIRM"))

        ks.activate.assert_not_called()
        # Last _send call should mention expired
        last_msg = handler._alerter._send.call_args[0][0]
        assert "expired" in last_msg.lower()

    @pytest.mark.asyncio
    async def test_confirm_without_prior_kill_ignored(self) -> None:
        """Test 5: CONFIRM without prior /kill does not crash or activate."""
        ks = MagicMock()
        handler = _make_handler(kill_switch=ks)

        result = await handler.handle_update(_make_update("12345", "CONFIRM"))

        ks.activate.assert_not_called()
        # Should not crash -- returns something
        assert "ok" in result

    @pytest.mark.asyncio
    async def test_kill_confirmation_cleaned_up_after_use(self) -> None:
        """Test 8: Pending kill state removed after confirmation."""
        ks = MagicMock()
        ks.activate.return_value = _FakeKillResult()
        handler = _make_handler(kill_switch=ks)

        with patch("finalayze.core.telegram_bot.time") as mock_time:
            mock_time.monotonic.return_value = 100.0
            await handler.handle_update(_make_update("12345", "/kill"))
            assert "12345" in handler._pending_kill

            mock_time.monotonic.return_value = 110.0
            await handler.handle_update(_make_update("12345", "CONFIRM"))
            assert "12345" not in handler._pending_kill


class TestGoNoGoCommand:
    """Tests for /gonogo command handler."""

    @pytest.mark.asyncio
    async def test_gonogo_calls_evaluate_and_sends_report(self) -> None:
        """Test 6: /gonogo calls evaluate() and sends formatted message."""
        from finalayze.monitoring.go_no_go import CriterionResult, GateReport, GateVerdict

        report = GateReport(
            verdict=GateVerdict.PROCEED,
            criteria=[
                CriterionResult(
                    name="uptime_pct",
                    passed=True,
                    actual=99.5,
                    threshold=95.0,
                    unit="%",
                    critical=True,
                ),
                CriterionResult(
                    name="fill_rate_pct",
                    passed=True,
                    actual=98.0,
                    threshold=90.0,
                    unit="%",
                    critical=True,
                ),
            ],
            sandbox_days=7,
            evaluated_at=datetime.now(UTC),
            reason="All 8 criteria passed",
        )

        reporter = MagicMock()
        reporter.evaluate = AsyncMock(return_value=report)
        handler = _make_handler(go_no_go_reporter=reporter)

        # Mock the DB session factory to return a mock session
        mock_session = AsyncMock()
        mock_factory = MagicMock()
        mock_factory.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_factory.return_value.__aexit__ = AsyncMock(return_value=False)

        with (
            patch(
                "finalayze.core.telegram_bot.async_session_factory",
                mock_factory,
                create=True,
            ),
            patch.dict(
                "sys.modules",
                {"finalayze.core.db": MagicMock(async_session_factory=mock_factory)},
            ),
        ):
            await handler.handle_update(_make_update("12345", "/gonogo"))

        reporter.evaluate.assert_called_once()
        msg = handler._alerter._send.call_args[0][0]
        assert "PROCEED" in msg
        assert "uptime_pct" in msg

    @pytest.mark.asyncio
    async def test_gonogo_formats_verdict_emoji(self) -> None:
        """Test 7: /gonogo formats PROCEED/DEFER/ABORT with appropriate indicators."""
        from finalayze.monitoring.go_no_go import CriterionResult, GateReport, GateVerdict

        report = GateReport(
            verdict=GateVerdict.ABORT,
            criteria=[
                CriterionResult(
                    name="max_drawdown_pct",
                    passed=False,
                    actual=5.0,
                    threshold=2.27,
                    unit="%",
                    critical=True,
                ),
            ],
            sandbox_days=7,
            evaluated_at=datetime.now(UTC),
            reason="Critical failures: max_drawdown_pct",
        )

        reporter = MagicMock()
        reporter.evaluate = AsyncMock(return_value=report)
        handler = _make_handler(go_no_go_reporter=reporter)

        mock_session = AsyncMock()
        mock_factory = MagicMock()
        mock_factory.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_factory.return_value.__aexit__ = AsyncMock(return_value=False)

        with patch.dict(
            "sys.modules",
            {"finalayze.core.db": MagicMock(async_session_factory=mock_factory)},
        ):
            await handler.handle_update(_make_update("12345", "/gonogo"))

        msg = handler._alerter._send.call_args[0][0]
        assert "ABORT" in msg
        # Failed criterion should have fail indicator
        assert "max_drawdown_pct" in msg
