"""Unit tests for TelegramAlerter message formatting and routing.

After the AlertQueue refactor, TelegramAlerter is a pure formatter:
- All on_*() methods route through send_alert() → queue.post()
- No direct HTTP calls — transport is tested separately in test_alerter_seam_refactor.py
- No-op when token is empty or no queue attached
"""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from finalayze.core.alerts import AlertPriority, TelegramAlerter, TelegramMessageQueue
from finalayze.execution.broker_base import OrderRequest, OrderResult
from finalayze.risk.circuit_breaker import CircuitLevel

# ── Constants ────────────────────────────────────────────────────────────────
VALID_TOKEN = "1234567890:AABBccDDeEFfGgHhIiJj"  # noqa: S105
VALID_CHAT_ID = "-1001234567890"
MARKET_US = "us"
MARKET_MOEX = "moex"
FILL_PRICE = Decimal("150.00")
ORDER_QTY = Decimal(10)
DRAWDOWN_PCT = 0.103
DAILY_PNL_US = Decimal(342)
DAILY_PNL_MOEX = Decimal(1200)
TOTAL_EQUITY = Decimal(51200)


def _make_order_result() -> OrderResult:
    return OrderResult(
        filled=True,
        fill_price=FILL_PRICE,
        symbol="AAPL",
        side="BUY",
        quantity=ORDER_QTY,
    )


def _make_order_request() -> OrderRequest:
    return OrderRequest(symbol="AAPL", side="BUY", quantity=ORDER_QTY)


def _make_alerter_with_queue() -> tuple[TelegramAlerter, MagicMock]:
    """Return (alerter, mock_queue) with queue attached."""
    alerter = TelegramAlerter(bot_token=VALID_TOKEN, chat_id=VALID_CHAT_ID)
    mock_queue = MagicMock()
    alerter.set_queue(mock_queue)
    return alerter, mock_queue


# ── No-op when disabled ──────────────────────────────────────────────────────


class TestTelegramAlerterNoOp:
    """When token is empty, all methods must silently do nothing."""

    def test_no_op_on_trade_filled(self) -> None:
        alerter = TelegramAlerter(bot_token="", chat_id=VALID_CHAT_ID)
        mock_queue = MagicMock()
        alerter.set_queue(mock_queue)
        alerter.on_trade_filled(_make_order_result(), MARKET_US, "alpaca")
        mock_queue.post.assert_not_called()

    def test_no_op_on_trade_rejected(self) -> None:
        alerter = TelegramAlerter(bot_token="", chat_id=VALID_CHAT_ID)
        mock_queue = MagicMock()
        alerter.set_queue(mock_queue)
        alerter.on_trade_rejected(_make_order_request(), "insufficient funds")
        mock_queue.post.assert_not_called()

    def test_no_op_on_circuit_breaker_trip(self) -> None:
        alerter = TelegramAlerter(bot_token="", chat_id=VALID_CHAT_ID)
        mock_queue = MagicMock()
        alerter.set_queue(mock_queue)
        alerter.on_circuit_breaker_trip(MARKET_US, CircuitLevel.HALTED, DRAWDOWN_PCT)
        mock_queue.post.assert_not_called()

    def test_no_op_on_circuit_breaker_reset(self) -> None:
        alerter = TelegramAlerter(bot_token="", chat_id=VALID_CHAT_ID)
        mock_queue = MagicMock()
        alerter.set_queue(mock_queue)
        alerter.on_circuit_breaker_reset(MARKET_US)
        mock_queue.post.assert_not_called()

    def test_no_op_on_daily_summary(self) -> None:
        alerter = TelegramAlerter(bot_token="", chat_id=VALID_CHAT_ID)
        mock_queue = MagicMock()
        alerter.set_queue(mock_queue)
        alerter.on_daily_summary(
            {MARKET_US: DAILY_PNL_US, MARKET_MOEX: DAILY_PNL_MOEX},
            TOTAL_EQUITY,
        )
        mock_queue.post.assert_not_called()

    def test_no_op_on_error(self) -> None:
        alerter = TelegramAlerter(bot_token="", chat_id=VALID_CHAT_ID)
        mock_queue = MagicMock()
        alerter.set_queue(mock_queue)
        alerter.on_error("NewsApiFetcher", "connection timeout")
        mock_queue.post.assert_not_called()

    def test_no_op_without_queue(self) -> None:
        alerter = TelegramAlerter(bot_token=VALID_TOKEN, chat_id=VALID_CHAT_ID)
        # No queue — must be silent
        alerter.send_alert("test")


# ── Message formatting ────────────────────────────────────────────────────────


class TestTelegramAlerterSendsMessages:
    """Each on_*() method must format a message and route through queue.post()."""

    def test_on_trade_filled_text(self) -> None:
        alerter, mock_queue = _make_alerter_with_queue()
        alerter.on_trade_filled(_make_order_result(), MARKET_US, "alpaca")
        mock_queue.post.assert_called_once()
        text = mock_queue.post.call_args[0][0]
        assert "AAPL" in text
        assert "150.00" in text

    def test_on_trade_filled_priority(self) -> None:
        alerter, mock_queue = _make_alerter_with_queue()
        alerter.on_trade_filled(_make_order_result(), MARKET_US, "alpaca")
        priority = mock_queue.post.call_args[0][1]
        assert priority == AlertPriority.IMPORTANT

    def test_on_trade_filled_moex_uses_ruble_symbol(self) -> None:
        alerter, mock_queue = _make_alerter_with_queue()
        alerter.on_trade_filled(_make_order_result(), MARKET_MOEX, "tinkoff")
        text = mock_queue.post.call_args[0][0]
        assert "₽" in text

    def test_on_trade_rejected_text(self) -> None:
        alerter, mock_queue = _make_alerter_with_queue()
        alerter.on_trade_rejected(_make_order_request(), "insufficient funds")
        text = mock_queue.post.call_args[0][0]
        assert "AAPL" in text
        assert "insufficient funds" in text

    def test_on_circuit_breaker_trip_text_and_priority(self) -> None:
        alerter, mock_queue = _make_alerter_with_queue()
        alerter.on_circuit_breaker_trip(MARKET_US, CircuitLevel.HALTED, DRAWDOWN_PCT)
        text, priority = mock_queue.post.call_args[0]
        assert MARKET_US.upper() in text
        assert priority == AlertPriority.CRITICAL

    def test_on_circuit_breaker_reset_text(self) -> None:
        alerter, mock_queue = _make_alerter_with_queue()
        alerter.on_circuit_breaker_reset(MARKET_US)
        text = mock_queue.post.call_args[0][0]
        assert "reset" in text.lower() or "resumed" in text.lower()

    def test_on_daily_summary_text(self) -> None:
        alerter, mock_queue = _make_alerter_with_queue()
        alerter.on_daily_summary(
            {MARKET_US: DAILY_PNL_US, MARKET_MOEX: DAILY_PNL_MOEX},
            TOTAL_EQUITY,
        )
        text = mock_queue.post.call_args[0][0]
        assert "Daily" in text

    def test_on_error_text_and_priority(self) -> None:
        alerter, mock_queue = _make_alerter_with_queue()
        alerter.on_error("NewsApiFetcher", "gRPC timeout")
        text, priority = mock_queue.post.call_args[0]
        assert "NewsApiFetcher" in text
        assert "gRPC timeout" in text
        assert priority == AlertPriority.CRITICAL

    def test_send_alert_never_raises(self) -> None:
        """send_alert must swallow all exceptions — never crash the caller."""
        alerter = TelegramAlerter(bot_token=VALID_TOKEN, chat_id=VALID_CHAT_ID)
        bad_queue = MagicMock()
        bad_queue.post.side_effect = RuntimeError("queue exploded")
        alerter.set_queue(bad_queue)
        # Must not raise
        alerter.send_alert("test message")
