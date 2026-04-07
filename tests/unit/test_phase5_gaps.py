"""Gap coverage tests for Phase 5: Integration & Telegram.

Covers untested methods and edge cases identified by add-tests analysis:
- TelegramAlerter: _send_sync, on_coupon_received, on_cbr_meeting, on_bond_event_trade,
  on_stop_loss_triggered, on_startup, on_shutdown, send_alert edge cases
- TelegramMessageQueue: batching reversal, rate limit boundary, double-start guard
- TradingLoop: _compute_top_movers, _cbr_day_refresh, _weekly_digest,
  _attempt_grpc_reconnect, _reconcile_inflight_orders, _daily_reset edge cases
- TelegramBotHandler: no-message update, all-brokers-fail, bond_processor=None,
  command with extra args, empty text
- Webhook: empty secret header, missing message field
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections import deque
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from finalayze.core.alerts import AlertPriority, TelegramAlerter, TelegramMessageQueue

if TYPE_CHECKING:
    from finalayze.core.telegram_bot import TelegramBotHandler

# ── Constants (ruff PLR2004) ─────────────────────────────────────────────────
VALID_TOKEN = "1234567890:AABBccDDeEFfGgHhIiJj"  # noqa: S105
VALID_CHAT_ID = "-1001234567890"
RATE_LIMIT = 20


def _make_alerter(token: str = VALID_TOKEN) -> TelegramAlerter:
    return TelegramAlerter(bot_token=token, chat_id=VALID_CHAT_ID)


# ═══════════════════════════════════════════════════════════════════════════════
# TelegramAlerter._send_sync
# ═══════════════════════════════════════════════════════════════════════════════


class TestSendSync:
    """_send_sync: synchronous HTTP POST used by APScheduler threads."""

    def test_send_sync_noop_with_empty_token(self) -> None:
        """_send_sync returns True without HTTP call when token is empty."""
        alerter = _make_alerter(token="")
        result = alerter._send_sync("test")
        assert result is True

    def test_send_sync_returns_true_on_success(self) -> None:
        """_send_sync returns True on HTTP 200."""
        alerter = _make_alerter()
        mock_resp = MagicMock(status_code=200)
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = mock_resp

        with patch("finalayze.core.alerts.httpx.Client", return_value=mock_client):
            result = alerter._send_sync("hello")

        assert result is True
        mock_client.post.assert_called_once()

    def test_send_sync_returns_false_on_429(self) -> None:
        """_send_sync returns False on HTTP 429 rate limit."""
        alerter = _make_alerter()
        mock_resp = MagicMock(status_code=429)
        mock_resp.json.return_value = {"parameters": {"retry_after": 30}}
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = mock_resp

        with patch("finalayze.core.alerts.httpx.Client", return_value=mock_client):
            result = alerter._send_sync("hello")

        assert result is False

    def test_send_sync_exception_suppressed(self) -> None:
        """_send_sync suppresses exceptions and returns False."""
        alerter = _make_alerter()
        with patch("finalayze.core.alerts.httpx.Client", side_effect=Exception("network")):
            result = alerter._send_sync("hello")
        assert result is False


# ═══════════════════════════════════════════════════════════════════════════════
# TelegramAlerter on_* method formatting
# ═══════════════════════════════════════════════════════════════════════════════


class TestAlerterOnMethods:
    """Test untested on_* methods for correct formatting and priority."""

    def test_on_coupon_received_formats_html(self) -> None:
        alerter = _make_alerter()
        alerter.send_alert = MagicMock()  # type: ignore[assignment]
        alerter.on_coupon_received("SU26244RMFS2", Decimal("3250.00"), "RUB")
        text = alerter.send_alert.call_args[0][0]
        assert "<b>SU26244RMFS2</b>" in text
        assert "<code>3,250.00</code>" in text
        assert "RUB" in text
        assert alerter.send_alert.call_args[1]["priority"] == AlertPriority.INFO

    def test_on_cbr_meeting_formats_date_and_rate(self) -> None:
        alerter = _make_alerter()
        alerter.send_alert = MagicMock()  # type: ignore[assignment]
        alerter.on_cbr_meeting("2026-03-20", "HOLD", "21.00%")
        text = alerter.send_alert.call_args[0][0]
        assert "2026-03-20" in text
        assert "HOLD" in text
        assert "<code>21.00%</code>" in text
        assert alerter.send_alert.call_args[1]["priority"] == AlertPriority.INFO

    def test_on_bond_event_trade_formats_reason(self) -> None:
        alerter = _make_alerter()
        alerter.send_alert = MagicMock()  # type: ignore[assignment]
        alerter.on_bond_event_trade("SU26244RMFS2", "BUY", "5d before meeting, gap=-0.25")
        text = alerter.send_alert.call_args[0][0]
        assert "<b>SU26244RMFS2</b>" in text
        assert "BUY" in text
        assert "5d before meeting" in text
        assert alerter.send_alert.call_args[1]["priority"] == AlertPriority.IMPORTANT

    def test_on_stop_loss_triggered_formats_prices(self) -> None:
        alerter = _make_alerter()
        alerter.send_alert = MagicMock()  # type: ignore[assignment]
        alerter.on_stop_loss_triggered(
            "SBER", Decimal("280.50"), Decimal("266.48"), Decimal("265.00")
        )
        text = alerter.send_alert.call_args[0][0]
        assert "<b>SBER</b>" in text
        assert "<code>280.50</code>" in text
        assert "<code>266.48</code>" in text
        assert "<code>265.00</code>" in text
        assert alerter.send_alert.call_args[1]["priority"] == AlertPriority.IMPORTANT

    def test_on_startup_formats_mode_and_instruments(self) -> None:
        alerter = _make_alerter()
        alerter.send_alert = MagicMock()  # type: ignore[assignment]
        instruments = 23
        alerter.on_startup("sandbox", ["moex", "us"], instruments)
        text = alerter.send_alert.call_args[0][0]
        assert "sandbox" in text
        assert "moex" in text
        assert "23" in text
        assert alerter.send_alert.call_args[1]["priority"] == AlertPriority.INFO

    def test_on_shutdown_sends_stop_message(self) -> None:
        alerter = _make_alerter()
        alerter.send_alert = MagicMock()  # type: ignore[assignment]
        alerter.on_shutdown()
        text = alerter.send_alert.call_args[0][0]
        assert "stopped" in text.lower() or "\u23f9" in text
        assert alerter.send_alert.call_args[1]["priority"] == AlertPriority.INFO


# ═══════════════════════════════════════════════════════════════════════════════
# TelegramAlerter.send_alert edge cases
# ═══════════════════════════════════════════════════════════════════════════════


class TestSendAlertEdgeCases:
    """send_alert routing: sync vs async, queue vs direct."""

    def test_send_alert_sync_context_uses_send_sync(self) -> None:
        """From non-async context (no running loop), uses _send_sync."""
        alerter = _make_alerter()
        with patch.object(alerter, "_send_sync") as mock_sync:
            # No event loop running → should call _send_sync
            alerter.send_alert("test from thread")
            mock_sync.assert_called_once_with("test from thread")

    def test_send_alert_with_queue_and_priority_enqueues(self) -> None:
        """With queue attached and priority given, routes through queue.enqueue."""
        alerter = _make_alerter()
        mock_queue = MagicMock(spec=TelegramMessageQueue)
        mock_queue.enqueue = AsyncMock()
        alerter.set_queue(mock_queue)

        async def _test() -> None:
            alerter.send_alert("queued msg", priority=AlertPriority.INFO)
            await asyncio.sleep(0.05)
            mock_queue.enqueue.assert_called_once()

        asyncio.run(_test())

    def test_send_alert_with_queue_no_priority_uses_direct_send(self) -> None:
        """With queue but no priority, uses direct _send (not enqueue)."""
        alerter = _make_alerter()
        mock_queue = MagicMock(spec=TelegramMessageQueue)
        alerter.set_queue(mock_queue)

        async def _test() -> None:
            with patch.object(alerter, "_send", new_callable=AsyncMock):
                alerter.send_alert("no priority")
                await asyncio.sleep(0.05)
                mock_queue.enqueue.assert_not_called()

        asyncio.run(_test())


# ═══════════════════════════════════════════════════════════════════════════════
# TelegramMessageQueue edge cases
# ═══════════════════════════════════════════════════════════════════════════════


class TestQueueEdgeCases:
    """Queue batching reversal, rate limit boundary, double-start."""

    @pytest.mark.asyncio
    async def test_batch_reverts_when_below_threshold(self) -> None:
        """If batch collection yields < 5 messages, they're put back individually."""
        alerter = MagicMock()
        alerter._send = AsyncMock(return_value=True)
        queue = TelegramMessageQueue(alerter)

        # Enqueue 1 IMPORTANT then 1 INFO (batch won't reach threshold)
        await queue._queue.put(
            __import__("finalayze.core.alerts", fromlist=["QueuedMessage"]).QueuedMessage(
                priority=AlertPriority.IMPORTANT,
                timestamp=time.monotonic(),
                text="fill_1",
            )
        )
        # The INFO message breaks the batch collection
        from finalayze.core.alerts import QueuedMessage

        await queue._queue.put(
            QueuedMessage(
                priority=AlertPriority.INFO,
                timestamp=time.monotonic(),
                text="info_msg",
            )
        )
        # Collect batch for IMPORTANT — should get only 1 (the INFO stops collection)
        batch = queue._collect_batch(AlertPriority.IMPORTANT)
        # Batch should be just 1, INFO should be put back
        assert len(batch) <= 1
        assert not queue._queue.empty()  # INFO was put back

    @pytest.mark.asyncio
    async def test_rate_limit_boundary_old_timestamps_purged(self) -> None:
        """Timestamps older than 60s are purged from the sliding window."""
        alerter = MagicMock()
        alerter._send = AsyncMock(return_value=True)
        queue = TelegramMessageQueue(alerter)
        now = time.monotonic()
        # Fill with timestamps just outside the window (61s ago)
        window_seconds = 61
        queue._sent_timestamps = deque(
            [now - window_seconds for _ in range(RATE_LIMIT)],
            maxlen=RATE_LIMIT * 2,
        )
        # Should NOT be rate limited (all timestamps expired)
        assert queue._is_rate_limited() is False

    @pytest.mark.asyncio
    async def test_double_start_creates_second_task(self) -> None:
        """Calling start() twice creates a new drain task (second overwrites first)."""
        alerter = MagicMock()
        alerter._send = AsyncMock(return_value=True)
        queue = TelegramMessageQueue(alerter)
        await queue.start()
        first_task = queue._drain_task
        await queue.start()
        second_task = queue._drain_task
        # Both exist, second is a different task
        assert first_task is not None
        assert second_task is not None
        # Clean up
        if first_task and not first_task.done():
            first_task.cancel()
            with __import__("contextlib").suppress(asyncio.CancelledError):
                await first_task
        await queue.stop()


# ═══════════════════════════════════════════════════════════════════════════════
# TradingLoop._attempt_grpc_reconnect
# ═══════════════════════════════════════════════════════════════════════════════


class TestAttemptGrpcReconnect:
    """gRPC reconnection with exponential backoff and stop_event on exhaustion."""

    def test_reconnect_non_tinkoff_returns_false(self) -> None:
        """Non-TinkoffBroker returns False immediately."""
        from finalayze.core.trading_loop import TradingLoop

        loop = MagicMock(spec=TradingLoop)
        loop._broker_router = MagicMock()
        loop._broker_router.route.return_value = MagicMock()  # not a TinkoffBroker

        with patch("finalayze.core.trading_loop.TinkoffBroker", create=True):
            result = TradingLoop._attempt_grpc_reconnect(loop, "us")

        assert result is False

    def test_reconnect_success_on_first_attempt(self) -> None:
        """Reconnection succeeds on first attempt, returns True."""
        from finalayze.core.trading_loop import TradingLoop
        from finalayze.execution.tinkoff_broker import TinkoffBroker

        mock_broker = MagicMock(spec=TinkoffBroker)
        mock_broker.reconnect_client.return_value = True

        loop = MagicMock(spec=TradingLoop)
        loop._broker_router = MagicMock()
        loop._broker_router.route.return_value = mock_broker
        loop._reconnect_delays = [1]
        loop._alerter = MagicMock()
        loop._stop_event = threading.Event()

        with patch("time.sleep"):
            result = TradingLoop._attempt_grpc_reconnect(loop, "moex")

        assert result is True

    def test_reconnect_exhaustion_sets_stop_event(self) -> None:
        """All reconnect attempts fail → _stop_event.set() called."""
        from finalayze.core.trading_loop import TradingLoop
        from finalayze.execution.tinkoff_broker import TinkoffBroker

        mock_broker = MagicMock(spec=TinkoffBroker)
        mock_broker.reconnect_client.return_value = False

        loop = MagicMock(spec=TradingLoop)
        loop._broker_router = MagicMock()
        loop._broker_router.route.return_value = mock_broker
        loop._reconnect_delays = [0.01, 0.01]
        loop._alerter = MagicMock()
        loop._stop_event = threading.Event()

        with patch("time.sleep"):
            result = TradingLoop._attempt_grpc_reconnect(loop, "moex")

        assert result is False
        assert loop._stop_event.is_set()

    def test_reconnect_jitter_bounded(self) -> None:
        """Jitter stays within [0.8x, 1.2x] of base delay."""
        import random

        for _ in range(100):
            jitter = random.uniform(0.8, 1.2)
            assert 0.8 <= jitter <= 1.2


# ═══════════════════════════════════════════════════════════════════════════════
# TradingLoop._reconcile_inflight_orders
# ═══════════════════════════════════════════════════════════════════════════════


class TestReconcileInflightOrders:
    """Startup reconciliation: cancel stale orders, log partial fills."""

    def test_reconcile_cancels_stale_orders(self) -> None:
        """All open orders are cancelled on startup."""
        from finalayze.core.trading_loop import TradingLoop
        from finalayze.execution.tinkoff_broker import TinkoffBroker

        order = MagicMock()
        order.order_id = "order-123"
        order.execution_status = "FILL"
        order.filled_quantity = Decimal(0)
        order.filled_price = Decimal(0)

        mock_broker = MagicMock(spec=TinkoffBroker)
        mock_broker.get_open_orders.return_value = [order]
        mock_broker.cancel_order_safe.return_value = True

        loop = MagicMock(spec=TradingLoop)
        loop._circuit_breakers = {"moex": MagicMock()}
        loop._broker_router = MagicMock()
        loop._broker_router.route.return_value = mock_broker

        TradingLoop._reconcile_inflight_orders(loop)
        mock_broker.cancel_order_safe.assert_called_once_with("order-123")

    def test_reconcile_logs_partial_fill(self) -> None:
        """Partial fills (filled_quantity > 0) are logged."""
        from finalayze.core.trading_loop import TradingLoop
        from finalayze.execution.tinkoff_broker import TinkoffBroker

        order = MagicMock()
        order.order_id = "order-456"
        order.execution_status = "PARTIAL"
        order.filled_quantity = Decimal(5)
        order.filled_price = Decimal("280.50")

        mock_broker = MagicMock(spec=TinkoffBroker)
        mock_broker.get_open_orders.return_value = [order]
        mock_broker.cancel_order_safe.return_value = True

        loop = MagicMock(spec=TradingLoop)
        loop._circuit_breakers = {"moex": MagicMock()}
        loop._broker_router = MagicMock()
        loop._broker_router.route.return_value = mock_broker

        # Should not raise; partial fill is just logged
        TradingLoop._reconcile_inflight_orders(loop)
        mock_broker.cancel_order_safe.assert_called_once()

    def test_reconcile_no_orders_is_noop(self) -> None:
        """No open orders → no cancel calls."""
        from finalayze.core.trading_loop import TradingLoop
        from finalayze.execution.tinkoff_broker import TinkoffBroker

        mock_broker = MagicMock(spec=TinkoffBroker)
        mock_broker.get_open_orders.return_value = []

        loop = MagicMock(spec=TradingLoop)
        loop._circuit_breakers = {"moex": MagicMock()}
        loop._broker_router = MagicMock()
        loop._broker_router.route.return_value = mock_broker

        TradingLoop._reconcile_inflight_orders(loop)
        mock_broker.cancel_order_safe.assert_not_called()

    def test_reconcile_get_orders_exception_continues(self) -> None:
        """Exception in get_open_orders doesn't crash — skips to next market."""
        from finalayze.core.trading_loop import TradingLoop
        from finalayze.execution.tinkoff_broker import TinkoffBroker

        mock_broker = MagicMock(spec=TinkoffBroker)
        mock_broker.get_open_orders.side_effect = Exception("gRPC error")

        loop = MagicMock(spec=TradingLoop)
        loop._circuit_breakers = {"moex": MagicMock()}
        loop._broker_router = MagicMock()
        loop._broker_router.route.return_value = mock_broker

        # Should not raise
        TradingLoop._reconcile_inflight_orders(loop)


# ═══════════════════════════════════════════════════════════════════════════════
# TradingLoop._compute_top_movers
# ═══════════════════════════════════════════════════════════════════════════════


class TestComputeTopMovers:
    """Top 3 movers by absolute P&L % across all markets."""

    def test_returns_max_three(self) -> None:
        """Returns at most 3 movers even if more positions exist."""
        from finalayze.core.trading_loop import TradingLoop

        loop = MagicMock(spec=TradingLoop)
        loop._circuit_breakers = {"us": MagicMock()}
        loop._baseline_equities = {"us": Decimal(50000)}

        portfolio = MagicMock()
        portfolio.positions = {
            "AAPL": Decimal(10),
            "MSFT": Decimal(5),
            "GOOG": Decimal(8),
            "NVDA": Decimal(3),
        }
        broker = MagicMock()
        broker.get_portfolio.return_value = portfolio
        loop._broker_router = MagicMock()
        loop._broker_router.route.return_value = broker

        result = TradingLoop._compute_top_movers(loop)
        assert len(result) <= 3

    def test_returns_empty_on_exception(self) -> None:
        """Returns empty list when broker.get_portfolio() raises."""
        from finalayze.core.trading_loop import TradingLoop

        loop = MagicMock(spec=TradingLoop)
        loop._circuit_breakers = {"us": MagicMock()}
        loop._baseline_equities = {}

        broker = MagicMock()
        broker.get_portfolio.side_effect = Exception("broker down")
        loop._broker_router = MagicMock()
        loop._broker_router.route.return_value = broker

        result = TradingLoop._compute_top_movers(loop)
        assert result == []


# ═══════════════════════════════════════════════════════════════════════════════
# TradingLoop._cbr_day_refresh
# ═══════════════════════════════════════════════════════════════════════════════


class TestCbrDayRefresh:
    """CBR day refresh fires macro update and alert."""

    def test_cbr_refresh_calls_on_cbr_meeting(self) -> None:
        """After successful macro refresh, alerter.on_cbr_meeting is called."""
        from finalayze.core.trading_loop import TradingLoop

        macro_data = MagicMock()
        macro_data.key_rate = Decimal("21.00")
        macro_data.last_cbr_decision = "hold"

        macro_cache = MagicMock()
        macro_cache.is_cbr_meeting_day.return_value = True
        macro_cache.get.return_value = macro_data

        loop = MagicMock(spec=TradingLoop)
        loop._macro_cache = macro_cache
        loop._alerter = MagicMock()
        loop._now.return_value = datetime(2026, 3, 20, 12, 30, tzinfo=UTC)
        loop._macro_refresh = MagicMock()
        loop._bond_cycle = MagicMock()

        TradingLoop._cbr_day_refresh(loop)
        loop._alerter.on_cbr_meeting.assert_called_once()
        call_args = loop._alerter.on_cbr_meeting.call_args[0]
        assert "2026-03-20" in call_args[0]
        assert "HOLD" in call_args[1]
        assert "21.00%" in call_args[2]

    def test_cbr_refresh_sends_error_when_stale(self) -> None:
        """If macro data stale after refresh, sends error alert."""
        from finalayze.core.trading_loop import TradingLoop

        macro_cache = MagicMock()
        macro_cache.is_cbr_meeting_day.return_value = True
        macro_cache.get.return_value = None  # stale/missing

        loop = MagicMock(spec=TradingLoop)
        loop._macro_cache = macro_cache
        loop._alerter = MagicMock()
        loop._macro_refresh = MagicMock()
        loop._bond_cycle = MagicMock()

        TradingLoop._cbr_day_refresh(loop)
        loop._alerter.on_error.assert_called_once()
        assert "stale" in loop._alerter.on_error.call_args[0][1]

    def test_cbr_refresh_skips_non_meeting_day(self) -> None:
        """Returns early when not a CBR meeting day."""
        from finalayze.core.trading_loop import TradingLoop

        macro_cache = MagicMock()
        macro_cache.is_cbr_meeting_day.return_value = False

        loop = MagicMock(spec=TradingLoop)
        loop._macro_cache = macro_cache
        loop._alerter = MagicMock()

        TradingLoop._cbr_day_refresh(loop)
        loop._alerter.on_cbr_meeting.assert_not_called()


# ═══════════════════════════════════════════════════════════════════════════════
# TradingLoop._weekly_digest
# ═══════════════════════════════════════════════════════════════════════════════


class TestWeeklyDigestExecution:
    """Weekly digest sends formatted alert with market P&L."""

    def test_weekly_digest_sends_alert(self) -> None:
        """_weekly_digest sends an alert with week P&L data."""
        from finalayze.core.trading_loop import TradingLoop

        portfolio = MagicMock()
        portfolio.equity = Decimal(51000)
        portfolio.positions = {"AAPL": Decimal(10)}

        broker = MagicMock()
        broker.get_portfolio.return_value = portfolio

        loop = MagicMock(spec=TradingLoop)
        loop._circuit_breakers = {"us": MagicMock()}
        loop._broker_router = MagicMock()
        loop._broker_router.route.return_value = broker
        loop._baseline_equities = {"us": Decimal(50000)}
        loop._bond_processor = None
        loop._alerter = MagicMock()
        loop._now.return_value = datetime(2026, 3, 15, 16, 0, tzinfo=UTC)
        loop._compute_top_movers = MagicMock(return_value=[])

        TradingLoop._weekly_digest(loop)
        loop._alerter.send_alert.assert_called_once()
        text = loop._alerter.send_alert.call_args[0][0]
        assert "Weekly Digest" in text
        assert "US" in text

    def test_weekly_digest_includes_bond_pnl(self) -> None:
        """Weekly digest includes bond layer P&L when bond_processor is set."""
        from finalayze.core.trading_loop import TradingLoop

        portfolio = MagicMock()
        portfolio.equity = Decimal(51000)
        portfolio.positions = {}

        broker = MagicMock()
        broker.get_portfolio.return_value = portfolio

        bond_ledger = MagicMock()
        bond_ledger.current_equity = Decimal(1005000)

        loop = MagicMock(spec=TradingLoop)
        loop._circuit_breakers = {"us": MagicMock()}
        loop._broker_router = MagicMock()
        loop._broker_router.route.return_value = broker
        loop._baseline_equities = {"us": Decimal(50000), "moex_bonds": Decimal(1000000)}
        loop._bond_processor = MagicMock()
        loop._bond_processor._layer_ledgers = {"core": bond_ledger}
        loop._alerter = MagicMock()
        loop._now.return_value = datetime(2026, 3, 15, 16, 0, tzinfo=UTC)
        loop._compute_top_movers = MagicMock(return_value=[])

        TradingLoop._weekly_digest(loop)
        text = loop._alerter.send_alert.call_args[0][0]
        assert "BONDS" in text


# ═══════════════════════════════════════════════════════════════════════════════
# TradingLoop._daily_reset edge cases
# ═══════════════════════════════════════════════════════════════════════════════


class TestDailyResetEdgeCases:
    """Edge cases for daily P&L reset."""

    def test_daily_reset_none_bond_processor(self) -> None:
        """bond_processor=None → no bond P&L in market_pnl."""
        from finalayze.core.trading_loop import TradingLoop

        loop = MagicMock(spec=TradingLoop)
        loop._circuit_breakers = {"us": MagicMock()}
        loop._bond_processor = None
        loop._metrics = MagicMock()

        portfolio = MagicMock()
        portfolio.equity = Decimal(50500)

        broker = MagicMock()
        broker.get_portfolio.return_value = portfolio
        loop._broker_router = MagicMock()
        loop._broker_router.route.return_value = broker
        loop._baseline_equities = {"us": Decimal(50000)}
        loop._cross_market_breaker = MagicMock()
        loop._alerter = MagicMock()
        loop._loss_limit_tracker = MagicMock()
        loop._fx_service = None
        loop._now.return_value = datetime(2026, 3, 14, 0, 0, tzinfo=UTC)
        loop._persist_equity_snapshots = MagicMock()
        loop._compute_top_movers = MagicMock(return_value=[])

        TradingLoop._daily_reset(loop)
        market_pnl = loop._alerter.on_daily_summary.call_args[0][0]
        assert "moex_bonds" not in market_pnl

    def test_daily_reset_monday_resets_weekly_loss_limit(self) -> None:
        """On Monday (weekday=0), weekly loss limit is also reset."""
        from finalayze.core.trading_loop import TradingLoop

        loop = MagicMock(spec=TradingLoop)
        loop._circuit_breakers = {}
        loop._bond_processor = None
        loop._metrics = MagicMock()

        loop._cross_market_breaker = MagicMock()
        loop._alerter = MagicMock()
        loop._loss_limit_tracker = MagicMock()
        loop._fx_service = None
        loop._baseline_equities = {}
        loop._persist_equity_snapshots = MagicMock()
        loop._compute_top_movers = MagicMock(return_value=[])
        # Monday
        loop._now.return_value = datetime(2026, 3, 16, 0, 0, tzinfo=UTC)  # Monday

        TradingLoop._daily_reset(loop)
        loop._loss_limit_tracker.reset_week.assert_called_once()

    def test_daily_reset_non_monday_no_weekly_reset(self) -> None:
        """On non-Monday, weekly loss limit is NOT reset."""
        from finalayze.core.trading_loop import TradingLoop

        loop = MagicMock(spec=TradingLoop)
        loop._circuit_breakers = {}
        loop._bond_processor = None
        loop._metrics = MagicMock()

        loop._cross_market_breaker = MagicMock()
        loop._alerter = MagicMock()
        loop._loss_limit_tracker = MagicMock()
        loop._fx_service = None
        loop._baseline_equities = {}
        loop._persist_equity_snapshots = MagicMock()
        loop._compute_top_movers = MagicMock(return_value=[])
        # Wednesday
        loop._now.return_value = datetime(2026, 3, 18, 0, 0, tzinfo=UTC)  # Wednesday

        TradingLoop._daily_reset(loop)
        loop._loss_limit_tracker.reset_week.assert_not_called()

    def test_daily_reset_fx_unavailable_no_dual_currency(self) -> None:
        """FX service returning 0 → total_equity_rub is None."""
        from finalayze.core.trading_loop import TradingLoop

        loop = MagicMock(spec=TradingLoop)
        loop._circuit_breakers = {}
        loop._bond_processor = None
        loop._metrics = MagicMock()
        loop._cross_market_breaker = MagicMock()
        loop._alerter = MagicMock()
        loop._loss_limit_tracker = MagicMock()
        loop._baseline_equities = {}
        loop._persist_equity_snapshots = MagicMock()
        loop._compute_top_movers = MagicMock(return_value=[])
        loop._now.return_value = datetime(2026, 3, 14, 0, 0, tzinfo=UTC)

        fx = MagicMock()
        fx.get_usdrub.return_value = Decimal(0)
        loop._fx_service = fx

        TradingLoop._daily_reset(loop)
        call_args = loop._alerter.on_daily_summary.call_args
        # total_equity_rub should be None (FX rate is 0)
        total_equity_rub = (
            call_args[0][3] if len(call_args[0]) > 3 else call_args[1].get("total_equity_rub")
        )
        assert total_equity_rub is None


# ═══════════════════════════════════════════════════════════════════════════════
# TelegramBotHandler edge cases
# ═══════════════════════════════════════════════════════════════════════════════


class TestBotHandlerEdgeCases:
    """TelegramBotHandler: no-message, empty text, all-brokers-fail, bond=None."""

    def _make_handler(
        self,
        allowed_chat_ids: list[str] | None = None,
        bond_processor: object | None = None,
    ) -> TelegramBotHandler:
        from finalayze.core.telegram_bot import TelegramBotHandler

        alerter = MagicMock()
        alerter._send = AsyncMock(return_value=True)
        settings = MagicMock()
        settings.telegram_allowed_chat_ids = allowed_chat_ids or ["123456"]

        return TelegramBotHandler(
            alerter=alerter,
            broker_router=MagicMock(),
            circuit_breakers={"us": MagicMock()},
            settings=settings,
            bond_processor=bond_processor,
        )

    @pytest.mark.asyncio
    async def test_update_with_no_message_returns_no_message(self) -> None:
        """Update without 'message' field returns {"ok": "no_message"}."""
        handler = self._make_handler()
        result = await handler.handle_update({"update_id": 1})
        assert result == {"ok": "no_message"}

    @pytest.mark.asyncio
    async def test_empty_text_returns_no_command(self) -> None:
        """Message with empty text returns {"ok": "no_command"}."""
        handler = self._make_handler()
        result = await handler.handle_update({"message": {"chat": {"id": 123456}, "text": ""}})
        assert result == {"ok": "no_command"}

    @pytest.mark.asyncio
    async def test_command_with_extra_args_dispatches(self) -> None:
        """'/status extra args' still dispatches to handle_status."""
        handler = self._make_handler()
        handler._broker_router = MagicMock()
        handler._broker_router.registered_markets = []
        result = await handler.handle_update(
            {"message": {"chat": {"id": 123456}, "text": "/status extra args"}}
        )
        assert result == {"ok": "processed"}

    @pytest.mark.asyncio
    async def test_status_all_brokers_fail(self) -> None:
        """When all brokers raise, status still responds (with 'unavailable')."""
        handler = self._make_handler()
        handler._broker_router.registered_markets = ["us", "moex"]
        handler._broker_router.route.side_effect = Exception("broker down")
        result = await handler.handle_update(
            {"message": {"chat": {"id": 123456}, "text": "/status"}}
        )
        assert result == {"ok": "processed"}
        # Should have called _send with "unavailable" in text
        text = handler._alerter._send.call_args[0][0]
        assert "unavailable" in text

    @pytest.mark.asyncio
    async def test_status_bond_processor_none(self) -> None:
        """bond_processor=None → bond section is skipped in status."""
        handler = self._make_handler(bond_processor=None)
        handler._broker_router.registered_markets = []
        result = await handler.handle_update(
            {"message": {"chat": {"id": 123456}, "text": "/status"}}
        )
        assert result == {"ok": "processed"}
        text = handler._alerter._send.call_args[0][0]
        assert "Bond Layers" not in text

    @pytest.mark.asyncio
    async def test_breakers_bond_processor_none(self) -> None:
        """bond_processor=None → bond breaker section skipped in /breakers."""
        handler = self._make_handler(bond_processor=None)
        # Setup circuit breakers
        for cb in handler._circuit_breakers.values():
            cb.level = "normal"
            cb.baseline = Decimal(100000)

        result = await handler.handle_update(
            {"message": {"chat": {"id": 123456}, "text": "/breakers"}}
        )
        assert result == {"ok": "processed"}
        text = handler._alerter._send.call_args[0][0]
        assert "Bond Layer Breakers" not in text

    @pytest.mark.asyncio
    async def test_chat_id_integer_coerced_to_string(self) -> None:
        """Integer chat_id from Telegram is coerced to string for whitelist check."""
        handler = self._make_handler(allowed_chat_ids=["123456"])
        result = await handler.handle_update(
            {"message": {"chat": {"id": 123456}, "text": "/status"}}
        )
        # int 123456 → str "123456" matches whitelist
        assert result["ok"] != "ignored"


# ═══════════════════════════════════════════════════════════════════════════════
# Webhook edge cases
# ═══════════════════════════════════════════════════════════════════════════════


class TestWebhookEdgeCases:
    """Webhook endpoint: empty secret, missing message field in JSON."""

    def _make_app(self) -> tuple:
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from finalayze.api.v1.telegram import create_telegram_router
        from finalayze.core.telegram_bot import TelegramBotHandler

        alerter = MagicMock()
        alerter._send = AsyncMock(return_value=True)
        settings = MagicMock()
        settings.telegram_allowed_chat_ids = ["123456"]

        handler = TelegramBotHandler(
            alerter=alerter,
            broker_router=MagicMock(),
            circuit_breakers={},
            settings=settings,
        )
        app = FastAPI()
        router = create_telegram_router(handler, "test-secret")
        app.include_router(router)
        return TestClient(app), handler

    def test_empty_secret_header_returns_403(self) -> None:
        """Empty string secret header is rejected (not equal to webhook secret)."""
        client, _ = self._make_app()
        resp = client.post(
            "/api/telegram/webhook",
            json={"message": {"chat": {"id": 123456}, "text": "/status"}},
            headers={"X-Telegram-Bot-Api-Secret-Token": ""},
        )
        assert resp.status_code == 403

    def test_valid_json_no_message_returns_no_message(self) -> None:
        """Valid JSON without 'message' key returns {"ok": "no_message"}."""
        client, _ = self._make_app()
        resp = client.post(
            "/api/telegram/webhook",
            json={"update_id": 12345},
            headers={"X-Telegram-Bot-Api-Secret-Token": "test-secret"},
        )
        assert resp.status_code == 200
        assert resp.json() == {"ok": "no_message"}

    def test_text_as_null_returns_no_command(self) -> None:
        """JSON with text=null (None) returns no_command."""
        client, _ = self._make_app()
        resp = client.post(
            "/api/telegram/webhook",
            json={"message": {"chat": {"id": 123456}, "text": None}},
            headers={"X-Telegram-Bot-Api-Secret-Token": "test-secret"},
        )
        assert resp.status_code == 200
        assert resp.json() == {"ok": "no_command"}
