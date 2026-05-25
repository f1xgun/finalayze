"""S1.1 — Order idempotency and Tinkoff execution-status mapping.

The previous live order path had two related bugs:

1. ``client_order_id`` was declared on neither schema nor brokers, so
   ``RetryPolicy.execute(lambda: broker.submit_order(order))`` could
   double-submit on any transient network error (UNAVAILABLE / timeout).
2. ``TinkoffBroker.submit_order`` returned ``OrderResult(filled=True, ...)``
   for *every* successful gRPC response without inspecting
   ``execution_report_status``. Rejections and queued (NEW) orders were
   silently recorded as fills.

These tests cover the new contract.
"""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

import pytest
from t_tech.invest.schemas import OrderExecutionReportStatus

from finalayze.execution.broker_base import OrderRequest
from finalayze.execution.tinkoff_broker import TinkoffBroker
from finalayze.markets.instruments import DEFAULT_MOEX_INSTRUMENTS, InstrumentRegistry


def _make_registry() -> InstrumentRegistry:
    registry = InstrumentRegistry()
    for inst in DEFAULT_MOEX_INSTRUMENTS:
        registry.register(inst)
    return registry


def _make_broker() -> TinkoffBroker:
    broker = TinkoffBroker(token="fake_token", registry=_make_registry(), sandbox=True)  # noqa: S106
    broker._account_id = "acc-sandbox-001"
    return broker


def _mock_post_order_result(
    status: int,
    *,
    order_id: str = "ord-tinkoff-1",
    executed_units: int = 270,
    executed_nano: int = 0,
    lots_executed: int = 1,
) -> MagicMock:
    """Build a mock t-invest PostOrderResponse with a specific execution status."""
    result = MagicMock()
    result.execution_report_status = status
    result.order_id = order_id
    result.executed_order_price.units = executed_units
    result.executed_order_price.nano = executed_nano
    result.lots_executed = lots_executed
    return result


# ── client_order_id contract ────────────────────────────────────────────


class TestOrderRequestClientId:
    def test_default_factory_generates_id(self) -> None:
        order = OrderRequest(symbol="SBER", side="BUY", quantity=Decimal(10))
        assert order.client_order_id, "default factory must populate client_order_id"

    def test_default_factory_unique_per_order(self) -> None:
        a = OrderRequest(symbol="SBER", side="BUY", quantity=Decimal(10))
        b = OrderRequest(symbol="SBER", side="BUY", quantity=Decimal(10))
        assert a.client_order_id != b.client_order_id

    def test_explicit_id_preserved(self) -> None:
        order = OrderRequest(
            symbol="SBER",
            side="BUY",
            quantity=Decimal(10),
            client_order_id="my-explicit-id",
        )
        assert order.client_order_id == "my-explicit-id"

    def test_frozen_after_construction(self) -> None:
        order = OrderRequest(symbol="SBER", side="BUY", quantity=Decimal(10))
        with pytest.raises(Exception):  # noqa: B017, PT011
            order.client_order_id = "mutated"  # type: ignore[misc]


# ── Tinkoff: client_order_id passed to post_order ───────────────────────


class TestTinkoffPassesClientOrderId:
    def test_post_order_receives_client_order_id(self) -> None:
        """tinkoff_broker.submit_order must forward client_order_id as order_id."""
        broker = _make_broker()
        captured: dict[str, object] = {}

        # Stub _post_order_async itself — capture kwargs, then return a
        # pre-built coroutine so the real submit_order path stays intact.
        async def _stub(**kwargs: object) -> MagicMock:
            captured.update(kwargs)
            return _mock_post_order_result(
                OrderExecutionReportStatus.EXECUTION_REPORT_STATUS_FILL,
            )

        broker._post_order_async = _stub  # type: ignore[method-assign]

        # _run_async would normally start an event loop; here we just await
        # the coroutine it was handed.
        def _sync_runner(coro: object) -> object:
            import asyncio  # noqa: PLC0415

            return asyncio.run(coro)  # type: ignore[arg-type]

        broker._run_async = MagicMock(side_effect=_sync_runner)  # type: ignore[method-assign]

        order = OrderRequest(
            symbol="SBER",
            side="BUY",
            quantity=Decimal(10),
            client_order_id="fnz-test-12345",
        )
        result = broker.submit_order(order)

        assert captured.get("order_id") == "fnz-test-12345"
        assert result.filled is True


# ── Tinkoff: execution_report_status mapping ────────────────────────────


class TestTinkoffStatusMapping:
    def test_fill_status_marks_filled(self) -> None:
        broker = _make_broker()
        broker._run_async = MagicMock(  # type: ignore[method-assign]
            return_value=_mock_post_order_result(
                OrderExecutionReportStatus.EXECUTION_REPORT_STATUS_FILL,
            ),
        )
        result = broker.submit_order(OrderRequest("SBER", "BUY", Decimal(10)))
        assert result.filled is True
        assert result.fill_price == Decimal(270)

    def test_partially_fill_marks_filled_with_partial_qty(self) -> None:
        broker = _make_broker()
        broker._run_async = MagicMock(  # type: ignore[method-assign]
            return_value=_mock_post_order_result(
                OrderExecutionReportStatus.EXECUTION_REPORT_STATUS_PARTIALLYFILL,
                lots_executed=1,  # only 1 of N lots filled
            ),
        )
        result = broker.submit_order(OrderRequest("SBER", "BUY", Decimal(10)))
        assert result.filled is True
        assert "partial" in result.reason.lower()

    def test_rejected_status_marks_not_filled(self) -> None:
        broker = _make_broker()
        broker._run_async = MagicMock(  # type: ignore[method-assign]
            return_value=_mock_post_order_result(
                OrderExecutionReportStatus.EXECUTION_REPORT_STATUS_REJECTED,
            ),
        )
        result = broker.submit_order(OrderRequest("SBER", "BUY", Decimal(10)))
        assert result.filled is False
        assert "reject" in result.reason.lower()

    def test_new_status_marks_not_filled(self) -> None:
        """NEW = queued at the exchange; not yet executed."""
        broker = _make_broker()
        broker._run_async = MagicMock(  # type: ignore[method-assign]
            return_value=_mock_post_order_result(
                OrderExecutionReportStatus.EXECUTION_REPORT_STATUS_NEW,
            ),
        )
        result = broker.submit_order(OrderRequest("SBER", "BUY", Decimal(10)))
        assert result.filled is False
        assert "queued" in result.reason.lower() or "new" in result.reason.lower()

    def test_cancelled_status_marks_not_filled(self) -> None:
        broker = _make_broker()
        broker._run_async = MagicMock(  # type: ignore[method-assign]
            return_value=_mock_post_order_result(
                OrderExecutionReportStatus.EXECUTION_REPORT_STATUS_CANCELLED,
            ),
        )
        result = broker.submit_order(OrderRequest("SBER", "BUY", Decimal(10)))
        assert result.filled is False
        assert "cancel" in result.reason.lower()

    def test_unspecified_status_marks_not_filled(self) -> None:
        broker = _make_broker()
        broker._run_async = MagicMock(  # type: ignore[method-assign]
            return_value=_mock_post_order_result(
                OrderExecutionReportStatus.EXECUTION_REPORT_STATUS_UNSPECIFIED,
            ),
        )
        result = broker.submit_order(OrderRequest("SBER", "BUY", Decimal(10)))
        assert result.filled is False
