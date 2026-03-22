"""Unit tests for TinkoffBroker.get_open_orders() and cancel_order() methods."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from finalayze.execution.tinkoff_broker import (
    OrderStateResult,
    TinkoffBroker,
    _TERMINAL_STATUSES,
)
from finalayze.markets.instruments import DEFAULT_MOEX_INSTRUMENTS, InstrumentRegistry


def _make_registry() -> InstrumentRegistry:
    registry = InstrumentRegistry()
    for inst in DEFAULT_MOEX_INSTRUMENTS:
        registry.register(inst)
    return registry


def _make_broker() -> TinkoffBroker:
    broker = TinkoffBroker(token="fake_token", registry=_make_registry(), sandbox=True)  # noqa: S106
    broker._account_id = "test-account"
    return broker


def _make_order_mock(
    order_id: str = "ord-1",
    status: int = 4,  # "new"
    lots_executed: int = 0,
    price_units: int = 0,
    price_nano: int = 0,
) -> MagicMock:
    """Create a mock T-Invest order object."""
    order = MagicMock()
    order.order_id = order_id
    order.execution_report_status = status
    order.lots_executed = lots_executed
    order.executed_order_price.units = price_units
    order.executed_order_price.nano = price_nano
    return order


class TestGetOpenOrders:
    def test_returns_non_terminal_orders(self) -> None:
        """get_open_orders() should return only non-terminal orders."""
        broker = _make_broker()
        mock_response = MagicMock()
        mock_response.orders = [
            _make_order_mock("ord-new", status=4),  # new
            _make_order_mock("ord-partial", status=2, lots_executed=5, price_units=100),  # partial
            _make_order_mock("ord-filled", status=1, lots_executed=10, price_units=200),  # fill
            _make_order_mock("ord-cancelled", status=3),  # cancelled
        ]

        with patch.object(broker, "_run_async", return_value=mock_response):
            orders = broker.get_open_orders()

        # Only "new" and "partially_fill" are non-terminal
        assert len(orders) == 2
        order_ids = [o.order_id for o in orders]
        assert "ord-new" in order_ids
        assert "ord-partial" in order_ids

    def test_returns_order_state_results(self) -> None:
        """get_open_orders() should return OrderStateResult instances."""
        broker = _make_broker()
        mock_response = MagicMock()
        mock_response.orders = [
            _make_order_mock("ord-1", status=4, lots_executed=0, price_units=0),
        ]

        with patch.object(broker, "_run_async", return_value=mock_response):
            orders = broker.get_open_orders()

        assert len(orders) == 1
        assert isinstance(orders[0], OrderStateResult)
        assert orders[0].order_id == "ord-1"
        assert orders[0].execution_status == "new"
        assert orders[0].is_terminal is False

    def test_returns_empty_list_on_error(self) -> None:
        """get_open_orders() should return empty list on API failure."""
        broker = _make_broker()

        with patch.object(broker, "_run_async", side_effect=RuntimeError("gRPC error")):
            orders = broker.get_open_orders()

        assert orders == []

    def test_returns_empty_list_when_no_orders(self) -> None:
        """get_open_orders() should return empty list when no orders exist."""
        broker = _make_broker()
        mock_response = MagicMock()
        mock_response.orders = []

        with patch.object(broker, "_run_async", return_value=mock_response):
            orders = broker.get_open_orders()

        assert orders == []

    def test_partial_fill_mapped_correctly(self) -> None:
        """Partially filled order should have correct quantity and price."""
        broker = _make_broker()
        mock_response = MagicMock()
        mock_response.orders = [
            _make_order_mock("ord-p", status=2, lots_executed=5, price_units=150, price_nano=500_000_000),
        ]

        with patch.object(broker, "_run_async", return_value=mock_response):
            orders = broker.get_open_orders()

        assert len(orders) == 1
        assert orders[0].filled_quantity == Decimal(5)
        assert orders[0].filled_price == Decimal("150.5")
        assert orders[0].execution_status == "partially_fill"


class TestCancelOrderBool:
    """Tests for the bool-returning cancel_order_safe() method."""

    def test_cancel_returns_true_on_success(self) -> None:
        """cancel_order_safe() should return True on successful cancel."""
        broker = _make_broker()

        with patch.object(broker, "_run_async", return_value=None):
            result = broker.cancel_order_safe("ord-1")

        assert result is True

    def test_cancel_returns_false_on_error(self) -> None:
        """cancel_order_safe() should return False on API failure."""
        broker = _make_broker()

        with patch.object(broker, "_run_async", side_effect=RuntimeError("cancel failed")):
            result = broker.cancel_order_safe("ord-1")

        assert result is False
