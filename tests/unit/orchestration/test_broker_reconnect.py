"""Tests for broker_reconnect module functions (Layer 5 -- orchestrator).

Tests the pure functions:
  - attempt_grpc_reconnect: exponential-backoff reconnection with alerts
  - reconcile_inflight_orders: query and cancel stale orders
"""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import pytest


class TestAttemptGrpcReconnect:
    """Pure function for exponential-backoff gRPC reconnection."""

    def test_non_tinkoff_broker_returns_false(self) -> None:
        """Non-TinkoffBroker returns False immediately."""
        from finalayze.orchestration.broker_reconnect import attempt_grpc_reconnect

        broker_router = MagicMock()
        broker_router.route.return_value = MagicMock()  # not TinkoffBroker
        alerter = MagicMock()
        stop_event = threading.Event()
        reconnect_delays = [1, 2]

        result = attempt_grpc_reconnect(
            broker_router=broker_router,
            alerter=alerter,
            stop_event=stop_event,
            reconnect_delays=reconnect_delays,
            broker_name="us",
        )

        assert result is False
        alerter.on_error.assert_not_called()

    def test_successful_reconnect_on_first_attempt(self) -> None:
        """Reconnection succeeds on first attempt, returns True."""
        from finalayze.execution.tinkoff_broker import TinkoffBroker
        from finalayze.orchestration.broker_reconnect import attempt_grpc_reconnect

        mock_broker = MagicMock(spec=TinkoffBroker)
        mock_broker.reconnect_client.return_value = True

        broker_router = MagicMock()
        broker_router.route.return_value = mock_broker
        alerter = MagicMock()
        stop_event = threading.Event()
        reconnect_delays = [1]

        result = attempt_grpc_reconnect(
            broker_router=broker_router,
            alerter=alerter,
            stop_event=stop_event,
            reconnect_delays=reconnect_delays,
            broker_name="moex",
        )

        assert result is True
        assert alerter.on_error.called

    def test_stop_event_set_during_wait_returns_false(self) -> None:
        """If stop_event is set during wait, exits early returning False."""
        from finalayze.execution.tinkoff_broker import TinkoffBroker
        from finalayze.orchestration.broker_reconnect import attempt_grpc_reconnect

        mock_broker = MagicMock(spec=TinkoffBroker)
        broker_router = MagicMock()
        broker_router.route.return_value = mock_broker
        alerter = MagicMock()
        stop_event = threading.Event()
        reconnect_delays = [1, 2]

        # Patch wait to return True (stop_event was set)
        with patch.object(stop_event, "wait", return_value=True):
            result = attempt_grpc_reconnect(
                broker_router=broker_router,
                alerter=alerter,
                stop_event=stop_event,
                reconnect_delays=reconnect_delays,
                broker_name="moex",
            )

        assert result is False
        mock_broker.reconnect_client.assert_not_called()

    def test_exhaustion_sets_stop_event(self) -> None:
        """All reconnect attempts fail → stop_event.set() called."""
        from finalayze.execution.tinkoff_broker import TinkoffBroker
        from finalayze.orchestration.broker_reconnect import attempt_grpc_reconnect

        mock_broker = MagicMock(spec=TinkoffBroker)
        mock_broker.reconnect_client.return_value = False

        broker_router = MagicMock()
        broker_router.route.return_value = mock_broker
        alerter = MagicMock()
        stop_event = threading.Event()
        reconnect_delays = [0.01, 0.01]

        result = attempt_grpc_reconnect(
            broker_router=broker_router,
            alerter=alerter,
            stop_event=stop_event,
            reconnect_delays=reconnect_delays,
            broker_name="moex",
        )

        assert result is False
        assert stop_event.is_set()

    def test_uses_stop_event_wait_not_time_sleep(self) -> None:
        """stop_event.wait(timeout=) must be called, not time.sleep."""
        from finalayze.execution.tinkoff_broker import TinkoffBroker
        from finalayze.orchestration.broker_reconnect import attempt_grpc_reconnect

        mock_broker = MagicMock(spec=TinkoffBroker)
        mock_broker.reconnect_client.return_value = True

        broker_router = MagicMock()
        broker_router.route.return_value = mock_broker
        alerter = MagicMock()
        stop_event = MagicMock()
        stop_event.wait.return_value = False
        reconnect_delays = [0.01]

        with patch("time.sleep") as mock_sleep:
            attempt_grpc_reconnect(
                broker_router=broker_router,
                alerter=alerter,
                stop_event=stop_event,
                reconnect_delays=reconnect_delays,
                broker_name="moex",
            )

        mock_sleep.assert_not_called()
        assert stop_event.wait.called


class TestReconcileInflightOrders:
    """Pure function for startup in-flight order reconciliation."""

    def test_skips_non_tinkoff_brokers(self) -> None:
        """Non-TinkoffBroker markets are skipped."""
        from finalayze.orchestration.broker_reconnect import reconcile_inflight_orders

        broker_router = MagicMock()
        broker_router.route.return_value = MagicMock()  # not TinkoffBroker
        circuit_breakers = {"us": MagicMock()}

        reconcile_inflight_orders(broker_router=broker_router, circuit_breakers=circuit_breakers)

        # No get_open_orders should be called on non-Tinkoff broker

    def test_cancels_stale_orders(self) -> None:
        """All open orders are cancelled on startup."""
        from decimal import Decimal

        from finalayze.execution.tinkoff_broker import TinkoffBroker
        from finalayze.orchestration.broker_reconnect import reconcile_inflight_orders

        order = MagicMock()
        order.order_id = "order-123"
        order.execution_status = "FILL"
        order.filled_quantity = Decimal(0)
        order.filled_price = Decimal(0)

        mock_broker = MagicMock(spec=TinkoffBroker)
        mock_broker.get_open_orders.return_value = [order]
        mock_broker.cancel_order_safe.return_value = True

        broker_router = MagicMock()
        broker_router.route.return_value = mock_broker
        circuit_breakers = {"moex": MagicMock()}

        reconcile_inflight_orders(broker_router=broker_router, circuit_breakers=circuit_breakers)

        mock_broker.cancel_order_safe.assert_called_once_with("order-123")

    def test_logs_partial_fills(self) -> None:
        """Partial fills (filled_quantity > 0) are logged."""
        from decimal import Decimal

        from finalayze.execution.tinkoff_broker import TinkoffBroker
        from finalayze.orchestration.broker_reconnect import reconcile_inflight_orders

        order = MagicMock()
        order.order_id = "order-456"
        order.execution_status = "PARTIAL"
        order.filled_quantity = Decimal(5)
        order.filled_price = Decimal("280.50")

        mock_broker = MagicMock(spec=TinkoffBroker)
        mock_broker.get_open_orders.return_value = [order]
        mock_broker.cancel_order_safe.return_value = True

        broker_router = MagicMock()
        broker_router.route.return_value = mock_broker
        circuit_breakers = {"moex": MagicMock()}

        # Should not raise; partial fill is just logged
        reconcile_inflight_orders(broker_router=broker_router, circuit_breakers=circuit_breakers)

        mock_broker.cancel_order_safe.assert_called_once()

    def test_no_orders_is_noop(self) -> None:
        """No open orders → no cancel calls."""
        from finalayze.execution.tinkoff_broker import TinkoffBroker
        from finalayze.orchestration.broker_reconnect import reconcile_inflight_orders

        mock_broker = MagicMock(spec=TinkoffBroker)
        mock_broker.get_open_orders.return_value = []

        broker_router = MagicMock()
        broker_router.route.return_value = mock_broker
        circuit_breakers = {"moex": MagicMock()}

        reconcile_inflight_orders(broker_router=broker_router, circuit_breakers=circuit_breakers)

        mock_broker.cancel_order_safe.assert_not_called()

    def test_get_orders_exception_continues(self) -> None:
        """Exception in get_open_orders doesn't crash — skips to next market."""
        from finalayze.execution.tinkoff_broker import TinkoffBroker
        from finalayze.orchestration.broker_reconnect import reconcile_inflight_orders

        mock_broker = MagicMock(spec=TinkoffBroker)
        mock_broker.get_open_orders.side_effect = Exception("gRPC error")

        broker_router = MagicMock()
        broker_router.route.return_value = mock_broker
        circuit_breakers = {"moex": MagicMock()}

        # Should not raise
        reconcile_inflight_orders(broker_router=broker_router, circuit_breakers=circuit_breakers)

    def test_route_exception_continues(self) -> None:
        """Exception in broker_router.route doesn't crash — skips to next market."""
        from finalayze.orchestration.broker_reconnect import reconcile_inflight_orders

        broker_router = MagicMock()
        broker_router.route.side_effect = Exception("router error")
        circuit_breakers = {"moex": MagicMock()}

        # Should not raise
        reconcile_inflight_orders(broker_router=broker_router, circuit_breakers=circuit_breakers)
