"""Tests for TradingLoop._attempt_grpc_reconnect non-blocking sleep (ASYNC-01)."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import pytest


class _FakeTinkoffBroker:
    """Stand-in for TinkoffBroker to satisfy isinstance checks."""

    def reconnect_client(self) -> bool:
        return True


class TestGrpcReconnectNonBlocking:
    """Verify _attempt_grpc_reconnect uses _stop_event.wait() instead of time.sleep."""

    def _make_loop(self) -> object:
        """Create a minimal TradingLoop with mocked dependencies."""
        with patch("finalayze.core.trading_loop.TradingLoop.__init__", return_value=None):
            from finalayze.core.trading_loop import TradingLoop

            loop = TradingLoop.__new__(TradingLoop)

        loop._broker_router = MagicMock()
        loop._alerter = MagicMock()
        loop._stop_event = threading.Event()
        loop._reconnect_delays = [30, 60]
        return loop

    def test_uses_stop_event_wait_not_time_sleep(self) -> None:
        """time.sleep must NOT be called; _stop_event.wait(timeout=) must be used."""
        loop = self._make_loop()

        mock_broker = MagicMock(spec=_FakeTinkoffBroker)
        mock_broker.reconnect_client.return_value = True
        loop._broker_router.route.return_value = mock_broker

        with (
            patch(
                "finalayze.execution.tinkoff_broker.TinkoffBroker",
                _FakeTinkoffBroker,
            ),
            patch.object(loop._stop_event, "wait", return_value=False) as mock_wait,
            patch("time.sleep") as mock_sleep,
        ):
            result = loop._attempt_grpc_reconnect("moex")

        assert result is True
        mock_sleep.assert_not_called()
        assert mock_wait.call_count >= 1

    def test_early_exit_when_stop_event_set(self) -> None:
        """If _stop_event is set during wait, reconnect exits early returning False."""
        loop = self._make_loop()

        mock_broker = MagicMock(spec=_FakeTinkoffBroker)
        loop._broker_router.route.return_value = mock_broker

        with (
            patch(
                "finalayze.execution.tinkoff_broker.TinkoffBroker",
                _FakeTinkoffBroker,
            ),
            patch.object(loop._stop_event, "wait", return_value=True),
        ):
            result = loop._attempt_grpc_reconnect("moex")

        assert result is False
        # reconnect_client should NOT be called since we exited early
        mock_broker.reconnect_client.assert_not_called()

    def test_successful_reconnect_after_wait(self) -> None:
        """Successful reconnect returns True after wait completes normally."""
        loop = self._make_loop()

        mock_broker = MagicMock(spec=_FakeTinkoffBroker)
        mock_broker.reconnect_client.return_value = True
        loop._broker_router.route.return_value = mock_broker

        with (
            patch(
                "finalayze.execution.tinkoff_broker.TinkoffBroker",
                _FakeTinkoffBroker,
            ),
            patch.object(loop._stop_event, "wait", return_value=False),
        ):
            result = loop._attempt_grpc_reconnect("moex")

        assert result is True
        mock_broker.reconnect_client.assert_called_once()
