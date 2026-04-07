"""Unit tests for TinkoffBroker.close() structured logging."""

from __future__ import annotations

import asyncio
import contextlib
import threading
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from finalayze.execution.tinkoff_broker import TinkoffBroker

# ---------- helpers ----------

FAKE_TOKEN = "fake_token"  # noqa: S105


def _make_broker() -> TinkoffBroker:
    """Create a TinkoffBroker with mocked dependencies for close() testing."""
    registry = MagicMock()
    return TinkoffBroker(token=FAKE_TOKEN, registry=registry, sandbox=True)


def _setup_broker_with_loop(broker: TinkoffBroker) -> asyncio.AbstractEventLoop:
    """Set up a broker with a running background event loop."""
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    broker._loop = loop
    broker._loop_thread = thread
    broker._client = MagicMock()
    broker._services = MagicMock()
    return loop


# ---------- tests ----------


class TestTinkoffBrokerClose:
    def test_close_logs_warning_on_aexit_failure(self) -> None:
        """close() must log warning with resource name and error type when __aexit__ fails."""
        broker = _make_broker()
        loop = _setup_broker_with_loop(broker)

        # Make __aexit__ raise
        broker._client.__aexit__ = AsyncMock(side_effect=RuntimeError("channel broken"))

        try:
            with patch("finalayze.execution.tinkoff_broker._log") as mock_log:
                broker.close()

            mock_log.warning.assert_any_call(
                "grpc_channel_close_failed",
                resource="grpc_client",
                error_type="RuntimeError",
                error="channel broken",
            )
        finally:
            if not loop.is_closed():
                loop.call_soon_threadsafe(loop.stop)

    def test_close_logs_warning_on_loop_stop_failure(self) -> None:
        """close() must log warning when loop.stop() fails."""
        broker = _make_broker()
        loop = _setup_broker_with_loop(broker)

        # Make __aexit__ succeed but loop.stop fails
        broker._client.__aexit__ = AsyncMock(return_value=None)

        original_call_soon = loop.call_soon_threadsafe

        def patched_call_soon(callback, *args):
            raise RuntimeError("loop already stopped")

        loop.call_soon_threadsafe = patched_call_soon

        try:
            with patch("finalayze.execution.tinkoff_broker._log") as mock_log:
                broker.close()

            mock_log.warning.assert_any_call(
                "event_loop_stop_failed",
                resource="event_loop",
                error_type="RuntimeError",
                error="loop already stopped",
            )
        finally:
            with contextlib.suppress(Exception):
                original_call_soon(loop.stop)

    def test_close_cleans_up_references_even_after_errors(self) -> None:
        """_client, _services, _loop must be None even after errors."""
        broker = _make_broker()
        _setup_broker_with_loop(broker)

        broker._client.__aexit__ = AsyncMock(side_effect=RuntimeError("boom"))

        broker.close()

        assert broker._client is None
        assert broker._services is None
        assert broker._loop is None
        assert broker._loop_thread is None

    def test_close_silent_when_no_errors(self) -> None:
        """close() must not log warnings when cleanup succeeds."""
        broker = _make_broker()
        _setup_broker_with_loop(broker)

        broker._client.__aexit__ = AsyncMock(return_value=None)

        with patch("finalayze.execution.tinkoff_broker._log") as mock_log:
            broker.close()

        # No warnings should be emitted
        mock_log.warning.assert_not_called()

    def test_close_noop_when_client_is_none(self) -> None:
        """close() should be a no-op if _client is already None."""
        broker = _make_broker()
        assert broker._client is None

        with patch("finalayze.execution.tinkoff_broker._log") as mock_log:
            broker.close()  # should not raise

        mock_log.warning.assert_not_called()
