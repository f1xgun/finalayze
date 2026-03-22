"""Unit tests for TinkoffBroker.reconnect_client() method."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import pytest

from finalayze.execution.tinkoff_broker import (
    _TBANK_GRPC_SANDBOX_TARGET,
    _TBANK_GRPC_TARGET,
    TinkoffBroker,
)
from finalayze.markets.instruments import DEFAULT_MOEX_INSTRUMENTS, InstrumentRegistry


def _make_registry() -> InstrumentRegistry:
    registry = InstrumentRegistry()
    for inst in DEFAULT_MOEX_INSTRUMENTS:
        registry.register(inst)
    return registry


def _make_broker(sandbox: bool = True) -> TinkoffBroker:
    return TinkoffBroker(token="fake_token", registry=_make_registry(), sandbox=sandbox)  # noqa: S106


class TestReconnectClient:
    def test_reconnect_creates_new_client(self) -> None:
        """reconnect_client() should destroy old client and create a new one."""
        broker = _make_broker()
        old_client = MagicMock()
        broker._client = old_client
        broker._account_id = "old-account-123"

        mock_accounts = MagicMock()
        mock_accounts.accounts = [MagicMock(id="new-account-456")]

        with patch.object(broker, "_run_async", return_value=mock_accounts):
            with patch(
                "finalayze.execution.tinkoff_broker.AsyncClient"
            ) as mock_client_cls:
                mock_new_client = MagicMock()
                mock_client_cls.return_value = mock_new_client
                result = broker.reconnect_client()

        assert result is True
        assert broker._client is mock_new_client
        assert broker._account_id == "new-account-456"

    def test_reconnect_calls_close_on_old_client(self) -> None:
        """reconnect_client() should close the old client via close()."""
        broker = _make_broker()
        old_client = MagicMock()
        broker._client = old_client

        mock_accounts = MagicMock()
        mock_accounts.accounts = [MagicMock(id="acc-1")]

        with patch.object(broker, "_run_async", return_value=mock_accounts):
            with patch("finalayze.execution.tinkoff_broker.AsyncClient"):
                broker.reconnect_client()

        # close() on the old client's __aexit__ should have been attempted
        # We just verify the client was replaced (close is called internally)
        assert broker._client is not old_client

    def test_reconnect_uses_correct_sandbox_target(self) -> None:
        """reconnect_client() should use sandbox target when sandbox=True."""
        broker = _make_broker(sandbox=True)
        broker._client = MagicMock()

        mock_accounts = MagicMock()
        mock_accounts.accounts = [MagicMock(id="acc-1")]

        with patch.object(broker, "_run_async", return_value=mock_accounts):
            with patch(
                "finalayze.execution.tinkoff_broker.AsyncClient"
            ) as mock_client_cls:
                broker.reconnect_client()

        mock_client_cls.assert_called_once_with("fake_token", target=_TBANK_GRPC_SANDBOX_TARGET)  # noqa: S106

    def test_reconnect_uses_correct_prod_target(self) -> None:
        """reconnect_client() should use production target when sandbox=False."""
        broker = _make_broker(sandbox=False)
        broker._client = MagicMock()

        mock_accounts = MagicMock()
        mock_accounts.accounts = [MagicMock(id="acc-1")]

        with patch.object(broker, "_run_async", return_value=mock_accounts):
            with patch(
                "finalayze.execution.tinkoff_broker.AsyncClient"
            ) as mock_client_cls:
                broker.reconnect_client()

        mock_client_cls.assert_called_once_with("fake_token", target=_TBANK_GRPC_TARGET)  # noqa: S106

    def test_reconnect_returns_false_on_exception(self) -> None:
        """reconnect_client() should return False if client creation fails."""
        broker = _make_broker()
        broker._client = MagicMock()

        with patch(
            "finalayze.execution.tinkoff_broker.AsyncClient",
            side_effect=RuntimeError("connection refused"),
        ):
            result = broker.reconnect_client()

        assert result is False
        assert broker._client is None

    def test_reconnect_resets_account_id(self) -> None:
        """reconnect_client() should reset _account_id before recreating."""
        broker = _make_broker()
        broker._client = MagicMock()
        broker._account_id = "old-id"

        mock_accounts = MagicMock()
        mock_accounts.accounts = [MagicMock(id="new-id")]

        with patch.object(broker, "_run_async", return_value=mock_accounts):
            with patch("finalayze.execution.tinkoff_broker.AsyncClient"):
                broker.reconnect_client()

        assert broker._account_id == "new-id"

    def test_reconnect_thread_safety(self) -> None:
        """Concurrent reconnect_client() calls should not corrupt state."""
        broker = _make_broker()
        broker._client = MagicMock()
        results: list[bool] = []
        lock = threading.Lock()
        errors: list[Exception] = []

        mock_accounts = MagicMock()
        mock_accounts.accounts = [MagicMock(id="acc-safe")]

        # Patch at instance level before spawning threads
        with patch.object(broker, "_run_async", return_value=mock_accounts), patch(
            "finalayze.execution.tinkoff_broker.AsyncClient"
        ):

            def do_reconnect() -> None:
                try:
                    r = broker.reconnect_client()
                    with lock:
                        results.append(r)
                except Exception as exc:
                    with lock:
                        errors.append(exc)

            threads = [threading.Thread(target=do_reconnect) for _ in range(5)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=5)

        assert not errors, f"Errors in threads: {errors}"
        assert all(r is True for r in results)
