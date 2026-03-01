"""Tests for 6D.1: Persistent Tinkoff async client with connection reuse."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from finalayze.data.fetchers.tinkoff_data import TinkoffFetcher
from finalayze.execution.tinkoff_broker import TinkoffBroker


def _make_registry() -> MagicMock:
    registry = MagicMock()
    instr = MagicMock()
    instr.figi = "BBG000B9XRY4"
    instr.lot_size = 1
    instr.symbol = "SBER"
    registry.get.return_value = instr
    return registry


class TestTinkoffBrokerPersistentClient:
    """Verify that TinkoffBroker reuses a single client instance."""

    def test_get_client_returns_same_instance(self) -> None:
        """Two calls to _get_client should return the same object."""
        broker = TinkoffBroker(
            token="test-token", registry=_make_registry(), sandbox=True
        )
        with patch(
            "finalayze.execution.tinkoff_broker.AsyncSandboxClient"
        ) as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client

            client1 = broker._get_client()
            client2 = broker._get_client()

            assert client1 is client2
            mock_cls.assert_called_once_with("test-token")

    def test_close_clears_client(self) -> None:
        """close() should set _client to None."""
        broker = TinkoffBroker(
            token="test-token", registry=_make_registry(), sandbox=True
        )
        with patch(
            "finalayze.execution.tinkoff_broker.AsyncSandboxClient"
        ) as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client

            broker._get_client()
            assert broker._client is not None

            broker.close()
            assert broker._client is None


class TestTinkoffFetcherPersistentClient:
    """Verify that TinkoffFetcher reuses a single client instance."""

    def test_get_client_returns_same_instance(self) -> None:
        """Two calls to _get_client should return the same object."""
        fetcher = TinkoffFetcher(
            token="test-token", registry=_make_registry(), sandbox=True
        )
        with patch(
            "finalayze.data.fetchers.tinkoff_data.AsyncSandboxClient"
        ) as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client

            client1 = fetcher._get_client()
            client2 = fetcher._get_client()

            assert client1 is client2
            mock_cls.assert_called_once_with("test-token")

    def test_close_clears_client(self) -> None:
        """close() should set _client to None."""
        fetcher = TinkoffFetcher(
            token="test-token", registry=_make_registry(), sandbox=True
        )
        with patch(
            "finalayze.data.fetchers.tinkoff_data.AsyncSandboxClient"
        ) as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client

            fetcher._get_client()
            assert fetcher._client is not None

            fetcher.close()
            assert fetcher._client is None

    def test_live_mode_uses_async_client(self) -> None:
        """sandbox=False should use AsyncClient, not AsyncSandboxClient."""
        fetcher = TinkoffFetcher(
            token="test-token", registry=_make_registry(), sandbox=False
        )
        with patch(
            "finalayze.data.fetchers.tinkoff_data.AsyncClient"
        ) as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client

            client = fetcher._get_client()
            assert client is mock_client
            mock_cls.assert_called_once_with("test-token")
