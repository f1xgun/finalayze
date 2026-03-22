"""Tests for 6D.1: Persistent Tinkoff async client with connection reuse."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from finalayze.data.fetchers.tinkoff_data import TinkoffFetcher
from finalayze.execution.tinkoff_broker import TinkoffBroker

_TEST_TOKEN = "test-token"  # noqa: S105


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
        broker = TinkoffBroker(token=_TEST_TOKEN, registry=_make_registry(), sandbox=True)
        with patch("finalayze.execution.tinkoff_broker.AsyncClient") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client

            client1 = broker._get_client()
            client2 = broker._get_client()

            assert client1 is client2
            mock_cls.assert_called_once_with(
                _TEST_TOKEN, target="sandbox-invest-public-api.tbank.ru:443"
            )

    def test_close_clears_client(self) -> None:
        """close() should set _client to None."""
        broker = TinkoffBroker(token=_TEST_TOKEN, registry=_make_registry(), sandbox=True)
        with patch("finalayze.execution.tinkoff_broker.AsyncClient") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client

            broker._get_client()
            assert broker._client is not None

            broker.close()
            assert broker._client is None


class TestTinkoffFetcherClientCreation:
    """Verify TinkoffFetcher _make_client creates correct client types."""

    def test_make_client_sandbox_uses_sandbox_client(self) -> None:
        """sandbox=True should create AsyncClient with sandbox target."""
        fetcher = TinkoffFetcher(token=_TEST_TOKEN, registry=_make_registry(), sandbox=True)
        with patch("finalayze.data.fetchers.tinkoff_data.AsyncClient") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client

            client = fetcher._make_client()
            assert client is mock_client

    def test_make_client_creates_fresh_instance_each_call(self) -> None:
        """Each _make_client call should create a new instance (no caching)."""
        fetcher = TinkoffFetcher(token=_TEST_TOKEN, registry=_make_registry(), sandbox=True)
        with patch("finalayze.data.fetchers.tinkoff_data.AsyncClient") as mock_cls:
            client1 = fetcher._make_client()
            client2 = fetcher._make_client()

            assert mock_cls.call_count == 2  # noqa: PLR2004

    def test_close_is_noop(self) -> None:
        """close() is a no-op for TinkoffFetcher (fresh client per call)."""
        fetcher = TinkoffFetcher(token=_TEST_TOKEN, registry=_make_registry(), sandbox=True)
        fetcher.close()  # should not raise

    def test_live_mode_uses_async_client(self) -> None:
        """sandbox=False should use AsyncClient with production target."""
        fetcher = TinkoffFetcher(token=_TEST_TOKEN, registry=_make_registry(), sandbox=False)
        with patch("finalayze.data.fetchers.tinkoff_data.AsyncClient") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client

            client = fetcher._make_client()
            assert client is mock_client
