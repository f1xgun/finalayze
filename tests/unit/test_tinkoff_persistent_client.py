"""Tests for persistent gRPC channel in TinkoffFetcher and TinkoffBroker."""

from __future__ import annotations

import threading
from unittest.mock import AsyncMock, MagicMock, patch

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


# ── TinkoffBroker tests (unchanged reference) ────────────────────────────────


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


# ── TinkoffFetcher persistent channel tests ───────────────────────────────────


class TestTinkoffFetcherPersistentChannel:
    """Verify TinkoffFetcher reuses a persistent background event loop."""

    def test_run_async_creates_loop_once(self) -> None:
        """_run_async should create a background event loop that persists across calls."""
        fetcher = TinkoffFetcher(token=_TEST_TOKEN, registry=_make_registry(), sandbox=True)

        async def _dummy() -> str:
            return "ok"

        result1 = fetcher._run_async(_dummy())
        loop_after_first = fetcher._loop

        result2 = fetcher._run_async(_dummy())
        loop_after_second = fetcher._loop

        assert result1 == "ok"
        assert result2 == "ok"
        assert loop_after_first is loop_after_second
        assert loop_after_first is not None
        assert isinstance(fetcher._loop_thread, threading.Thread)

        fetcher.close()

    def test_close_stops_loop_and_clears_state(self) -> None:
        """close() should stop the background event loop and nil out state."""
        fetcher = TinkoffFetcher(token=_TEST_TOKEN, registry=_make_registry(), sandbox=True)

        async def _dummy() -> str:
            return "ok"

        fetcher._run_async(_dummy())
        assert fetcher._loop is not None

        fetcher.close()

        assert fetcher._loop is None
        assert fetcher._loop_thread is None
        assert fetcher._client is None
        assert fetcher._services is None

    def test_recovery_after_close(self) -> None:
        """After close(), a new _run_async call creates a fresh event loop."""
        fetcher = TinkoffFetcher(token=_TEST_TOKEN, registry=_make_registry(), sandbox=True)

        async def _dummy() -> str:
            return "ok"

        fetcher._run_async(_dummy())
        old_loop = fetcher._loop
        fetcher.close()

        result = fetcher._run_async(_dummy())
        assert result == "ok"
        assert fetcher._loop is not old_loop
        assert fetcher._loop is not None

        fetcher.close()

    def test_make_client_called_once_for_multiple_fetches(self) -> None:
        """_make_client should be called only once when _get_services_async caches services."""
        fetcher = TinkoffFetcher(token=_TEST_TOKEN, registry=_make_registry(), sandbox=True)

        mock_services = MagicMock()
        mock_response = MagicMock()
        mock_response.candles = []
        mock_services.market_data.get_candles = AsyncMock(return_value=mock_response)

        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_services)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch.object(fetcher, "_make_client", return_value=mock_client) as mock_make:
            # First call -- creates client
            fetcher._run_async(fetcher._get_services_async())
            # Second call -- reuses cached services
            fetcher._run_async(fetcher._get_services_async())

            mock_make.assert_called_once()

        fetcher.close()


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

    def test_close_tears_down_state(self) -> None:
        """close() should clean up persistent state."""
        fetcher = TinkoffFetcher(token=_TEST_TOKEN, registry=_make_registry(), sandbox=True)
        fetcher.close()  # should not raise even with no active loop

    def test_live_mode_uses_async_client(self) -> None:
        """sandbox=False should use AsyncClient with production target."""
        fetcher = TinkoffFetcher(token=_TEST_TOKEN, registry=_make_registry(), sandbox=False)
        with patch("finalayze.data.fetchers.tinkoff_data.AsyncClient") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client

            client = fetcher._make_client()
            assert client is mock_client


class TestTinkoffFetcherBondMethodsPersistent:
    """Verify bond methods use the persistent channel, not asyncio.run()."""

    def test_fetch_bond_candles_uses_run_async(self) -> None:
        """fetch_bond_candles should use _run_async, not asyncio.run."""
        import inspect

        source = inspect.getsource(TinkoffFetcher.fetch_bond_candles)
        assert "self._run_async" in source, "fetch_bond_candles must use _run_async"
        assert "asyncio.run(" not in source, "fetch_bond_candles must not use asyncio.run"

    def test_fetch_bond_info_uses_run_async(self) -> None:
        """fetch_bond_info should use _run_async, not asyncio.run."""
        import inspect

        source = inspect.getsource(TinkoffFetcher.fetch_bond_info)
        assert "self._run_async" in source, "fetch_bond_info must use _run_async"
        assert "asyncio.run(" not in source, "fetch_bond_info must not use asyncio.run"

    def test_fetch_bond_coupons_uses_run_async(self) -> None:
        """fetch_bond_coupons should use _run_async, not asyncio.run."""
        import inspect

        source = inspect.getsource(TinkoffFetcher.fetch_bond_coupons)
        assert "self._run_async" in source, "fetch_bond_coupons must use _run_async"
        assert "asyncio.run(" not in source, "fetch_bond_coupons must not use asyncio.run"

    def test_fetch_accrued_interest_uses_run_async(self) -> None:
        """fetch_accrued_interest should use _run_async, not asyncio.run."""
        import inspect

        source = inspect.getsource(TinkoffFetcher.fetch_accrued_interest)
        assert "self._run_async" in source, "fetch_accrued_interest must use _run_async"
        assert "asyncio.run(" not in source, "fetch_accrued_interest must not use asyncio.run"

    def test_bond_async_methods_use_get_services(self) -> None:
        """Bond async methods should use _get_services_async, not _make_client."""
        import inspect

        for method_name in [
            "_fetch_bond_candles_async",
            "_fetch_bond_info_async",
            "_fetch_bond_coupons_async",
            "_fetch_accrued_interest_async",
            "_fetch_amortization_async",
        ]:
            method = getattr(TinkoffFetcher, method_name)
            source = inspect.getsource(method)
            assert "_get_services_async" in source, (
                f"{method_name} must use _get_services_async"
            )
            assert "self._make_client()" not in source, (
                f"{method_name} must not create its own client"
            )
