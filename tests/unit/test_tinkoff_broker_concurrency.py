"""Tests for TinkoffBroker concurrency safety (CONC-02, CONC-03).

Verifies:
- asyncio.Lock used in async code paths (not threading.Lock)
- threading.Lock used only in sync code paths (_get_client, _loop_init_lock)
- Event loop creation guarded by double-check locking pattern
"""

from __future__ import annotations

import asyncio
import inspect
import threading
from unittest.mock import MagicMock

from finalayze.execution.tinkoff_broker import TinkoffBroker


def _make_broker() -> TinkoffBroker:
    """Create a TinkoffBroker with dummy args for inspection."""
    registry = MagicMock()
    return TinkoffBroker(token="test-token", registry=registry, sandbox=True)  # noqa: S106


class TestAsyncLockType:
    """CONC-02: _get_services_async must use asyncio.Lock, not threading.Lock."""

    def test_async_lock_is_asyncio_lock(self) -> None:
        broker = _make_broker()
        assert type(broker._async_lock).__name__ == "Lock"
        assert type(broker._async_lock).__module__ == "asyncio.locks"

    def test_client_lock_is_threading_lock(self) -> None:
        """Sync _get_client must still use threading.Lock."""
        broker = _make_broker()
        lock_name = type(broker._client_lock).__name__.lower()
        assert "_rlock" in lock_name or "lock" in lock_name
        # threading.Lock() returns a _thread.lock object
        assert hasattr(broker._client_lock, "acquire")
        assert hasattr(broker._client_lock, "release")

    def test_get_services_async_uses_async_lock(self) -> None:
        """Verify source code of _get_services_async contains async with self._async_lock."""
        source = inspect.getsource(TinkoffBroker._get_services_async)
        assert "async with self._async_lock" in source

    def test_no_threading_lock_in_async_methods(self) -> None:
        """No async method should use threading.Lock (self._client_lock)."""
        for name, method in inspect.getmembers(TinkoffBroker, predicate=inspect.isfunction):
            if asyncio.iscoroutinefunction(method):
                source = inspect.getsource(method)
                assert "self._client_lock" not in source, (
                    f"Async method {name} uses threading lock self._client_lock"
                )


class TestLoopInitLock:
    """CONC-03: _run_async must guard event loop creation with threading.Lock."""

    def test_loop_init_lock_is_threading_lock(self) -> None:
        broker = _make_broker()
        assert hasattr(broker._loop_init_lock, "acquire")
        assert hasattr(broker._loop_init_lock, "release")
        # Must NOT be an asyncio.Lock
        assert type(broker._loop_init_lock).__module__ != "asyncio.locks"

    def test_run_async_uses_loop_init_lock(self) -> None:
        """Verify source code of _run_async contains with self._loop_init_lock."""
        source = inspect.getsource(TinkoffBroker._run_async)
        assert "with self._loop_init_lock" in source

    def test_run_async_has_double_check_pattern(self) -> None:
        """_run_async should have double-check (check outside lock, check inside lock)."""
        source = inspect.getsource(TinkoffBroker._run_async)
        # Should have two checks for self._loop
        loop_checks = source.count("self._loop is None")
        assert loop_checks >= 2, f"Expected >= 2 loop None checks, got {loop_checks}"
