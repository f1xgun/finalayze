"""Async/gRPC event-loop runtime management for TradingLoop.

Extracted from trading_loop.py to manage:
- Persistent background event loop for non-gRPC async work (HTTP, DB, Telegram)
- Dedicated gRPC event loop to isolate PollerCompletionQueue from other async work

This prevents BlockingIOError contention from causing trading cycle drift.
"""

from __future__ import annotations

import asyncio
import threading
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from collections.abc import Callable

_log = structlog.get_logger()


class AsyncRuntime:
    """Manages persistent background event loops for async and gRPC operations.

    Owns two event loops (PUBLIC attributes):
    - async_loop: for non-gRPC async work (HTTP, DB, Telegram)
    - grpc_loop: for gRPC operations (TinkoffBroker, TinkoffFetcher)
    - async_thread: thread running async_loop
    - grpc_thread: thread running grpc_loop

    Both are created lazily on first use and run in daemon threads.
    """

    def __init__(
        self,
        grpc_loop: asyncio.AbstractEventLoop | None = None,
        on_async_loop_created: Callable[[asyncio.AbstractEventLoop], None] | None = None,
    ) -> None:
        """Initialize AsyncRuntime with optional injected gRPC loop.

        Args:
            grpc_loop: Pre-created gRPC event loop. If provided, it is used
                directly without creating a new thread. If None, one will be
                created lazily on first run_grpc call.
            on_async_loop_created: Optional callback fired exactly once when
                the async loop is lazily created (in run_async or ensure_async_loop).
                Receives the newly created loop as its only argument.
        """
        # Persistent background event loop for non-gRPC async calls (PUBLIC)
        self.async_loop: asyncio.AbstractEventLoop | None = None
        self.async_thread: threading.Thread | None = None

        # Dedicated gRPC event loop to isolate PollerCompletionQueue (PUBLIC)
        self.grpc_loop: asyncio.AbstractEventLoop | None = grpc_loop
        self.grpc_thread: threading.Thread | None = None

        # Callback fired once when async loop is lazily created
        self._on_async_loop_created = on_async_loop_created
        self._async_loop_created = False

    def ensure_async_loop(self) -> asyncio.AbstractEventLoop:
        """Lazily create and return the async event loop.

        Creates a daemon thread with its own event loop on first call.
        Fires the on_async_loop_created callback exactly once.

        Returns:
            The async event loop (never None).
        """
        if self.async_loop is None or self.async_loop.is_closed():
            loop = asyncio.new_event_loop()
            self.async_loop = loop
            thread = threading.Thread(target=loop.run_forever, daemon=True)
            thread.start()
            self.async_thread = thread
            # Fire callback exactly once
            if not self._async_loop_created and self._on_async_loop_created is not None:
                self._async_loop_created = True
                self._on_async_loop_created(loop)
        return self.async_loop

    def run_async(self, coro: Any, *, timeout: int = 30) -> Any:
        """Run an async coroutine on the persistent background event loop.

        Lazily creates a daemon thread with its own event loop on first call.

        Args:
            coro: Coroutine to execute.
            timeout: Timeout in seconds (default 30).

        Returns:
            Result of the coroutine.

        Raises:
            TimeoutError: If the coroutine exceeds the timeout.
        """
        loop = self.ensure_async_loop()
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        return future.result(timeout=timeout)

    def init_grpc_loop(self) -> asyncio.AbstractEventLoop:
        """Create a dedicated background event loop for all gRPC operations.

        Isolated from async_loop to prevent PollerCompletionQueue BlockingIOError
        from starving HTTP/DB/Telegram coroutines and causing strategy cycle drift.

        Returns:
            The newly created gRPC event loop.
        """
        loop = asyncio.new_event_loop()

        # Suppress benign BlockingIOError from gRPC PollerCompletionQueue
        def _grpc_exception_handler(
            loop: asyncio.AbstractEventLoop, context: dict[str, Any]
        ) -> None:
            exc = context.get("exception")
            if isinstance(exc, BlockingIOError):
                return  # benign EAGAIN from PollerCompletionQueue
            loop.default_exception_handler(context)

        loop.set_exception_handler(_grpc_exception_handler)
        thread = threading.Thread(target=loop.run_forever, daemon=True, name="grpc-loop")
        thread.start()
        self.grpc_loop = loop
        self.grpc_thread = thread
        return loop

    def run_grpc(self, coro: Any, *, timeout: int = 30) -> Any:
        """Run a gRPC coroutine on the dedicated gRPC event loop.

        Use this for all TinkoffBroker and TinkoffFetcher calls.
        Non-gRPC async work (HTTP, DB, Telegram) should use run_async().

        Args:
            coro: Coroutine to execute.
            timeout: Timeout in seconds (default 30).

        Returns:
            Result of the coroutine.

        Raises:
            TimeoutError: If the coroutine exceeds the timeout.
        """
        if self.grpc_loop is None or self.grpc_loop.is_closed():
            self.init_grpc_loop()
        # grpc_loop is guaranteed to be non-None here (created in init_grpc_loop)
        loop = self.grpc_loop
        assert loop is not None
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        return future.result(timeout=timeout)

    def shutdown(self) -> None:
        """Shut down both event loops idempotently.

        Stops and joins both async and gRPC loops, then resets them to None.
        """
        if self.async_loop is not None and not self.async_loop.is_closed():
            self.async_loop.call_soon_threadsafe(self.async_loop.stop)
            if self.async_thread is not None:
                self.async_thread.join(timeout=5)
        self.async_loop = None
        self.async_thread = None

        if self.grpc_loop is not None and not self.grpc_loop.is_closed():
            self.grpc_loop.call_soon_threadsafe(self.grpc_loop.stop)
            if self.grpc_thread is not None:
                self.grpc_thread.join(timeout=5)
        self.grpc_loop = None
        self.grpc_thread = None
