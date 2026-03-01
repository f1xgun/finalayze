"""Exponential backoff retry policy for transient broker errors (Layer 5)."""

from __future__ import annotations

import asyncio
import random
import time
from typing import TYPE_CHECKING, TypeVar

import httpx
import structlog

from finalayze.core.exceptions import InstrumentNotFoundError, InsufficientFundsError

try:
    import grpc  # type: ignore[import-untyped]

    _GRPC_RETRYABLE: tuple[type[Exception], ...] = (grpc.RpcError,)
except ImportError:
    _GRPC_RETRYABLE: tuple[type[Exception], ...] = ()  # type: ignore[no-redef]

_RETRYABLE_EXCEPTIONS: tuple[type[Exception], ...] = (
    ConnectionError,
    TimeoutError,
    httpx.ConnectError,
    httpx.TimeoutException,
    *_GRPC_RETRYABLE,
)

if TYPE_CHECKING:
    from collections.abc import Callable

_log = structlog.get_logger()

_T = TypeVar("_T")

_DEFAULT_MAX_RETRIES = 3
_DEFAULT_BASE_DELAY = 1.0
_DEFAULT_MAX_DELAY = 30.0
_JITTER_MIN = 0.5
_JITTER_MAX = 1.5


class RetryPolicy:
    """Exponential backoff with jitter for transient broker errors.

    Retries on ``ConnectionError``, ``TimeoutError``, ``httpx.ConnectError``,
    and ``httpx.TimeoutException``. Non-retryable exceptions (e.g.
    ``InsufficientFundsError``, ``InstrumentNotFoundError``) are raised
    immediately.
    """

    def __init__(
        self,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        base_delay: float = _DEFAULT_BASE_DELAY,
        max_delay: float = _DEFAULT_MAX_DELAY,
        non_retryable: tuple[type[Exception], ...] = (
            InsufficientFundsError,
            InstrumentNotFoundError,
        ),
    ) -> None:
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._max_delay = max_delay
        self._non_retryable = non_retryable

    def execute(self, fn: Callable[[], _T]) -> _T:
        """Execute ``fn`` with synchronous retry logic."""
        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                return fn()
            except self._non_retryable:
                raise
            except _RETRYABLE_EXCEPTIONS as exc:
                last_exc = exc
                if attempt < self._max_retries:
                    delay = self._compute_delay(attempt)
                    _log.warning(
                        "Retry %d/%d after %.2fs: %s",
                        attempt + 1,
                        self._max_retries,
                        delay,
                        exc,
                    )
                    time.sleep(delay)
            except Exception:
                raise

        assert last_exc is not None
        msg = f"All {self._max_retries} retries exhausted"
        raise type(last_exc)(msg) from last_exc

    async def aexecute(self, fn: Callable[[], _T]) -> _T:
        """Execute ``fn`` with async retry logic (non-blocking sleep)."""
        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                return fn()
            except self._non_retryable:
                raise
            except _RETRYABLE_EXCEPTIONS as exc:
                last_exc = exc
                if attempt < self._max_retries:
                    delay = self._compute_delay(attempt)
                    _log.warning(
                        "Retry %d/%d after %.2fs: %s",
                        attempt + 1,
                        self._max_retries,
                        delay,
                        exc,
                    )
                    await asyncio.sleep(delay)
            except Exception:
                raise

        assert last_exc is not None
        msg = f"All {self._max_retries} retries exhausted"
        raise type(last_exc)(msg) from last_exc

    def _compute_delay(self, attempt: int) -> float:
        """Compute delay with exponential backoff and jitter."""
        jitter = random.uniform(_JITTER_MIN, _JITTER_MAX)  # noqa: S311
        delay = self._base_delay * float(2**attempt) * jitter
        return min(delay, self._max_delay)
