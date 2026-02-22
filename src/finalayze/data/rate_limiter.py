"""Token-bucket rate limiter for API calls (Layer 2)."""

from __future__ import annotations

import time

from finalayze.core.exceptions import ConfigurationError


class RateLimiter:
    """Token bucket rate limiter.

    Limits calls to `rate` per second with a burst capacity of `capacity`.
    Thread-safe NOT guaranteed — designed for single-threaded async use.
    """

    def __init__(self, name: str, rate: float, capacity: float | None = None) -> None:
        if rate <= 0:
            msg = f"Rate must be positive, got {rate}"
            raise ConfigurationError(msg)
        self._name = name
        self._rate = rate  # tokens per second
        self._capacity = capacity if capacity is not None else rate
        self._tokens = self._capacity
        self._last_refill = time.monotonic()

    @property
    def name(self) -> str:
        """Return the limiter's name."""
        return self._name

    def acquire(self, tokens: float = 1.0) -> None:
        """Block until `tokens` tokens are available."""
        self._refill()
        if self._tokens >= tokens:
            self._tokens -= tokens
            return
        # Need to wait for enough tokens
        deficit = tokens - self._tokens
        wait_time = deficit / self._rate
        time.sleep(wait_time)
        self._tokens = 0.0
        self._last_refill = time.monotonic()

    def __enter__(self) -> RateLimiter:
        """Acquire one token and enter the context."""
        self.acquire()
        return self

    def __exit__(self, *args: object) -> None:
        """Exit context — nothing to release in a rate limiter."""

    def _refill(self) -> None:
        """Lazily refill tokens based on elapsed wall-clock time."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        new_tokens = elapsed * self._rate
        self._tokens = min(self._capacity, self._tokens + new_tokens)
        self._last_refill = now
