"""Unit tests for the token-bucket RateLimiter (TDD — RED phase)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from finalayze.core.exceptions import ConfigurationError
from finalayze.data.rate_limiter import RateLimiter

# ---------------------------------------------------------------------------
# Constants (no magic numbers per ruff PLR2004)
# ---------------------------------------------------------------------------
RATE_10_PER_SECOND = 10.0
CAPACITY_10 = 10.0
TOKENS_WITHIN_LIMIT = 5
TOKENS_EXCEEDING_LIMIT = 11
SLEEP_CALL_COUNT = 1
MONOTONIC_T0 = 0.0
MONOTONIC_T1 = 1.0


def _make_limiter_with_mock_time(
    mock_time: MagicMock,
    name: str = "test",
    rate: float = RATE_10_PER_SECOND,
    capacity: float | None = None,
) -> RateLimiter:
    """Create a RateLimiter while time.monotonic is already mocked at T0."""
    mock_time.monotonic.return_value = MONOTONIC_T0
    mock_time.sleep = MagicMock()
    return RateLimiter(name=name, rate=rate, capacity=capacity)


class TestRateLimiterBasic:
    """Basic functionality of RateLimiter."""

    def test_acquire_within_limit_does_not_block(self) -> None:
        """Acquiring tokens within capacity should not call sleep."""
        with patch("finalayze.data.rate_limiter.time") as mock_time:
            limiter = _make_limiter_with_mock_time(mock_time)
            for _ in range(TOKENS_WITHIN_LIMIT):
                limiter.acquire()
            mock_time.sleep.assert_not_called()

    def test_acquire_exceeds_limit_waits(self) -> None:
        """Acquiring more tokens than capacity should call time.sleep."""
        with patch("finalayze.data.rate_limiter.time") as mock_time:
            limiter = _make_limiter_with_mock_time(mock_time, capacity=CAPACITY_10)
            # Drain all tokens first (capacity=10), then the 11th requires sleep
            for _ in range(TOKENS_EXCEEDING_LIMIT):
                limiter.acquire()
            assert mock_time.sleep.call_count >= SLEEP_CALL_COUNT

    def test_rate_limiter_name(self) -> None:
        """RateLimiter must expose a name attribute."""
        limiter = RateLimiter(name="my-limiter", rate=RATE_10_PER_SECOND)
        assert limiter.name == "my-limiter"

    def test_rate_limiter_rejects_zero_rate(self) -> None:
        """Creating a RateLimiter with rate=0 must raise ConfigurationError."""
        with pytest.raises(ConfigurationError):
            RateLimiter(name="bad", rate=0)

    def test_rate_limiter_rejects_negative_rate(self) -> None:
        """Creating a RateLimiter with rate<0 must raise ConfigurationError."""
        with pytest.raises(ConfigurationError):
            RateLimiter(name="bad", rate=-1.0)

    def test_acquire_returns_self_for_context_manager(self) -> None:
        """RateLimiter can be used as a context manager with `with limiter:`."""
        with patch("finalayze.data.rate_limiter.time") as mock_time:
            limiter = _make_limiter_with_mock_time(mock_time, name="ctx")
            with limiter as ctx:
                assert ctx is limiter

    def test_multiple_limiters_independent(self) -> None:
        """Two RateLimiter instances must not share token state."""
        with patch("finalayze.data.rate_limiter.time") as mock_time:
            limiter_a = _make_limiter_with_mock_time(mock_time, name="a", capacity=CAPACITY_10)
            limiter_b = _make_limiter_with_mock_time(mock_time, name="b", capacity=CAPACITY_10)

            # Drain limiter_a completely
            for _ in range(int(CAPACITY_10)):
                limiter_a.acquire()

            # limiter_b should still have full capacity — no sleep needed
            mock_time.sleep.reset_mock()
            limiter_b.acquire()
            mock_time.sleep.assert_not_called()


class TestRateLimiterRefill:
    """Token refill behaviour over time."""

    def test_tokens_refill_over_time(self) -> None:
        """After elapsed time, tokens should refill up to capacity."""
        with patch("finalayze.data.rate_limiter.time") as mock_time:
            limiter = _make_limiter_with_mock_time(mock_time, capacity=CAPACITY_10)

            # Drain all tokens at T0
            for _ in range(int(CAPACITY_10)):
                limiter.acquire()

            # 1 second later, should have refilled 10 tokens — no sleep needed
            mock_time.monotonic.return_value = MONOTONIC_T1
            mock_time.sleep.reset_mock()
            limiter.acquire()
            mock_time.sleep.assert_not_called()
