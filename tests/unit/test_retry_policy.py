"""Tests for RetryPolicy (execution/retry.py)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from finalayze.core.exceptions import InstrumentNotFoundError, InsufficientFundsError
from finalayze.execution.retry import RetryPolicy

_MAX_RETRIES = 3


class TestRetrySuccess:
    """Test that successful calls are returned immediately."""

    def test_success_on_first_try(self) -> None:
        policy = RetryPolicy(max_retries=_MAX_RETRIES)
        fn = MagicMock(return_value="ok")
        result = policy.execute(fn)
        assert result == "ok"
        fn.assert_called_once()

    def test_success_after_transient_failure(self) -> None:
        policy = RetryPolicy(max_retries=_MAX_RETRIES, base_delay=0.001)
        fn = MagicMock(side_effect=[ConnectionError("down"), "recovered"])
        with patch("finalayze.execution.retry.time.sleep"):
            result = policy.execute(fn)
        assert result == "recovered"
        assert fn.call_count == 2


class TestRetryExhausted:
    """Test behavior when all retries are exhausted."""

    def test_max_retries_exhausted(self) -> None:
        policy = RetryPolicy(max_retries=2, base_delay=0.001)
        fn = MagicMock(side_effect=ConnectionError("always down"))
        with (
            patch("finalayze.execution.retry.time.sleep"),
            pytest.raises(ConnectionError, match="retries exhausted"),
        ):
            policy.execute(fn)
        expected_calls = 3  # initial + 2 retries
        assert fn.call_count == expected_calls

    def test_timeout_error_retried(self) -> None:
        policy = RetryPolicy(max_retries=1, base_delay=0.001)
        fn = MagicMock(side_effect=TimeoutError("timeout"))
        with (
            patch("finalayze.execution.retry.time.sleep"),
            pytest.raises(TimeoutError, match="retries exhausted"),
        ):
            policy.execute(fn)
        expected_calls = 2  # initial + 1 retry
        assert fn.call_count == expected_calls


class TestNonRetryable:
    """Test non-retryable exceptions are raised immediately."""

    def test_insufficient_funds_not_retried(self) -> None:
        policy = RetryPolicy(max_retries=_MAX_RETRIES)
        fn = MagicMock(side_effect=InsufficientFundsError("no funds"))
        with pytest.raises(InsufficientFundsError, match="no funds"):
            policy.execute(fn)
        fn.assert_called_once()

    def test_instrument_not_found_not_retried(self) -> None:
        policy = RetryPolicy(max_retries=_MAX_RETRIES)
        fn = MagicMock(side_effect=InstrumentNotFoundError("not found"))
        with pytest.raises(InstrumentNotFoundError, match="not found"):
            policy.execute(fn)
        fn.assert_called_once()

    def test_non_transient_exception_not_retried(self) -> None:
        """Non-transient, non-excluded exceptions are raised immediately."""
        policy = RetryPolicy(max_retries=_MAX_RETRIES)
        fn = MagicMock(side_effect=ValueError("bad value"))
        with pytest.raises(ValueError, match="bad value"):
            policy.execute(fn)
        fn.assert_called_once()


class TestBackoffTiming:
    """Test exponential backoff with jitter."""

    def test_delay_increases_exponentially(self) -> None:
        policy = RetryPolicy(max_retries=_MAX_RETRIES, base_delay=1.0, max_delay=30.0)
        delays: list[float] = []
        for attempt in range(4):
            delay = policy._compute_delay(attempt)
            delays.append(delay)

        # Each delay should generally increase (with jitter)
        # delay = base * 2^attempt * jitter(0.5, 1.5)
        # attempt 0: 1 * 1 * [0.5, 1.5] = [0.5, 1.5]
        # attempt 1: 1 * 2 * [0.5, 1.5] = [1.0, 3.0]
        # attempt 2: 1 * 4 * [0.5, 1.5] = [2.0, 6.0]
        min_first = 0.5
        max_third = 6.0
        assert delays[0] >= min_first
        assert delays[2] <= max_third

    def test_delay_capped_at_max(self) -> None:
        policy = RetryPolicy(max_retries=10, base_delay=10.0, max_delay=5.0)
        delay = policy._compute_delay(5)
        assert delay <= 5.0


class TestAsyncRetry:
    """Test async retry variant."""

    @pytest.mark.asyncio
    async def test_async_success(self) -> None:
        policy = RetryPolicy(max_retries=_MAX_RETRIES)
        fn = MagicMock(return_value="async_ok")
        result = await policy.aexecute(fn)
        assert result == "async_ok"

    @pytest.mark.asyncio
    async def test_async_retry_on_connection_error(self) -> None:
        policy = RetryPolicy(max_retries=2, base_delay=0.001)
        fn = MagicMock(side_effect=[ConnectionError("down"), "recovered"])
        with patch("finalayze.execution.retry.asyncio.sleep"):
            result = await policy.aexecute(fn)
        assert result == "recovered"

    @pytest.mark.asyncio
    async def test_async_non_retryable(self) -> None:
        policy = RetryPolicy(max_retries=_MAX_RETRIES)
        fn = MagicMock(side_effect=InsufficientFundsError("no funds"))
        with pytest.raises(InsufficientFundsError):
            await policy.aexecute(fn)
        fn.assert_called_once()
