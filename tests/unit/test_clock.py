"""Unit tests for the Clock abstraction (TDD — RED phase)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from finalayze.core.clock import Clock, RealClock, SimulatedClock

# ---------------------------------------------------------------------------
# Constants (no magic numbers per ruff PLR2004)
# ---------------------------------------------------------------------------
ADVANCE_SECONDS = 60.0
ADVANCE_HOURS = 1
START_YEAR = 2024
START_MONTH = 1
START_DAY = 15
START_HOUR = 10
START_MINUTE = 30
NEGATIVE_SECONDS = -100.0


def _start_dt() -> datetime:
    """Return a fixed UTC-aware start datetime for tests."""
    return datetime(START_YEAR, START_MONTH, START_DAY, START_HOUR, START_MINUTE, tzinfo=UTC)


class TestRealClock:
    """Tests for RealClock."""

    def test_real_clock_returns_utc_now(self) -> None:
        """RealClock.now() must return a UTC-aware datetime close to the current time."""
        clock = RealClock()
        before = datetime.now(tz=UTC)
        result = clock.now()
        after = datetime.now(tz=UTC)
        assert result.tzinfo is not None
        assert before <= result <= after

    def test_real_clock_is_utc_aware(self) -> None:
        """RealClock.now() must return a timezone-aware datetime."""
        clock = RealClock()
        result = clock.now()
        assert result.tzinfo is not None


class TestSimulatedClock:
    """Tests for SimulatedClock."""

    def test_simulated_clock_returns_start(self) -> None:
        """SimulatedClock(start).now() returns the start datetime."""
        start = _start_dt()
        clock = SimulatedClock(start=start)
        assert clock.now() == start

    def test_simulated_clock_advance_by_seconds(self) -> None:
        """advance(seconds=60) moves clock forward by 60 seconds."""
        start = _start_dt()
        clock = SimulatedClock(start=start)
        clock.advance(seconds=ADVANCE_SECONDS)
        expected = start + timedelta(seconds=ADVANCE_SECONDS)
        assert clock.now() == expected

    def test_simulated_clock_advance_by_timedelta(self) -> None:
        """advance(delta=timedelta(hours=1)) moves clock forward by 1 hour."""
        start = _start_dt()
        clock = SimulatedClock(start=start)
        delta = timedelta(hours=ADVANCE_HOURS)
        clock.advance(delta=delta)
        expected = start + delta
        assert clock.now() == expected

    def test_simulated_clock_is_frozen_between_advances(self) -> None:
        """Two consecutive calls to now() without advance return the same value."""
        clock = SimulatedClock(start=_start_dt())
        first = clock.now()
        second = clock.now()
        assert first == second

    def test_simulated_clock_rejects_naive_datetime(self) -> None:
        """SimulatedClock must raise ValueError for a naive (no-tzinfo) datetime."""
        # Construct an aware datetime then strip tzinfo to produce a naive one
        naive = _start_dt().replace(tzinfo=None)
        with pytest.raises(ValueError, match="timezone-aware"):
            SimulatedClock(start=naive)

    def test_simulated_clock_multiple_advances(self) -> None:
        """Multiple advances accumulate correctly."""
        start = _start_dt()
        clock = SimulatedClock(start=start)
        clock.advance(seconds=ADVANCE_SECONDS)
        clock.advance(seconds=ADVANCE_SECONDS)
        expected = start + timedelta(seconds=ADVANCE_SECONDS * 2)
        assert clock.now() == expected

    def test_advance_negative_seconds_raises(self) -> None:
        """advance(seconds=-100) must raise ValueError."""
        clock = SimulatedClock(start=_start_dt())
        with pytest.raises(ValueError, match="negative"):
            clock.advance(seconds=NEGATIVE_SECONDS)

    def test_advance_negative_timedelta_raises(self) -> None:
        """advance(delta=timedelta(seconds=-100)) must raise ValueError."""
        clock = SimulatedClock(start=_start_dt())
        with pytest.raises(ValueError, match="negative"):
            clock.advance(delta=timedelta(seconds=NEGATIVE_SECONDS))

    def test_advance_zero_seconds_is_allowed(self) -> None:
        """advance(seconds=0) is valid and does not move the clock."""
        start = _start_dt()
        clock = SimulatedClock(start=start)
        clock.advance(seconds=0.0)
        assert clock.now() == start


class TestClockProtocol:
    """Verify both clocks satisfy the Clock protocol."""

    def test_real_clock_satisfies_protocol(self) -> None:
        """RealClock must be usable wherever Clock is expected."""
        clock: Clock = RealClock()
        result = clock.now()
        assert isinstance(result, datetime)

    def test_simulated_clock_satisfies_protocol(self) -> None:
        """SimulatedClock must be usable wherever Clock is expected."""
        clock: Clock = SimulatedClock(start=_start_dt())
        result = clock.now()
        assert isinstance(result, datetime)

    def test_clock_protocol_is_runtime_checkable(self) -> None:
        """isinstance(obj, Clock) must not raise TypeError (runtime_checkable)."""
        real = RealClock()
        simulated = SimulatedClock(start=_start_dt())
        assert isinstance(real, Clock)
        assert isinstance(simulated, Clock)
