"""Clock abstraction for real-time vs simulated time (Layer 0).

Used in sandbox mode for historical replay at configurable speed.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Protocol


class Clock(Protocol):
    """Protocol for time providers.

    Any class that implements ``now() -> datetime`` satisfies this protocol.
    """

    def now(self) -> datetime:
        """Return the current time as a UTC-aware datetime."""
        ...  # pragma: no cover


class RealClock:
    """Returns the current UTC time from the system clock."""

    def now(self) -> datetime:
        """Return the current UTC time."""
        return datetime.now(tz=UTC)


class SimulatedClock:
    """A manually-advanced clock for backtesting and sandbox replay."""

    def __init__(self, start: datetime) -> None:
        if start.tzinfo is None:
            msg = "start datetime must be timezone-aware"
            raise ValueError(msg)
        self._current = start

    def now(self) -> datetime:
        """Return the current simulated time."""
        return self._current

    def advance(self, seconds: float = 0.0, *, delta: timedelta | None = None) -> None:
        """Advance the simulated clock by ``seconds`` or a ``timedelta``.

        Args:
            seconds: Number of seconds to advance (ignored when ``delta`` is given).
            delta: Optional timedelta to advance by. Takes precedence over ``seconds``.
        """
        if delta is not None:
            self._current += delta
        else:
            self._current += timedelta(seconds=seconds)
