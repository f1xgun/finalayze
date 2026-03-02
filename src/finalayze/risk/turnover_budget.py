"""Turnover budget: limits trading frequency per symbol per month (Layer 4)."""

from __future__ import annotations

from datetime import datetime  # noqa: TC003


class TurnoverBudget:
    """Limits trading frequency per symbol per month.

    Each symbol is allowed a maximum number of round-trips (buy + sell)
    per calendar month.  Exceeding the budget blocks further trades for
    that symbol until the next month.
    """

    _MAX_ROUND_TRIPS_PER_MONTH = 2

    def __init__(self, max_round_trips: int = _MAX_ROUND_TRIPS_PER_MONTH) -> None:
        self._max = max_round_trips
        # symbol -> {month_key -> round_trip_count}
        self._counts: dict[str, dict[str, int]] = {}

    def can_trade(self, symbol: str, timestamp: datetime) -> bool:
        """Check if *symbol* has remaining turnover budget this month."""
        month_key = timestamp.strftime("%Y-%m")
        count = self._counts.get(symbol, {}).get(month_key, 0)
        return count < self._max

    def record_round_trip(self, symbol: str, timestamp: datetime) -> None:
        """Record a completed round trip (buy + sell)."""
        month_key = timestamp.strftime("%Y-%m")
        if symbol not in self._counts:
            self._counts[symbol] = {}
        self._counts[symbol][month_key] = self._counts[symbol].get(month_key, 0) + 1

    def reset(self) -> None:
        """Clear all counts."""
        self._counts.clear()
