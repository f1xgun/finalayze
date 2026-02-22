"""Trading hours scheduler for supported markets (Layer 2).

See docs/architecture/OVERVIEW.md for market definitions.
"""

from __future__ import annotations

from datetime import UTC, datetime, time, timedelta
from zoneinfo import ZoneInfo

_FIRST_WEEKEND_WEEKDAY = 5  # Monday=0 … Friday=4, Saturday=5, Sunday=6


class MarketSchedule:
    """Trading hours for a single market/exchange.

    All public methods accept and return UTC-aware :class:`datetime` objects.
    Internally, comparisons are performed in the market's local timezone.
    """

    def __init__(
        self,
        market_id: str,
        open_time: time,
        close_time: time,
        tz: str,
    ) -> None:
        self._market_id = market_id
        self._open_time = open_time
        self._close_time = close_time
        self._tz = ZoneInfo(tz)

    # ── Public API ───────────────────────────────────────────────────────

    def is_market_open(self, dt: datetime | None = None) -> bool:
        """Return True if *dt* falls within trading hours.

        Args:
            dt: UTC-aware datetime to check.  If ``None``, the current UTC
                time is used.

        Returns:
            ``True`` when the market local time satisfies
            ``open_time <= local_time < close_time``, ``False`` otherwise.
        """
        utc_dt = dt if dt is not None else datetime.now(tz=UTC)
        local_dt = utc_dt.astimezone(self._tz)
        if local_dt.weekday() >= _FIRST_WEEKEND_WEEKDAY:
            return False
        local_time = local_dt.time()
        return self._open_time <= local_time < self._close_time

    def next_open(self, dt: datetime | None = None) -> datetime:
        """Return the next market open as a UTC-aware :class:`datetime`.

        If the market is currently before open on the given day, returns the
        open time for the **same** calendar day (in the market's timezone).
        Otherwise advances to the next calendar day.

        Args:
            dt: UTC-aware datetime from which to search.  Defaults to the
                current UTC time when ``None``.

        Returns:
            UTC-aware :class:`datetime` of the next market open.
        """
        utc_dt = dt if dt is not None else datetime.now(tz=UTC)
        local_dt = utc_dt.astimezone(self._tz)

        # Try today first; advance by one day if already past or at open time.
        candidate = local_dt.replace(
            hour=self._open_time.hour,
            minute=self._open_time.minute,
            second=0,
            microsecond=0,
        )
        if candidate <= local_dt:
            candidate += timedelta(days=1)

        # Skip weekends
        while candidate.weekday() >= _FIRST_WEEKEND_WEEKDAY:
            candidate += timedelta(days=1)

        return candidate.astimezone(UTC)


# ── Pre-built schedules ──────────────────────────────────────────────────

US_MARKET_SCHEDULE = MarketSchedule(
    market_id="us",
    open_time=time(9, 30),  # 9:30 AM Eastern
    close_time=time(16, 0),  # 4:00 PM Eastern
    tz="America/New_York",
)

MOEX_MARKET_SCHEDULE = MarketSchedule(
    market_id="moex",
    open_time=time(10, 0),  # 10:00 AM Moscow
    close_time=time(18, 40),  # 6:40 PM Moscow
    tz="Europe/Moscow",
)

SCHEDULES: dict[str, MarketSchedule] = {
    "us": US_MARKET_SCHEDULE,
    "moex": MOEX_MARKET_SCHEDULE,
}
