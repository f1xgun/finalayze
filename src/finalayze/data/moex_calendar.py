"""MOEX trading calendar — static Russian public holiday list (Layer 2).

Provides is_moex_holiday() and trading_days_gap() for holiday-aware
feature engineering (e.g. Brent holiday suppression in Task 12).

Updated annually. Next update: 2026-01.
Sources: Russian government decree on transfer of weekend days.
"""

from __future__ import annotations

from datetime import date, timedelta

_WEEKEND_WEEKDAY_MIN: int = 5  # weekday() returns 5=Saturday, 6=Sunday

# Russian public holidays by (month, day). Repeats every year.
# Additional transferred holidays vary by year — not included (minor impact).
_FIXED_HOLIDAYS: frozenset[tuple[int, int]] = frozenset(
    {
        (1, 1),  # New Year's Day
        (1, 2),  # New Year holiday block
        (1, 3),  # New Year holiday block
        (1, 4),  # New Year holiday block
        (1, 5),  # New Year holiday block
        (1, 6),  # New Year holiday block
        (1, 7),  # Orthodox Christmas
        (1, 8),  # New Year holiday block ends
        (2, 23),  # Defender of the Fatherland Day
        (3, 8),  # International Women's Day
        (5, 1),  # Spring and Labour Day
        (5, 9),  # Victory Day
        (6, 12),  # Russia Day
        (11, 4),  # National Unity Day
    }
)


def is_moex_holiday(d: date) -> bool:
    """Return True if d is a Russian public holiday (MOEX closed).

    Does NOT check weekends — use d.weekday() >= 5 for that.
    Does NOT account for government-transferred holidays (minor, ~2-3/year).
    """
    return (d.month, d.day) in _FIXED_HOLIDAYS


def trading_days_gap(d1: date, d2: date) -> int:
    """Count non-trading days between d1 (exclusive) and d2 (exclusive).

    Non-trading = weekend OR public holiday.
    Returns 0 if d1 >= d2.
    """
    if d1 >= d2:
        return 0
    count = 0
    current = d1 + timedelta(days=1)
    while current < d2:
        if current.weekday() >= _WEEKEND_WEEKDAY_MIN or is_moex_holiday(current):
            count += 1
        current += timedelta(days=1)
    return count
