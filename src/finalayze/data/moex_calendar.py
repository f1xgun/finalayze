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


# Per-year transferred holidays from Russian government decrees.
# When a public holiday falls on a weekend, the government transfers the
# day off to a nearby weekday. These vary each year (2020-2026).
_TRANSFERRED_HOLIDAYS: dict[int, frozenset[tuple[int, int]]] = {
    2020: frozenset({(3, 9), (5, 4), (5, 5), (5, 11), (6, 15), (11, 5)}),
    2021: frozenset({(2, 22), (3, 9), (5, 3), (5, 10), (6, 14), (11, 3), (11, 5), (12, 31)}),
    2022: frozenset({(3, 7), (5, 2), (5, 3), (5, 10), (6, 13), (11, 3)}),
    2023: frozenset({(2, 24), (5, 8), (6, 13), (11, 6)}),
    2024: frozenset({(4, 29), (4, 30), (5, 10), (12, 30), (12, 31)}),
    2025: frozenset({(5, 2), (5, 8), (6, 13), (11, 3), (12, 31)}),
    2026: frozenset({(3, 9), (5, 11), (1, 9)}),  # preliminary
}


def is_moex_holiday(d: date) -> bool:
    """Return True if d is a Russian public holiday (MOEX closed).

    Does NOT check weekends — use d.weekday() >= 5 for that.
    Checks both fixed holidays and per-year transferred holidays.
    """
    if (d.month, d.day) in _FIXED_HOLIDAYS:
        return True
    year_transferred = _TRANSFERRED_HOLIDAYS.get(d.year)
    return bool(year_transferred and (d.month, d.day) in year_transferred)


def is_moex_trading_day(d: date) -> bool:
    """Return True if d is a MOEX trading day.

    Checks weekends, fixed holidays, and per-year transferred holidays.
    This is the unified check for both backtest and live trading.
    """
    if d.weekday() >= _WEEKEND_WEEKDAY_MIN:
        return False
    return not is_moex_holiday(d)


def trading_days_gap(d1: date, d2: date) -> int:
    """Count non-trading days between d1 (exclusive) and d2 (exclusive).

    Non-trading = weekend OR public holiday OR transferred holiday.
    Returns 0 if d1 >= d2.
    """
    if d1 >= d2:
        return 0
    count = 0
    current = d1 + timedelta(days=1)
    while current < d2:
        if not is_moex_trading_day(current):
            count += 1
        current += timedelta(days=1)
    return count
