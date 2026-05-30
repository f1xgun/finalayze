"""Fundamental disclosure-date machinery (Layer 2, pure / no I/O).

Mirrors ``cbr.py``'s CPI publication-date pattern (``_effective_cpi_publication_date``
+ ``get_latest_published_cpi_month``) for company fundamentals (BACKFILL-H-02 / D-04).

Point-in-time dating: a ``FundamentalSnapshot`` must be stamped with the date the
figure was actually *disclosed*, never with the fiscal-period end (that would leak
future information into a backtest — the look-ahead trap H-02). When SmartLab
supplies an explicit "Дата отчета" cell, that is used directly; when the cell is
empty, this module supplies the conservative fallback: fiscal-quarter-end + 75 days.

Pure functions only — no network, no filesystem. Layer 2.
"""

from __future__ import annotations

from datetime import date, timedelta

# Approximate MOEX-issuer disclosure lag: IFRS/RAS statements land roughly a
# quarter after the fiscal-period end. 75 days is the conservative fallback when
# no exact disclosure date is recorded (mirrors cbr.py's _CPI_PUBLICATION_LAG_MONTHS
# pattern, expressed in days for per-quarter granularity).
_FUNDAMENTAL_DISCLOSURE_LAG_DAYS = 75

# Fiscal-quarter -> (month, day) of the quarter-end.
_QUARTER_END: dict[int, tuple[int, int]] = {
    1: (3, 31),
    2: (6, 30),
    3: (9, 30),
    4: (12, 31),
}

# Known exact disclosure dates, keyed by (symbol, fiscal_period e.g. "2025Q2").
# Seed verified dates here; absent entries fall back to quarter-end + 75d.
# Only dates on/before a backtest's as_of may use that period's fundamentals
# (look-ahead boundary), exactly mirroring CPI_PUBLICATION_DATES in cbr.py.
FUNDAMENTAL_PUBLICATION_DATES: dict[tuple[str, str], date] = {
    # ("SBER", "2025Q2"): date(2025, 7, 29),  # seed known dates here
}


def get_effective_disclosure_date(symbol: str, period: str) -> date:
    """Return the effective disclosure date for *symbol*'s fiscal *period*.

    If an exact date is recorded in ``FUNDAMENTAL_PUBLICATION_DATES`` it is
    returned verbatim; otherwise the conservative fallback (fiscal-quarter-end +
    ``_FUNDAMENTAL_DISCLOSURE_LAG_DAYS``) is used.

    The fallback is NEVER the bare fiscal-quarter-end — it is always end + 75d, so
    a snapshot built from this date cannot be read before the figure could have
    been public (BACKFILL-H-02, look-ahead safe).

    Args:
        symbol: Ticker, e.g. ``"SBER"``.
        period: Fiscal period as ``"YYYYQN"``, e.g. ``"2025Q1"``.

    Returns:
        The effective disclosure ``date``.
    """
    recorded = FUNDAMENTAL_PUBLICATION_DATES.get((symbol, period))
    if recorded is not None:
        return recorded
    year, quarter = int(period[:4]), int(period[5])
    month, day = _QUARTER_END[quarter]
    return date(year, month, day) + timedelta(days=_FUNDAMENTAL_DISCLOSURE_LAG_DAYS)
