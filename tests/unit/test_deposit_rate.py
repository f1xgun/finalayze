"""RED scaffold: as-of deposit-rate helper (DEP-01 / D-04 / D-17).

Pins the L2 ``deposit_rate_as_of`` contract before it exists:
- the annual deposit rate is ``(key_rate - spread) / 100`` read as-of via the
  look-ahead-safe CBR meeting calendar (most recent meeting on/before the date);
- a date between two CBR meetings uses the EARLIER (already-decided) rate, never
  the future meeting (no look-ahead, D-17);
- a date before the first meeting in the calendar returns ``Decimal(0)``.

Expected key-rate values are read from the SAME ``MacroContextProvider`` the
helper uses, so the assertions carry no hand-copied magic rate numbers.

RED now: ``deposit_rate_as_of`` does not exist in ``finalayze.data.fetchers.cbr``.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

from finalayze.data.fetchers.cbr import MacroContextProvider, deposit_rate_as_of

# ── Constants ───────────────────────────────────────────────────────────────

_SPREAD_PP = Decimal("1.0")
_PCT = Decimal(100)

# A date after a decided CBR meeting (2024-10-25 -> 21.00pp) with a known rate.
_RATED_DATE = date(2024, 11, 1)

# A date strictly between two meetings: 2024-07-26 (18.00) and 2024-09-13 (19.00).
# The as-of rate must use the EARLIER meeting, never the later one.
_BETWEEN_DATE = date(2024, 8, 1)
_NEXT_MEETING_DATE = date(2024, 9, 13)

# A date before the first meeting in the calendar (2022-02-28).
_PRE_CALENDAR_DATE = date(2020, 1, 1)


def test_deposit_rate_is_key_rate_minus_spread() -> None:
    """deposit_rate_as_of(d) == (key_rate(d) - spread) / 100, as-of (DEP-01)."""
    snap = MacroContextProvider().get_snapshot(_RATED_DATE)
    assert snap.key_rate is not None
    expected = (snap.key_rate - _SPREAD_PP) / _PCT
    assert deposit_rate_as_of(_RATED_DATE, spread_pp=_SPREAD_PP) == expected


def test_no_future_meeting_leak() -> None:
    """Between two meetings, the rate uses the earlier decision, not the future one (D-17)."""
    provider = MacroContextProvider()
    as_of_rate = provider.get_snapshot(_BETWEEN_DATE).key_rate
    future_rate = provider.get_snapshot(_NEXT_MEETING_DATE).key_rate
    assert as_of_rate is not None
    assert future_rate is not None
    # Sanity: the future meeting genuinely raised the rate, so a leak would show.
    assert as_of_rate < future_rate

    expected = (as_of_rate - _SPREAD_PP) / _PCT
    assert deposit_rate_as_of(_BETWEEN_DATE, spread_pp=_SPREAD_PP) == expected
    # Explicitly not the future rate.
    assert (
        deposit_rate_as_of(_BETWEEN_DATE, spread_pp=_SPREAD_PP) != (future_rate - _SPREAD_PP) / _PCT
    )


def test_before_first_meeting_returns_zero() -> None:
    """A date before the first CBR meeting in the calendar -> Decimal(0)."""
    assert deposit_rate_as_of(_PRE_CALENDAR_DATE) == Decimal(0)
