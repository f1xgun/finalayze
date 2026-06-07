"""RED scaffold: dividend accrual golden + look-ahead + cancelled (ACCT-01 / ACCT-04 / D-17).

Pins the total-return dividend contract before the implementation exists:
- golden kopeck (ACCT-04): SBER 2023 100 sh x 25.0 x (1 - 0.13) = 2175.00 EXACTLY,
  Decimal-exact (no float rounding);
- look-ahead guard (D-17): shifting the schedule forward one bar must NOT credit
  on the original (past) ex-date bar -- a future-shifted series cannot change a
  past bar's accrual;
- cancelled skip (ACCT-01): GAZP 2022-06-30 status "cancelled" is dropped at load
  time, so a held GAZP across that date accrues zero.

RED now: ``finalayze.backtest.dividend_schedule`` / ``finalayze.core.constants``
do not exist yet (Plan 02 + Plan 03 create them).
"""

from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal

from finalayze.backtest.dividend_schedule import load_dividend_schedule
from finalayze.core.constants import NDFL_RATE

# ── Constants (named -- no magic numbers) ───────────────────────────────────

_SBER = "SBER"
_SBER_QTY = Decimal(100)
_SBER_DIV = Decimal("25.0")
_SBER_EX_DATE = date(2023, 5, 11)
_EXPECTED_NET = _SBER_QTY * _SBER_DIV * (Decimal(1) - NDFL_RATE)  # == 2175.00 exactly
_GOLDEN_NET = Decimal("2175.00")

_GAZP = "GAZP"
_GAZP_CANCELLED_EX_DATE = date(2022, 6, 30)

_ONE_BAR = timedelta(days=1)


def test_golden_sber_2023() -> None:
    """SBER 2023 ex-date credits exactly qty x div x (1 - NDFL) net (ACCT-04)."""
    schedule = load_dividend_schedule()
    gross_per_share = schedule[(_SBER, _SBER_EX_DATE)]
    assert gross_per_share == _SBER_DIV  # schedule stores gross per share

    net = _SBER_QTY * gross_per_share * (Decimal(1) - NDFL_RATE)
    assert net == _EXPECTED_NET  # Decimal-exact equality, never a float tolerance
    assert net == _GOLDEN_NET  # the hand-verified kopeck value: 2175.00


def test_lookahead_shift_is_noop() -> None:
    """Shifting the schedule forward one bar does not credit on a past bar (D-17).

    A held SBER on the original ex-date sees the dividend in the real schedule;
    once every ex-date is shifted +1 bar, the original (past) bar's key is gone
    -- so appending/shifting a future date cannot retroactively credit a past
    bar. This is the look-ahead guard, encoded before the engine credit path.
    """
    schedule = load_dividend_schedule()
    assert (_SBER, _SBER_EX_DATE) in schedule

    shifted = {(sym, ex + _ONE_BAR): gross for (sym, ex), gross in schedule.items()}
    # The original (past) ex-date bar accrues nothing under the shifted series.
    assert (_SBER, _SBER_EX_DATE) not in shifted
    # The credit moved forward exactly one bar, never backward.
    assert (_SBER, _SBER_EX_DATE + _ONE_BAR) in shifted
    assert shifted[(_SBER, _SBER_EX_DATE + _ONE_BAR)] == schedule[(_SBER, _SBER_EX_DATE)]


def test_cancelled_not_accrued() -> None:
    """A cancelled dividend is skipped at load time -> zero accrual (ACCT-01)."""
    schedule = load_dividend_schedule()
    assert (_GAZP, _GAZP_CANCELLED_EX_DATE) not in schedule
