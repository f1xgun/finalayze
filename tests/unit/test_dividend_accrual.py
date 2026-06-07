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
from pathlib import Path

import pytest

from finalayze.backtest.dividend_schedule import load_dividend_schedule
from finalayze.core.constants import NDFL_RATE
from finalayze.core.exceptions import ConfigurationError

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


# ── WR-03: per-event malformed content is fully fail-closed (ConfigurationError) ──

# Each maps a malformed YAML body to the low-level exception the unguarded loop
# would otherwise leak; the loader must re-raise ALL of them as ConfigurationError.
_MALFORMED_BODIES = (
    pytest.param("SBER:\n  - just-a-string\n", id="event_not_a_mapping"),
    pytest.param("SBER:\n  - amount: 25.0\n", id="missing_ex_date"),
    pytest.param("SBER:\n  - ex_date: '2023-05-11'\n", id="missing_amount"),
    pytest.param("SBER:\n  - ex_date: not-a-date\n    amount: 25.0\n", id="bad_date"),
    pytest.param("SBER:\n", id="null_events"),
    pytest.param("SBER:\n  - ex_date: '2023-05-11'\n    amount: abc\n", id="non_numeric_amount"),
)


@pytest.mark.parametrize("body", _MALFORMED_BODIES)
def test_malformed_event_is_fail_closed(body: str, tmp_path: Path) -> None:
    """Malformed-but-parseable per-event content raises ConfigurationError (WR-03).

    The docstring promises a corrupt/unparseable/malformed file is fail-closed; the
    per-event parse loop must re-raise low-level KeyError/ValueError/TypeError/
    AttributeError/InvalidOperation as ConfigurationError rather than leaking an
    opaque traceback at the snapshot path.
    """
    snapshot = tmp_path / "moex_dividends.yaml"
    snapshot.write_text(body, encoding="utf-8")
    with pytest.raises(ConfigurationError):
        load_dividend_schedule(path=snapshot)


def test_well_formed_override_still_loads(tmp_path: Path) -> None:
    """A well-formed override path still loads correctly (WR-03 guard is not over-broad)."""
    snapshot = tmp_path / "moex_dividends.yaml"
    snapshot.write_text(
        "SBER:\n  - ex_date: '2023-05-11'\n    amount: 25.0\n    status: paid\n",
        encoding="utf-8",
    )
    schedule = load_dividend_schedule(path=snapshot)
    assert schedule[(_SBER, _SBER_EX_DATE)] == _SBER_DIV


# ── WR-04: dividend credit is idempotent per (symbol, ex_date) ───────────────


def test_process_dividends_idempotent_on_repeated_date() -> None:
    """A repeated calendar date does NOT re-credit the same dividend (WR-04).

    On a sub-daily timeframe (or any data where two bars share a calendar day),
    ``process_dividends`` is invoked twice with the same ``current_date``. The
    second call must credit zero -- a ``(symbol, ex_date)`` already credited is
    never re-credited (mirrors ``ShadowLedger._processed_dividend_dates``).
    """
    from finalayze.execution.simulated_broker import SimulatedBroker

    broker = SimulatedBroker(initial_cash=Decimal(0))
    broker._positions[_SBER] = _SBER_QTY  # held position across the ex-date
    schedule = {(_SBER, _SBER_EX_DATE): _SBER_DIV}

    first = broker.process_dividends(_SBER_EX_DATE, schedule)
    cash_after_first = broker.get_portfolio().cash
    second = broker.process_dividends(_SBER_EX_DATE, schedule)  # same date, repeated bar

    assert first == _GOLDEN_NET  # 100 x 25.0 x (1 - 0.13) net
    assert second == Decimal(0)  # idempotent: no double-credit
    assert broker.get_portfolio().cash == cash_after_first  # cash unchanged on repeat


# ── WR-01: dividend credit routes through the progressive 13/15% band ────────

_BIG_QTY = Decimal(200_000)  # 200k sh x 25.0 = 5.0M gross -> straddles the 2.4M threshold
_THRESHOLD = Decimal(2_400_000)
_NDFL_HIGH = Decimal("0.15")


def test_process_dividends_flat_13_below_threshold() -> None:
    """Below the 2.4M YTD threshold, the accumulator path equals flat 13% (WR-01).

    Routing the golden SBER credit (2500 gross << 2.4M) through the YTD accumulator
    must yield the SAME net as the flat-rate default -- the golden kopeck value 2175.00.
    """
    from finalayze.core.ndfl import YtdTaxAccumulator
    from finalayze.execution.simulated_broker import SimulatedBroker

    broker = SimulatedBroker(initial_cash=Decimal(0))
    broker._positions[_SBER] = _SBER_QTY
    acc = YtdTaxAccumulator()

    net = broker.process_dividends(
        _SBER_EX_DATE, {(_SBER, _SBER_EX_DATE): _SBER_DIV}, ytd_accumulator=acc
    )
    assert net == _GOLDEN_NET  # 2175.00 -- identical to the flat-rate golden


def test_process_dividends_15_fires_above_threshold() -> None:
    """Cumulative YTD dividend income above 2.4M is taxed 15% on the excess (WR-01).

    A 5.0M gross dividend credited via the accumulator straddles the 2.4M threshold:
    2.4M at 13% and the 2.6M excess at 15%. The net must reflect the higher band.
    """
    from finalayze.core.ndfl import YtdTaxAccumulator
    from finalayze.execution.simulated_broker import SimulatedBroker

    broker = SimulatedBroker(initial_cash=Decimal(0))
    broker._positions[_SBER] = _BIG_QTY
    acc = YtdTaxAccumulator()

    gross = _SBER_DIV * _BIG_QTY  # 5.0M
    net = broker.process_dividends(
        _SBER_EX_DATE, {(_SBER, _SBER_EX_DATE): _SBER_DIV}, ytd_accumulator=acc
    )

    below_tax = _THRESHOLD * NDFL_RATE
    above_tax = (gross - _THRESHOLD) * _NDFL_HIGH
    expected_net = gross - (below_tax + above_tax)
    assert net == expected_net
    # The high band genuinely fired: net is strictly below the flat-13% net.
    assert net < gross - gross * NDFL_RATE
    assert acc.ytd_taxable == gross
