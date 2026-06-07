"""RED scaffold: D-06 / R-6 lockup-respecting funding-order (Phase 72 Wave-0).

Pins the L5 rebalance funding-order contract before it exists: a quarterly
rebalance that needs cash to fund an underweight leg draws sources in the
lockup-respecting order (D-06):

  1. matured tranches (principal + accrued, no penalty),
  2. accrued income / liquid demand cash,
  3. last-resort break of a LOCKED tranche -- forfeiting accrued to the demand
     rate (the W1 ``break_tranche`` penalty, Pitfall 7: never a silent full-rate
     sale).

R-6: the ladder builder must open tranches on quarter boundaries so a matured
source actually exists at the next rebalance; a mid-quarter tranche is NOT
available at the boundary rebalance, and the degraded path (no matured tranche)
must fall through to income/cash gracefully without breaking a locked rung while
cheaper sources cover the need.

RED now: the funding-order helper ``finalayze.orchestration.allocation.fund_underweight``
(Plan 05) does not exist yet. (``DepositSimulatedBroker`` / ``DepositTranche`` /
``DEPOSIT_DEMAND_RATE`` already exist from W1; the RED is on the missing helper.
Helper path pinned here = ``finalayze.orchestration.allocation.fund_underweight``;
Plan 05 wires this exact path or its ``AllocationOrchestrator`` method equivalent.)
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

from dateutil.relativedelta import relativedelta

from finalayze.core.constants import DEPOSIT_DEMAND_RATE
from finalayze.core.schemas import DepositTranche
from finalayze.execution.deposit_broker import DepositSimulatedBroker
from finalayze.orchestration.allocation import fund_underweight

# -- Constants (named -- no magic numbers, ruff PLR2004) ----------------------

_INITIAL_CASH = Decimal(0)
_PRINCIPAL = Decimal(1_000_000)
_ANNUAL_RATE = Decimal("0.20")
_TERM_3M = 3
_ZERO = Decimal(0)

# Quarter-boundary calendar (R-6).
_REBALANCE_TS = date(2023, 4, 1)  # the Apr-1 quarterly rebalance
_QUARTER_OPEN = date(2023, 1, 1)  # opened on a quarter boundary
_QUARTER_MATURITY = date(2023, 4, 1)  # 3mo later = the next quarter boundary -> matured
_MID_QUARTER_OPEN = date(2023, 2, 15)  # opened mid-quarter
_MID_QUARTER_MATURITY = date(2023, 5, 15)  # 3mo later -> NOT available at Apr-1

# A funding need the matured tranche can satisfy from principal + accrued alone.
_FUNDING_NEED = Decimal(500_000)


def _make_tranche(
    *,
    open_date: date,
    maturity_date: date,
    principal: Decimal = _PRINCIPAL,
    broken: bool = False,
) -> DepositTranche:
    return DepositTranche(
        principal=principal,
        term_months=_TERM_3M,
        annual_rate=_ANNUAL_RATE,
        open_date=open_date,
        maturity_date=maturity_date,
        broken=broken,
    )


def test_matured_tranche_funds_first() -> None:
    """A matured tranche funds the need first, with NO break penalty (D-06 step 1)."""
    matured = _make_tranche(open_date=_QUARTER_OPEN, maturity_date=_QUARTER_MATURITY)
    broker = DepositSimulatedBroker(initial_cash=_INITIAL_CASH, tranches=[matured])

    funded = fund_underweight(broker, _FUNDING_NEED, _REBALANCE_TS)

    assert funded >= _FUNDING_NEED  # the need is satisfied
    assert matured.broken is False  # the matured tranche was not BROKEN -- it matured
    assert broker.interest_income_net >= _ZERO  # no negative penalty bookkeeping


def test_falls_through_to_income_then_cash() -> None:
    """No matured tranche -> draw income then liquid cash, never break (R-6 degraded path)."""
    # Liquid demand cash already covers the need; the locked tranche stays untouched.
    locked = _make_tranche(open_date=_MID_QUARTER_OPEN, maturity_date=_MID_QUARTER_MATURITY)
    broker = DepositSimulatedBroker(initial_cash=_FUNDING_NEED, tranches=[locked])

    funded = fund_underweight(broker, _FUNDING_NEED, _REBALANCE_TS)

    assert funded >= _FUNDING_NEED
    # A cheaper source (cash) covered the need -> the locked tranche is NOT broken.
    assert locked.broken is False


def test_last_resort_breaks_locked_with_penalty() -> None:
    """Insufficient matured+income+cash -> break a LOCKED tranche with penalty (D-06 step 4)."""
    locked = _make_tranche(open_date=_MID_QUARTER_OPEN, maturity_date=_MID_QUARTER_MATURITY)
    # No liquid cash, no matured tranche, no accrued income -> the locked rung is
    # the only source, so it must be broken (Pitfall 7: never a silent full-rate sale).
    broker = DepositSimulatedBroker(initial_cash=_INITIAL_CASH, tranches=[locked])
    deposit_before = broker.deposit_value()

    fund_underweight(broker, _FUNDING_NEED, _REBALANCE_TS)

    assert locked.broken is True
    # The penalty is actually charged: accrued forfeited to the demand rate.
    assert locked.accrued_net == locked.principal * DEPOSIT_DEMAND_RATE
    # The deposit class value drops (the broken rung is no longer a full-rate asset).
    assert broker.deposit_value() <= deposit_before


def test_quarter_boundary_ladder_construction() -> None:
    """Quarter-boundary tranches mature at the next boundary; mid-quarter ones do not (R-6)."""
    quarter_tranche = _make_tranche(
        open_date=_QUARTER_OPEN, maturity_date=_QUARTER_OPEN + relativedelta(months=_TERM_3M)
    )
    mid_tranche = _make_tranche(
        open_date=_MID_QUARTER_OPEN,
        maturity_date=_MID_QUARTER_OPEN + relativedelta(months=_TERM_3M),
    )
    # A boundary-opened 3mo tranche IS matured at the Apr-1 rebalance...
    assert quarter_tranche.maturity_date <= _REBALANCE_TS
    # ...a mid-quarter 3mo tranche is NOT -- it is still locked at Apr-1.
    assert mid_tranche.maturity_date > _REBALANCE_TS

    broker = DepositSimulatedBroker(
        initial_cash=_INITIAL_CASH, tranches=[quarter_tranche, mid_tranche]
    )
    funded = fund_underweight(broker, _FUNDING_NEED, _REBALANCE_TS)
    assert funded >= _FUNDING_NEED
    # Step-1 funding draws the matured (boundary) tranche; the mid-quarter rung
    # is NOT broken because the matured source already covers the need.
    assert mid_tranche.broken is False
