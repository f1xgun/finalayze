"""RED scaffold: deposit sleeve accrual / mark / break / roll (ACCT-02 / DEP-01 / DEP-02 / R-4).

Pins the L5 ``DepositSimulatedBroker`` contract before it exists:
- daily-compounding net-of-NDFL interest accrues to cash per bar (ACCT-02); the
  first bar is below the floor so net == gross;
- the deposit mark = principal + accrued net interest and NEVER reprices on
  market input (DEP-01 / D-05: zero market risk);
- breaking a tranche pre-maturity forfeits accrued interest down to the demand
  rate (DEP-02 / D-03);
- a matured tranche auto-rolls into a fresh same-term tranche at the as-of rate,
  with no penalty and full accrued principal carried forward (R-4).

RED now: ``finalayze.execution.deposit_broker`` (Plan 04) +
``finalayze.core.schemas.DepositTranche`` / ``finalayze.core.constants``
(Plan 02) + ``deposit_rate_as_of`` (Plan 03) do not exist yet.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal

from finalayze.core.constants import DEPOSIT_DEMAND_RATE, NDFL_RATE
from finalayze.core.schemas import Candle, DepositTranche
from finalayze.data.fetchers.cbr import deposit_rate_as_of
from finalayze.execution.deposit_broker import DepositSimulatedBroker

# ── Constants (named -- no magic numbers) ───────────────────────────────────

_INITIAL_CASH = Decimal(0)
_PRINCIPAL = Decimal(1_000_000)
_ANNUAL_RATE = Decimal("0.20")
_TERM_3M = 3
_TRADING_DAYS = Decimal(252)
_ONE = Decimal(1)
_ZERO = Decimal(0)

_OPEN_DATE = date(2024, 1, 9)
_MATURITY_DATE = date(2024, 4, 9)  # ~3 months after open
_ACCRUE_BAR = date(2024, 1, 10)
_ACCRUE_BARS = (
    date(2024, 1, 10),
    date(2024, 1, 11),
    date(2024, 1, 12),
    date(2024, 1, 15),
)
# Daily-compounding convention (portfolio_orchestrator.py:358):
_DAILY_RATE = (_ONE + _ANNUAL_RATE) ** (_ONE / _TRADING_DAYS) - _ONE
_FIRST_BAR_GROSS = _PRINCIPAL * _DAILY_RATE  # below the floor -> net == gross

# Arbitrary market candle whose price must NOT move the deposit mark.
_ARBITRARY_CANDLE = Candle(
    symbol="SBER",
    market_id="moex",
    timeframe="1d",
    timestamp=datetime(2024, 1, 10, 7, 0, tzinfo=UTC),
    open=Decimal(999),
    high=Decimal(9999),
    low=Decimal(1),
    close=Decimal(9999),
    volume=1_000_000,
)


def _make_tranche(
    principal: Decimal = _PRINCIPAL,
    annual_rate: Decimal = _ANNUAL_RATE,
    term_months: int = _TERM_3M,
    open_date: date = _OPEN_DATE,
    maturity_date: date = _MATURITY_DATE,
) -> DepositTranche:
    return DepositTranche(
        principal=principal,
        term_months=term_months,
        annual_rate=annual_rate,
        open_date=open_date,
        maturity_date=maturity_date,
    )


def _make_broker(tranche: DepositTranche) -> DepositSimulatedBroker:
    return DepositSimulatedBroker(initial_cash=_INITIAL_CASH, tranches=[tranche])


def test_daily_accrual_net() -> None:
    """One bar accrues net-of-NDFL interest into the MARK, not cash (CR-01 / ACCT-02).

    Mark-only contract (CR-01): a locked deposit's interest stays INSIDE the
    deposit -- ``accrue`` compounds the net into ``deposit_value()`` and does NOT
    sweep it to ``self._cash`` (which would double-represent it once
    ``deposit_value`` also carries the accrued mark). ``accrue`` still RETURNS the
    net for callers who want the bar's credit.
    """
    broker = _make_broker(_make_tranche())
    cash_before = broker.get_portfolio().cash
    mark_before = broker.deposit_value()

    net = broker.accrue(_ACCRUE_BAR)

    assert broker.interest_income_net > _ZERO
    # First bar is below the non-taxable floor -> no tax -> net == gross.
    assert net == _FIRST_BAR_GROSS
    assert broker.interest_income_net == _FIRST_BAR_GROSS
    assert broker.tax_paid == _ZERO
    # Mark-only: cash is UNCHANGED; the mark grew by exactly the net (no double-count).
    assert broker.get_portfolio().cash == cash_before
    assert broker.deposit_value() - mark_before == net


def test_accrue_compounds_inside_the_tranche() -> None:
    """Net interest compounds on (principal + accrued_net) -- true compounding (CR-01).

    A locked deposit's accrued interest stays in the deposit and itself earns
    interest, so the second bar's gross is computed on the grown base, not the
    flat principal. The first bar (flat principal) is therefore strictly smaller
    than the second bar's gross.
    """
    broker = _make_broker(_make_tranche())

    first = broker.accrue(_ACCRUE_BARS[0])
    second = broker.accrue(_ACCRUE_BARS[1])

    # Both bars are below the floor (no tax) so net == gross; the second bar
    # compounds on (principal + first net) > principal -> strictly larger.
    assert second > first


def test_cash_plus_mark_reconciled_before_and_after_break() -> None:
    """``cash + deposit_value()`` is reconciled with no double-count (CR-01 regression).

    Before a break: total == initial_cash + principal + cumulative_net (the accrued
    interest lives ONLY in the mark, never also in cash). After a pre-maturity
    break: total == initial_cash + principal + demand penalty (the accrued interest
    is forfeited to the demand rate; there is no swept cash to claw back).
    """
    tranche = _make_tranche()
    broker = _make_broker(tranche)

    cumulative_net = _ZERO
    for bar in _ACCRUE_BARS:
        cumulative_net += broker.accrue(bar)

    cash = broker.get_portfolio().cash
    total_before = cash + broker.deposit_value()
    assert total_before == _INITIAL_CASH + _PRINCIPAL + cumulative_net

    broker.break_tranche(tranche, _ACCRUE_BARS[-1])

    demand_penalty = tranche.principal * DEPOSIT_DEMAND_RATE
    total_after = broker.get_portfolio().cash + broker.deposit_value()
    assert total_after == _INITIAL_CASH + _PRINCIPAL + demand_penalty


def test_mark_is_principal_plus_accrued() -> None:
    """Deposit mark = principal + accrued net; market prices never move it (D-05)."""
    tranche = _make_tranche()
    broker = _make_broker(tranche)
    broker.accrue(_ACCRUE_BAR)

    expected_mark = tranche.principal + tranche.accrued_net
    assert broker.deposit_value() == expected_mark

    # An arbitrary market price must NOT reprice the deposit.
    broker.update_prices(_ARBITRARY_CANDLE)
    assert broker.deposit_value() == expected_mark


def test_break_forfeits_interest() -> None:
    """Pre-maturity break forfeits accrued interest to the demand rate (DEP-02 / D-03)."""
    tranche = _make_tranche()
    broker = _make_broker(tranche)
    for bar in _ACCRUE_BARS:
        broker.accrue(bar)
    assert tranche.accrued_net > _ZERO  # there was accrued interest to forfeit

    broker.break_tranche(tranche, _ACCRUE_BARS[-1])

    assert tranche.broken is True
    assert tranche.accrued_net == tranche.principal * DEPOSIT_DEMAND_RATE


def test_roll_at_maturity() -> None:
    """A matured tranche rolls into a fresh same-term tranche at the as-of rate (R-4)."""
    tranche = _make_tranche()
    broker = _make_broker(tranche)
    for bar in _ACCRUE_BARS:
        broker.accrue(bar)
    accrued_at_maturity = tranche.accrued_net

    rolled = broker.roll_at_maturity(tranche, _MATURITY_DATE)

    # Principal carries forward with full accrued interest (no penalty at maturity).
    assert rolled.principal == tranche.principal + accrued_at_maturity
    assert rolled.term_months == tranche.term_months
    assert rolled.annual_rate == deposit_rate_as_of(_MATURITY_DATE)
    assert rolled.broken is False
    # The rolled tranche opens on the maturity bar and matures strictly later.
    assert rolled.open_date == _MATURITY_DATE
    assert rolled.maturity_date > _MATURITY_DATE


def test_accrual_curve_monotone_by_net_interest() -> None:
    """Net interest income only ever grows -- a smooth compounding line (R-4 / D-16)."""
    broker = _make_broker(_make_tranche())
    previous = _ZERO
    for bar in _ACCRUE_BARS:
        broker.accrue(bar)
        assert broker.interest_income_net >= previous
        previous = broker.interest_income_net
    assert NDFL_RATE > _ZERO  # the tax rate exists for above-floor bars
