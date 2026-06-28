"""Deposit-interest NDFL progressive-band regression tests (audit 2026-06-28, MEDIUM).

Deposit interest was charged a flat 13% (``ndfl_on_deposit_interest``), inconsistent
with dividends/coupons which route through the cross-sleeve progressive 13/15% band
(``YtdTaxAccumulator``). The taxable excess (above the running-max floor) now stacks on
the same shared YTD when an accumulator is supplied; below the 2.4M RUB threshold the
marginal band == flat 13% (sub-threshold runs stay byte-identical).
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

from finalayze.core.constants import NDFL_PROGRESSIVE_THRESHOLD
from finalayze.core.ndfl import YtdTaxAccumulator, ndfl_on_deposit_interest
from finalayze.core.schemas import DepositTranche
from finalayze.execution.deposit_broker import DepositSimulatedBroker

_FLAT = Decimal("0.13")
_HIGH = Decimal("0.15")


def test_backcompat_flat_13_without_accumulator() -> None:
    # No accumulator -> flat 13% on the taxable excess (floor 0 -> all taxable).
    assert ndfl_on_deposit_interest(Decimal(100), Decimal(0), Decimal(0)) == Decimal(100) * _FLAT


def test_floor_shields_below_threshold() -> None:
    # Excess over a 30 floor is 70 -> 70 * 13%.
    assert ndfl_on_deposit_interest(Decimal(100), Decimal(0), Decimal(30)) == Decimal(70) * _FLAT


def test_below_threshold_marginal_equals_flat() -> None:
    acc = YtdTaxAccumulator()
    # Fresh YTD, small amount -> entirely in the 13% band -> identical to flat.
    progressive = ndfl_on_deposit_interest(
        Decimal(1000), Decimal(0), Decimal(0), tax_acc=acc, year=2025
    )
    assert progressive == Decimal(1000) * _FLAT


def test_above_threshold_charges_15_on_excess() -> None:
    acc = YtdTaxAccumulator()
    # Pre-load YTD to 50 below the 2.4M threshold (e.g. from dividends earlier in the year).
    acc.tax(NDFL_PROGRESSIVE_THRESHOLD - Decimal(50), 2025)
    # Deposit taxable excess of 100 straddles the threshold: 50 @ 13% + 50 @ 15%.
    tax = ndfl_on_deposit_interest(Decimal(100), Decimal(0), Decimal(0), tax_acc=acc, year=2025)
    assert tax == Decimal(50) * _FLAT + Decimal(50) * _HIGH
    assert tax > Decimal(100) * _FLAT  # strictly more than the old flat-13% result


def test_broker_accrue_routes_taxable_excess_through_accumulator() -> None:
    tranche = DepositTranche(
        principal=Decimal(1_000_000),
        term_months=12,
        annual_rate=Decimal("0.18"),
        open_date=date(2025, 1, 1),
        maturity_date=date(2026, 1, 1),
    )
    broker = DepositSimulatedBroker(initial_cash=Decimal(0), tranches=[tranche])
    # Force YTD well above any running-max floor so the day's gross is fully taxable.
    broker._ytd_deposit_gross = Decimal(10_000_000)
    broker._current_year = 2025

    acc = YtdTaxAccumulator()
    broker.accrue(date(2025, 6, 2), tax_acc=acc)

    # The day's taxable deposit excess advanced the shared cross-sleeve YTD.
    assert acc.ytd_taxable > Decimal(0)
