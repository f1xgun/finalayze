"""Tests for LDV eligibility + Kcb exempt headroom (design section 4.3 steps 5-6).

INVARIANT 3: LDV headroom uses Kcb for mixed holding periods; only lots with
N>=3 full years AND known cost-basis AND Russian issuer AND non-IIS qualify;
foreign/EAEU lots are FLAGGED not credited.
INVARIANT 4: INPUT_SECURITIES (no cost/date) -> LDV clock + cost unknown -> not
eligible (flagged elsewhere), never a fabricated number.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

from finalayze.tax.ldv import (
    LDV_ANNUAL_LIMIT,
    LDV_EARLIEST_ACQUIRE,
    LDV_MIN_FULL_YEARS,
    LdvHeadroom,
    LdvHoldingItem,
    kcb_coefficient,
    ldv_eligible,
    ldv_headroom,
)
from finalayze.tax.lots import TaxLot

TICKER = "SBER"
FIGI = "BBG004730N88"
CCY = "RUB"

QTY = Decimal(100)
PRICE = Decimal(200)
COMM = Decimal(10)

TODAY = date(2026, 6, 1)
# > 3 full years before today
ACQ_4Y = date(2021, 1, 10)
# exactly ~3 full years (boundary) before today
ACQ_3Y = date(2023, 5, 1)
# < 3 years -> not eligible
ACQ_2Y = date(2024, 6, 1)
# acquired before 2014 -> not eligible
ACQ_PRE2014 = date(2013, 6, 1)

FINREZ_1M = Decimal(1_000_000)
FINREZ_2M = Decimal(2_000_000)
FINREZ_5M = Decimal(5_000_000)


def _lot(
    acquire: date,
    *,
    russian: bool = True,
    on_iis: bool = False,
    cost_known: bool = True,
) -> TaxLot:
    return TaxLot(
        figi=FIGI,
        ticker=TICKER,
        acquire_date=acquire,
        quantity=QTY,
        price_per_unit=PRICE,
        commission_buy=COMM,
        currency=CCY,
        russian_issuer=russian,
        on_iis=on_iis,
        cost_basis_known=cost_known,
    )


def test_min_full_years_is_three() -> None:
    assert LDV_MIN_FULL_YEARS == 3
    assert date(2014, 1, 1) == LDV_EARLIEST_ACQUIRE
    assert Decimal(3_000_000) == LDV_ANNUAL_LIMIT


def test_eligible_four_year_russian_non_iis_known_cost() -> None:
    assert ldv_eligible(_lot(ACQ_4Y), TODAY) is True


def test_not_eligible_under_three_years() -> None:
    assert ldv_eligible(_lot(ACQ_2Y), TODAY) is False


def test_not_eligible_pre_2014() -> None:
    assert ldv_eligible(_lot(ACQ_PRE2014), TODAY) is False


def test_not_eligible_foreign_issuer() -> None:
    # foreign (incl EAEU) -> flagged not credited
    assert ldv_eligible(_lot(ACQ_4Y, russian=False), TODAY) is False


def test_not_eligible_on_iis() -> None:
    assert ldv_eligible(_lot(ACQ_4Y, on_iis=True), TODAY) is False


def test_not_eligible_cost_basis_unknown() -> None:
    # INPUT_SECURITIES transferred-in lot: cost/date unknown
    assert ldv_eligible(_lot(ACQ_4Y, cost_known=False), TODAY) is False


def test_boundary_buffer_rejects_just_under_three_years() -> None:
    # a lot whose 3rd anniversary is only a couple of days away must NOT
    # be credited (conservative T+2 buffer)
    just_under = date(TODAY.year - LDV_MIN_FULL_YEARS, TODAY.month, TODAY.day)
    # move acquire one day LATER so 3 full years have not quite elapsed
    almost = just_under.replace(day=just_under.day + 1)
    assert ldv_eligible(_lot(almost), TODAY) is False


def test_kcb_coefficient_mixed_holding_periods() -> None:
    # Vi = positive finrez, i = full years held
    # lot1: V=1M, i=3 ; lot2: V=2M, i=5
    items = [
        LdvHoldingItem(positive_finrez=FINREZ_1M, full_years=3),
        LdvHoldingItem(positive_finrez=FINREZ_2M, full_years=5),
    ]
    kcb = kcb_coefficient(items)
    # Kcb = (1M*3 + 2M*5) / (1M+2M) = (3M+10M)/3M = 13/3
    expected = (FINREZ_1M * 3 + FINREZ_2M * 5) / (FINREZ_1M + FINREZ_2M)
    assert kcb == expected


def test_headroom_uses_kcb_times_annual_limit() -> None:
    items = [
        LdvHoldingItem(positive_finrez=FINREZ_1M, full_years=3),
        LdvHoldingItem(positive_finrez=FINREZ_2M, full_years=5),
    ]
    hr: LdvHeadroom = ldv_headroom(items)
    kcb = (FINREZ_1M * 3 + FINREZ_2M * 5) / (FINREZ_1M + FINREZ_2M)
    assert hr.kcb == kcb
    assert hr.cap == kcb * LDV_ANNUAL_LIMIT
    # exempt = min(total positive finrez, cap)
    total_finrez = FINREZ_1M + FINREZ_2M
    assert hr.exempt_amount == min(total_finrez, kcb * LDV_ANNUAL_LIMIT)


def test_headroom_single_period_is_years_times_three_million() -> None:
    # a single 3-year lot: Kcb == 3, cap == 3 * 3M = 9M
    items = [LdvHoldingItem(positive_finrez=FINREZ_5M, full_years=3)]
    hr = ldv_headroom(items)
    assert hr.kcb == Decimal(3)
    assert hr.cap == Decimal(3) * LDV_ANNUAL_LIMIT
    # finrez 5M < cap 9M -> all exempt
    assert hr.exempt_amount == FINREZ_5M


def test_headroom_empty_is_zero() -> None:
    hr = ldv_headroom([])
    assert hr.exempt_amount == Decimal(0)
    assert hr.cap == Decimal(0)
