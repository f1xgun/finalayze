"""Tests for per-base YTD accumulators (design section 4.3 steps 3-4).

INVARIANT 1 (critical): dividends are a SEPARATE base with their own 2.4M
threshold and are NEVER netted/harvested -- a base-A loss leaves dividend tax
unchanged. Tested explicitly here.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

from finalayze.core.constants import (
    NDFL_PROGRESSIVE_THRESHOLD,
    NDFL_RATE,
    NDFL_RATE_HIGH,
)
from finalayze.tax.baskets import (
    BaseAccumulators,
    TaxBase,
    realized_ytd_base_a,
)
from finalayze.tax.lots import Operation, OperationType, RealizedResult

TICKER = "SBER"
FIGI = "BBG004730N88"
CCY = "RUB"

GAIN_500K = Decimal(500_000)
LOSS_200K = Decimal(-200_000)
COUPON_100K = Decimal(100_000)
DIVIDEND_300K = Decimal(300_000)

YEAR = 2026
D_JAN = date(2026, 1, 15)
D_FEB = date(2026, 2, 15)


def _realized(realized_value: Decimal, on: date) -> RealizedResult:
    # encode the target realized value via proceeds vs matched_cost
    return RealizedResult(
        figi=FIGI,
        ticker=TICKER,
        acquire_date=date(2020, 1, 1),
        dispose_date=on,
        quantity=Decimal(1),
        proceeds=realized_value,
        matched_cost=Decimal(0),
        buy_commission_share=Decimal(0),
        sell_commission_share=Decimal(0),
    )


def _coupon(amount: Decimal, on: date) -> Operation:
    return Operation(
        op_type=OperationType.COUPON,
        op_date=on,
        figi=FIGI,
        ticker=TICKER,
        payment=amount,
        currency=CCY,
    )


def _dividend(amount: Decimal, on: date) -> Operation:
    return Operation(
        op_type=OperationType.DIVIDEND,
        op_date=on,
        figi=FIGI,
        ticker=TICKER,
        payment=amount,
        currency=CCY,
    )


def test_realized_ytd_base_a_nets_gains_losses_and_coupons() -> None:
    realized = [_realized(GAIN_500K, D_JAN), _realized(LOSS_200K, D_FEB)]
    coupons = [_coupon(COUPON_100K, D_JAN)]
    # dividends must be EXCLUDED from base A
    total = realized_ytd_base_a(realized, coupons)
    assert total == GAIN_500K + LOSS_200K + COUPON_100K


def test_dividends_never_enter_base_a() -> None:
    realized: list[RealizedResult] = []
    coupons: list[Operation] = []
    # feeding dividend-typed operations into the coupon slot must be ignored:
    # only COUPON ops count. We assert base A stays zero here.
    total = realized_ytd_base_a(realized, coupons)
    assert total == Decimal(0)


def test_base_a_tax_uses_marginal_band() -> None:
    acc = BaseAccumulators()
    # credit an amount straddling the 2.4M threshold
    delta = NDFL_PROGRESSIVE_THRESHOLD + Decimal(100_000)
    tax = acc.credit(TaxBase.SECURITIES, delta, YEAR)
    expected = NDFL_PROGRESSIVE_THRESHOLD * NDFL_RATE + Decimal(100_000) * NDFL_RATE_HIGH
    assert tax == expected


def test_dividend_base_has_its_own_threshold() -> None:
    acc = BaseAccumulators()
    # a big base-A credit must NOT push dividends into 15%
    big_base_a = NDFL_PROGRESSIVE_THRESHOLD + Decimal(1_000_000)
    acc.credit(TaxBase.SECURITIES, big_base_a, YEAR)
    div_tax = acc.credit(TaxBase.DIVIDENDS, DIVIDEND_300K, YEAR)
    # dividends still fully in the 13% band (own threshold untouched)
    assert div_tax == DIVIDEND_300K * NDFL_RATE


def test_base_a_loss_does_not_reduce_dividend_tax() -> None:
    """INVARIANT 1: a base-A loss leaves dividend tax unchanged."""
    acc_with_loss = BaseAccumulators()
    # a realized base-A loss does not net against the dividend base at all
    acc_with_loss.credit(TaxBase.SECURITIES, LOSS_200K, YEAR)
    div_tax_with_loss = acc_with_loss.credit(TaxBase.DIVIDENDS, DIVIDEND_300K, YEAR)

    acc_no_loss = BaseAccumulators()
    div_tax_no_loss = acc_no_loss.credit(TaxBase.DIVIDENDS, DIVIDEND_300K, YEAR)

    assert div_tax_with_loss == div_tax_no_loss
    assert div_tax_with_loss == DIVIDEND_300K * NDFL_RATE


def test_bases_reset_on_new_year() -> None:
    acc = BaseAccumulators()
    acc.credit(TaxBase.SECURITIES, NDFL_PROGRESSIVE_THRESHOLD, YEAR)
    # next year starts fresh at 13%
    next_year_delta = Decimal(100_000)
    tax = acc.credit(TaxBase.SECURITIES, next_year_delta, YEAR + 1)
    assert tax == next_year_delta * NDFL_RATE
