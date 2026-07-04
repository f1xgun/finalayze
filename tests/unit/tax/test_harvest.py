"""Tests for tax-loss harvesting vs YTD base-A result (design section 4.3 step 7).

savings = min(harvestable_loss, positive_YTD_base_A) * effective_rate.
A harvested loss offsets coupons (base A) but NEVER dividends. Warn on the
FIFO-effect: a harvest sale can reset another lot's LDV clock.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from finalayze.core.constants import NDFL_RATE, NDFL_RATE_HIGH
from finalayze.tax.harvest import (
    HarvestCandidate,
    HarvestError,
    HarvestResult,
    harvestable,
)
from finalayze.tax.lots import TaxLot

TICKER = "GAZP"
FIGI = "BBG004730RP0"
CCY = "RUB"

YTD_POSITIVE_500K = Decimal(500_000)
YTD_POSITIVE_100K = Decimal(100_000)
LOSS_300K = Decimal(-300_000)
LOSS_800K = Decimal(-800_000)

QTY = Decimal(50)
PRICE = Decimal(300)
COMM = Decimal(5)
ACQ_OLD = date(2021, 1, 1)  # would be LDV-eligible soon
ACQ_NEW = date(2025, 1, 1)


def _candidate(loss: Decimal, *, resets_ldv: bool = False) -> HarvestCandidate:
    lot = TaxLot(
        figi=FIGI,
        ticker=TICKER,
        acquire_date=ACQ_OLD if resets_ldv else ACQ_NEW,
        quantity=QTY,
        price_per_unit=PRICE,
        commission_buy=COMM,
        currency=CCY,
    )
    return HarvestCandidate(lot=lot, unrealized_loss=loss, breaks_ldv_clock=resets_ldv)


def test_savings_capped_by_positive_ytd_base_a() -> None:
    # loss 800k but only 500k positive YTD -> only 500k is usable
    res: HarvestResult = harvestable(YTD_POSITIVE_500K, [_candidate(LOSS_800K)])
    assert res.offset_used == YTD_POSITIVE_500K
    # effective rate = 13% below threshold
    assert res.savings_estimate == YTD_POSITIVE_500K * NDFL_RATE


def test_savings_capped_by_harvestable_loss() -> None:
    # loss 300k, positive YTD 500k -> only 300k of loss usable
    res = harvestable(YTD_POSITIVE_500K, [_candidate(LOSS_300K)])
    assert res.offset_used == abs(LOSS_300K)
    assert res.savings_estimate == abs(LOSS_300K) * NDFL_RATE


def test_no_savings_when_no_positive_ytd() -> None:
    res = harvestable(Decimal(0), [_candidate(LOSS_300K)])
    assert res.offset_used == Decimal(0)
    assert res.savings_estimate == Decimal(0)


def test_no_savings_when_ytd_negative() -> None:
    res = harvestable(Decimal(-100_000), [_candidate(LOSS_300K)])
    assert res.offset_used == Decimal(0)
    assert res.savings_estimate == Decimal(0)


def test_effective_rate_uses_15_above_threshold() -> None:
    # a large positive YTD already above the 2.4M threshold: marginal offset
    # relieves tax at 15%
    big_ytd = Decimal(3_000_000)
    res = harvestable(big_ytd, [_candidate(Decimal(-500_000))])
    # offset 500k comes off the TOP (above threshold) -> relieved at 15%
    assert res.savings_estimate == Decimal(500_000) * NDFL_RATE_HIGH


def test_dividends_never_harvested() -> None:
    # harvestable() must accept NO dividend argument and only touch base A.
    # We assert the offset only ever equals min(loss, positive base-A YTD),
    # independent of any dividend income (which is not an input here at all).
    res = harvestable(YTD_POSITIVE_100K, [_candidate(LOSS_800K)])
    assert res.offset_used == YTD_POSITIVE_100K
    assert res.savings_estimate == YTD_POSITIVE_100K * NDFL_RATE


def test_ldv_clock_reset_warning_emitted() -> None:
    res = harvestable(YTD_POSITIVE_500K, [_candidate(LOSS_300K, resets_ldv=True)])
    assert any("LDV" in w for w in res.warnings)


def test_ldv_clock_warning_does_not_misattribute_sold_lot_date() -> None:
    """IN-05: the warning must NOT claim the reset lot was 'acquired {sold-lot date}'.

    The FIFO sale of the loss lot resets ANOTHER lot's LDV clock; naming the sold
    loss lot's own acquire date as the affected lot's date misleads the operator
    about which position is affected (design 2.4).
    """
    candidate = _candidate(LOSS_300K, resets_ldv=True)
    res = harvestable(YTD_POSITIVE_500K, [candidate])
    ldv_warnings = [w for w in res.warnings if "LDV" in w]
    assert ldv_warnings
    # the sold loss lot's own acquire date must not be presented as the reset lot's
    assert f"acquired {candidate.lot.acquire_date}" not in ldv_warnings[0]
    # but the sold ticker (the lot being harvested) should still be named
    assert candidate.lot.ticker in ldv_warnings[0]


def test_no_ldv_warning_when_clock_not_broken() -> None:
    res = harvestable(YTD_POSITIVE_500K, [_candidate(LOSS_300K, resets_ldv=False)])
    assert not any("LDV" in w for w in res.warnings)


def test_positive_pnl_candidate_rejected() -> None:
    """IN-03: a candidate carrying a GAIN (positive P&L) is not harvestable -> raise.

    Passing a positive value would inflate offset_used via abs(); a gain is not a
    harvestable loss, so it must raise rather than silently over-count savings.
    """
    gain = Decimal(100_000)
    bad_candidate = _candidate(gain)
    with pytest.raises(HarvestError):
        harvestable(YTD_POSITIVE_500K, [bad_candidate])


def test_zero_pnl_candidate_is_ignored_not_an_error() -> None:
    """IN-03: a zero-loss candidate contributes nothing and is not an error."""
    res = harvestable(YTD_POSITIVE_500K, [_candidate(Decimal(0))])
    assert res.offset_used == Decimal(0)
    assert res.savings_estimate == Decimal(0)
