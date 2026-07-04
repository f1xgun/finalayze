"""Tests for tax lots + strict-FIFO matching (design section 4.3 steps 1-2).

Named constants only (no magic numbers). All money math is Decimal.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from finalayze.tax.lots import (
    FifoMatchError,
    Operation,
    OperationType,
    RealizedResult,
    TaxLot,
    fifo_match,
    open_lots_after_match,
)

# --- named constants (no magic numbers) ---
FIGI_SBER = "BBG004730N88"
TICKER_SBER = "SBER"
CCY_RUB = "RUB"

QTY_100 = Decimal(100)
QTY_60 = Decimal(60)
QTY_40 = Decimal(40)
QTY_20 = Decimal(20)
QTY_10 = Decimal(10)
QTY_150 = Decimal(150)
QTY_NEG_5 = Decimal(-5)
QTY_NEG_10 = Decimal(-10)
QTY_ZERO = Decimal(0)

PRICE_100 = Decimal(100)
PRICE_200 = Decimal(200)
PRICE_210 = Decimal(210)
PRICE_250 = Decimal(250)
PRICE_300 = Decimal(300)

COMMISSION_10 = Decimal(10)
COMMISSION_20 = Decimal(20)
COMMISSION_ZERO = Decimal(0)

DATE_BUY_1 = date(2021, 3, 1)
DATE_BUY_2 = date(2022, 6, 1)
DATE_SELL_1 = date(2026, 4, 1)

# --- CR-01 chronology constants: a SELL dated BEFORE a later BUY ---
DATE_JAN = date(2024, 1, 1)
DATE_FEB = date(2024, 2, 1)
DATE_MAR = date(2024, 3, 1)
DATE_APR = date(2024, 4, 1)


def _buy(qty: Decimal, price: Decimal, commission: Decimal, on: date) -> Operation:
    return Operation(
        op_type=OperationType.BUY,
        op_date=on,
        figi=FIGI_SBER,
        ticker=TICKER_SBER,
        quantity=qty,
        price_per_unit=price,
        commission=commission,
        payment=-(qty * price) - commission,
        currency=CCY_RUB,
    )


def _sell(qty: Decimal, price: Decimal, commission: Decimal, on: date) -> Operation:
    return Operation(
        op_type=OperationType.SELL,
        op_date=on,
        figi=FIGI_SBER,
        ticker=TICKER_SBER,
        quantity=qty,
        price_per_unit=price,
        commission=commission,
        payment=(qty * price) - commission,
        currency=CCY_RUB,
    )


def test_taxlot_is_frozen_decimal_dataclass() -> None:
    lot = TaxLot(
        figi=FIGI_SBER,
        ticker=TICKER_SBER,
        acquire_date=DATE_BUY_1,
        quantity=QTY_100,
        price_per_unit=PRICE_200,
        commission_buy=COMMISSION_10,
        currency=CCY_RUB,
    )
    assert lot.quantity == QTY_100
    assert isinstance(lot.price_per_unit, Decimal)
    with pytest.raises((AttributeError, TypeError)):
        lot.quantity = QTY_60  # type: ignore[misc]


def test_operation_carries_signed_payment_and_type() -> None:
    op = _buy(QTY_100, PRICE_200, COMMISSION_10, DATE_BUY_1)
    assert op.op_type is OperationType.BUY
    # payment for a BUY is a cash outflow (negative)
    assert op.payment < COMMISSION_ZERO
    assert op.cost_basis_known is True


def test_fifo_single_full_close_realized_with_commissions() -> None:
    buys = [_buy(QTY_100, PRICE_200, COMMISSION_10, DATE_BUY_1)]
    sells = [_sell(QTY_100, PRICE_250, COMMISSION_20, DATE_SELL_1)]
    results = fifo_match(buys, sells)
    assert len(results) == 1
    r = results[0]
    # proceeds 100*250=25000; cost 100*200=20000; buy comm 10; sell comm 20
    expected = Decimal(25_000) - Decimal(20_000) - COMMISSION_10 - COMMISSION_20
    assert r.realized == expected
    assert r.quantity == QTY_100


def test_fifo_partial_close_oldest_lot_first() -> None:
    # two buys; one sell of 150 closes ALL of oldest 100 + 50 of the next lot
    buys = [
        _buy(QTY_100, PRICE_200, COMMISSION_10, DATE_BUY_1),
        _buy(QTY_100, PRICE_210, COMMISSION_10, DATE_BUY_2),
    ]
    sells = [_sell(QTY_150, PRICE_250, COMMISSION_20, DATE_SELL_1)]
    results = fifo_match(buys, sells)
    # one sell may produce multiple realized fragments (one per matched lot)
    assert sum(r.quantity for r in results) == QTY_150
    total_realized = sum((r.realized for r in results), Decimal(0))
    # proceeds = 150 * 250 = 37500
    # cost matched = 100*200 + 50*210 = 20000 + 10500 = 30500
    # buy comm share: full 10 for lot1 + (50/100)*10 = 5 for lot2 = 15
    # sell comm 20 (charged once against the sell)
    expected = Decimal(37_500) - Decimal(30_500) - Decimal(15) - COMMISSION_20
    assert total_realized == expected
    # the oldest lot must be fully consumed first
    first = results[0]
    assert first.acquire_date == DATE_BUY_1
    assert first.quantity == QTY_100


def test_fifo_loss_is_negative_realized() -> None:
    buys = [_buy(QTY_100, PRICE_250, COMMISSION_10, DATE_BUY_1)]
    sells = [_sell(QTY_100, PRICE_200, COMMISSION_10, DATE_SELL_1)]
    results = fifo_match(buys, sells)
    # proceeds 20000; cost 25000; comms 20 -> loss
    assert results[0].realized == Decimal(20_000) - Decimal(25_000) - COMMISSION_20
    assert results[0].realized < COMMISSION_ZERO


def test_fifo_oversell_without_basis_raises() -> None:
    buys = [_buy(QTY_100, PRICE_200, COMMISSION_10, DATE_BUY_1)]
    sells = [_sell(QTY_150, PRICE_250, COMMISSION_20, DATE_SELL_1)]
    with pytest.raises(FifoMatchError):
        fifo_match(buys, sells)


def test_realized_result_records_dates_for_ldv_clock_warning() -> None:
    buys = [_buy(QTY_100, PRICE_200, COMMISSION_10, DATE_BUY_1)]
    sells = [_sell(QTY_100, PRICE_250, COMMISSION_20, DATE_SELL_1)]
    r: RealizedResult = fifo_match(buys, sells)[0]
    assert r.acquire_date == DATE_BUY_1
    assert r.dispose_date == DATE_SELL_1


# ---------- CR-01: FIFO look-ahead (a SELL must NOT match a future BUY) ----------


def test_fifo_oversell_does_not_consume_future_buy() -> None:
    """CR-01: a Feb-1 sell of 20 when only 10 held (a Mar-1 buy exists) must RAISE.

    The March lot is dated strictly AFTER the February sell, so it did not exist
    on the sell date and must NOT be used to cover the oversell. This is the
    look-ahead-free guarantee (design 2.6 / R-3).
    """
    buys = [
        _buy(QTY_10, PRICE_100, COMMISSION_ZERO, DATE_JAN),
        _buy(QTY_10, PRICE_300, COMMISSION_ZERO, DATE_MAR),
    ]
    sells = [_sell(QTY_20, PRICE_250, COMMISSION_ZERO, DATE_FEB)]
    with pytest.raises(FifoMatchError):
        fifo_match(buys, sells)


def test_fifo_sell_never_matches_a_lot_acquired_after_the_sell() -> None:
    """CR-01: a partial sell before a later buy consumes ONLY the prior lot.

    Feb-1 SELL of 10 with a Jan-1 lot (10) and a Mar-1 lot (10): the fragment must
    match ONLY the Jan-1 lot; the Mar-1 lot must remain fully open afterwards.
    """
    buys = [
        _buy(QTY_10, PRICE_100, COMMISSION_ZERO, DATE_JAN),
        _buy(QTY_10, PRICE_300, COMMISSION_ZERO, DATE_MAR),
    ]
    sells = [_sell(QTY_10, PRICE_250, COMMISSION_ZERO, DATE_FEB)]
    results = fifo_match(buys, sells)
    assert len(results) == 1
    assert results[0].acquire_date == DATE_JAN
    assert results[0].matched_cost == QTY_10 * PRICE_100
    # the future March lot must survive fully open
    remaining = open_lots_after_match(buys, sells)
    assert len(remaining) == 1
    assert remaining[0].acquire_date == DATE_MAR
    assert remaining[0].quantity == QTY_10


def test_fifo_sell_before_any_buy_raises() -> None:
    """CR-01: a SELL strictly before any BUY has no matchable lot -> oversell."""
    buys = [_buy(QTY_10, PRICE_300, COMMISSION_ZERO, DATE_MAR)]
    sells = [_sell(QTY_10, PRICE_250, COMMISSION_ZERO, DATE_JAN)]
    with pytest.raises(FifoMatchError):
        fifo_match(buys, sells)


def test_fifo_same_day_buy_then_sell_is_matchable() -> None:
    """CR-01: a same-day buy settles BEFORE a same-day sell (buy-then-sell works)."""
    buys = [_buy(QTY_10, PRICE_100, COMMISSION_ZERO, DATE_FEB)]
    sells = [_sell(QTY_10, PRICE_250, COMMISSION_ZERO, DATE_FEB)]
    results = fifo_match(buys, sells)
    assert len(results) == 1
    assert results[0].acquire_date == DATE_FEB
    assert results[0].quantity == QTY_10


def test_fifo_interleaved_sells_only_match_prior_lots() -> None:
    """CR-01: interleaved buy/sell/buy/sell matches each sell only to prior lots."""
    buys = [
        _buy(QTY_10, PRICE_100, COMMISSION_ZERO, DATE_JAN),
        _buy(QTY_10, PRICE_300, COMMISSION_ZERO, DATE_MAR),
    ]
    sells = [
        _sell(QTY_10, PRICE_200, COMMISSION_ZERO, DATE_FEB),
        _sell(QTY_10, PRICE_250, COMMISSION_ZERO, DATE_APR),
    ]
    results = fifo_match(buys, sells)
    assert len(results) == 2
    by_dispose = sorted(results, key=lambda r: r.dispose_date)
    # Feb sell matched the Jan lot; Apr sell matched the Mar lot
    assert by_dispose[0].dispose_date == DATE_FEB
    assert by_dispose[0].acquire_date == DATE_JAN
    assert by_dispose[1].dispose_date == DATE_APR
    assert by_dispose[1].acquire_date == DATE_MAR


def test_open_lots_after_match_oversell_does_not_consume_future_buy() -> None:
    """CR-01: open_lots_after_match must raise on the same Feb oversell, not eat Mar."""
    buys = [
        _buy(QTY_10, PRICE_100, COMMISSION_ZERO, DATE_JAN),
        _buy(QTY_10, PRICE_300, COMMISSION_ZERO, DATE_MAR),
    ]
    sells = [_sell(QTY_20, PRICE_250, COMMISSION_ZERO, DATE_FEB)]
    with pytest.raises(FifoMatchError):
        open_lots_after_match(buys, sells)


# ---------- WR-05: malformed trade quantities must RAISE, not be swallowed ----------


def test_fifo_negative_sell_quantity_raises() -> None:
    buys = [_buy(QTY_100, PRICE_200, COMMISSION_10, DATE_BUY_1)]
    sells = [_sell(QTY_NEG_5, PRICE_250, COMMISSION_10, DATE_SELL_1)]
    with pytest.raises(FifoMatchError):
        fifo_match(buys, sells)


def test_fifo_zero_sell_quantity_raises() -> None:
    buys = [_buy(QTY_100, PRICE_200, COMMISSION_10, DATE_BUY_1)]
    sells = [_sell(QTY_ZERO, PRICE_250, COMMISSION_10, DATE_SELL_1)]
    with pytest.raises(FifoMatchError):
        fifo_match(buys, sells)


def test_fifo_negative_buy_quantity_raises() -> None:
    buys = [_buy(QTY_NEG_10, PRICE_200, COMMISSION_10, DATE_BUY_1)]
    sells = [_sell(QTY_100, PRICE_250, COMMISSION_10, DATE_SELL_1)]
    with pytest.raises(FifoMatchError):
        fifo_match(buys, sells)


def test_open_lots_after_match_negative_buy_quantity_raises() -> None:
    buys = [_buy(QTY_NEG_10, PRICE_200, COMMISSION_10, DATE_BUY_1)]
    sells: list[Operation] = []
    with pytest.raises(FifoMatchError):
        open_lots_after_match(buys, sells)
