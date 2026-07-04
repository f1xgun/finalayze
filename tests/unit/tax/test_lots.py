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
)

# --- named constants (no magic numbers) ---
FIGI_SBER = "BBG004730N88"
TICKER_SBER = "SBER"
CCY_RUB = "RUB"

QTY_100 = Decimal(100)
QTY_60 = Decimal(60)
QTY_40 = Decimal(40)
QTY_150 = Decimal(150)

PRICE_200 = Decimal(200)
PRICE_210 = Decimal(210)
PRICE_250 = Decimal(250)

COMMISSION_10 = Decimal(10)
COMMISSION_20 = Decimal(20)
COMMISSION_ZERO = Decimal(0)

DATE_BUY_1 = date(2021, 3, 1)
DATE_BUY_2 = date(2022, 6, 1)
DATE_SELL_1 = date(2026, 4, 1)


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
