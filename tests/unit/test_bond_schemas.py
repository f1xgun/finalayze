"""Unit tests for bond-related Pydantic schemas (Layer 0)."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from uuid import uuid4

import pytest
from pydantic import ValidationError

from finalayze.core.schemas import (
    AccruedInterest,
    BondInfo,
    CouponPayment,
    InstrumentType,
    Signal,
    SignalDirection,
    TradeResult,
)

# ── Constants (ruff PLR2004: no magic numbers) ──────────────────────────

FACE_VALUE = Decimal(1000)
COUPON_RATE = Decimal("7.10")
COUPON_FREQUENCY_SEMIANNUAL = 2
COUPON_AMOUNT = Decimal("35.50")
COUPON_NUMBER = 5
NKD_VALUE = Decimal("12.34")
NKD_PERCENT = Decimal("1.234")
SIGNAL_CONFIDENCE = 0.85
TRADE_QUANTITY = Decimal(10)
TRADE_ENTRY = Decimal("980.00")
TRADE_EXIT = Decimal("990.00")
TRADE_PNL = Decimal("100.00")
TRADE_PNL_PCT = Decimal("1.02")
COUPON_INCOME = Decimal("35.50")


# ── InstrumentType ──────────────────────────────────────────────────────


class TestInstrumentType:
    def test_stock_is_valid(self) -> None:
        value: InstrumentType = "stock"
        assert value == "stock"

    def test_etf_is_valid(self) -> None:
        value: InstrumentType = "etf"
        assert value == "etf"

    def test_bond_is_valid(self) -> None:
        value: InstrumentType = "bond"
        assert value == "bond"


# ── BondInfo ────────────────────────────────────────────────────────────


class TestBondInfo:
    @pytest.fixture
    def bond_info(self) -> BondInfo:
        return BondInfo(
            figi="BBG00T22WKV5",
            ticker="SU26238RMFS4",
            isin="RU000A1038V6",
            name="OFZ 26238",
            face_value=FACE_VALUE,
            coupon_rate=COUPON_RATE,
            coupon_frequency=COUPON_FREQUENCY_SEMIANNUAL,
            maturity_date=date(2041, 5, 15),
        )

    def test_creation(self, bond_info: BondInfo) -> None:
        assert bond_info.figi == "BBG00T22WKV5"
        assert bond_info.ticker == "SU26238RMFS4"
        assert bond_info.isin == "RU000A1038V6"
        assert bond_info.name == "OFZ 26238"
        assert bond_info.face_value == FACE_VALUE
        assert bond_info.coupon_rate == COUPON_RATE
        assert bond_info.coupon_frequency == COUPON_FREQUENCY_SEMIANNUAL
        assert bond_info.maturity_date == date(2041, 5, 15)

    def test_defaults(self, bond_info: BondInfo) -> None:
        assert bond_info.floating_coupon is False
        assert bond_info.class_code == "TQOB"
        assert bond_info.currency == "RUB"

    def test_frozen(self, bond_info: BondInfo) -> None:
        with pytest.raises(ValidationError):
            bond_info.ticker = "CHANGED"  # type: ignore[misc]

    def test_decimal_fields(self, bond_info: BondInfo) -> None:
        assert isinstance(bond_info.face_value, Decimal)
        assert isinstance(bond_info.coupon_rate, Decimal)

    def test_floating_coupon(self) -> None:
        bond = BondInfo(
            figi="BBG00TEST001",
            ticker="SU29014RMFS2",
            isin="RU000A105F30",
            name="OFZ 29014 (float)",
            face_value=FACE_VALUE,
            coupon_rate=Decimal(0),
            coupon_frequency=COUPON_FREQUENCY_SEMIANNUAL,
            maturity_date=date(2030, 3, 18),
            floating_coupon=True,
        )
        assert bond.floating_coupon is True


# ── CouponPayment ──────────────────────────────────────────────────────


class TestCouponPayment:
    @pytest.fixture
    def coupon(self) -> CouponPayment:
        return CouponPayment(
            bond_figi="BBG00T22WKV5",
            coupon_date=date(2025, 5, 15),
            record_date=date(2025, 5, 13),
            amount_per_bond=COUPON_AMOUNT,
            coupon_number=COUPON_NUMBER,
        )

    def test_creation(self, coupon: CouponPayment) -> None:
        assert coupon.bond_figi == "BBG00T22WKV5"
        assert coupon.coupon_date == date(2025, 5, 15)
        assert coupon.record_date == date(2025, 5, 13)
        assert coupon.amount_per_bond == COUPON_AMOUNT
        assert coupon.coupon_number == COUPON_NUMBER

    def test_frozen(self, coupon: CouponPayment) -> None:
        with pytest.raises(ValidationError):
            coupon.coupon_number = 99  # type: ignore[misc]

    def test_decimal_amount(self, coupon: CouponPayment) -> None:
        assert isinstance(coupon.amount_per_bond, Decimal)


# ── AccruedInterest ────────────────────────────────────────────────────


class TestAccruedInterest:
    @pytest.fixture
    def nkd(self) -> AccruedInterest:
        return AccruedInterest(
            bond_figi="BBG00T22WKV5",
            date=date(2025, 4, 10),
            value=NKD_VALUE,
            value_percent=NKD_PERCENT,
        )

    def test_creation(self, nkd: AccruedInterest) -> None:
        assert nkd.bond_figi == "BBG00T22WKV5"
        assert nkd.date == date(2025, 4, 10)
        assert nkd.value == NKD_VALUE
        assert nkd.value_percent == NKD_PERCENT

    def test_frozen(self, nkd: AccruedInterest) -> None:
        with pytest.raises(ValidationError):
            nkd.value = Decimal(0)  # type: ignore[misc]

    def test_decimal_fields(self, nkd: AccruedInterest) -> None:
        assert isinstance(nkd.value, Decimal)
        assert isinstance(nkd.value_percent, Decimal)


# ── Signal instrument_type field ───────────────────────────────────────


class TestSignalInstrumentType:
    def test_default_is_stock(self) -> None:
        signal = Signal(
            strategy_name="momentum_v1",
            symbol="AAPL",
            market_id="us",
            segment_id="us_tech",
            direction=SignalDirection.BUY,
            confidence=SIGNAL_CONFIDENCE,
            strategy_payload={"rsi": 65.0},
            reasoning="test",
        )
        assert signal.instrument_type == "stock"

    def test_bond_instrument_type(self) -> None:
        signal = Signal(
            strategy_name="bond_carry",
            symbol="SU26238RMFS4",
            market_id="moex",
            segment_id="ru_bonds",
            direction=SignalDirection.BUY,
            confidence=SIGNAL_CONFIDENCE,
            strategy_payload={},
            reasoning="OFZ carry",
            instrument_type="bond",
        )
        assert signal.instrument_type == "bond"


# ── TradeResult bond fields ────────────────────────────────────────────


class TestTradeResultBondFields:
    def test_default_coupon_income_is_zero(self) -> None:
        trade = TradeResult(
            signal_id=uuid4(),
            symbol="AAPL",
            side="buy",
            quantity=TRADE_QUANTITY,
            entry_price=TRADE_ENTRY,
            exit_price=TRADE_EXIT,
            pnl=TRADE_PNL,
            pnl_pct=TRADE_PNL_PCT,
        )
        assert trade.coupon_income == Decimal(0)
        assert trade.instrument_type == "stock"

    def test_bond_trade_with_coupon_income(self) -> None:
        trade = TradeResult(
            signal_id=uuid4(),
            symbol="SU26238RMFS4",
            side="buy",
            quantity=TRADE_QUANTITY,
            entry_price=TRADE_ENTRY,
            exit_price=TRADE_EXIT,
            pnl=TRADE_PNL,
            pnl_pct=TRADE_PNL_PCT,
            coupon_income=COUPON_INCOME,
            instrument_type="bond",
        )
        assert trade.coupon_income == COUPON_INCOME
        assert trade.instrument_type == "bond"
        assert isinstance(trade.coupon_income, Decimal)
