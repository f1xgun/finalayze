"""Tests for Task 1: fetch_all_bonds, CouponEvent schema, STREAM_COUPONS, ORM models."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from finalayze.core.schemas import CouponEvent


class TestCouponEventSchema:
    """CouponEvent schema must have all required fields."""

    def test_coupon_event_has_required_fields(self) -> None:
        event = CouponEvent(
            bond_figi="BBG123",
            bond_ticker="SU26238RMFS4",
            coupon_date=date(2026, 6, 15),
            record_date=date(2026, 6, 11),
            amount_per_bond=Decimal("35.50"),
            coupon_number=5,
        )
        assert event.bond_figi == "BBG123"
        assert event.bond_ticker == "SU26238RMFS4"
        assert event.coupon_date == date(2026, 6, 15)
        assert event.record_date == date(2026, 6, 11)
        assert event.amount_per_bond == Decimal("35.50")
        assert event.coupon_number == 5
        assert event.is_floating is False

    def test_coupon_event_is_floating(self) -> None:
        event = CouponEvent(
            bond_figi="BBG123",
            bond_ticker="SU29007RMFS0",
            coupon_date=date(2026, 6, 15),
            record_date=date(2026, 6, 11),
            amount_per_bond=Decimal("80.00"),
            coupon_number=3,
            is_floating=True,
        )
        assert event.is_floating is True

    def test_coupon_event_is_frozen(self) -> None:
        event = CouponEvent(
            bond_figi="BBG123",
            bond_ticker="SU26238RMFS4",
            coupon_date=date(2026, 6, 15),
            record_date=date(2026, 6, 11),
            amount_per_bond=Decimal("35.50"),
            coupon_number=5,
        )
        with pytest.raises(Exception):  # noqa: B017
            event.bond_figi = "BBG456"  # type: ignore[misc]


class TestStreamCouponsConstant:
    """EventBus must have STREAM_COUPONS constant."""

    def test_stream_coupons_exists(self) -> None:
        from finalayze.core.events import EventBus

        assert hasattr(EventBus, "STREAM_COUPONS")
        assert EventBus.STREAM_COUPONS == "coupons"


class TestBondCandleModel:
    """BondCandleModel ORM must have correct fields."""

    def test_bond_candle_model_fields(self) -> None:
        from finalayze.core.models import BondCandleModel

        model = BondCandleModel()
        # Verify field names exist on the class
        assert hasattr(BondCandleModel, "bond_figi")
        assert hasattr(BondCandleModel, "date")
        assert hasattr(BondCandleModel, "open")
        assert hasattr(BondCandleModel, "high")
        assert hasattr(BondCandleModel, "low")
        assert hasattr(BondCandleModel, "close")
        assert hasattr(BondCandleModel, "volume")

    def test_bond_candle_model_tablename(self) -> None:
        from finalayze.core.models import BondCandleModel

        assert BondCandleModel.__tablename__ == "bond_candles"


class TestCouponScheduleModel:
    """CouponScheduleModel ORM must have correct fields."""

    def test_coupon_schedule_model_fields(self) -> None:
        from finalayze.core.models import CouponScheduleModel

        assert hasattr(CouponScheduleModel, "bond_figi")
        assert hasattr(CouponScheduleModel, "coupon_number")
        assert hasattr(CouponScheduleModel, "coupon_date")
        assert hasattr(CouponScheduleModel, "record_date")
        assert hasattr(CouponScheduleModel, "amount_per_bond")

    def test_coupon_schedule_model_tablename(self) -> None:
        from finalayze.core.models import CouponScheduleModel

        assert CouponScheduleModel.__tablename__ == "coupon_schedules"


class TestAmortizationEventModel:
    """AmortizationEventModel ORM must have correct fields."""

    def test_amortization_event_model_fields(self) -> None:
        from finalayze.core.models import AmortizationEventModel

        assert hasattr(AmortizationEventModel, "bond_figi")
        assert hasattr(AmortizationEventModel, "event_date")
        assert hasattr(AmortizationEventModel, "event_number")
        assert hasattr(AmortizationEventModel, "remaining_nominal_pct")

    def test_amortization_event_model_tablename(self) -> None:
        from finalayze.core.models import AmortizationEventModel

        assert AmortizationEventModel.__tablename__ == "amortization_events"


class TestFetchAllBonds:
    """TinkoffFetcher.fetch_all_bonds() must exist and return bond metadata."""

    def test_fetch_all_bonds_method_exists(self) -> None:
        from finalayze.data.fetchers.tinkoff_data import TinkoffFetcher

        assert hasattr(TinkoffFetcher, "fetch_all_bonds")

    def test_fetch_amortization_schedule_method_exists(self) -> None:
        from finalayze.data.fetchers.tinkoff_data import TinkoffFetcher

        assert hasattr(TinkoffFetcher, "fetch_amortization_schedule")
