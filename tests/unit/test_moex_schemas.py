"""Tests for MOEX data models (FXRate, KeyRateRecord, TurnoverRecord, MoexMarketData)."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest
from pydantic_core import ValidationError

from finalayze.core.schemas import (
    FXRate,
    KeyRateRecord,
    MarketContext,
    MoexMarketData,
    TurnoverRecord,
)


class TestFXRate:
    def test_create_valid(self) -> None:
        rate = FXRate(
            timestamp=datetime(2024, 1, 15, 0, 0, tzinfo=UTC),
            pair="USDRUB",
            rate=Decimal("89.50"),
        )
        assert rate.pair == "USDRUB"
        assert rate.rate == Decimal("89.50")
        assert rate.timestamp == datetime(2024, 1, 15, 0, 0, tzinfo=UTC)

    def test_frozen(self) -> None:
        rate = FXRate(
            timestamp=datetime(2024, 1, 15, 0, 0, tzinfo=UTC),
            pair="USDRUB",
            rate=Decimal("89.50"),
        )
        with pytest.raises(ValidationError):
            rate.rate = Decimal("90.00")  # type: ignore[misc]

    def test_naive_timestamp_rejected(self) -> None:
        with pytest.raises(ValueError, match="timezone-aware"):
            FXRate(
                timestamp=datetime(2024, 1, 15, 0, 0),  # noqa: DTZ001
                pair="USDRUB",
                rate=Decimal("89.50"),
            )


class TestKeyRateRecord:
    def test_create_valid(self) -> None:
        rec = KeyRateRecord(
            timestamp=datetime(2024, 1, 1, 0, 0, tzinfo=UTC),
            rate=Decimal("0.16"),  # 16% stored as decimal fraction
        )
        assert rec.rate == Decimal("0.16")

    def test_frozen(self) -> None:
        rec = KeyRateRecord(
            timestamp=datetime(2024, 1, 1, 0, 0, tzinfo=UTC),
            rate=Decimal("0.16"),
        )
        with pytest.raises(ValidationError):
            rec.rate = Decimal("0.21")  # type: ignore[misc]

    def test_naive_timestamp_rejected(self) -> None:
        with pytest.raises(ValueError, match="timezone-aware"):
            KeyRateRecord(
                timestamp=datetime(2024, 1, 1, 0, 0),  # noqa: DTZ001
                rate=Decimal("0.16"),
            )


class TestTurnoverRecord:
    def test_create_valid(self) -> None:
        rec = TurnoverRecord(
            timestamp=datetime(2024, 1, 15, 0, 0, tzinfo=UTC),
            volume_rub=Decimal(1500000000000),
        )
        assert rec.volume_rub == Decimal(1500000000000)

    def test_frozen(self) -> None:
        rec = TurnoverRecord(
            timestamp=datetime(2024, 1, 15, 0, 0, tzinfo=UTC),
            volume_rub=Decimal(1500000000000),
        )
        with pytest.raises(ValidationError):
            rec.volume_rub = Decimal(0)  # type: ignore[misc]

    def test_naive_timestamp_rejected(self) -> None:
        with pytest.raises(ValueError, match="timezone-aware"):
            TurnoverRecord(
                timestamp=datetime(2024, 1, 15, 0, 0),  # noqa: DTZ001
                volume_rub=Decimal(100),
            )


class TestMoexMarketData:
    def test_create_empty(self) -> None:
        data = MoexMarketData()
        assert data.fx_rates is None
        assert data.key_rates is None
        assert data.commodity_candles is None
        assert data.turnover is None

    def test_frozen(self) -> None:
        from dataclasses import FrozenInstanceError

        data = MoexMarketData()
        with pytest.raises(FrozenInstanceError):
            data.fx_rates = ()  # type: ignore[misc]

    def test_create_with_fx(self) -> None:
        fx = (
            FXRate(
                timestamp=datetime(2024, 1, 15, 0, 0, tzinfo=UTC),
                pair="USDRUB",
                rate=Decimal("89.50"),
            ),
        )
        data = MoexMarketData(fx_rates=fx)
        assert data.fx_rates is not None
        assert len(data.fx_rates) == 1


class TestMarketContextExtended:
    def test_moex_data_field(self) -> None:
        moex = MoexMarketData()
        ctx = MarketContext(moex_data=moex)
        assert ctx.moex_data is not None

    def test_backward_compat_no_moex(self) -> None:
        ctx = MarketContext()
        assert ctx.moex_data is None
        assert ctx.benchmark_candles is None
        assert ctx.vix_candles is None
