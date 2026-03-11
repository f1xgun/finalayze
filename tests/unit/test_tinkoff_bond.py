"""Unit tests for TinkoffFetcher bond data methods."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from finalayze.core.exceptions import DataFetchError
from finalayze.core.schemas import AccruedInterest, BondInfo, CouponPayment
from finalayze.data.fetchers.tinkoff_data import TinkoffFetcher
from finalayze.markets.instruments import DEFAULT_MOEX_INSTRUMENTS, Instrument, InstrumentRegistry

# ---------- constants ----------

FAKE_TOKEN = "fake_token"  # noqa: S105
FAKE_FIGI = "BBG00T22WKV5"
FAKE_BOND_SYMBOL = "SU99999RMFS0"
NANO_HALF = 500_000_000
FACE_VALUE_UNITS = 1000
COUPON_AMOUNT_UNITS = 35
COUPON_AMOUNT_NANO = 500_000_000  # 35.50
NKD_VALUE_UNITS = 12
NKD_VALUE_NANO = 300_000_000  # 12.30
NKD_PERCENT_UNITS = 1
NKD_PERCENT_NANO = 230_000_000  # 1.23


# ---------- helpers ----------


def _make_registry() -> InstrumentRegistry:
    registry = InstrumentRegistry()
    for inst in DEFAULT_MOEX_INSTRUMENTS:
        registry.register(inst)
    # Register a fake bond so fetch_bond_info/coupons/NKD can resolve symbol → FIGI
    registry.register(
        Instrument(
            symbol=FAKE_BOND_SYMBOL,
            market_id="moex",
            name="Fake OFZ",
            instrument_type="bond",
            figi=FAKE_FIGI,
            currency="RUB",
        )
    )
    return registry


def _make_fetcher() -> TinkoffFetcher:
    return TinkoffFetcher(token=FAKE_TOKEN, registry=_make_registry(), sandbox=True)


def _make_fake_money(units: int, nano: int, currency: str = "RUB") -> MagicMock:
    """Build a fake MoneyValue."""
    m = MagicMock()
    m.units = units
    m.nano = nano
    m.currency = currency
    return m


def _make_fake_quotation(units: int, nano: int) -> MagicMock:
    """Build a fake Quotation."""
    q = MagicMock()
    q.units = units
    q.nano = nano
    return q


# ---------- _money_to_decimal ----------


class TestMoneyToDecimal:
    def test_whole_number(self) -> None:
        fetcher = _make_fetcher()
        m = _make_fake_money(units=FACE_VALUE_UNITS, nano=0)
        result = fetcher._money_to_decimal(m)
        assert result == Decimal(FACE_VALUE_UNITS)

    def test_fractional(self) -> None:
        fetcher = _make_fetcher()
        m = _make_fake_money(units=COUPON_AMOUNT_UNITS, nano=COUPON_AMOUNT_NANO)
        result = fetcher._money_to_decimal(m)
        assert result == Decimal("35.5")

    def test_zero(self) -> None:
        fetcher = _make_fetcher()
        m = _make_fake_money(units=0, nano=0)
        result = fetcher._money_to_decimal(m)
        assert result == Decimal(0)


# ---------- _business_days_before ----------


class TestBusinessDaysBefore:
    def test_simple_weekday(self) -> None:
        """Wednesday - 2 business days = Monday."""
        d = date(2026, 3, 11)  # Wednesday
        result = TinkoffFetcher._business_days_before(d, 2)
        assert result == date(2026, 3, 9)  # Monday

    def test_crosses_weekend(self) -> None:
        """Monday - 2 business days = Thursday (skips Sat, Sun)."""
        d = date(2026, 3, 9)  # Monday
        result = TinkoffFetcher._business_days_before(d, 2)
        assert result == date(2026, 3, 5)  # Thursday

    def test_tuesday_minus_two(self) -> None:
        """Tuesday - 2 business days = Friday (skips Sat, Sun)."""
        d = date(2026, 3, 10)  # Tuesday
        result = TinkoffFetcher._business_days_before(d, 2)
        assert result == date(2026, 3, 6)  # Friday

    def test_zero_days(self) -> None:
        """0 business days back = same date."""
        d = date(2026, 3, 11)
        result = TinkoffFetcher._business_days_before(d, 0)
        assert result == d

    def test_one_day_from_monday(self) -> None:
        """Monday - 1 business day = Friday."""
        d = date(2026, 3, 9)  # Monday
        result = TinkoffFetcher._business_days_before(d, 1)
        assert result == date(2026, 3, 6)  # Friday


# ---------- fetch_bond_info ----------


class TestFetchBondInfo:
    def test_grpc_error_raises_data_fetch_error(self) -> None:
        """gRPC failure must be wrapped in DataFetchError."""
        fetcher = _make_fetcher()
        with (
            patch(
                "finalayze.data.fetchers.tinkoff_data.asyncio.run",
                side_effect=RuntimeError("gRPC connection refused"),
            ),
            pytest.raises(DataFetchError, match="gRPC error fetching bond info"),
        ):
            fetcher.fetch_bond_info(FAKE_BOND_SYMBOL)

    def test_returns_bond_info(self) -> None:
        """Mocked async method returns correct BondInfo."""
        fetcher = _make_fetcher()

        expected = BondInfo(
            figi=FAKE_FIGI,
            ticker="SU26238RMFS4",
            isin="RU000A1038V6",
            name="OFZ 26238",
            face_value=Decimal(FACE_VALUE_UNITS),
            coupon_rate=Decimal(0),
            coupon_frequency=2,
            maturity_date=date(2041, 5, 15),
            floating_coupon=False,
            class_code="TQOB",
            currency="RUB",
        )

        with patch(
            "finalayze.data.fetchers.tinkoff_data.asyncio.run",
            return_value=expected,
        ):
            result = fetcher.fetch_bond_info(FAKE_BOND_SYMBOL)

        assert isinstance(result, BondInfo)
        assert result.figi == FAKE_FIGI
        assert result.ticker == "SU26238RMFS4"
        assert result.face_value == Decimal(FACE_VALUE_UNITS)
        assert result.coupon_frequency == 2
        assert result.maturity_date == date(2041, 5, 15)

    def test_rate_limiter_called(self) -> None:
        """Rate limiter acquire() must be called before the API call."""
        mock_limiter = MagicMock()
        fetcher = TinkoffFetcher(
            token=FAKE_TOKEN,
            registry=_make_registry(),
            sandbox=True,
            rate_limiter=mock_limiter,
        )

        expected = BondInfo(
            figi=FAKE_FIGI,
            ticker="SU26238RMFS4",
            isin="RU000A1038V6",
            name="OFZ 26238",
            face_value=Decimal(FACE_VALUE_UNITS),
            coupon_rate=Decimal(0),
            coupon_frequency=2,
            maturity_date=date(2041, 5, 15),
            floating_coupon=False,
            class_code="TQOB",
            currency="RUB",
        )

        with patch(
            "finalayze.data.fetchers.tinkoff_data.asyncio.run",
            return_value=expected,
        ):
            fetcher.fetch_bond_info(FAKE_BOND_SYMBOL)

        mock_limiter.acquire.assert_called_once()


# ---------- fetch_bond_coupons ----------


class TestFetchBondCoupons:
    def test_grpc_error_raises_data_fetch_error(self) -> None:
        fetcher = _make_fetcher()
        start = datetime(2024, 1, 1, tzinfo=UTC)
        end = datetime(2024, 12, 31, tzinfo=UTC)
        with (
            patch(
                "finalayze.data.fetchers.tinkoff_data.asyncio.run",
                side_effect=RuntimeError("gRPC error"),
            ),
            pytest.raises(DataFetchError, match="gRPC error fetching coupons"),
        ):
            fetcher.fetch_bond_coupons(FAKE_BOND_SYMBOL, start, end)

    def test_returns_coupon_payments(self) -> None:
        """Mocked async method returns list of CouponPayment."""
        fetcher = _make_fetcher()
        start = datetime(2024, 1, 1, tzinfo=UTC)
        end = datetime(2024, 12, 31, tzinfo=UTC)

        expected = [
            CouponPayment(
                bond_figi=FAKE_FIGI,
                coupon_date=date(2024, 5, 15),
                record_date=date(2024, 5, 13),
                amount_per_bond=Decimal("35.50"),
                coupon_number=1,
            ),
            CouponPayment(
                bond_figi=FAKE_FIGI,
                coupon_date=date(2024, 11, 15),
                record_date=date(2024, 11, 13),
                amount_per_bond=Decimal("35.50"),
                coupon_number=2,
            ),
        ]

        with patch(
            "finalayze.data.fetchers.tinkoff_data.asyncio.run",
            return_value=expected,
        ):
            result = fetcher.fetch_bond_coupons(FAKE_BOND_SYMBOL, start, end)

        assert len(result) == 2
        assert all(isinstance(c, CouponPayment) for c in result)
        assert result[0].bond_figi == FAKE_FIGI
        assert result[0].coupon_date == date(2024, 5, 15)
        assert result[0].amount_per_bond == Decimal("35.50")
        assert result[1].coupon_number == 2


# ---------- fetch_accrued_interest ----------


class TestFetchAccruedInterest:
    def test_grpc_error_raises_data_fetch_error(self) -> None:
        fetcher = _make_fetcher()
        start = datetime(2024, 1, 1, tzinfo=UTC)
        end = datetime(2024, 1, 5, tzinfo=UTC)
        with (
            patch(
                "finalayze.data.fetchers.tinkoff_data.asyncio.run",
                side_effect=RuntimeError("gRPC error"),
            ),
            pytest.raises(DataFetchError, match="gRPC error fetching NKD"),
        ):
            fetcher.fetch_accrued_interest(FAKE_BOND_SYMBOL, start, end)

    def test_returns_accrued_interest_list(self) -> None:
        """Mocked async method returns list of AccruedInterest."""
        fetcher = _make_fetcher()
        start = datetime(2024, 1, 1, tzinfo=UTC)
        end = datetime(2024, 1, 3, tzinfo=UTC)

        expected = [
            AccruedInterest(
                bond_figi=FAKE_FIGI,
                date=date(2024, 1, 1),
                value=Decimal("12.30"),
                value_percent=Decimal("1.23"),
            ),
            AccruedInterest(
                bond_figi=FAKE_FIGI,
                date=date(2024, 1, 2),
                value=Decimal("12.50"),
                value_percent=Decimal("1.25"),
            ),
        ]

        with patch(
            "finalayze.data.fetchers.tinkoff_data.asyncio.run",
            return_value=expected,
        ):
            result = fetcher.fetch_accrued_interest(FAKE_BOND_SYMBOL, start, end)

        assert len(result) == 2
        assert all(isinstance(ai, AccruedInterest) for ai in result)
        assert result[0].bond_figi == FAKE_FIGI
        assert result[0].date == date(2024, 1, 1)
        assert result[0].value == Decimal("12.30")
        assert result[0].value_percent == Decimal("1.23")


# ---------- async methods integration (mock SDK responses) ----------


class TestFetchBondInfoAsync:
    """Test _fetch_bond_info_async with mocked SDK client."""

    def test_async_maps_sdk_response(self) -> None:
        fetcher = _make_fetcher()

        # Build mock bond instrument
        mock_bond = MagicMock()
        mock_bond.figi = FAKE_FIGI
        mock_bond.ticker = "SU26238RMFS4"
        mock_bond.isin = "RU000A1038V6"
        mock_bond.name = "OFZ 26238"
        mock_bond.nominal = _make_fake_money(FACE_VALUE_UNITS, 0)
        mock_bond.coupon_quantity_per_year = 2
        mock_bond.maturity_date = datetime(2041, 5, 15, tzinfo=UTC)
        mock_bond.floating_coupon_flag = False
        mock_bond.class_code = "TQOB"
        mock_bond.currency = "rub"

        mock_resp = MagicMock()
        mock_resp.instrument = mock_bond

        # Mock SDK client
        mock_services = MagicMock()
        mock_services.instruments.bond_by = AsyncMock(return_value=mock_resp)

        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_services)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch.object(fetcher, "_make_client", return_value=mock_client):
            import asyncio

            result = asyncio.run(fetcher._fetch_bond_info_async(FAKE_FIGI))

        assert isinstance(result, BondInfo)
        assert result.figi == FAKE_FIGI
        assert result.face_value == Decimal(FACE_VALUE_UNITS)
        assert result.maturity_date == date(2041, 5, 15)
        assert result.coupon_frequency == 2
        assert result.floating_coupon is False


class TestFetchBondCouponsAsync:
    """Test _fetch_bond_coupons_async with mocked SDK client."""

    def test_async_maps_sdk_response(self) -> None:
        fetcher = _make_fetcher()
        start = datetime(2024, 1, 1, tzinfo=UTC)
        end = datetime(2024, 12, 31, tzinfo=UTC)

        # Build mock coupon events
        mock_coupon = MagicMock()
        mock_coupon.pay_one_bond = _make_fake_money(COUPON_AMOUNT_UNITS, COUPON_AMOUNT_NANO)
        mock_coupon.coupon_date = datetime(2024, 5, 15, tzinfo=UTC)
        mock_coupon.coupon_number = 1

        mock_resp = MagicMock()
        mock_resp.events = [mock_coupon]

        mock_services = MagicMock()
        mock_services.instruments.get_bond_coupons = AsyncMock(return_value=mock_resp)

        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_services)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch.object(fetcher, "_make_client", return_value=mock_client):
            import asyncio

            result = asyncio.run(fetcher._fetch_bond_coupons_async(FAKE_FIGI, start, end))

        assert len(result) == 1
        assert isinstance(result[0], CouponPayment)
        assert result[0].bond_figi == FAKE_FIGI
        assert result[0].amount_per_bond == Decimal("35.5")
        assert result[0].coupon_date == date(2024, 5, 15)
        # record_date = 2 business days before May 15 (Wed) = May 13 (Mon)
        assert result[0].record_date == date(2024, 5, 13)
        assert result[0].coupon_number == 1


class TestFetchAccruedInterestAsync:
    """Test _fetch_accrued_interest_async with mocked SDK client."""

    def test_async_maps_sdk_response(self) -> None:
        fetcher = _make_fetcher()
        start = datetime(2024, 1, 1, tzinfo=UTC)
        end = datetime(2024, 1, 3, tzinfo=UTC)

        mock_ai = MagicMock()
        mock_ai.value = _make_fake_money(NKD_VALUE_UNITS, NKD_VALUE_NANO)
        mock_ai.value_percent = _make_fake_quotation(NKD_PERCENT_UNITS, NKD_PERCENT_NANO)
        mock_ai.date = datetime(2024, 1, 2, tzinfo=UTC)

        mock_resp = MagicMock()
        mock_resp.accrued_interests = [mock_ai]

        mock_services = MagicMock()
        mock_services.instruments.get_accrued_interests = AsyncMock(return_value=mock_resp)

        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_services)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch.object(fetcher, "_make_client", return_value=mock_client):
            import asyncio

            result = asyncio.run(fetcher._fetch_accrued_interest_async(FAKE_FIGI, start, end))

        assert len(result) == 1
        assert isinstance(result[0], AccruedInterest)
        assert result[0].bond_figi == FAKE_FIGI
        assert result[0].date == date(2024, 1, 2)
        assert result[0].value == Decimal("12.3")
        assert result[0].value_percent == Decimal("1.23")
