"""Unit tests for BondDiscoveryService."""

from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from finalayze.core.schemas import BondInfo
from finalayze.markets.instruments import InstrumentRegistry

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TODAY = date(2026, 3, 14)
FUTURE_MATURITY = TODAY + timedelta(days=365)
NEAR_MATURITY = TODAY + timedelta(days=60)  # < 3 months

OFZ_FIGI = "BBG011FJ4HS6"
CORP_FIGI = "BBG_CORP_001"


def _make_bond_dict(
    *,
    figi: str = OFZ_FIGI,
    ticker: str = "SU26238RMFS4",
    isin: str = "RU000A105YH5",
    name: str = "OFZ 26238",
    currency: str = "rub",
    maturity_date: date | None = None,
    risk_level: int = 1,
    liquidity_flag: bool = True,
    perpetual_flag: bool = False,
    api_trade_available_flag: bool = True,
    class_code: str = "TQOB",
    sector: str = "government",
    floating_coupon_flag: bool = False,
    amortization_flag: bool = False,
    coupon_quantity_per_year: int = 2,
    nominal: Decimal = Decimal(1000),
    initial_nominal: Decimal = Decimal(1000),
    bond_type: str = "",
    subordinated_flag: bool = False,
    lot: int = 1,
    aci_value: Decimal = Decimal("10.0"),
    call_date: date | None = None,
) -> dict:
    return {
        "figi": figi,
        "ticker": ticker,
        "isin": isin,
        "name": name,
        "currency": currency,
        "maturity_date": maturity_date or FUTURE_MATURITY,
        "risk_level": risk_level,
        "liquidity_flag": liquidity_flag,
        "perpetual_flag": perpetual_flag,
        "api_trade_available_flag": api_trade_available_flag,
        "class_code": class_code,
        "sector": sector,
        "floating_coupon_flag": floating_coupon_flag,
        "amortization_flag": amortization_flag,
        "coupon_quantity_per_year": coupon_quantity_per_year,
        "nominal": nominal,
        "initial_nominal": initial_nominal,
        "bond_type": bond_type,
        "subordinated_flag": subordinated_flag,
        "lot": lot,
        "aci_value": aci_value,
        "call_date": call_date,
    }


def _make_corporate_bond(**kwargs) -> dict:
    defaults = {
        "figi": CORP_FIGI,
        "ticker": "RU000A106A",
        "name": "Corporate Bond",
        "class_code": "TQCB",
        "sector": "corporate",
    }
    defaults.update(kwargs)
    return _make_bond_dict(**defaults)


def _make_mock_fetcher(bonds: list[dict] | None = None) -> MagicMock:
    fetcher = MagicMock()
    fetcher.fetch_all_bonds.return_value = bonds or []
    fetcher.fetch_amortization_schedule.return_value = []
    return fetcher


def _make_mock_event_bus() -> MagicMock:
    bus = MagicMock()
    bus.publish = AsyncMock(return_value="msg-id-1")
    return bus


class TestBondDiscoveryFilters:
    """Test individual and combined filter logic."""

    def test_filters_out_near_maturity(self) -> None:
        from finalayze.data.bond_discovery import BondDiscoveryService

        bond = _make_bond_dict(maturity_date=NEAR_MATURITY)
        fetcher = _make_mock_fetcher([bond])
        registry = InstrumentRegistry()
        service = BondDiscoveryService(fetcher, registry)
        import asyncio

        result = asyncio.run(service.discover(today=TODAY))
        assert result.filtered_count == 0

    def test_filters_out_high_risk(self) -> None:
        from finalayze.data.bond_discovery import BondDiscoveryService

        bond = _make_bond_dict(risk_level=3)
        fetcher = _make_mock_fetcher([bond])
        registry = InstrumentRegistry()
        service = BondDiscoveryService(fetcher, registry)
        import asyncio

        result = asyncio.run(service.discover(today=TODAY))
        assert result.filtered_count == 0

    def test_filters_out_non_rub(self) -> None:
        from finalayze.data.bond_discovery import BondDiscoveryService

        bond = _make_bond_dict(currency="usd")
        fetcher = _make_mock_fetcher([bond])
        registry = InstrumentRegistry()
        service = BondDiscoveryService(fetcher, registry)
        import asyncio

        result = asyncio.run(service.discover(today=TODAY))
        assert result.filtered_count == 0

    def test_filters_out_not_tradeable(self) -> None:
        from finalayze.data.bond_discovery import BondDiscoveryService

        bond = _make_bond_dict(api_trade_available_flag=False)
        fetcher = _make_mock_fetcher([bond])
        registry = InstrumentRegistry()
        service = BondDiscoveryService(fetcher, registry)
        import asyncio

        result = asyncio.run(service.discover(today=TODAY))
        assert result.filtered_count == 0

    def test_filters_out_illiquid(self) -> None:
        from finalayze.data.bond_discovery import BondDiscoveryService

        bond = _make_bond_dict(liquidity_flag=False)
        fetcher = _make_mock_fetcher([bond])
        registry = InstrumentRegistry()
        service = BondDiscoveryService(fetcher, registry)
        import asyncio

        result = asyncio.run(service.discover(today=TODAY))
        assert result.filtered_count == 0

    def test_filters_out_perpetual(self) -> None:
        from finalayze.data.bond_discovery import BondDiscoveryService

        bond = _make_bond_dict(perpetual_flag=True)
        fetcher = _make_mock_fetcher([bond])
        registry = InstrumentRegistry()
        service = BondDiscoveryService(fetcher, registry)
        import asyncio

        result = asyncio.run(service.discover(today=TODAY))
        assert result.filtered_count == 0

    def test_passes_valid_bond(self) -> None:
        from finalayze.data.bond_discovery import BondDiscoveryService

        bond = _make_bond_dict()
        fetcher = _make_mock_fetcher([bond])
        registry = InstrumentRegistry()
        service = BondDiscoveryService(fetcher, registry)
        import asyncio

        result = asyncio.run(service.discover(today=TODAY))
        assert result.filtered_count == 1


class TestBondDiscoverySegmentClassification:
    """Test OFZ vs corporate segment classification."""

    def test_ofz_segment_by_class_code(self) -> None:
        from finalayze.data.bond_discovery import BondDiscoveryService

        bond = _make_bond_dict(class_code="TQOB", sector="government")
        fetcher = _make_mock_fetcher([bond])
        registry = InstrumentRegistry()
        service = BondDiscoveryService(fetcher, registry)
        import asyncio

        result = asyncio.run(service.discover(today=TODAY))
        assert len(result.ofz) == 1
        assert len(result.corporate) == 0

    def test_ofz_segment_by_tqod(self) -> None:
        from finalayze.data.bond_discovery import BondDiscoveryService

        bond = _make_bond_dict(class_code="TQOD", sector="")
        fetcher = _make_mock_fetcher([bond])
        registry = InstrumentRegistry()
        service = BondDiscoveryService(fetcher, registry)
        import asyncio

        result = asyncio.run(service.discover(today=TODAY))
        assert len(result.ofz) == 1

    def test_corporate_segment(self) -> None:
        from finalayze.data.bond_discovery import BondDiscoveryService

        bond = _make_corporate_bond()
        fetcher = _make_mock_fetcher([bond])
        registry = InstrumentRegistry()
        service = BondDiscoveryService(fetcher, registry)
        import asyncio

        result = asyncio.run(service.discover(today=TODAY))
        assert len(result.ofz) == 0
        assert len(result.corporate) == 1

    def test_mixed_segments(self) -> None:
        from finalayze.data.bond_discovery import BondDiscoveryService

        ofz = _make_bond_dict(figi="FIGI_OFZ")
        corp = _make_corporate_bond(figi="FIGI_CORP")
        fetcher = _make_mock_fetcher([ofz, corp])
        registry = InstrumentRegistry()
        service = BondDiscoveryService(fetcher, registry)
        import asyncio

        result = asyncio.run(service.discover(today=TODAY))
        assert len(result.ofz) == 1
        assert len(result.corporate) == 1
        assert result.filtered_count == 2


class TestRegisterDiscoveredBonds:
    """Test Instrument registration."""

    def test_register_creates_instruments(self) -> None:
        from finalayze.data.bond_discovery import BondDiscoveryService, register_discovered_bonds

        bond = _make_bond_dict()
        fetcher = _make_mock_fetcher([bond])
        registry = InstrumentRegistry()
        service = BondDiscoveryService(fetcher, registry)
        import asyncio

        result = asyncio.run(service.discover(today=TODAY))
        count = register_discovered_bonds(result, registry)
        assert count == 1
        # Instrument should be in registry
        instruments = registry.list_by_type("moex", "bond")
        assert len(instruments) == 1
        inst = instruments[0]
        assert inst.instrument_type == "bond"
        assert inst.figi == OFZ_FIGI
        assert inst.segment_id == "ru_ofz"

    def test_register_corporate_segment(self) -> None:
        from finalayze.data.bond_discovery import BondDiscoveryService, register_discovered_bonds

        bond = _make_corporate_bond()
        fetcher = _make_mock_fetcher([bond])
        registry = InstrumentRegistry()
        service = BondDiscoveryService(fetcher, registry)
        import asyncio

        result = asyncio.run(service.discover(today=TODAY))
        count = register_discovered_bonds(result, registry)
        assert count == 1
        instruments = registry.list_by_type("moex", "bond")
        assert instruments[0].segment_id == "ru_corporate"


class TestAmortizationHandling:
    """Test amortizing bond handling."""

    def test_amortizing_bond_fetches_schedule(self) -> None:
        from finalayze.data.bond_discovery import BondDiscoveryService

        bond = _make_bond_dict(amortization_flag=True)
        fetcher = _make_mock_fetcher([bond])
        fetcher.fetch_amortization_schedule.return_value = [
            {"event_date": date(2027, 6, 15), "pay_one_bond": Decimal("50.0"), "event_number": 1}
        ]
        registry = InstrumentRegistry()
        service = BondDiscoveryService(fetcher, registry)
        import asyncio

        result = asyncio.run(service.discover(today=TODAY))
        assert result.filtered_count == 1
        fetcher.fetch_amortization_schedule.assert_called_once()


class TestEmptyBondList:
    """Test edge case: no bonds from API."""

    def test_empty_bond_list_returns_empty(self) -> None:
        from finalayze.data.bond_discovery import BondDiscoveryService

        fetcher = _make_mock_fetcher([])
        registry = InstrumentRegistry()
        service = BondDiscoveryService(fetcher, registry)
        import asyncio

        result = asyncio.run(service.discover(today=TODAY))
        assert result.total_count == 0
        assert result.filtered_count == 0
        assert len(result.ofz) == 0
        assert len(result.corporate) == 0


class TestCouponEventEmission:
    """Test coupon event emission logic."""

    def test_emits_on_matching_record_date(self) -> None:
        from finalayze.data.bond_discovery import BondDiscoveryService

        bond_info = BondInfo(
            figi=OFZ_FIGI,
            ticker="SU26238RMFS4",
            isin="RU000A105YH5",
            name="OFZ 26238",
            face_value=Decimal(1000),
            coupon_rate=Decimal("7.10"),
            coupon_frequency=2,
            maturity_date=FUTURE_MATURITY,
        )
        coupon_schedules = {
            OFZ_FIGI: [
                {
                    "coupon_date": date(2026, 3, 16),
                    "record_date": TODAY,
                    "amount_per_bond": Decimal("35.50"),
                    "coupon_number": 5,
                    "is_floating": False,
                }
            ]
        }

        bus = _make_mock_event_bus()
        fetcher = _make_mock_fetcher()
        registry = InstrumentRegistry()
        service = BondDiscoveryService(fetcher, registry, event_bus=bus)
        import asyncio

        count = asyncio.run(
            service.check_and_emit_coupon_events([bond_info], coupon_schedules, today=TODAY)
        )
        assert count == 1
        bus.publish.assert_called_once()

    def test_no_emit_on_non_matching_date(self) -> None:
        from finalayze.data.bond_discovery import BondDiscoveryService

        bond_info = BondInfo(
            figi=OFZ_FIGI,
            ticker="SU26238RMFS4",
            isin="RU000A105YH5",
            name="OFZ 26238",
            face_value=Decimal(1000),
            coupon_rate=Decimal("7.10"),
            coupon_frequency=2,
            maturity_date=FUTURE_MATURITY,
        )
        coupon_schedules = {
            OFZ_FIGI: [
                {
                    "coupon_date": date(2026, 6, 15),
                    "record_date": date(2026, 6, 11),
                    "amount_per_bond": Decimal("35.50"),
                    "coupon_number": 5,
                    "is_floating": False,
                }
            ]
        }

        bus = _make_mock_event_bus()
        fetcher = _make_mock_fetcher()
        registry = InstrumentRegistry()
        service = BondDiscoveryService(fetcher, registry, event_bus=bus)
        import asyncio

        count = asyncio.run(
            service.check_and_emit_coupon_events([bond_info], coupon_schedules, today=TODAY)
        )
        assert count == 0
        bus.publish.assert_not_called()

    def test_no_emit_when_no_event_bus(self) -> None:
        from finalayze.data.bond_discovery import BondDiscoveryService

        bond_info = BondInfo(
            figi=OFZ_FIGI,
            ticker="SU26238RMFS4",
            isin="RU000A105YH5",
            name="OFZ 26238",
            face_value=Decimal(1000),
            coupon_rate=Decimal("7.10"),
            coupon_frequency=2,
            maturity_date=FUTURE_MATURITY,
        )
        coupon_schedules = {
            OFZ_FIGI: [
                {
                    "coupon_date": date(2026, 3, 16),
                    "record_date": TODAY,
                    "amount_per_bond": Decimal("35.50"),
                    "coupon_number": 5,
                    "is_floating": False,
                }
            ]
        }

        fetcher = _make_mock_fetcher()
        registry = InstrumentRegistry()
        service = BondDiscoveryService(fetcher, registry, event_bus=None)
        import asyncio

        count = asyncio.run(
            service.check_and_emit_coupon_events([bond_info], coupon_schedules, today=TODAY)
        )
        assert count == 0


class TestInstrumentRegistryBondType:
    """Test that list_by_type returns discovered bonds."""

    def test_list_by_type_returns_registered_bonds(self) -> None:
        from finalayze.data.bond_discovery import BondDiscoveryService, register_discovered_bonds

        ofz = _make_bond_dict(figi="FIGI1", ticker="BOND1")
        corp = _make_corporate_bond(figi="FIGI2", ticker="BOND2")
        fetcher = _make_mock_fetcher([ofz, corp])
        registry = InstrumentRegistry()
        service = BondDiscoveryService(fetcher, registry)
        import asyncio

        result = asyncio.run(service.discover(today=TODAY))
        register_discovered_bonds(result, registry)

        bonds = registry.list_by_type("moex", "bond")
        assert len(bonds) == 2
        for b in bonds:
            assert b.instrument_type == "bond"
