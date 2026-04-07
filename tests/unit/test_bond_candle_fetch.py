"""Tests for bond candle fetching and cache population."""

from __future__ import annotations

from datetime import UTC, date, datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from finalayze.core.schemas import BondInfo

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BOND_FIGI = "BBG011FJ4HS6"
OPEN_PRICE_UNITS = 95
OPEN_PRICE_NANO = 500_000_000  # 95.5
CLOSE_PRICE_UNITS = 96
CLOSE_PRICE_NANO = 250_000_000  # 96.25
HIGH_PRICE_UNITS = 97
HIGH_PRICE_NANO = 0
LOW_PRICE_UNITS = 94
LOW_PRICE_NANO = 750_000_000  # 94.75
FAKE_VOLUME = 50_000
EXPECTED_CANDLE_COUNT = 1


def _make_bond_info(figi: str = BOND_FIGI) -> BondInfo:
    return BondInfo(
        figi=figi,
        ticker="SU26238RMFS4",
        isin="RU000A105YH5",
        name="OFZ 26238",
        face_value=Decimal(1000),
        coupon_rate=Decimal("7.10"),
        coupon_frequency=2,
        maturity_date=date(2041, 5, 15),
    )


def _make_fake_candle() -> MagicMock:
    """Build a fake T-Invest HistoricCandle for bonds."""
    candle = MagicMock()
    candle.open.units = OPEN_PRICE_UNITS
    candle.open.nano = OPEN_PRICE_NANO
    candle.close.units = CLOSE_PRICE_UNITS
    candle.close.nano = CLOSE_PRICE_NANO
    candle.high.units = HIGH_PRICE_UNITS
    candle.high.nano = HIGH_PRICE_NANO
    candle.low.units = LOW_PRICE_UNITS
    candle.low.nano = LOW_PRICE_NANO
    candle.volume = FAKE_VOLUME
    candle.time = datetime(2026, 3, 10, tzinfo=UTC)
    return candle


class TestFetchBondCandles:
    """TinkoffFetcher.fetch_bond_candles() tests."""

    def test_fetch_returns_candle_dicts(self) -> None:
        from finalayze.data.fetchers.tinkoff_data import TinkoffFetcher
        from finalayze.markets.instruments import InstrumentRegistry

        _make_fake_candle()
        registry = InstrumentRegistry()
        fetcher = TinkoffFetcher(
            token="fake",  # noqa: S106
            registry=registry,
            sandbox=True,
        )

        candle_data = [
            {
                "date": date(2026, 3, 10),
                "open": Decimal("95.5"),
                "high": Decimal(97),
                "low": Decimal("94.75"),
                "close": Decimal("96.25"),
                "volume": FAKE_VOLUME,
            }
        ]
        with patch.object(fetcher, "_run_async", return_value=candle_data):
            candles = fetcher.fetch_bond_candles(BOND_FIGI, date(2026, 3, 1), date(2026, 3, 14))

        assert len(candles) == EXPECTED_CANDLE_COUNT
        c = candles[0]
        assert isinstance(c, dict)
        assert "date" in c
        assert "open" in c
        assert "close" in c
        assert "volume" in c

    def test_fetch_empty_response(self) -> None:
        from finalayze.data.fetchers.tinkoff_data import TinkoffFetcher
        from finalayze.markets.instruments import InstrumentRegistry

        registry = InstrumentRegistry()
        fetcher = TinkoffFetcher(
            token="fake",  # noqa: S106
            registry=registry,
        )

        with patch.object(fetcher, "_run_async", return_value=[]):
            candles = fetcher.fetch_bond_candles(BOND_FIGI, date(2026, 3, 1), date(2026, 3, 14))

        assert candles == []

    def test_fetch_converts_quotation_to_decimal(self) -> None:
        from finalayze.data.fetchers.tinkoff_data import TinkoffFetcher
        from finalayze.markets.instruments import InstrumentRegistry

        registry = InstrumentRegistry()
        fetcher = TinkoffFetcher(
            token="fake",  # noqa: S106
            registry=registry,
        )

        candle_data = [
            {
                "date": date(2026, 3, 10),
                "open": Decimal("95.5"),
                "high": Decimal(97),
                "low": Decimal("94.75"),
                "close": Decimal("96.25"),
                "volume": FAKE_VOLUME,
            }
        ]
        with patch.object(fetcher, "_run_async", return_value=candle_data):
            candles = fetcher.fetch_bond_candles(BOND_FIGI, date(2026, 3, 1), date(2026, 3, 14))

        assert isinstance(candles[0]["open"], Decimal)
        assert isinstance(candles[0]["close"], Decimal)

    def test_fetch_handles_grpc_error(self) -> None:
        from finalayze.data.fetchers.tinkoff_data import TinkoffFetcher
        from finalayze.markets.instruments import InstrumentRegistry

        registry = InstrumentRegistry()
        fetcher = TinkoffFetcher(
            token="fake",  # noqa: S106
            registry=registry,
        )

        with patch.object(fetcher, "_run_async", side_effect=RuntimeError("gRPC error")):
            candles = fetcher.fetch_bond_candles(BOND_FIGI, date(2026, 3, 1), date(2026, 3, 14))

        assert candles == []


class TestPopulateCandleCache:
    """BondDiscoveryService.populate_candle_cache() tests."""

    def test_populate_writes_candle_models(self) -> None:
        from finalayze.data.bond_discovery import BondDiscoveryService
        from finalayze.markets.instruments import InstrumentRegistry

        bond_info = _make_bond_info()
        fetcher = MagicMock()
        fetcher.fetch_bond_candles.return_value = [
            {
                "date": date(2026, 3, 10),
                "open": Decimal("95.5"),
                "high": Decimal(97),
                "low": Decimal("94.75"),
                "close": Decimal("96.25"),
                "volume": FAKE_VOLUME,
            }
        ]

        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        _scalar_none = MagicMock(scalar=MagicMock(return_value=None))
        mock_session.execute = AsyncMock(return_value=_scalar_none)
        mock_session.add_all = MagicMock()
        mock_session.commit = AsyncMock()

        mock_factory = MagicMock(return_value=mock_session)

        registry = InstrumentRegistry()
        service = BondDiscoveryService(fetcher, registry)
        import asyncio

        count = asyncio.run(service.populate_candle_cache([bond_info], mock_factory))
        assert count == EXPECTED_CANDLE_COUNT

    def test_populate_handles_fetch_error_for_individual_bond(self) -> None:
        from finalayze.data.bond_discovery import BondDiscoveryService
        from finalayze.markets.instruments import InstrumentRegistry

        bond1 = _make_bond_info(figi="FIGI_OK")
        bond2 = _make_bond_info(figi="FIGI_FAIL")

        fetcher = MagicMock()

        def fetch_side_effect(figi, *args, **kwargs):
            if figi == "FIGI_FAIL":
                raise RuntimeError("fetch error")
            return [
                {
                    "date": date(2026, 3, 10),
                    "open": Decimal("95.5"),
                    "high": Decimal(97),
                    "low": Decimal("94.75"),
                    "close": Decimal("96.25"),
                    "volume": FAKE_VOLUME,
                }
            ]

        fetcher.fetch_bond_candles.side_effect = fetch_side_effect

        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        _scalar_none = MagicMock(scalar=MagicMock(return_value=None))
        mock_session.execute = AsyncMock(return_value=_scalar_none)
        mock_session.add_all = MagicMock()
        mock_session.commit = AsyncMock()

        mock_factory = MagicMock(return_value=mock_session)

        registry = InstrumentRegistry()
        service = BondDiscoveryService(fetcher, registry)
        import asyncio

        # Should not raise -- logs warning and continues
        count = asyncio.run(service.populate_candle_cache([bond1, bond2], mock_factory))
        assert count == EXPECTED_CANDLE_COUNT  # only bond1 succeeded
