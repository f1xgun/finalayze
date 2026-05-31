"""Unit tests for TinkoffFetcher."""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from finalayze.core.exceptions import DataFetchError, InstrumentNotFoundError
from finalayze.data.fetchers.tinkoff_data import TinkoffFetcher
from finalayze.markets.instruments import DEFAULT_MOEX_INSTRUMENTS, InstrumentRegistry

# ---------- helpers ----------

FAKE_TOKEN = "fake_token"  # noqa: S105
SBER_SYMBOL = "SBER"
SBER_FIGI = "BBG004730N88"
UNKNOWN_SYMBOL = "UNKNOWN"
OPEN_PRICE = 270
CLOSE_PRICE = 275
HIGH_PRICE = 280
LOW_PRICE = 265
FAKE_VOLUME = 1_000_000
FAKE_TIMESTAMP = 1_700_000_000
NANO_HALF = 500_000_000


def _make_registry() -> InstrumentRegistry:
    registry = InstrumentRegistry()
    for inst in DEFAULT_MOEX_INSTRUMENTS:
        registry.register(inst)
    return registry


def _make_fetcher(sandbox: bool = True) -> TinkoffFetcher:
    return TinkoffFetcher(token=FAKE_TOKEN, registry=_make_registry(), sandbox=sandbox)


def _make_fake_candle(
    open_u: int,
    open_n: int,
    close_u: int,
    close_n: int,
    high_u: int,
    high_n: int,
    low_u: int,
    low_n: int,
    volume: int,
    time_seconds: int,
) -> MagicMock:
    """Build a fake Tinkoff HistoricCandle object."""
    candle = MagicMock()
    candle.open.units = open_u
    candle.open.nano = open_n
    candle.close.units = close_u
    candle.close.nano = close_n
    candle.high.units = high_u
    candle.high.nano = high_n
    candle.low.units = low_u
    candle.low.nano = low_n
    candle.volume = volume
    candle.time.seconds = time_seconds
    candle.time.nanos = 0
    return candle


# ---------- unit tests ----------


class TestTinkoffFetcherQuotationToDecimal:
    def test_whole_number(self) -> None:
        fetcher = _make_fetcher()
        q = MagicMock()
        q.units = OPEN_PRICE
        q.nano = 0
        assert fetcher._quotation_to_decimal(q) == Decimal(OPEN_PRICE)

    def test_fractional(self) -> None:
        fetcher = _make_fetcher()
        q = MagicMock()
        q.units = OPEN_PRICE
        q.nano = NANO_HALF  # 0.5
        assert fetcher._quotation_to_decimal(q) == Decimal("270.5")

    def test_sub_nano(self) -> None:
        """nano=1 -> 0.000000001, result should be greater than 1."""
        fetcher = _make_fetcher()
        q = MagicMock()
        q.units = 1
        q.nano = 1
        result = fetcher._quotation_to_decimal(q)
        assert result > Decimal(1)


class TestTinkoffFetcherSymbolToFigi:
    def test_known_symbol(self) -> None:
        fetcher = _make_fetcher()
        assert fetcher._symbol_to_figi(SBER_SYMBOL) == SBER_FIGI

    def test_unknown_symbol_raises(self) -> None:
        fetcher = _make_fetcher()
        with pytest.raises(InstrumentNotFoundError):
            fetcher._symbol_to_figi(UNKNOWN_SYMBOL)


class TestTinkoffFetchCandles:
    def test_fetch_returns_candles(self) -> None:
        fake_candle = _make_fake_candle(
            open_u=OPEN_PRICE,
            open_n=0,
            close_u=CLOSE_PRICE,
            close_n=0,
            high_u=HIGH_PRICE,
            high_n=0,
            low_u=LOW_PRICE,
            low_n=0,
            volume=FAKE_VOLUME,
            time_seconds=FAKE_TIMESTAMP,
        )

        fetcher = _make_fetcher()
        with patch.object(fetcher, "_run_async", return_value=[fake_candle]):
            start = datetime(2024, 1, 1, tzinfo=UTC)
            end = datetime(2024, 2, 1, tzinfo=UTC)
            candles = fetcher.fetch_candles(SBER_SYMBOL, start, end, timeframe="1d")

        assert len(candles) == 1
        c = candles[0]
        assert c.symbol == SBER_SYMBOL
        assert c.market_id == "moex"
        assert c.source == "tinkoff"
        assert c.open == Decimal(OPEN_PRICE)
        assert c.close == Decimal(CLOSE_PRICE)
        assert c.volume == FAKE_VOLUME

    def test_fetch_unknown_symbol_raises(self) -> None:
        fetcher = _make_fetcher()
        start = datetime(2024, 1, 1, tzinfo=UTC)
        end = datetime(2024, 2, 1, tzinfo=UTC)
        with pytest.raises(InstrumentNotFoundError):
            fetcher.fetch_candles(UNKNOWN_SYMBOL, start, end)

    def test_fetch_propagates_sdk_error(self) -> None:
        fetcher = _make_fetcher()
        with patch.object(fetcher, "_run_async", side_effect=RuntimeError("gRPC error")):
            start = datetime(2024, 1, 1, tzinfo=UTC)
            end = datetime(2024, 2, 1, tzinfo=UTC)
            with pytest.raises(DataFetchError, match="gRPC error"):
                fetcher.fetch_candles(SBER_SYMBOL, start, end)

    def test_invalid_timeframe_raises(self) -> None:
        fetcher = _make_fetcher()
        start = datetime(2024, 1, 1, tzinfo=UTC)
        end = datetime(2024, 2, 1, tzinfo=UTC)
        with pytest.raises(DataFetchError, match="timeframe"):
            fetcher.fetch_candles(SBER_SYMBOL, start, end, timeframe="5m")


class TestTinkoffFetcherGrpcTimeout:
    """Tests for configurable gRPC timeout on TinkoffFetcher."""

    def test_default_timeout_is_60(self) -> None:
        """Default gRPC timeout must be 60 seconds."""
        fetcher = _make_fetcher()
        assert fetcher._grpc_timeout == 60.0

    def test_custom_timeout_via_constructor(self) -> None:
        """Custom timeout can be passed via constructor parameter."""
        registry = _make_registry()
        fetcher = TinkoffFetcher(
            token=FAKE_TOKEN, registry=registry, sandbox=True, grpc_timeout=30.0
        )
        assert fetcher._grpc_timeout == 30.0

    def test_fetch_async_uses_wait_for_with_timeout(self) -> None:
        """_fetch_async must wrap gRPC call with asyncio.wait_for(timeout=self._grpc_timeout)."""
        import inspect

        source = inspect.getsource(TinkoffFetcher._fetch_async)
        assert "wait_for" in source, "_fetch_async must use asyncio.wait_for"
        assert "timeout" in source, "_fetch_async must pass timeout parameter"

    def test_timeout_error_converted_to_data_fetch_error(self) -> None:
        """asyncio.TimeoutError from wait_for must be converted to DataFetchError."""
        import asyncio as _asyncio

        fetcher = _make_fetcher()
        with patch.object(fetcher, "_run_async", side_effect=TimeoutError()):
            start = datetime(2024, 1, 1, tzinfo=UTC)
            end = datetime(2024, 2, 1, tzinfo=UTC)
            with pytest.raises(DataFetchError, match="timeout"):
                fetcher.fetch_candles(SBER_SYMBOL, start, end)


class TestTinkoffFetcherSandboxClientSelection:
    """Verify that sandbox flag controls which AsyncClient class is used."""

    def test_sandbox_true_uses_sandbox_target(self) -> None:
        """When sandbox=True, _make_client must pass sandbox target to AsyncClient."""
        with patch("finalayze.data.fetchers.tinkoff_data.AsyncClient") as mock_cls:
            mock_cls.return_value = MagicMock()
            fetcher = _make_fetcher(sandbox=True)
            fetcher._make_client()

            mock_cls.assert_called_once()
            call_kwargs = mock_cls.call_args
            assert "sandbox" in str(call_kwargs)  # target contains "sandbox"

    def test_sandbox_false_uses_production_client(self) -> None:
        """When sandbox=False, _make_client must pass production target to AsyncClient."""
        with patch("finalayze.data.fetchers.tinkoff_data.AsyncClient") as mock_cls:
            mock_cls.return_value = MagicMock()
            fetcher = _make_fetcher(sandbox=False)
            fetcher._make_client()

            mock_cls.assert_called_once()
            call_kwargs = mock_cls.call_args
            assert "sandbox" not in str(call_kwargs)  # production target


class TestTinkoffFetcherErrorTypeLogging:
    """ERR-03: Verify structured error logging includes error_type field."""

    def test_fetch_candles_logs_error_type(self) -> None:
        """fetch_candles exception log must include error_type field."""
        fetcher = _make_fetcher()
        with (
            patch.object(fetcher, "_run_async", side_effect=RuntimeError("gRPC error")),
            patch("finalayze.data.fetchers.tinkoff_data._log") as mock_log,
        ):
            start = datetime(2024, 1, 1, tzinfo=UTC)
            end = datetime(2024, 2, 1, tzinfo=UTC)
            with pytest.raises(DataFetchError):
                fetcher.fetch_candles(SBER_SYMBOL, start, end)
            mock_log.exception.assert_called_once()
            call_kwargs = mock_log.exception.call_args
            assert call_kwargs[1].get("error_type") == "RuntimeError"
            assert call_kwargs[1].get("timeframe") == "1d"

    def test_fetch_all_bonds_logs_error_type(self) -> None:
        """fetch_all_bonds exception log must include error_type field."""
        fetcher = _make_fetcher()
        with (
            patch.object(fetcher, "_run_async", side_effect=ConnectionError("connection lost")),
            patch("finalayze.data.fetchers.tinkoff_data._log") as mock_log,
        ):
            result = fetcher.fetch_all_bonds()
            assert result == []
            mock_log.exception.assert_called_once()
            call_kwargs = mock_log.exception.call_args
            assert call_kwargs[1].get("error_type") == "ConnectionError"

    def test_fetch_amortization_logs_error_type(self) -> None:
        """fetch_amortization_schedule exception log must include error_type and instrument_id."""
        fetcher = _make_fetcher()
        with (
            patch.object(fetcher, "_run_async", side_effect=ValueError("bad data")),
            patch("finalayze.data.fetchers.tinkoff_data._log") as mock_log,
        ):
            result = fetcher.fetch_amortization_schedule("test-instrument-id")
            assert result == []
            mock_log.exception.assert_called_once()
            call_kwargs = mock_log.exception.call_args
            assert call_kwargs[1].get("error_type") == "ValueError"
            assert call_kwargs[1].get("instrument_id") == "test-instrument-id"

    def test_fetch_bond_candles_logs_error_type(self) -> None:
        """fetch_bond_candles exception log must include error_type and figi."""
        from datetime import date as date_type

        fetcher = _make_fetcher()
        with (
            patch.object(fetcher, "_run_async", side_effect=TimeoutError("timed out")),
            patch("finalayze.data.fetchers.tinkoff_data._log") as mock_log,
        ):
            result = fetcher.fetch_bond_candles(
                "BBG00FAKE123", date_type(2024, 1, 1), date_type(2024, 2, 1)
            )
            assert result == []
            mock_log.exception.assert_called_once()
            call_kwargs = mock_log.exception.call_args
            assert call_kwargs[1].get("error_type") == "TimeoutError"
            assert call_kwargs[1].get("figi") == "BBG00FAKE123"


# ── All-asset-class discovery (UNIV-03) ─────────────────────────────────


def _make_fake_inst(*, real_exchange: int, **fields: object) -> MagicMock:
    """Build a fake T-Bank instrument object with a MOEX/non-MOEX flag.

    ``getattr`` on a MagicMock auto-creates child mocks, so any attribute the
    fetcher reads via ``getattr(inst, name, None)`` would be truthy. We restrict
    the mock to an explicit spec so absent fields surface as the default.
    """
    inst = MagicMock(spec=[*fields.keys(), "real_exchange"])
    inst.real_exchange = real_exchange
    for name, value in fields.items():
        setattr(inst, name, value)
    return inst


def _patch_services_with(instruments_method: str, resp_instruments: list[MagicMock]) -> object:
    """Return an AsyncMock services object whose ``instruments.<method>`` resp
    carries ``resp_instruments``."""
    resp = MagicMock()
    resp.instruments = resp_instruments
    services = MagicMock()
    setattr(services.instruments, instruments_method, AsyncMock(return_value=resp))
    return services


class TestFetchAllAssetClasses:
    """fetch_all_shares/etfs/futures/currencies: MOEX-filtered, error-safe."""

    def test_methods_exist(self) -> None:
        for name in (
            "fetch_all_shares",
            "fetch_all_etfs",
            "fetch_all_futures",
            "fetch_all_currencies",
        ):
            assert hasattr(TinkoffFetcher, name)

    def test_shares_filters_non_moex(self) -> None:
        from t_tech.invest.schemas import RealExchange

        fetcher = _make_fetcher()
        moex = _make_fake_inst(
            real_exchange=int(RealExchange.REAL_EXCHANGE_MOEX),
            figi="BBG004730N88",
            ticker="SBER",
            isin="RU0009029540",
            class_code="TQBR",
            name="Sberbank",
            lot=10,
            currency="rub",
            asset_uid="uid-sber",
            first_1day_candle_date=datetime(2018, 1, 1, tzinfo=UTC),
        )
        rts = _make_fake_inst(
            real_exchange=int(RealExchange.REAL_EXCHANGE_RTS),
            figi="BBG000NONMOEX",
            ticker="AAPL",
            isin="US0378331005",
            class_code="SPBXM",
            name="Apple",
            lot=1,
            currency="usd",
            asset_uid="uid-aapl",
            first_1day_candle_date=None,
        )
        services = _patch_services_with("shares", [moex, rts])
        with patch.object(fetcher, "_get_services_async", AsyncMock(return_value=services)):
            result = fetcher.fetch_all_shares()
        assert len(result) == 1
        row = result[0]
        assert row["ticker"] == "SBER"
        for key in ("figi", "ticker", "isin", "class_code", "name", "lot", "currency", "asset_uid"):
            assert key in row

    def test_etfs_filters_non_moex(self) -> None:
        from t_tech.invest.schemas import RealExchange

        fetcher = _make_fetcher()
        moex = _make_fake_inst(
            real_exchange=int(RealExchange.REAL_EXCHANGE_MOEX),
            figi="BBG_ETF",
            ticker="TMOS",
            isin="RU000A101X76",
            class_code="TQTF",
            name="Tinkoff iMOEX",
            lot=1,
            currency="rub",
            asset_uid="uid-tmos",
            first_1day_candle_date=None,
        )
        rts = _make_fake_inst(real_exchange=int(RealExchange.REAL_EXCHANGE_RTS), ticker="SPY")
        services = _patch_services_with("etfs", [moex, rts])
        with patch.object(fetcher, "_get_services_async", AsyncMock(return_value=services)):
            result = fetcher.fetch_all_etfs()
        assert len(result) == 1
        assert result[0]["ticker"] == "TMOS"
        assert result[0]["asset_uid"] == "uid-tmos"

    def test_futures_carry_expiration_and_basic_asset(self) -> None:
        from t_tech.invest.schemas import RealExchange

        fetcher = _make_fetcher()
        moex = _make_fake_inst(
            real_exchange=int(RealExchange.REAL_EXCHANGE_MOEX),
            figi="FUTSBER",
            ticker="SBRF-12.25",
            class_code="SPBFUT",
            name="Sberbank Futures",
            lot=1,
            currency="rub",
            basic_asset="SBER",
            expiration_date=datetime(2025, 12, 19, tzinfo=UTC),
        )
        services = _patch_services_with("futures", [moex])
        with patch.object(fetcher, "_get_services_async", AsyncMock(return_value=services)):
            result = fetcher.fetch_all_futures()
        assert len(result) == 1
        row = result[0]
        assert row["ticker"] == "SBRF-12.25"
        assert row["basic_asset"] == "SBER"
        assert row["expiration_date"] == date(2025, 12, 19)

    def test_currencies_carry_isin(self) -> None:
        from t_tech.invest.schemas import RealExchange

        fetcher = _make_fetcher()
        nominal = MagicMock()
        nominal.units = 1
        nominal.nano = 0
        moex = _make_fake_inst(
            real_exchange=int(RealExchange.REAL_EXCHANGE_MOEX),
            figi="BBG_USD",
            ticker="USD000UTSTOM",
            class_code="CETS",
            name="USD/RUB",
            lot=1000,
            currency="rub",
            isin="",
            nominal=nominal,
        )
        services = _patch_services_with("currencies", [moex])
        with patch.object(fetcher, "_get_services_async", AsyncMock(return_value=services)):
            result = fetcher.fetch_all_currencies()
        assert len(result) == 1
        row = result[0]
        assert row["ticker"] == "USD000UTSTOM"
        for key in ("figi", "ticker", "currency", "lot", "name"):
            assert key in row

    def test_shares_returns_empty_on_grpc_error(self) -> None:
        fetcher = _make_fetcher()
        with (
            patch.object(fetcher, "_run_async", side_effect=ConnectionError("gRPC down")),
            patch("finalayze.data.fetchers.tinkoff_data._log") as mock_log,
        ):
            assert fetcher.fetch_all_shares() == []
            mock_log.exception.assert_called_once()
            assert mock_log.exception.call_args[1].get("error_type") == "ConnectionError"
