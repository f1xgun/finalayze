"""Unit tests for TinkoffFetcher."""

from __future__ import annotations

from datetime import UTC, datetime
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

        with patch(
            "finalayze.data.fetchers.tinkoff_data.asyncio.run",
            return_value=[fake_candle],
        ):
            fetcher = _make_fetcher()
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
        with patch(
            "finalayze.data.fetchers.tinkoff_data.asyncio.run",
            side_effect=RuntimeError("gRPC error"),
        ):
            fetcher = _make_fetcher()
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


class TestTinkoffFetcherSandboxClientSelection:
    """Verify that sandbox flag controls which AsyncClient class is used."""

    def _make_client_mock(self, fake_candle: MagicMock) -> MagicMock:
        """Build a mock client that returns candles directly (persistent client pattern)."""
        mock_response = MagicMock()
        mock_response.candles = [fake_candle]
        mock_client = MagicMock()
        mock_client.market_data.get_candles = AsyncMock(return_value=mock_response)
        return mock_client

    def test_sandbox_true_uses_sandbox_client(self) -> None:
        """When sandbox=True, _get_client must use AsyncSandboxClient."""
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

        mock_client = self._make_client_mock(fake_candle)
        mock_sandbox_cls = MagicMock(return_value=mock_client)
        mock_prod_cls = MagicMock()

        with (
            patch(
                "finalayze.data.fetchers.tinkoff_data.AsyncSandboxClient",
                mock_sandbox_cls,
            ),
            patch(
                "finalayze.data.fetchers.tinkoff_data.AsyncClient",
                mock_prod_cls,
            ),
        ):
            fetcher = _make_fetcher(sandbox=True)
            start = datetime(2024, 1, 1, tzinfo=UTC)
            end = datetime(2024, 2, 1, tzinfo=UTC)
            fetcher.fetch_candles(SBER_SYMBOL, start, end, timeframe="1d")

        mock_sandbox_cls.assert_called_once()
        mock_prod_cls.assert_not_called()

    def test_sandbox_false_uses_production_client(self) -> None:
        """When sandbox=False, _get_client must use AsyncClient (production)."""
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

        mock_client = self._make_client_mock(fake_candle)
        mock_prod_cls = MagicMock(return_value=mock_client)
        mock_sandbox_cls = MagicMock()

        with (
            patch(
                "finalayze.data.fetchers.tinkoff_data.AsyncClient",
                mock_prod_cls,
            ),
            patch(
                "finalayze.data.fetchers.tinkoff_data.AsyncSandboxClient",
                mock_sandbox_cls,
            ),
        ):
            fetcher = _make_fetcher(sandbox=False)
            start = datetime(2024, 1, 1, tzinfo=UTC)
            end = datetime(2024, 2, 1, tzinfo=UTC)
            fetcher.fetch_candles(SBER_SYMBOL, start, end, timeframe="1d")

        mock_prod_cls.assert_called_once()
        mock_sandbox_cls.assert_not_called()
