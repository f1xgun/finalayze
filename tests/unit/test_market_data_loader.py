"""Tests for MarketDataLoader."""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal
from unittest.mock import MagicMock

from finalayze.core.exceptions import DataFetchError
from finalayze.core.schemas import Candle, FXRate, KeyRateRecord, TurnoverRecord
from finalayze.data.loader import MarketDataLoader


def _make_candle(symbol: str = "SPY", day: int = 15) -> Candle:
    return Candle(
        symbol=symbol,
        market_id="us",
        timeframe="1d",
        timestamp=datetime(2024, 1, day, 14, 30, tzinfo=UTC),
        open=Decimal(100),
        high=Decimal(101),
        low=Decimal(99),
        close=Decimal("100.5"),
        volume=1000000,
    )


def _make_segment(market: str = "us") -> MagicMock:
    cfg = MagicMock()
    cfg.market = market
    return cfg


class TestMarketDataLoaderUS:
    def test_load_us_segment(self) -> None:
        yf = MagicMock()
        yf.fetch_candles.side_effect = [[_make_candle("SPY")], [_make_candle("^VIX")]]

        loader = MarketDataLoader(yfinance_fetcher=yf)
        ctx = loader.load(_make_segment("us"), date(2024, 1, 1), date(2024, 2, 1))

        assert ctx.benchmark_candles is not None
        assert ctx.vix_candles is not None
        assert ctx.moex_data is None
        assert loader.fetch_failures == []

    def test_us_benchmark_failure_graceful(self) -> None:
        yf = MagicMock()
        yf.fetch_candles.side_effect = [DataFetchError("fail"), [_make_candle("^VIX")]]

        loader = MarketDataLoader(yfinance_fetcher=yf)
        ctx = loader.load(_make_segment("us"), date(2024, 1, 1), date(2024, 2, 1))

        assert ctx.benchmark_candles is None
        assert ctx.vix_candles is not None
        assert "yfinance.SPY" in loader.fetch_failures


class TestMarketDataLoaderMOEX:
    def test_load_moex_segment(self) -> None:
        moex_candles = MagicMock()
        moex_candles.fetch_candles.return_value = [_make_candle("IMOEX")]

        moex_raw = MagicMock()
        moex_raw.fetch_market_turnover.return_value = [
            TurnoverRecord(
                timestamp=datetime(2024, 1, 15, 0, 0, tzinfo=UTC),
                volume_rub=Decimal("1.5e12"),
            )
        ]

        cbr = MagicMock()
        cbr.fetch_fx_rates.return_value = [
            FXRate(
                timestamp=datetime(2024, 1, 15, 0, 0, tzinfo=UTC),
                pair="USDRUB",
                rate=Decimal("89.50"),
            )
        ]
        cbr.fetch_key_rate.return_value = [
            KeyRateRecord(timestamp=datetime(2024, 1, 1, 0, 0, tzinfo=UTC), rate=Decimal("0.16"))
        ]

        yf = MagicMock()
        yf.fetch_candles.return_value = [_make_candle("BZ=F")]

        loader = MarketDataLoader(
            moex_iss_candles=moex_candles,
            moex_iss_raw=moex_raw,
            cbr=cbr,
            yfinance_fetcher=yf,
        )
        ctx = loader.load(_make_segment("moex"), date(2024, 1, 1), date(2024, 2, 1))

        assert ctx.benchmark_candles is not None
        assert ctx.vix_candles is None
        assert ctx.moex_data is not None
        assert ctx.moex_data.fx_rates is not None
        assert ctx.moex_data.key_rates is not None
        assert ctx.moex_data.turnover is not None
        assert ctx.moex_data.commodity_candles is not None
        assert loader.fetch_failures == []

    def test_moex_partial_failure(self) -> None:
        moex_candles = MagicMock()
        moex_candles.fetch_candles.return_value = [_make_candle("IMOEX")]

        moex_raw = MagicMock()
        moex_raw.fetch_market_turnover.side_effect = DataFetchError("ISS down")

        cbr = MagicMock()
        cbr.fetch_fx_rates.return_value = [
            FXRate(
                timestamp=datetime(2024, 1, 15, 0, 0, tzinfo=UTC),
                pair="USDRUB",
                rate=Decimal("89.50"),
            )
        ]
        cbr.fetch_key_rate.side_effect = DataFetchError("CBR down")

        yf = MagicMock()
        yf.fetch_candles.return_value = [_make_candle("BZ=F")]

        loader = MarketDataLoader(
            moex_iss_candles=moex_candles,
            moex_iss_raw=moex_raw,
            cbr=cbr,
            yfinance_fetcher=yf,
        )
        ctx = loader.load(_make_segment("moex"), date(2024, 1, 1), date(2024, 2, 1))

        assert ctx.benchmark_candles is not None
        assert ctx.moex_data.fx_rates is not None
        assert ctx.moex_data.key_rates is None
        assert ctx.moex_data.turnover is None
        assert len(loader.fetch_failures) == 2  # noqa: PLR2004

    def test_fetch_failures_cleared_on_load(self) -> None:
        yf = MagicMock()
        yf.fetch_candles.side_effect = DataFetchError("fail")

        loader = MarketDataLoader(yfinance_fetcher=yf)
        loader.load(_make_segment("us"), date(2024, 1, 1), date(2024, 2, 1))
        assert len(loader.fetch_failures) > 0

        yf.fetch_candles.side_effect = [[_make_candle()], [_make_candle("^VIX")]]
        loader.load(_make_segment("us"), date(2024, 1, 1), date(2024, 2, 1))
        assert loader.fetch_failures == []
