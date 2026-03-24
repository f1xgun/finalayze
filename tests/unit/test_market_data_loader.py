"""Tests for MarketDataLoader."""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from finalayze.core.exceptions import DataFetchError
from finalayze.core.schemas import Candle, FXRate, KeyRateRecord, MarketContext, TurnoverRecord
from finalayze.data.fetchers._cache_utils import GenericFileCache
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


class TestRunIterationIntegration:
    """Smoke tests for the MarketDataLoader usage pattern in run_iteration.py.

    These tests replicate the construction and usage pattern that run_iteration.py
    uses after the refactor — loader creation, load() per segment, close() cleanup.
    """

    def test_us_segment_via_simplenamespace(self) -> None:
        """Loader works when segment config is a SimpleNamespace (plain market attr)."""
        yf = MagicMock()
        yf.fetch_candles.side_effect = [
            [_make_candle("SPY")],
            [_make_candle("^VIX")],
        ]

        loader = MarketDataLoader(yfinance_fetcher=yf)
        seg = SimpleNamespace(market="us")
        ctx = loader.load(seg, date(2024, 1, 1), date(2024, 6, 30))

        assert isinstance(ctx, MarketContext)
        assert ctx.moex_data is None
        assert ctx.benchmark_candles is not None
        assert ctx.vix_candles is not None
        loader.close()

    def test_moex_segment_via_simplenamespace(self, tmp_path: Path) -> None:
        """Loader works for MOEX segment using SimpleNamespace with market='moex'."""
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
            KeyRateRecord(
                timestamp=datetime(2024, 1, 1, 0, 0, tzinfo=UTC),
                rate=Decimal("0.16"),
            )
        ]

        yf = MagicMock()
        yf.fetch_candles.return_value = [_make_candle("BZ=F")]

        turnover_cache = GenericFileCache(tmp_path / "turnover")
        cbr_cache = GenericFileCache(tmp_path / "cbr")

        loader = MarketDataLoader(
            moex_iss_candles=moex_candles,
            moex_iss_raw=moex_raw,
            cbr=cbr,
            yfinance_fetcher=yf,
            turnover_cache=turnover_cache,
            cbr_cache=cbr_cache,
        )
        seg = SimpleNamespace(market="moex")
        ctx = loader.load(seg, date(2024, 1, 1), date(2024, 6, 30))

        assert isinstance(ctx, MarketContext)
        assert ctx.moex_data is not None
        assert ctx.benchmark_candles is not None
        assert ctx.vix_candles is None
        loader.close()

    def test_loader_close_is_safe_with_no_fetchers(self) -> None:
        """close() must not raise when all fetchers are None."""
        loader = MarketDataLoader()
        loader.close()  # no-op, must not raise

class TestBrentCaching:
    """Tests for Brent crude caching via _cached_fetch."""

    def test_brent_uses_cached_fetch(self) -> None:
        """_load_moex() must call _cached_fetch for Brent data, not _safe_fetch."""
        import inspect

        source = inspect.getsource(MarketDataLoader._load_moex)
        # Find the brent assignment -- it should use _cached_fetch
        lines = source.split("\n")
        brent_idx = None
        for i, line in enumerate(lines):
            if line.strip().startswith("brent = self."):
                brent_idx = i
                break
        assert brent_idx is not None, "Could not find brent assignment in _load_moex"
        brent_line = lines[brent_idx].strip()
        assert "_cached_fetch" in brent_line, (
            f"Brent must use _cached_fetch, got: {brent_line}"
        )
        assert "_safe_fetch" not in brent_line, (
            f"Brent must not use _safe_fetch, got: {brent_line}"
        )

    def test_second_call_uses_cache(self, tmp_path: Path) -> None:
        """Second call to _load_moex() with same date range must not call yfinance for Brent."""
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
            KeyRateRecord(
                timestamp=datetime(2024, 1, 1, 0, 0, tzinfo=UTC),
                rate=Decimal("0.16"),
            )
        ]

        brent_candle = _make_candle("BZ=F")
        yf = MagicMock()
        yf.fetch_candles.return_value = [brent_candle]

        brent_cache = GenericFileCache(tmp_path / "brent")

        loader = MarketDataLoader(
            moex_iss_candles=moex_candles,
            moex_iss_raw=moex_raw,
            cbr=cbr,
            yfinance_fetcher=yf,
            brent_cache=brent_cache,
        )

        seg = SimpleNamespace(market="moex")
        # First call -- fetches from yfinance
        ctx1 = loader.load(seg, date(2024, 1, 1), date(2024, 2, 1))
        assert ctx1.moex_data is not None
        assert ctx1.moex_data.commodity_candles is not None
        first_call_count = yf.fetch_candles.call_count

        # Second call -- should use cache, not call yfinance again for Brent
        ctx2 = loader.load(seg, date(2024, 1, 1), date(2024, 2, 1))
        assert ctx2.moex_data is not None
        assert ctx2.moex_data.commodity_candles is not None
        # yfinance should NOT have been called again for Brent
        assert yf.fetch_candles.call_count == first_call_count


class TestRunIterationIntegrationExtra:
    def test_us_segment_benchmark_only_no_vix(self) -> None:
        """US load with VIX fetch failure still returns benchmark_candles."""
        yf = MagicMock()
        yf.fetch_candles.side_effect = [
            [_make_candle("SPY")],
            DataFetchError("VIX unavailable"),
        ]

        loader = MarketDataLoader(yfinance_fetcher=yf)
        seg = SimpleNamespace(market="us")
        ctx = loader.load(seg, date(2024, 1, 1), date(2024, 6, 30))

        assert ctx.benchmark_candles is not None
        assert ctx.vix_candles is None
        assert "yfinance.VIX" in loader.fetch_failures
