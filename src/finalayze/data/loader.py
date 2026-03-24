"""MarketDataLoader — orchestrates ambient market data for a segment (Layer 2).

Not thread-safe — one loader per backtest run.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import TYPE_CHECKING, Any, Protocol

import structlog

from finalayze.core.exceptions import DataFetchError
from finalayze.core.schemas import (
    Candle,
    FXRate,
    KeyRateRecord,
    MarketContext,
    MoexMarketData,
    TurnoverRecord,
)
from finalayze.data.fetchers._cache_utils import GenericFileCache

if TYPE_CHECKING:
    from collections.abc import Callable

    from finalayze.data.fetchers.base import BaseFetcher
    from finalayze.data.fetchers.cbr import CBRFetcher
    from finalayze.data.fetchers.moex_iss import MoexISSFetcher

_log = structlog.get_logger()


class _HasMarket(Protocol):
    """Minimal protocol for segment config — avoids cross-layer import."""

    market: str


class MarketDataLoader:
    """Orchestrates loading of all ambient market data for a segment.

    Not thread-safe. Close all fetchers when done via close().
    """

    def __init__(
        self,
        moex_iss_candles: BaseFetcher | None = None,
        moex_iss_raw: MoexISSFetcher | None = None,
        cbr: CBRFetcher | None = None,
        yfinance_fetcher: BaseFetcher | None = None,
        turnover_cache: GenericFileCache | None = None,
        cbr_cache: GenericFileCache | None = None,
        brent_cache: GenericFileCache | None = None,
    ) -> None:
        self._moex_candles = moex_iss_candles  # CachingFetcher(MoexISSFetcher)
        self._moex_raw = moex_iss_raw  # Raw MoexISSFetcher for turnover
        self._cbr = cbr
        self._yf = yfinance_fetcher
        self._turnover_cache = turnover_cache
        self._cbr_cache = cbr_cache
        self._brent_cache = brent_cache
        self.fetch_failures: list[str] = []

    def close(self) -> None:
        """Close all owned fetchers."""
        for fetcher in (self._moex_candles, self._moex_raw, self._cbr, self._yf):
            if fetcher is not None and hasattr(fetcher, "close"):
                fetcher.close()

    def load(self, segment_config: _HasMarket, start: date, end: date) -> MarketContext:
        """Load all ambient data. Routes by segment_config.market."""
        self.fetch_failures.clear()
        start_dt = datetime(start.year, start.month, start.day, tzinfo=UTC)
        end_dt = datetime(end.year, end.month, end.day, tzinfo=UTC)
        if segment_config.market == "moex":
            return self._load_moex(start_dt, end_dt)
        return self._load_us(start_dt, end_dt)

    # ── Market-specific loaders ──────────────────────────────────────────────

    def _load_us(self, start: datetime, end: datetime) -> MarketContext:
        benchmark = self._safe_fetch("yfinance.SPY", lambda: self._yf_fetch("SPY", start, end))
        vix = self._safe_fetch("yfinance.VIX", lambda: self._yf_fetch("^VIX", start, end))
        return MarketContext(
            benchmark_candles=benchmark or None,
            vix_candles=vix or None,
            moex_data=None,
        )

    def _load_moex(self, start: datetime, end: datetime) -> MarketContext:
        benchmark = self._safe_fetch(
            "moex_iss.IMOEX",
            lambda: self._moex_candles_fetch("IMOEX", start, end),
        )
        fx = self._cached_fetch(
            "cbr.fx_rates",
            self._cbr_cache,
            FXRate,
            "cbr",
            "USDRUB",
            start,
            end,
            fn=lambda: self._cbr_fx_fetch("USD", start, end),
        )
        key_rate = self._cached_fetch(
            "cbr.key_rate",
            self._cbr_cache,
            KeyRateRecord,
            "cbr",
            "key_rate",
            start,
            end,
            fn=lambda: self._cbr_key_rate_fetch(start, end),
        )
        brent = self._cached_fetch(
            "yfinance.brent",
            self._brent_cache,
            Candle,
            "yfinance",
            "BZ_F",
            start,
            end,
            fn=lambda: self._yf_fetch("BZ=F", start, end),
        )
        turnover = self._cached_fetch(
            "moex_iss.turnover",
            self._turnover_cache,
            TurnoverRecord,
            "turnover",
            "total",
            start,
            end,
            fn=lambda: self._moex_turnover_fetch(start, end),
        )
        return MarketContext(
            benchmark_candles=benchmark or None,
            vix_candles=None,
            moex_data=MoexMarketData(
                fx_rates=tuple(fx) if fx else None,
                key_rates=tuple(key_rate) if key_rate else None,
                commodity_candles={"BZ=F": tuple(brent)} if brent else None,
                turnover=tuple(turnover) if turnover else None,
            ),
        )

    # ── Delegate helpers ─────────────────────────────────────────────────────

    def _yf_fetch(self, symbol: str, start: datetime, end: datetime) -> list[Any]:
        if self._yf is None:
            raise DataFetchError("yfinance_fetcher not configured")
        return self._yf.fetch_candles(symbol, start, end)

    def _moex_candles_fetch(self, symbol: str, start: datetime, end: datetime) -> list[Any]:
        if self._moex_candles is None:
            raise DataFetchError("moex_iss_candles fetcher not configured")
        return self._moex_candles.fetch_candles(symbol, start, end)

    def _moex_turnover_fetch(self, start: datetime, end: datetime) -> list[Any]:
        if self._moex_raw is None:
            raise DataFetchError("moex_iss_raw fetcher not configured")
        return self._moex_raw.fetch_market_turnover(start, end)

    def _cbr_fx_fetch(self, currency: str, start: datetime, end: datetime) -> list[Any]:
        if self._cbr is None:
            raise DataFetchError("cbr fetcher not configured")
        return self._cbr.fetch_fx_rates(currency, start, end)

    def _cbr_key_rate_fetch(self, start: datetime, end: datetime) -> list[Any]:
        if self._cbr is None:
            raise DataFetchError("cbr fetcher not configured")
        return self._cbr.fetch_key_rate(start, end)

    # ── Core fetch logic ─────────────────────────────────────────────────────

    def _safe_fetch(self, source_name: str, fn: Callable[[], list[Any]]) -> list[Any]:
        """Call fn(), gracefully catching any exception."""
        try:
            return fn()
        except Exception as exc:  # Broad catch: yfinance may raise ValueError/ConnectionError
            _log.error("market_data_fetch_failed", source=source_name, error=str(exc))
            self.fetch_failures.append(source_name)
            return []

    def _cached_fetch(
        self,
        source_name: str,
        cache: GenericFileCache | None,
        model_class: type[Any],
        cache_source: str,
        cache_id: str,
        start: datetime,
        end: datetime,
        *,
        fn: Callable[[], list[Any]],
    ) -> list[Any]:
        """Fetch with optional file cache. `fn` is keyword-only to prevent ordering bugs."""
        key = GenericFileCache.make_key(cache_source, cache_id, start.date(), end.date())
        if cache is not None:
            cached = cache.get(key, model_class)
            if cached is not None:
                return cached
        result = self._safe_fetch(source_name, fn)
        if result and cache is not None:
            cache.set(key, result)
        return result
