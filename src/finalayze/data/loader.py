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

    from config.settings import Settings

    from finalayze.core.schemas import FundamentalSnapshot
    from finalayze.data.fetchers.base import BaseFetcher
    from finalayze.data.fetchers.cbr import CBRFetcher
    from finalayze.data.fetchers.moex_iss import MoexISSFetcher

_log = structlog.get_logger()


class _HasMarket(Protocol):
    """Minimal protocol for segment config — avoids cross-layer import.

    ``symbols`` is optional (read via ``getattr``); when present and non-empty
    it scopes the fundamental-snapshot peer set loaded for MOEX segments.
    """

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
        settings: Settings | None = None,
    ) -> None:
        self._moex_candles = moex_iss_candles  # CachingFetcher(MoexISSFetcher)
        self._moex_raw = moex_iss_raw  # Raw MoexISSFetcher for turnover
        self._cbr = cbr
        self._yf = yfinance_fetcher
        self._turnover_cache = turnover_cache
        self._cbr_cache = cbr_cache
        self._brent_cache = brent_cache
        self._settings = settings  # for DB-backed fundamental_snapshots reads
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
            symbols = getattr(segment_config, "symbols", None)
            return self._load_moex(start_dt, end_dt, symbols)
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

    def _load_moex(
        self, start: datetime, end: datetime, symbols: list[str] | None = None
    ) -> MarketContext:
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
        snaps: list[Any] = []
        if symbols and self._settings is not None:
            snaps = self._safe_fetch(
                "db.fundamental_snapshots",
                lambda: list(self._fundamentals_fetch(symbols, end)),
            )
        return MarketContext(
            benchmark_candles=benchmark or None,
            vix_candles=None,
            moex_data=MoexMarketData(
                fx_rates=tuple(fx) if fx else None,
                key_rates=tuple(key_rate) if key_rate else None,
                commodity_candles={"BZ=F": tuple(brent)} if brent else None,
                turnover=tuple(turnover) if turnover else None,
                fundamentals=tuple(snaps) if snaps else None,
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

    def _fundamentals_fetch(
        self, symbols: list[str], end: datetime
    ) -> tuple[FundamentalSnapshot, ...]:
        """Read the full look-back window of fundamental snapshots for ``symbols``.

        Reads every snapshot with ``as_of <= end`` (per-window slicing is
        downstream in ``_slice_market_context``). Returns ``()`` when no settings
        handle is configured. DB errors propagate to ``_safe_fetch`` so a failure
        degrades to ``()`` and is appended to ``fetch_failures``.
        """
        if self._settings is None:
            return ()
        return _read_fundamental_snapshots(symbols, end, self._settings)

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


def _orm_to_fundamental(row: Any) -> FundamentalSnapshot:
    """Map a FundamentalSnapshotModel ORM row to a FundamentalSnapshot schema.

    None stays None — values are NEVER fabricated (V5 input-validation, T-64-03).
    """
    from finalayze.core.schemas import FundamentalSnapshot  # noqa: PLC0415

    def _f(x: Any | None) -> float | None:
        return float(x) if x is not None else None

    return FundamentalSnapshot(
        symbol=row.symbol,
        as_of=row.as_of,
        pe_ratio=_f(row.pe_ratio),
        ev_ebitda=_f(row.ev_ebitda),
        revenue_ttm=_f(row.revenue_ttm),
        net_margin=_f(row.net_margin),
        roe=_f(row.roe),
        eps_ttm=_f(row.eps_ttm),
        dividend_yield=_f(row.dividend_yield),
        market_cap=_f(row.market_cap),
        currency=row.currency,
    )


async def read_fundamental_snapshots_async(
    peer_symbols: list[str], end_dt: datetime, settings: Settings
) -> tuple[FundamentalSnapshot, ...]:
    """Async read of the FULL look-back window of fundamental snapshots.

    Selects every snapshot with ``as_of <= end_dt`` for the given peer symbols,
    ordered by ``as_of`` (per-window slicing is downstream). The async engine is
    always disposed (no connection-pool leak across calls). DB errors propagate
    to the caller — the live path runs under ``_safe_fetch``, which logs the
    failure, records it in ``fetch_failures`` and degrades to ``()`` — so a DB
    misconfiguration is observable rather than silently swallowed.
    """
    from sqlalchemy import select  # noqa: PLC0415
    from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine  # noqa: PLC0415

    from finalayze.core.models import FundamentalSnapshotModel  # noqa: PLC0415

    engine = create_async_engine(settings.database_url, echo=False)
    try:
        async with AsyncSession(engine) as session:
            result = await session.execute(
                select(FundamentalSnapshotModel)
                .where(
                    FundamentalSnapshotModel.symbol.in_(peer_symbols),
                    FundamentalSnapshotModel.as_of <= end_dt,
                )
                .order_by(FundamentalSnapshotModel.as_of)
            )
            rows = result.scalars().all()
            return tuple(_orm_to_fundamental(row) for row in rows)
    finally:
        await engine.dispose()


def _read_fundamental_snapshots(
    peer_symbols: list[str], end_dt: datetime, settings: Settings
) -> tuple[FundamentalSnapshot, ...]:
    """Synchronous wrapper around :func:`read_fundamental_snapshots_async`."""
    import asyncio  # noqa: PLC0415

    return asyncio.run(read_fundamental_snapshots_async(peer_symbols, end_dt, settings))
