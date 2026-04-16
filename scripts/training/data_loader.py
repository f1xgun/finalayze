"""Data fetching and loading utilities for the training pipeline.

Handles fetching candles from database, Tinkoff API, and yfinance,
as well as benchmark/VIX data and market data loader construction.
"""

from __future__ import annotations

import asyncio
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from finalayze.core.models import CandleModel
from finalayze.core.schemas import Candle
from finalayze.data.fetchers.yfinance import YFinanceFetcher
from finalayze.data.loader import MarketDataLoader

if TYPE_CHECKING:
    from config.settings import Settings

# Lookback windows
LOOKBACK_DAYS = 1825  # 5 years of history for US segments
MOEX_LOOKBACK_DAYS = 730  # 2 years for MOEX (post-sanctions structural break)

# Benchmark tickers for market-neutral (excess return) labels
US_BENCHMARK = "SPY"
MOEX_BENCHMARK = "IMOEX"  # Moscow Exchange index
VIX_TICKER = "^VIX"  # CBOE Volatility Index (US only)


def is_moex_segment(segment_id: str) -> bool:
    """Return True if segment_id is a MOEX/Russian segment."""
    return segment_id.startswith("ru_")


def get_lookback_days(segment_id: str) -> int:
    """Return lookback days: 2 years for MOEX, 5 years for US."""
    return MOEX_LOOKBACK_DAYS if is_moex_segment(segment_id) else LOOKBACK_DAYS


def orm_to_candle(row: CandleModel) -> Candle:
    """Convert a CandleModel ORM row to a Candle schema object."""
    return Candle(
        symbol=row.symbol,
        market_id=row.market_id,
        timeframe=row.timeframe,
        timestamp=row.timestamp,
        open=row.open,
        high=row.high,
        low=row.low,
        close=row.close,
        volume=row.volume,
    )


async def fetch_from_db(symbol: str, market_id: str, settings: Settings) -> list[Candle]:
    """Try to load candles from DB. Returns empty list on failure."""
    from sqlalchemy import select  # noqa: PLC0415
    from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine  # noqa: PLC0415

    try:
        engine = create_async_engine(settings.database_url, echo=False)
        async with AsyncSession(engine) as session:
            result = await session.execute(
                select(CandleModel)
                .where(CandleModel.symbol == symbol, CandleModel.market_id == market_id)
                .order_by(CandleModel.timestamp)
            )
            rows = result.scalars().all()
            return [orm_to_candle(row) for row in rows]
    except Exception:
        return []


def align_benchmark_candles(
    stock_candles: list[Candle],
    benchmark_candles: list[Candle],
) -> list[Candle]:
    """Align benchmark candles to stock candles by date (timestamp-based join).

    For each stock candle, find the benchmark candle with the closest date
    that is <= stock date. This prevents look-ahead bias and handles
    missing benchmark dates (holidays, halts).

    If benchmark has no data at all, returns an empty list.
    If a stock candle's date is before the earliest benchmark date,
    the earliest benchmark candle is used (back-fill edge case).

    Returns a list of benchmark candles with the same length as stock_candles,
    with each entry corresponding to the aligned benchmark candle.
    """
    if not benchmark_candles or not stock_candles:
        return []

    # Build date -> candle mapping using date part only (ignore time)
    bench_by_date: dict[datetime, Candle] = {}
    for c in benchmark_candles:
        # Use date at midnight UTC for consistent matching
        key = c.timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
        bench_by_date[key] = c

    # Sort benchmark dates for forward-fill lookup
    sorted_bench_dates = sorted(bench_by_date.keys())
    if not sorted_bench_dates:
        return []

    aligned: list[Candle] = []
    last_bench: Candle = bench_by_date[sorted_bench_dates[0]]

    # Build a forward-filled map: iterate through all dates in order
    # and carry forward the last known benchmark candle
    from datetime import timedelta as _td  # noqa: PLC0415

    # Pre-build forward-filled lookup for efficiency
    min_date = min(
        sorted_bench_dates[0],
        stock_candles[0].timestamp.replace(hour=0, minute=0, second=0, microsecond=0),
    )
    max_date = max(
        sorted_bench_dates[-1],
        stock_candles[-1].timestamp.replace(hour=0, minute=0, second=0, microsecond=0),
    )

    ffill_map: dict[datetime, Candle] = {}
    current = min_date
    current_bench = bench_by_date.get(sorted_bench_dates[0])
    assert current_bench is not None

    while current <= max_date:
        if current in bench_by_date:
            current_bench = bench_by_date[current]
        ffill_map[current] = current_bench
        current += _td(days=1)

    for stock_c in stock_candles:
        stock_date = stock_c.timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
        if stock_date in ffill_map:
            aligned.append(ffill_map[stock_date])
        else:
            # Edge case: stock date outside range -- use last known
            aligned.append(last_bench)

    return aligned


def fetch_benchmark_candles(
    segment_id: str,
) -> list[Candle] | None:
    """Fetch benchmark candles for excess-return labeling.

    US segments: SPY via YFinanceFetcher.
    MOEX segments: IMOEX via TinkoffFetcher (requires token, else None).

    Returns None if benchmark cannot be fetched.
    """
    if is_moex_segment(segment_id):
        return fetch_moex_benchmark(segment_id)
    return fetch_us_benchmark(segment_id)


def fetch_moex_benchmark(segment_id: str) -> list[Candle] | None:
    """Fetch IMOEX benchmark for MOEX segments via Tinkoff API."""
    token = os.environ.get("FINALAYZE_TINKOFF_TOKEN")
    if not token:
        print(f"  [{segment_id}] FINALAYZE_TINKOFF_TOKEN not set, skipping MOEX benchmark (IMOEX).")
        return None
    try:
        from finalayze.data.fetchers.tinkoff_data import (  # noqa: PLC0415
            TinkoffFetcher,
        )
        from finalayze.markets.instruments import (  # noqa: PLC0415
            build_default_registry,
        )

        registry = build_default_registry()
        fetcher = TinkoffFetcher(token=token, registry=registry, sandbox=False)
        end = datetime.now(tz=UTC)
        start = end - timedelta(days=MOEX_LOOKBACK_DAYS)
        candles = fetcher.fetch_candles(MOEX_BENCHMARK, start, end)
        if candles:
            print(f"  [{segment_id}] Fetched {len(candles)} benchmark candles ({MOEX_BENCHMARK}).")
            return candles
        print(
            f"  [{segment_id}] No benchmark candles for {MOEX_BENCHMARK}, skipping excess returns."
        )
        return None
    except Exception as exc:
        print(f"  [{segment_id}] Failed to fetch MOEX benchmark: {exc}, skipping excess returns.")
        return None


def fetch_us_benchmark(segment_id: str) -> list[Candle] | None:
    """Fetch SPY benchmark for US segments via yfinance."""
    lookback = get_lookback_days(segment_id)
    end = datetime.now(tz=UTC)
    start = end - timedelta(days=lookback)
    market_id = segment_id.split("_", maxsplit=1)[0]
    fetcher = YFinanceFetcher(market_id=market_id)
    try:
        candles = fetcher.fetch_candles(US_BENCHMARK, start, end)
        if candles:
            print(f"  [{segment_id}] Fetched {len(candles)} benchmark candles ({US_BENCHMARK}).")
            return candles
        print(f"  [{segment_id}] No benchmark candles for {US_BENCHMARK}, skipping excess returns.")
        return None
    except Exception as exc:
        print(
            f"  [{segment_id}] Failed to fetch benchmark "
            f"({US_BENCHMARK}): {exc}, skipping excess returns."
        )
        return None


def fetch_vix_candles(segment_id: str) -> list[Candle] | None:
    """Fetch VIX candles for regime features (US segments only).

    MOEX segments return None since VIX is a US-specific index.
    """
    if is_moex_segment(segment_id):
        return None
    lookback = get_lookback_days(segment_id)
    end = datetime.now(tz=UTC)
    start = end - timedelta(days=lookback)
    market_id = segment_id.split("_", maxsplit=1)[0]
    fetcher = YFinanceFetcher(market_id=market_id)
    try:
        candles = fetcher.fetch_candles(VIX_TICKER, start, end)
        if candles:
            print(f"  [{segment_id}] Fetched {len(candles)} VIX candles ({VIX_TICKER}).")
            return candles
        print(f"  [{segment_id}] No VIX candles for {VIX_TICKER}, skipping VIX features.")
        return None
    except Exception as exc:
        print(f"  [{segment_id}] Failed to fetch VIX ({VIX_TICKER}): {exc}, skipping.")
        return None


def fetch_tinkoff_candles(symbol: str) -> list[Candle]:
    """Fetch candles from Tinkoff Invest API for MOEX symbols.

    Uses TinkoffFetcher which handles FIGI resolution, correct API endpoint
    (invest-public-api.tbank.ru:443), and GRPC_DNS_RESOLVER=native.
    Requires FINALAYZE_TINKOFF_TOKEN environment variable.

    Strips '.ME' suffix if present (yfinance convention) since the instrument
    registry uses plain MOEX tickers (SBER, GAZP, etc.).
    """
    token = os.environ.get("FINALAYZE_TINKOFF_TOKEN")
    if not token:
        print(f"  [warn] FINALAYZE_TINKOFF_TOKEN not set, skipping Tinkoff fetch for {symbol}")
        return []

    # Strip yfinance .ME suffix -- registry uses plain tickers
    clean_symbol = symbol.removesuffix(".ME")

    try:
        from finalayze.data.fetchers.tinkoff_data import TinkoffFetcher  # noqa: PLC0415
        from finalayze.markets.instruments import build_default_registry  # noqa: PLC0415

        registry = build_default_registry()
        fetcher = TinkoffFetcher(token=token, registry=registry, sandbox=False)
        end = datetime.now(tz=UTC)
        start = end - timedelta(days=MOEX_LOOKBACK_DAYS)
        return fetcher.fetch_candles(clean_symbol, start, end)
    except Exception as exc:
        print(f"  [warn] Tinkoff fetch failed for {symbol}: {exc}")
        return []


def fetch_symbol_candles(
    symbol: str,
    market_id: str,
    settings: Settings,
    segment_id: str | None = None,
) -> list[Candle]:
    """Fetch candles for a single symbol: DB first, then API fallback.

    For MOEX segments, tries Tinkoff API before yfinance. Uses segment-aware
    lookback (2 years for MOEX, 5 years for US).
    """
    candles = asyncio.run(fetch_from_db(symbol, market_id, settings))
    if candles:
        return candles

    lookback = get_lookback_days(segment_id) if segment_id else LOOKBACK_DAYS

    # For MOEX segments, try Tinkoff first
    if segment_id and is_moex_segment(segment_id):
        tinkoff_candles = fetch_tinkoff_candles(symbol)
        if tinkoff_candles:
            return tinkoff_candles

    # Fallback to yfinance
    end = datetime.now(tz=UTC)
    start = end - timedelta(days=lookback)
    fetcher = YFinanceFetcher(market_id=market_id)
    try:
        return fetcher.fetch_candles(symbol, start, end)
    except Exception as exc:
        print(f"  [warn] Could not fetch {symbol} from yfinance: {exc}")
        return []


def fetch_candles(
    segment_id: str, symbols: list[str], settings: Settings | None = None
) -> list[Candle]:
    """Fetch candles for all symbols in a segment, processing each independently."""
    from config.settings import Settings as _Settings  # noqa: PLC0415

    if settings is None:
        settings = _Settings()
    market_id = segment_id.split("_", maxsplit=1)[0]
    candles: list[Candle] = []
    for symbol in symbols:
        symbol_candles = fetch_symbol_candles(symbol, market_id, settings, segment_id=segment_id)
        candles.extend(symbol_candles)
    return candles


def build_market_data_loader(segment_ids: list[str]) -> MarketDataLoader:
    """Create a MarketDataLoader appropriate for the given set of segments.

    MOEX-specific fetchers (ISS + CBR) are only instantiated when at least one
    segment is MOEX, to avoid importing heavy gRPC deps unnecessarily.
    """
    from finalayze.data.fetchers._cache_utils import GenericFileCache  # noqa: PLC0415
    from finalayze.data.fetchers.caching import CachingFetcher  # noqa: PLC0415
    from finalayze.data.rate_limiter import RateLimiter  # noqa: PLC0415

    has_moex = any(sid.startswith("ru_") for sid in segment_ids)
    if has_moex:
        from finalayze.data.fetchers.cbr import CBRFetcher  # noqa: PLC0415
        from finalayze.data.fetchers.moex_iss import MoexISSFetcher  # noqa: PLC0415

        _moex_iss = MoexISSFetcher(rate_limiter=RateLimiter("moex_iss", rate=0.5, capacity=5))
        return MarketDataLoader(
            moex_iss_candles=CachingFetcher(_moex_iss, cache_dir=Path(".cache/moex_iss")),
            moex_iss_raw=_moex_iss,
            cbr=CBRFetcher(rate_limiter=RateLimiter("cbr", rate=0.2, capacity=3)),
            yfinance_fetcher=CachingFetcher(YFinanceFetcher(market_id="us")),
            turnover_cache=GenericFileCache(Path(".cache/turnover")),
            cbr_cache=GenericFileCache(Path(".cache/cbr")),
        )
    return MarketDataLoader(
        yfinance_fetcher=CachingFetcher(YFinanceFetcher(market_id="us")),
    )
