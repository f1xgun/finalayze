"""Train XGBoost + LightGBM + CatBoost models per market segment.

Usage:
    uv run python scripts/train_models.py
    uv run python scripts/train_models.py --segment us_tech
    uv run python scripts/train_models.py --segment us_tech --output-dir models/
    uv run python scripts/train_models.py --label-mode direction  # old next-bar labels
    uv run python scripts/train_models.py --label-mode triple_barrier  # default
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

# Ensure src/ and project root are importable when run directly
_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))  # for config.settings

# torch must be imported before lightgbm to prevent OpenMP thread-pool conflicts
import torch  # noqa: F401
from config.settings import Settings
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from finalayze.core.models import CandleModel
from finalayze.core.schemas import Candle
from finalayze.data.fetchers.yfinance import YFinanceFetcher
from finalayze.ml.features.technical import compute_features  # noqa: F401
from finalayze.ml.models.catboost_model import CatBoostModel
from finalayze.ml.models.lightgbm_model import LightGBMModel
from finalayze.ml.models.xgboost_model import XGBoostModel
from finalayze.ml.training import DEFAULT_WINDOW_SIZE, build_windows
from finalayze.ml.training.feature_selection import select_features_mi
from finalayze.ml.training.labeling import build_triple_barrier_dataset
from finalayze.ml.training.quality_gates import FoldMetrics
from finalayze.ml.training.sample_weights import compute_decay_weights

_WINDOW_SIZE = DEFAULT_WINDOW_SIZE
_TRAIN_RATIO = 0.70
_CALIBRATION_RATIO = 0.15
_TEST_RATIO = 0.15
_TUNED_PARAMS_DIR = Path(__file__).parent.parent / "results" / "tuned_params"

# Triple barrier parameters (match engine execution params)
_TB_UPPER_ATR_MULT = 2.0  # match ml_ensemble ATR stop
_TB_LOWER_ATR_MULT = 2.0  # symmetric barriers
_TB_MAX_HOLD = 20  # match DEFAULT_STRATEGY_HOLD_BARS["ml_ensemble"]
_TB_ATR_PERIOD = 14  # standard
_MOEX_ATR_UPLIFT = 1.2  # MOEX 1.2x uplift for wider barriers

# Label mode choices
LABEL_MODE_TRIPLE_BARRIER = "triple_barrier"
LABEL_MODE_DIRECTION = "direction"

# Benchmark tickers for market-neutral (excess return) labels
_US_BENCHMARK = "SPY"
_MOEX_BENCHMARK = "IMOEX"  # Moscow Exchange index


def _load_tuned_params(segment_id: str, model_type: str) -> dict | None:
    """Load Optuna-tuned params if available, else return None."""
    path = _TUNED_PARAMS_DIR / segment_id / f"{model_type}.json"
    if path.exists():
        with path.open() as f:
            return json.loads(f.read())
    return None


_LOOKBACK_DAYS = 1825  # 5 years of history for US segments
_MOEX_LOOKBACK_DAYS = 730  # 2 years for MOEX (post-sanctions structural break)
_DEFAULT_OUTPUT_DIR = "models/"
_MIN_CANDLES = _WINDOW_SIZE + 1  # need at least WINDOW_SIZE + 1 for one sample
_PURGE_GAP = _WINDOW_SIZE + _TB_MAX_HOLD  # 80 bars: feature window + label horizon

# XGBoost max_depth: shallower for MOEX (smaller dataset, prevent overfit)
_US_MAX_DEPTH = 5
_MOEX_MAX_DEPTH = 3

# MI feature selection: fewer features for MOEX (smaller dataset, 50:1 sample-to-feature ratio)
_US_MAX_FEATURES = 15
_MOEX_MAX_FEATURES = 10

# Map segment_id -> representative symbols for training data
_SEGMENT_SYMBOLS: dict[str, list[str]] = {
    "us_tech": [
        "AAPL",
        "MSFT",
        "GOOGL",
        "NVDA",
        "META",
        "AMZN",
        "TSLA",
        "CRM",
        "ADBE",
        "INTC",
        "AMD",
        "AVGO",
        "CSCO",
        "ORCL",
        "QCOM",
    ],
    "us_healthcare": [
        "JNJ",
        "PFE",
        "UNH",
        "ABBV",
        "MRK",
        "LLY",
        "TMO",
        "ABT",
        "BMY",
        "AMGN",
        "GILD",
        "MDT",
    ],
    "us_finance": [
        "JPM",
        "BAC",
        "GS",
        "MS",
        "WFC",
        "C",
        "BLK",
        "SCHW",
        "AXP",
        "USB",
        "PNC",
        "TFC",
    ],
    "us_broad": ["SPY", "QQQ", "DIA", "IWM", "VTI"],
    "ru_blue_chips": ["SBER", "GAZP", "LKOH", "GMKN", "ROSN", "NVTK", "PLZL", "MGNT"],
    "ru_energy": ["ROSN", "TATN", "NVTK", "LKOH", "SNGS", "SIBN"],
    "ru_tech": ["YNDX", "OZON", "VKCO", "CIAN"],
    "ru_finance": ["SBER", "VTBR", "TCSG", "MOEX", "CBOM"],
}


def _is_moex_segment(segment_id: str) -> bool:
    """Return True if segment_id is a MOEX/Russian segment."""
    return segment_id.startswith("ru_")


def _get_lookback_days(segment_id: str) -> int:
    """Return lookback days: 2 years for MOEX, 5 years for US."""
    return _MOEX_LOOKBACK_DAYS if _is_moex_segment(segment_id) else _LOOKBACK_DAYS


def _get_max_features(segment_id: str) -> int:
    """Return max MI-selected features: 10 for MOEX, 15 for US."""
    return _MOEX_MAX_FEATURES if _is_moex_segment(segment_id) else _US_MAX_FEATURES


def _get_xgboost_max_depth(segment_id: str) -> int:
    """Return XGBoost max_depth: 3 for MOEX, 5 for US."""
    return _MOEX_MAX_DEPTH if _is_moex_segment(segment_id) else _US_MAX_DEPTH


# CatBoost depth: shallower for MOEX (smaller dataset)
_US_CATBOOST_DEPTH = 4
_MOEX_CATBOOST_DEPTH = 3


def _get_catboost_depth(segment_id: str) -> int:
    """Return CatBoost depth: 3 for MOEX, 4 for US."""
    return _MOEX_CATBOOST_DEPTH if _is_moex_segment(segment_id) else _US_CATBOOST_DEPTH


def _align_benchmark_candles(
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


def _fetch_benchmark_candles(
    segment_id: str,
) -> list[Candle] | None:
    """Fetch benchmark candles for excess-return labeling.

    US segments: SPY via YFinanceFetcher.
    MOEX segments: IMOEX via TinkoffFetcher (requires token, else None).

    Returns None if benchmark cannot be fetched.
    """
    if _is_moex_segment(segment_id):
        return _fetch_moex_benchmark(segment_id)
    return _fetch_us_benchmark(segment_id)


def _fetch_moex_benchmark(segment_id: str) -> list[Candle] | None:
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
        start = end - timedelta(days=_MOEX_LOOKBACK_DAYS)
        candles = fetcher.fetch_candles(_MOEX_BENCHMARK, start, end)
        if candles:
            print(f"  [{segment_id}] Fetched {len(candles)} benchmark candles ({_MOEX_BENCHMARK}).")
            return candles
        print(
            f"  [{segment_id}] No benchmark candles for {_MOEX_BENCHMARK}, skipping excess returns."
        )
        return None
    except Exception as exc:
        print(f"  [{segment_id}] Failed to fetch MOEX benchmark: {exc}, skipping excess returns.")
        return None


def _fetch_us_benchmark(segment_id: str) -> list[Candle] | None:
    """Fetch SPY benchmark for US segments via yfinance."""
    lookback = _get_lookback_days(segment_id)
    end = datetime.now(tz=UTC)
    start = end - timedelta(days=lookback)
    market_id = segment_id.split("_", maxsplit=1)[0]
    fetcher = YFinanceFetcher(market_id=market_id)
    try:
        candles = fetcher.fetch_candles(_US_BENCHMARK, start, end)
        if candles:
            print(f"  [{segment_id}] Fetched {len(candles)} benchmark candles ({_US_BENCHMARK}).")
            return candles
        print(
            f"  [{segment_id}] No benchmark candles for {_US_BENCHMARK}, skipping excess returns."
        )
        return None
    except Exception as exc:
        print(
            f"  [{segment_id}] Failed to fetch benchmark "
            f"({_US_BENCHMARK}): {exc}, skipping excess returns."
        )
        return None


def _fetch_tinkoff_candles(symbol: str) -> list[Candle]:
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
        start = end - timedelta(days=_MOEX_LOOKBACK_DAYS)
        return fetcher.fetch_candles(clean_symbol, start, end)
    except Exception as exc:
        print(f"  [warn] Tinkoff fetch failed for {symbol}: {exc}")
        return []


def _orm_to_candle(row: CandleModel) -> Candle:
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


async def _fetch_from_db(symbol: str, market_id: str, settings: Settings) -> list[Candle]:
    """Try to load candles from DB. Returns empty list on failure."""
    try:
        engine = create_async_engine(settings.database_url, echo=False)
        async with AsyncSession(engine) as session:
            result = await session.execute(
                select(CandleModel)
                .where(CandleModel.symbol == symbol, CandleModel.market_id == market_id)
                .order_by(CandleModel.timestamp)
            )
            rows = result.scalars().all()
            return [_orm_to_candle(row) for row in rows]
    except Exception:
        return []


def _fetch_symbol_candles(
    symbol: str,
    market_id: str,
    settings: Settings,
    segment_id: str | None = None,
) -> list[Candle]:
    """Fetch candles for a single symbol: DB first, then API fallback.

    For MOEX segments, tries Tinkoff API before yfinance. Uses segment-aware
    lookback (2 years for MOEX, 5 years for US).
    """
    candles = asyncio.run(_fetch_from_db(symbol, market_id, settings))
    if candles:
        return candles

    lookback = _get_lookback_days(segment_id) if segment_id else _LOOKBACK_DAYS

    # For MOEX segments, try Tinkoff first
    if segment_id and _is_moex_segment(segment_id):
        tinkoff_candles = _fetch_tinkoff_candles(symbol)
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


def _fetch_candles(
    segment_id: str, symbols: list[str], settings: Settings | None = None
) -> list[Candle]:
    """Fetch candles for all symbols in a segment, processing each independently."""
    if settings is None:
        settings = Settings()
    market_id = segment_id.split("_", maxsplit=1)[0]
    candles: list[Candle] = []
    for symbol in symbols:
        symbol_candles = _fetch_symbol_candles(symbol, market_id, settings, segment_id=segment_id)
        candles.extend(symbol_candles)
    return candles


def _build_windows(
    candles: list[Candle],
) -> tuple[list[dict[str, float]], list[int]]:
    """Build (features, labels) from a single contiguous candle series.

    Delegates to the shared ``build_windows`` utility in ``finalayze.ml.training``.
    Discards timestamps (used only for multi-symbol temporal ordering).
    """
    features, labels, _ts = build_windows(candles, _WINDOW_SIZE)
    return features, labels


def _get_triple_barrier_params(segment_id: str) -> dict[str, float | int | bool]:
    """Return triple barrier parameters for a segment.

    MOEX segments get 1.2x ATR uplift for wider barriers (higher volatility).
    """
    if _is_moex_segment(segment_id):
        upper = _TB_UPPER_ATR_MULT * _MOEX_ATR_UPLIFT
        lower = _TB_LOWER_ATR_MULT * _MOEX_ATR_UPLIFT
    else:
        upper = _TB_UPPER_ATR_MULT
        lower = _TB_LOWER_ATR_MULT
    return {
        "upper_atr_mult": upper,
        "lower_atr_mult": lower,
        "max_hold": _TB_MAX_HOLD,
        "atr_period": _TB_ATR_PERIOD,
        "atr_scale": True,
    }


def _compute_uniqueness_from_hold_bars(hold_bars: list[int]) -> np.ndarray:  # type: ignore[type-arg]
    """Compute sample uniqueness from hold bar counts.

    Uses a sliding window approach: sample i spans bars [i, i + hold_bars[i]).
    Concurrency at each bar = number of active samples.
    Uniqueness = 1 / mean(concurrency over sample's span).

    O(n * max_hold) instead of O(n^2).
    """
    import numpy as _np  # noqa: PLC0415

    n = len(hold_bars)
    if n == 0:
        return _np.array([], dtype=_np.float64)

    max_bar = n + max(hold_bars) if hold_bars else n
    concurrency = _np.zeros(max_bar, dtype=_np.float64)

    # Count concurrent samples at each bar
    for i, hb in enumerate(hold_bars):
        if hb > 0:
            concurrency[i : i + hb] += 1.0

    # Compute uniqueness for each sample
    uniqueness = _np.empty(n, dtype=_np.float64)
    for i, hb in enumerate(hold_bars):
        if hb <= 0:
            uniqueness[i] = 1.0
            continue
        avg_conc = float(concurrency[i : i + hb].mean())
        uniqueness[i] = 1.0 / avg_conc if avg_conc > 0 else 1.0

    return uniqueness


def _build_dataset(
    segment_id: str,
    symbols: list[str],
    settings: Settings | None = None,
    label_mode: str = LABEL_MODE_TRIPLE_BARRIER,
    *,
    excess_returns: bool = False,
) -> tuple[list[dict[str, float]], list[int], np.ndarray | None, list[int] | None]:
    """Build (features, labels, barrier_weights, hold_bars) per symbol.

    Collects windows from all symbols and sorts by timestamp to maintain
    proper temporal ordering for train/test splits (no future leakage).

    Returns:
        Tuple of (features, labels, barrier_weights, hold_bars).
        barrier_weights is non-None only in triple_barrier mode (abs(pnl_pct)).
        hold_bars is non-None only in triple_barrier mode.
    """
    features, labels, weights, hold_bars, _timestamps = _build_dataset_with_timestamps(
        segment_id, symbols, settings, label_mode, excess_returns=excess_returns
    )
    return features, labels, weights, hold_bars


def _build_dataset_with_timestamps(
    segment_id: str,
    symbols: list[str],
    settings: Settings | None = None,
    label_mode: str = LABEL_MODE_TRIPLE_BARRIER,
    *,
    excess_returns: bool = False,
) -> tuple[list[dict[str, float]], list[int], np.ndarray | None, list[int] | None, list[datetime]]:
    """Build dataset with timestamps for calendar-date splitting (D4).

    Returns:
        Tuple of (features, labels, barrier_weights, hold_bars, timestamps).
    """
    if settings is None:
        settings = Settings()
    market_id = segment_id.split("_", maxsplit=1)[0]

    if label_mode == LABEL_MODE_TRIPLE_BARRIER:
        return _build_dataset_triple_barrier(
            segment_id,
            symbols,
            market_id,
            settings,
            excess_returns=excess_returns,
        )
    return _build_dataset_direction(segment_id, symbols, market_id, settings)


def _build_dataset_direction(
    segment_id: str,
    symbols: list[str],
    market_id: str,
    settings: Settings,
) -> tuple[list[dict[str, float]], list[int], np.ndarray | None, list[int] | None, list[datetime]]:
    """Build dataset with simple next-bar direction labels (old behavior)."""
    rows: list[tuple[datetime, dict[str, float], int]] = []
    for symbol in symbols:
        candles = _fetch_symbol_candles(symbol, market_id, settings, segment_id=segment_id)
        if len(candles) < _MIN_CANDLES:
            continue
        x_sym, y_sym, ts_sym = build_windows(candles, _WINDOW_SIZE)
        for ts, feat, lbl in zip(ts_sym, x_sym, y_sym, strict=True):
            rows.append((ts, feat, lbl))
    rows.sort(key=lambda r: r[0])
    features_out = [r[1] for r in rows]
    labels_out = [r[2] for r in rows]
    timestamps_out = [r[0] for r in rows]
    return features_out, labels_out, None, None, timestamps_out


def _build_dataset_triple_barrier(
    segment_id: str,
    symbols: list[str],
    market_id: str,
    settings: Settings,
    *,
    excess_returns: bool = False,
) -> tuple[list[dict[str, float]], list[int], np.ndarray | None, list[int] | None, list[datetime]]:
    """Build dataset with triple barrier labels.

    When excess_returns=True, fetches benchmark candles (SPY for US, IMOEX
    for MOEX) and aligns them per-symbol to produce market-neutral labels.
    """
    import numpy as _np  # noqa: PLC0415

    tb_params = _get_triple_barrier_params(segment_id)
    min_candles_tb = _WINDOW_SIZE + int(tb_params["max_hold"]) + 1
    rows: list[tuple[datetime, dict[str, float], int, float, int]] = []

    # Fetch benchmark candles once if excess returns requested
    raw_benchmark: list[Candle] | None = None
    if excess_returns:
        raw_benchmark = _fetch_benchmark_candles(segment_id)
        if raw_benchmark is None:
            print(f"  [{segment_id}] Could not fetch benchmark, falling back to absolute returns.")

    for symbol in symbols:
        candles = _fetch_symbol_candles(symbol, market_id, settings, segment_id=segment_id)
        if len(candles) < min_candles_tb:
            print(
                f"  [{segment_id}] {symbol}: only {len(candles)} candles, "
                f"need {min_candles_tb}+ for triple barrier -- skipping."
            )
            continue

        # Align benchmark to this symbol's candles
        aligned_bench: list[Candle] | None = None
        if raw_benchmark:
            aligned_bench = _align_benchmark_candles(candles, raw_benchmark)
            if len(aligned_bench) != len(candles):
                print(
                    f"  [{segment_id}] {symbol}: benchmark alignment "
                    f"mismatch ({len(aligned_bench)} vs {len(candles)}), "
                    "falling back to absolute returns."
                )
                aligned_bench = None

        x_sym, y_sym, w_sym, ts_sym, hb_sym = build_triple_barrier_dataset(
            candles,
            window_size=_WINDOW_SIZE,
            upper_atr_mult=float(tb_params["upper_atr_mult"]),
            lower_atr_mult=float(tb_params["lower_atr_mult"]),
            max_hold=int(tb_params["max_hold"]),
            atr_period=int(tb_params["atr_period"]),
            atr_scale=bool(tb_params["atr_scale"]),
            benchmark_candles=aligned_bench,
        )

        label_type = "excess-return" if aligned_bench else "absolute"
        pos_rate = f"{sum(y_sym) / len(y_sym):.1%}" if y_sym else "N/A"
        print(
            f"  [{segment_id}] {symbol}: {len(x_sym)} triple barrier samples "
            f"({label_type}, {pos_rate} positive)"
        )
        for ts, feat, lbl, wt, hb in zip(ts_sym, x_sym, y_sym, w_sym, hb_sym, strict=True):
            rows.append((ts, feat, lbl, wt, hb))

    rows.sort(key=lambda r: r[0])
    features_out = [r[1] for r in rows]
    labels_out = [r[2] for r in rows]
    weights_out = _np.array([r[3] for r in rows], dtype=float) if rows else None
    hold_bars_out = [r[4] for r in rows] if rows else None
    timestamps_out = [r[0] for r in rows]

    # Log overall label distribution
    if labels_out:
        pos_count = sum(labels_out)
        total = len(labels_out)
        label_mode_str = "Market-neutral" if raw_benchmark else "Absolute"
        print(
            f"  [{segment_id}] {label_mode_str} labels: "
            f"{pos_count / total:.1%} positive "
            f"({pos_count}/{total})"
        )

    return features_out, labels_out, weights_out, hold_bars_out, timestamps_out


# Walk-forward parameters (D1)
_WF_TRAIN_MONTHS = 12
_WF_CAL_MONTHS = 2
_WF_TEST_MONTHS = 4
_WF_STEP_MONTHS = 3

# BH correction (D3)
_BH_FDR = 0.10


def _generate_walk_forward_folds(
    timestamps: list[datetime],
) -> list[tuple[list[int], list[int], list[int]]]:
    """Generate walk-forward fold indices split by calendar date (D4).

    Each fold has: train indices, calibration indices, test indices.
    Purge gaps are applied between splits to prevent label leakage.

    Returns list of (train_idx, cal_idx, test_idx) tuples.
    """
    if not timestamps:
        return []

    start_date = timestamps[0]
    end_date = timestamps[-1]

    folds: list[tuple[list[int], list[int], list[int]]] = []
    fold_start = start_date

    while True:
        train_end = fold_start + timedelta(days=_WF_TRAIN_MONTHS * 30)
        purge1_end = train_end + timedelta(days=_PURGE_GAP)
        cal_end = purge1_end + timedelta(days=_WF_CAL_MONTHS * 30)
        purge2_end = cal_end + timedelta(days=_PURGE_GAP)
        test_end = purge2_end + timedelta(days=_WF_TEST_MONTHS * 30)

        if test_end > end_date + timedelta(days=1):
            break

        # Calendar-date split (D4): indices by date range, not row index
        train_idx = [i for i, ts in enumerate(timestamps) if fold_start <= ts < train_end]
        cal_idx = [i for i, ts in enumerate(timestamps) if purge1_end <= ts < cal_end]
        test_idx = [i for i, ts in enumerate(timestamps) if purge2_end <= ts < test_end]

        if train_idx and test_idx:
            folds.append((train_idx, cal_idx, test_idx))

        fold_start += timedelta(days=_WF_STEP_MONTHS * 30)

    return folds


def _apply_bh_correction(
    p_values: list[float],
    fdr: float = _BH_FDR,
) -> list[bool]:
    """Apply Benjamini-Hochberg FDR correction (D3).

    Returns a list of booleans: True if the model passes at that index.
    """
    if not p_values:
        return []

    n = len(p_values)
    # Sort p-values with original indices
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    results = [False] * n

    for rank, (orig_idx, pval) in enumerate(indexed, start=1):
        threshold = (rank / n) * fdr
        if pval <= threshold:
            results[orig_idx] = True
        else:
            # Once we fail, all higher p-values also fail
            break

    return results


def _compute_model_weights(
    results: dict[str, str],
) -> dict[str, float]:
    """Compute performance-weighted averaging weights from accuracy results.

    Weight = max(0, accuracy - 0.50)^2. Auto-excludes coin-flip models.
    """
    weights: dict[str, float] = {}
    for name, result_str in results.items():
        acc = 0.5
        for part in result_str.split():
            if part.startswith("acc="):
                acc = float(part.split("=")[1])
        weights[name.lower()] = max(0.0, acc - 0.50) ** 2
    return weights


def _evaluate_fold_metrics(
    models: list[XGBoostModel | LightGBMModel | CatBoostModel],
    test_features: list[dict[str, float]],
    test_labels: list[int],
    mean_uniqueness: float = 1.0,
) -> FoldMetrics:
    """Evaluate models on a test fold and compute FoldMetrics for quality gates."""
    probas_all: list[float] = []
    for feat in test_features:
        probs = []
        for m in models:
            trained = getattr(m, "_trained", None) or getattr(m, "_model", None)
            if trained is None:
                continue
            try:
                probs.append(m.predict_proba(feat))
            except Exception:
                continue
        probas_all.append(sum(probs) / len(probs) if probs else 0.5)

    preds = [round(p) for p in probas_all]
    n_test = len(test_labels)
    n_pos = sum(test_labels)
    n_neg = n_test - n_pos

    acc = float(accuracy_score(test_labels, preds)) if n_test > 0 else 0.5
    brier = float(brier_score_loss(test_labels, probas_all)) if n_test > 0 else 0.25

    # Sensitivity / specificity
    tp = sum(1 for p, y in zip(preds, test_labels, strict=True) if p == 1 and y == 1)
    tn = sum(1 for p, y in zip(preds, test_labels, strict=True) if p == 0 and y == 0)
    sensitivity = tp / n_pos if n_pos > 0 else 0.0
    specificity = tn / n_neg if n_neg > 0 else 0.0

    buy_count = sum(preds)
    buy_ratio = buy_count / n_test if n_test > 0 else 0.5

    return FoldMetrics(
        accuracy=acc,
        brier_score=brier,
        log_loss=float(log_loss(test_labels, probas_all, labels=[0, 1])) if n_test > 0 else 1.0,
        n_test=n_test,
        mean_uniqueness=mean_uniqueness,
        buy_ratio=buy_ratio,
        sensitivity=sensitivity,
        specificity=specificity,
        signal_count=n_test,
    )


def train_walk_forward(  # noqa: PLR0912, PLR0915
    segment_id: str,
    symbols: list[str],
    output_dir: Path,
    settings: Settings | None = None,
    label_mode: str = LABEL_MODE_TRIPLE_BARRIER,
    *,
    excess_returns: bool = False,
) -> dict[str, float] | None:
    """Train models using walk-forward validation (D1).

    Aligned with backtest walk-forward: 12mo train, 2mo cal, 4mo test, 3mo step.
    Returns per-gate pass rates, or None if insufficient data.
    """
    import numpy as _np  # noqa: PLC0415

    from finalayze.ml.training.quality_gates import (  # noqa: PLC0415
        evaluate_fold,
        evaluate_walk_forward,
    )

    if settings is None:
        settings = Settings()

    print(f"\n[{segment_id}] Walk-forward training (label_mode={label_mode})...")
    features, labels, barrier_weights, hold_bars, timestamps = _build_dataset_with_timestamps(
        segment_id,
        symbols,
        settings,
        label_mode,
        excess_returns=excess_returns,
    )
    if not features:
        print(f"[{segment_id}] No samples -- skipping.")
        return None

    folds = _generate_walk_forward_folds(timestamps)
    if not folds:
        min_months = _WF_TRAIN_MONTHS + _WF_CAL_MONTHS + _WF_TEST_MONTHS
        print(f"[{segment_id}] No valid WF folds (need {min_months}+ months of data).")
        return None

    print(f"[{segment_id}] {len(folds)} walk-forward folds")

    all_fold_results = []
    best_acc = 0.0
    best_models: list[XGBoostModel | LightGBMModel | CatBoostModel] | None = None
    best_selected_features: list[str] | None = None

    for fold_idx, (train_idx, cal_idx, test_idx) in enumerate(folds):
        train_f = [features[i] for i in train_idx]
        train_l = [labels[i] for i in train_idx]
        cal_f = [features[i] for i in cal_idx]
        test_f = [features[i] for i in test_idx]
        test_l = [labels[i] for i in test_idx]

        if len(train_f) < _WINDOW_SIZE:
            print(f"[{segment_id}] Fold {fold_idx}: too few train ({len(train_f)}), skip.")
            continue

        # Feature selection on train data only
        import pandas as pd  # noqa: PLC0415

        train_df = pd.DataFrame(train_f)
        train_series = pd.Series(train_l)
        max_feats = _get_max_features(segment_id)
        selected = select_features_mi(train_df, train_series, max_features=max_feats)

        if selected:
            train_f = [{k: row[k] for k in selected} for row in train_f]
            cal_f = [{k: row[k] for k in selected} for row in cal_f]
            test_f = [{k: row[k] for k in selected} for row in test_f]

        # Sample weights
        decay_w = compute_decay_weights(len(train_f))
        if hold_bars is not None:
            train_hb = [hold_bars[i] for i in train_idx if i < len(hold_bars)]
            if train_hb:
                uniq = _compute_uniqueness_from_hold_bars(train_hb)
                u_mean = float(uniq.mean()) if len(uniq) > 0 else 1.0
                uniq = uniq / u_mean if u_mean > 0 else uniq
            else:
                uniq = _np.ones(len(train_f), dtype=_np.float64)
        else:
            uniq = _np.ones(len(train_f), dtype=_np.float64)

        if barrier_weights is not None:
            bw_idx = [i for i in train_idx if i < len(barrier_weights)]
            train_bw = _np.array([barrier_weights[i] for i in bw_idx])
            dampened = _np.sqrt(_np.abs(train_bw))
            bw_mean = float(dampened.mean()) if len(dampened) > 0 else 1.0
            norm_bw = dampened / bw_mean if bw_mean > 0 else dampened
        else:
            norm_bw = _np.ones(len(train_f), dtype=_np.float64)

        sw = decay_w * uniq[: len(decay_w)] * norm_bw[: len(decay_w)]

        # Train models
        xgb = XGBoostModel(segment_id=segment_id, max_depth=_get_xgboost_max_depth(segment_id))
        lgbm = LightGBMModel(segment_id=segment_id)
        cat = CatBoostModel(segment_id=segment_id, depth=_get_catboost_depth(segment_id))

        xgb.fit(train_f, train_l, sample_weight=sw)
        lgbm.fit(train_f, train_l, sample_weight=sw)
        cat.fit(train_f, train_l, sample_weight=sw)

        models = [xgb, lgbm, cat]
        mean_uniq = float(uniq.mean()) if len(uniq) > 0 else 1.0

        # Evaluate on test fold
        if test_f:
            fold_metrics = _evaluate_fold_metrics(models, test_f, test_l, mean_uniq)
            gate_results = evaluate_fold(fold_metrics)
            all_fold_results.append(gate_results)

            passed_count = sum(1 for r in gate_results if r.passed)
            total_gates = len(gate_results)
            print(
                f"[{segment_id}] Fold {fold_idx}: acc={fold_metrics.accuracy:.3f}, "
                f"brier={fold_metrics.brier_score:.3f}, "
                f"gates={passed_count}/{total_gates}, "
                f"train={len(train_f)}, cal={len(cal_f)}, test={len(test_f)}"
            )

            if fold_metrics.accuracy > best_acc:
                best_acc = fold_metrics.accuracy
                best_models = models
                best_selected_features = selected

    if not all_fold_results:
        print(f"[{segment_id}] No folds produced results.")
        return None

    overall_passed, gate_pass_rates = evaluate_walk_forward(all_fold_results)

    status_str = "PASS" if overall_passed else "FAIL"
    print(f"\n[{segment_id}] Walk-forward results (overall: {status_str}):")
    for gate_name, rate in sorted(gate_pass_rates.items()):
        status = "PASS" if rate >= 0.60 else "FAIL"  # noqa: PLR2004
        print(f"  {gate_name:>20s}: {rate:.1%} [{status}]")

    # Save best models from walk-forward
    if best_models:
        segment_dir = output_dir / segment_id
        segment_dir.mkdir(parents=True, exist_ok=True)

        best_models[0].save(segment_dir / "xgb.pkl")
        best_models[1].save(segment_dir / "lgbm.pkl")
        best_models[2].save(segment_dir / "catboost.pkl")  # type: ignore[union-attr]

        if best_selected_features:
            (segment_dir / "selected_features.json").write_text(json.dumps(best_selected_features))

        # Save gate results for downstream BH correction
        gate_results_path = segment_dir / "wf_gate_results.json"
        gate_results_path.write_text(
            json.dumps(
                {
                    "overall_passed": overall_passed,
                    "gate_pass_rates": gate_pass_rates,
                    "n_folds": len(all_fold_results),
                    "best_accuracy": best_acc,
                },
                indent=2,
            )
        )

    return gate_pass_rates


def train_one_segment(  # noqa: PLR0915
    segment_id: str,
    symbols: list[str],
    output_dir: Path,
    settings: Settings | None = None,
    label_mode: str = LABEL_MODE_TRIPLE_BARRIER,
    *,
    excess_returns: bool = False,
) -> None:
    """Train and save models for a single segment."""
    import numpy as _np  # noqa: PLC0415

    if settings is None:
        settings = Settings()
    print(f"\n[{segment_id}] Fetching candles for {symbols} (label_mode={label_mode})...")

    features_list, label_list, barrier_weights, hold_bars = _build_dataset(
        segment_id,
        symbols,
        settings,
        label_mode=label_mode,
        excess_returns=excess_returns,
    )
    if not features_list:
        print(f"[{segment_id}] No samples -- skipping.")
        return

    if len(features_list) < _WINDOW_SIZE:
        print(f"[{segment_id}] Only {len(features_list)} samples, need {_WINDOW_SIZE}+, skipping.")
        return

    print(
        f"[{segment_id}] Total samples: {len(features_list)} "
        f"(label balance: {sum(label_list)}/{len(label_list)} positive)"
    )

    # Three-way temporal split: train / calibration / test
    # Purge gaps between each split prevent label leakage
    n = len(features_list)
    train_end = int(n * _TRAIN_RATIO)
    cal_start = min(train_end + _PURGE_GAP, n)
    cal_end = int(n * (_TRAIN_RATIO + _CALIBRATION_RATIO))
    test_start = min(cal_end + _PURGE_GAP, n)

    train_features = features_list[:train_end]
    train_labels = label_list[:train_end]
    cal_features = features_list[cal_start:cal_end]
    cal_labels = label_list[cal_start:cal_end]
    test_features = features_list[test_start:]
    test_labels = label_list[test_start:]

    print(
        f"[{segment_id}] Split: train={len(train_features)}, "
        f"cal={len(cal_features)}, test={len(test_features)} "
        f"(purge_gap={_PURGE_GAP})"
    )

    # Feature selection on TRAIN data only (no leakage -- design doc 2.3)
    selected_features: list[str] | None = None
    if train_features:
        import pandas as pd  # noqa: PLC0415

        feature_names = sorted(train_features[0].keys())
        train_df = pd.DataFrame(train_features)
        train_series = pd.Series(train_labels)
        max_feats = _get_max_features(segment_id)
        selected_features = select_features_mi(train_df, train_series, max_features=max_feats)
        if selected_features:
            print(
                f"[{segment_id}] Selected {len(selected_features)}/{len(feature_names)} "
                f"features (max={max_feats})"
            )
            train_features = [{k: row[k] for k in selected_features} for row in train_features]
            cal_features = [{k: row[k] for k in selected_features} for row in cal_features]
            test_features = [{k: row[k] for k in selected_features} for row in test_features]

    # Compute sample weights: combine decay, uniqueness, and barrier weights
    decay_weights = compute_decay_weights(len(train_features))

    # Uniqueness from overlapping labels (A6)
    if hold_bars is not None and len(hold_bars) >= train_end:
        train_hold_bars = hold_bars[:train_end]
        uniqueness = _compute_uniqueness_from_hold_bars(train_hold_bars)
        # Normalize to mean=1
        u_mean = float(uniqueness.mean()) if len(uniqueness) > 0 else 1.0
        uniqueness = uniqueness / u_mean if u_mean > 0 else uniqueness
    else:
        uniqueness = _np.ones(len(train_features), dtype=_np.float64)

    # Barrier weights: use sqrt to dampen extreme PnL values (A6)
    if barrier_weights is not None and len(barrier_weights) > 0:
        train_bw = barrier_weights[:train_end]
        dampened_bw = _np.sqrt(_np.abs(train_bw))
        bw_mean = float(dampened_bw.mean()) if len(dampened_bw) > 0 else 1.0
        normalized_bw = dampened_bw / bw_mean if bw_mean > 0 else dampened_bw
    else:
        normalized_bw = _np.ones(len(train_features), dtype=_np.float64)

    sample_weights = decay_weights * uniqueness * normalized_bw

    segment_dir = output_dir / segment_id
    segment_dir.mkdir(parents=True, exist_ok=True)

    # Persist MI-selected features for inference-time filtering (feature mismatch fix)
    if selected_features:
        features_path = segment_dir / "selected_features.json"
        features_path.write_text(json.dumps(selected_features))
        print(f"[{segment_id}] Saved selected_features.json ({len(selected_features)} features)")

    results = _train_and_evaluate_models(
        segment_id,
        segment_dir,
        train_features,
        train_labels,
        cal_features,
        cal_labels,
        test_features,
        test_labels,
        sample_weights=sample_weights,
    )
    summary = " | ".join(f"{k}: {v}" for k, v in results.items())
    print(f"[{segment_id}] {summary}")

    # Compute and save performance-weighted model weights
    model_weights = _compute_model_weights(results)
    weights_path = segment_dir / "model_weights.json"
    weights_path.write_text(json.dumps(model_weights, indent=2))
    print(f"[{segment_id}] Saved model_weights.json: {model_weights}")


def _evaluate_model(
    model: XGBoostModel | LightGBMModel | CatBoostModel,
    test_features: list[dict[str, float]],
    test_labels: list[int],
) -> str:
    """Evaluate a model and return a formatted summary string."""
    probas = [model.predict_proba(f) for f in test_features]
    preds = [round(p) for p in probas]
    acc = float(accuracy_score(test_labels, preds))
    brier = float(brier_score_loss(test_labels, probas))
    ll = float(log_loss(test_labels, probas, labels=[0, 1]))
    return f"acc={acc:.3f} brier={brier:.3f} logloss={ll:.3f}"


def _train_and_evaluate_models(  # noqa: PLR0912
    segment_id: str,
    segment_dir: Path,
    train_features: list[dict[str, float]],
    train_labels: list[int],
    cal_features: list[dict[str, float]],
    cal_labels: list[int],
    test_features: list[dict[str, float]],
    test_labels: list[int],
    sample_weights: np.ndarray | None = None,  # type: ignore[type-arg]
) -> dict[str, str]:
    """Train XGBoost, LightGBM, and LSTM; return evaluation results."""
    results: dict[str, str] = {}

    # Check for Optuna-tuned hyperparameters
    xgb_tuned = _load_tuned_params(segment_id, "xgboost")
    lgbm_tuned = _load_tuned_params(segment_id, "lightgbm")
    if xgb_tuned:
        print(f"[{segment_id}] Using tuned XGBoost params: {xgb_tuned}")
    if lgbm_tuned:
        print(f"[{segment_id}] Using tuned LightGBM params: {lgbm_tuned}")

    default_depth = _get_xgboost_max_depth(segment_id)
    xgb_kwargs: dict[str, int | float] = {"max_depth": default_depth}
    if xgb_tuned:
        xgb_kwargs["max_depth"] = xgb_tuned.get("max_depth", default_depth)
        for key in (
            "n_estimators",
            "learning_rate",
            "subsample",
            "colsample_bytree",
            "min_child_weight",
            "gamma",
            "reg_alpha",
            "reg_lambda",
        ):
            if key in xgb_tuned:
                xgb_kwargs[key] = xgb_tuned[key]
    xgb = XGBoostModel(segment_id=segment_id, **xgb_kwargs)  # type: ignore[arg-type]
    xgb.fit(train_features, train_labels, sample_weight=sample_weights)
    xgb.save(segment_dir / "xgb.pkl")

    # Log top-10 feature importances from XGBoost
    if xgb._model is not None and xgb._feature_names is not None:
        importances = xgb._model.feature_importances_
        feat_imp = sorted(
            zip(xgb._feature_names, importances, strict=True),
            key=lambda x: x[1],
            reverse=True,
        )
        print(f"[{segment_id}] XGBoost top-10 feature importances:")
        for name, score in feat_imp[:10]:
            print(f"  {name:>25s}: {score:.4f}")

    if test_features:
        results["XGB"] = _evaluate_model(xgb, test_features, test_labels)

    lgbm_kwargs: dict[str, int | float] = {}
    if lgbm_tuned:
        for key in (
            "n_estimators",
            "max_depth",
            "learning_rate",
            "num_leaves",
            "subsample",
            "colsample_bytree",
            "min_child_samples",
            "reg_alpha",
            "reg_lambda",
        ):
            if key in lgbm_tuned:
                lgbm_kwargs[key] = lgbm_tuned[key]
        # LightGBM uses feature_fraction/bagging_fraction in Optuna search space
        if "feature_fraction" in lgbm_tuned:
            lgbm_kwargs["colsample_bytree"] = lgbm_tuned["feature_fraction"]
        if "bagging_fraction" in lgbm_tuned:
            lgbm_kwargs["subsample"] = lgbm_tuned["bagging_fraction"]
    lgbm = LightGBMModel(segment_id=segment_id, **lgbm_kwargs)  # type: ignore[arg-type]
    lgbm.fit(train_features, train_labels, sample_weight=sample_weights)
    lgbm.save(segment_dir / "lgbm.pkl")
    if test_features:
        results["LGBM"] = _evaluate_model(lgbm, test_features, test_labels)

    catboost_depth = _get_catboost_depth(segment_id)
    catboost = CatBoostModel(segment_id=segment_id, depth=catboost_depth)
    catboost.fit(train_features, train_labels, sample_weight=sample_weights)
    catboost.save(segment_dir / "catboost.pkl")
    if test_features:
        results["CATBOOST"] = _evaluate_model(catboost, test_features, test_labels)

    # Fit EnsembleCalibrator on CALIBRATION set raw probabilities (out-of-sample)
    if cal_features and cal_labels:
        _fit_and_save_calibrator(
            segment_id,
            segment_dir,
            [xgb, lgbm, catboost],
            cal_features,
            cal_labels,
        )

    return results


def _fit_and_save_calibrator(
    segment_id: str,
    segment_dir: Path,
    models: list[XGBoostModel | LightGBMModel | CatBoostModel],
    test_features: list[dict[str, float]],
    test_labels: list[int],
) -> None:
    """Fit EnsembleCalibrator on out-of-sample ensemble probabilities and save it.

    Uses the TEST split to avoid data leakage: the calibrator sees the model's
    out-of-sample probability distribution, not the training distribution.
    """
    import numpy as _np  # noqa: PLC0415

    from finalayze.ml.calibration import EnsembleCalibrator  # noqa: PLC0415
    from finalayze.ml.loader import _atomic_save  # noqa: PLC0415

    raw_probas: list[float] = []
    for feat in test_features:
        probs: list[float] = []
        for m in models:
            trained = getattr(m, "_trained", None) or getattr(m, "_model", None)
            if trained is None:
                continue
            try:
                probs.append(m.predict_proba(feat))
            except Exception:
                continue
        if probs:
            raw_probas.append(sum(probs) / len(probs))
        else:
            raw_probas.append(0.5)

    calibrator = EnsembleCalibrator()
    calibrator.fit(_np.array(raw_probas), _np.array(test_labels))

    if calibrator.is_fitted:
        _atomic_save(calibrator, segment_dir / "calibrator.pkl")
        # Show calibration effect
        cal_low = calibrator.calibrate(0.2)
        cal_high = calibrator.calibrate(0.8)
        print(
            f"[{segment_id}] Calibrator fitted on {len(test_features)} OOS samples: "
            f"raw 0.2 -> {cal_low:.3f}, raw 0.8 -> {cal_high:.3f}"
        )
    else:
        print(f"[{segment_id}] Calibrator skipped (insufficient OOS data or single class)")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train XGBoost + LightGBM + CatBoost models per segment"
    )
    parser.add_argument(
        "--segment",
        default=None,
        help="Segment ID to train (default: all segments)",
    )
    parser.add_argument(
        "--output-dir",
        default=_DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {_DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--label-mode",
        default=LABEL_MODE_TRIPLE_BARRIER,
        choices=[LABEL_MODE_TRIPLE_BARRIER, LABEL_MODE_DIRECTION],
        help=(
            f"Labeling mode: '{LABEL_MODE_TRIPLE_BARRIER}' uses ATR-scaled triple barrier "
            f"labels (default), '{LABEL_MODE_DIRECTION}' uses simple next-bar direction labels."
        ),
    )
    parser.add_argument(
        "--walk-forward",
        action="store_true",
        default=False,
        help="Use walk-forward validation (D1) instead of single split.",
    )
    parser.add_argument(
        "--excess-returns",
        action="store_true",
        default=False,
        help=(
            "Use market-neutral (excess return) labels by subtracting "
            "benchmark return (SPY for US, IMOEX for MOEX). "
            "Only applies to triple_barrier label mode."
        ),
    )
    return parser.parse_args(argv)


def main() -> None:
    """Entry point."""
    args = _parse_args()
    output_dir = Path(args.output_dir)
    label_mode: str = args.label_mode
    walk_forward: bool = args.walk_forward
    excess_returns: bool = args.excess_returns

    if args.segment:
        segments = {args.segment: _SEGMENT_SYMBOLS.get(args.segment, [])}
    else:
        segments = _SEGMENT_SYMBOLS

    print(
        f"Label mode: {label_mode}, Walk-forward: {walk_forward}, Excess returns: {excess_returns}"
    )

    # Collect p-values for BH correction (D3) across all segments
    segment_accuracies: dict[str, float] = {}

    for segment_id, symbols in segments.items():
        try:
            if walk_forward:
                gate_rates = train_walk_forward(
                    segment_id=segment_id,
                    symbols=symbols,
                    output_dir=output_dir,
                    label_mode=label_mode,
                    excess_returns=excess_returns,
                )
                if gate_rates and "accuracy" in gate_rates:
                    # Load best accuracy from saved results
                    results_path = output_dir / segment_id / "wf_gate_results.json"
                    if results_path.exists():
                        wf_data = json.loads(results_path.read_text())
                        segment_accuracies[segment_id] = wf_data.get("best_accuracy", 0.5)
            else:
                train_one_segment(
                    segment_id=segment_id,
                    symbols=symbols,
                    output_dir=output_dir,
                    label_mode=label_mode,
                    excess_returns=excess_returns,
                )
        except FileNotFoundError as exc:
            print(f"[{segment_id}] FileNotFoundError -- {exc}, skipping.")
        except Exception as exc:
            print(f"[{segment_id}] Unexpected error -- {exc}, skipping.")

    # BH correction across all segments (D3)
    if walk_forward and segment_accuracies:
        _apply_bh_across_segments(segment_accuracies, output_dir)


def _apply_bh_across_segments(
    segment_accuracies: dict[str, float],
    output_dir: Path,
) -> None:
    """Apply BH multiple testing correction across all segments (D3).

    Converts accuracies to p-values using binomial test, then applies BH correction.
    Disables models in segments that fail the correction.
    """
    from scipy.stats import binom_test  # noqa: PLC0415

    segment_ids = list(segment_accuracies.keys())
    p_values: list[float] = []

    for seg_id in segment_ids:
        acc = segment_accuracies[seg_id]
        # Load n_test from wf results
        results_path = output_dir / seg_id / "wf_gate_results.json"
        n_folds = 1
        if results_path.exists():
            wf_data = json.loads(results_path.read_text())
            n_folds = wf_data.get("n_folds", 1)

        # Approximate: accuracy > 0.5 is the null hypothesis test
        n_correct = int(acc * n_folds * 100)  # approximate
        n_total = n_folds * 100
        p_val = float(binom_test(n_correct, n_total, 0.5, alternative="greater"))
        p_values.append(p_val)

    passes = _apply_bh_correction(p_values, fdr=_BH_FDR)

    print("\n=== BH Multiple Testing Correction (D3) ===")
    for seg_id, p_val, passed in zip(segment_ids, p_values, passes, strict=True):
        status = "PASS" if passed else "FAIL (disabled)"
        print(f"  {seg_id:>20s}: p={p_val:.4f} [{status}]")
        if not passed:
            # Mark segment as failed in its results file
            results_path = output_dir / seg_id / "wf_gate_results.json"
            if results_path.exists():
                wf_data = json.loads(results_path.read_text())
                wf_data["bh_passed"] = False
                results_path.write_text(json.dumps(wf_data, indent=2))


if __name__ == "__main__":
    main()
