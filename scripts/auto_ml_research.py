"""Autonomous ML experiment loop — autoresearch-inspired.

Runs unattended, trying feature/model configurations, evaluating with
walk-forward quality gates, and keeping improvements.  Logs every experiment
to ``results/experiments/experiment_log.jsonl``.

Inspired by karpathy/autoresearch: hypothesis → experiment → keep/discard → repeat.

Usage::

    # Feature ablation (drop features one at a time)
    uv run python scripts/auto_ml_research.py --segment us_tech --strategy ablation

    # Efficiency-driven feature selection (Pareto-optimal)
    uv run python scripts/auto_ml_research.py --segment us_tech --strategy efficiency

    # Hyperparameter perturbation (coordinate descent)
    uv run python scripts/auto_ml_research.py --segment us_tech --strategy hyperparameter

    # Random feature subsets
    uv run python scripts/auto_ml_research.py --segment us_tech --strategy random_subset

    # Run all strategies sequentially
    uv run python scripts/auto_ml_research.py --segment us_tech --strategy all

    # Limit experiments
    uv run python scripts/auto_ml_research.py --segment us_tech --max-experiments 20
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

# Ensure src/ and project root are importable
_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as _np
import pandas as pd
import structlog

# torch must be imported before lightgbm to prevent OpenMP thread-pool conflicts
import torch  # noqa: F401
from config.segments import DEFAULT_SEGMENTS
from sklearn.metrics import accuracy_score, brier_score_loss

from finalayze.core.schemas import (
    Candle,
    FXRate,
    KeyRateRecord,
    MarketContext,
    MoexMarketData,
    TurnoverRecord,
)
from finalayze.data.fetchers.yfinance import YFinanceFetcher
from finalayze.ml.models.catboost_model import CatBoostModel
from finalayze.ml.models.lightgbm_model import LightGBMModel
from finalayze.ml.models.xgboost_model import XGBoostModel
from finalayze.ml.training.feature_complexity import (
    summarize_complexity,
)
from finalayze.ml.training.feature_selection import (
    select_features_efficient,
)
from finalayze.ml.training.labeling import build_triple_barrier_dataset
from finalayze.ml.training.quality_gates import (
    FoldMetrics,
    evaluate_fold,
    evaluate_walk_forward,
)
from finalayze.ml.training.sample_weights import compute_decay_weights

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_WINDOW_SIZE = 80
_TB_UPPER_ATR_MULT = 2.0
_TB_LOWER_ATR_MULT = 2.0
_TB_MAX_HOLD = 20
_TB_ATR_PERIOD = 14
_MOEX_ATR_UPLIFT = 1.2
_LOOKBACK_DAYS = 1825
_MOEX_LOOKBACK_DAYS = 730
_US_MAX_FEATURES = 15
_MOEX_MAX_FEATURES = 10
_ENSEMBLE_WEIGHTS_MIN_FOLDS = 4  # minimum folds required to use optimized ensemble weights

_WF_TRAIN_MONTHS = 12
_WF_CAL_MONTHS = 2
_WF_TEST_MONTHS = 4
_WF_STEP_MONTHS = 3
_PURGE_GAP = 100

_MOEX_WF_TRAIN_MONTHS = 8
_MOEX_WF_CAL_MONTHS = 1
_MOEX_WF_TEST_MONTHS = 3
_MOEX_WF_STEP_MONTHS = 2
_MOEX_PURGE_GAP = 21
_MOEX_MIN_SIGNALS = 15
_US_MIN_SIGNALS = 50

_US_BENCHMARK = "SPY"
_MOEX_BENCHMARK = "IMOEX"
_VIX_TICKER = "^VIX"

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
}

# Populate ru_* equity segments from config/segments.py (single source of truth).
# Bond segments (instrument_type != "stock") are intentionally excluded.
for _seg in DEFAULT_SEGMENTS:
    if _seg.segment_id.startswith("ru_") and _seg.instrument_type == "stock":
        _SEGMENT_SYMBOLS[_seg.segment_id] = list(_seg.symbols)

_RESULTS_DIR = _PROJECT_ROOT / "results" / "experiments"

_MARKET_SPECIFIC_KEYWORDS = ("vix", "usdrub", "brent", "cbr", "imoex", "turnover")

# Default XGBoost / LightGBM / CatBoost hyperparameters
_DEFAULT_HPARAMS = {
    "xgb_max_depth": 5,
    "xgb_n_estimators": 200,
    "xgb_learning_rate": 0.05,
    "lgbm_n_estimators": 200,
    "lgbm_learning_rate": 0.05,
    "lgbm_num_leaves": 31,
    "cat_depth": 4,
    "cat_iterations": 200,
    "cat_learning_rate": 0.05,
}


# ---------------------------------------------------------------------------
# Segment helpers
# ---------------------------------------------------------------------------


def _is_moex_segment(segment_id: str) -> bool:
    """Return True if segment_id is a MOEX/Russian segment."""
    return segment_id.startswith("ru_")


def _get_lookback_days(segment_id: str) -> int:
    """Return lookback days: 730 for MOEX, 1825 for US."""
    return _MOEX_LOOKBACK_DAYS if _is_moex_segment(segment_id) else _LOOKBACK_DAYS


def _get_max_features(segment_id: str) -> int:
    """Return max MI-selected features: 10 for MOEX, 15 for US."""
    return _MOEX_MAX_FEATURES if _is_moex_segment(segment_id) else _US_MAX_FEATURES


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ExperimentConfig:
    """What to try in this experiment."""

    name: str
    description: str
    strategy: str  # ablation, efficiency, hyperparameter, random_subset
    feature_subset: list[str] | None = None
    max_features: int = _US_MAX_FEATURES
    hparams: dict[str, float | int] = field(default_factory=lambda: dict(_DEFAULT_HPARAMS))


@dataclass
class ExperimentResult:
    """What we got from the experiment."""

    config: ExperimentConfig
    n_folds: int = 0
    gate_pass_rates: dict[str, float] = field(default_factory=dict)
    overall_passed: bool = False
    avg_accuracy: float = 0.0
    avg_brier: float = 0.0
    avg_profit_factor: float = 0.0
    feature_count: int = 0
    features_used: list[str] = field(default_factory=list)
    complexity_summary: dict[str, float] = field(default_factory=dict)
    score: float = 0.0  # composite score for comparison
    status: str = "crash"  # keep, discard, crash
    duration_seconds: float = 0.0
    timestamp: str = ""
    error: str = ""


# ---------------------------------------------------------------------------
# Data loading (one-time)
# ---------------------------------------------------------------------------


def _fetch_us_candles(segment_id: str, symbols: list[str]) -> dict[str, list[Candle]]:
    """Fetch candles per symbol via yfinance."""
    market_id = segment_id.split("_", maxsplit=1)[0]
    fetcher = YFinanceFetcher(market_id=market_id)
    end = datetime.now(tz=UTC)
    start = end - timedelta(days=_LOOKBACK_DAYS)
    candles_by_sym: dict[str, list[Candle]] = {}
    for sym in symbols:
        try:
            candles = fetcher.fetch_candles(sym, start, end)
            if candles:
                candles_by_sym[sym] = candles
                print(f"  Fetched {len(candles)} candles for {sym}")
        except Exception as exc:
            print(f"  Failed to fetch {sym}: {exc}")
    return candles_by_sym


def _fetch_benchmark(segment_id: str) -> list[Candle] | None:
    """Fetch SPY benchmark candles."""
    market_id = segment_id.split("_", maxsplit=1)[0]
    fetcher = YFinanceFetcher(market_id=market_id)
    end = datetime.now(tz=UTC)
    start = end - timedelta(days=_LOOKBACK_DAYS)
    try:
        candles = fetcher.fetch_candles(_US_BENCHMARK, start, end)
        print(f"  Benchmark ({_US_BENCHMARK}): {len(candles)} candles")
        return candles
    except Exception as exc:
        print(f"  Benchmark fetch failed: {exc}")
        return None


def _fetch_vix(segment_id: str) -> list[Candle] | None:
    """Fetch VIX candles for regime features."""
    market_id = segment_id.split("_", maxsplit=1)[0]
    fetcher = YFinanceFetcher(market_id=market_id)
    end = datetime.now(tz=UTC)
    start = end - timedelta(days=_LOOKBACK_DAYS)
    try:
        candles = fetcher.fetch_candles(_VIX_TICKER, start, end)
        print(f"  VIX: {len(candles)} candles")
        return candles
    except Exception as exc:
        print(f"  VIX fetch failed: {exc}")
        return None


def _fetch_moex_candles(segment_id: str, symbols: list[str]) -> dict[str, list[Candle]]:
    """Fetch candles per symbol via TinkoffFetcher for MOEX segments.

    Requires FINALAYZE_TINKOFF_TOKEN env var. Returns empty dict if token missing.
    Token is never logged or printed (T-40-01 mitigation).
    """
    token = os.environ.get("FINALAYZE_TINKOFF_TOKEN")
    if not token:
        print(f"  ERROR: FINALAYZE_TINKOFF_TOKEN not set — cannot fetch MOEX data for {segment_id}")
        return {}
    from finalayze.data.fetchers.tinkoff_data import TinkoffFetcher  # noqa: PLC0415
    from finalayze.markets.instruments import build_default_registry  # noqa: PLC0415

    registry = build_default_registry()
    fetcher = TinkoffFetcher(token=token, registry=registry, sandbox=False)
    end = datetime.now(tz=UTC)
    start = end - timedelta(days=_MOEX_LOOKBACK_DAYS)
    candles_by_sym: dict[str, list[Candle]] = {}
    for sym in symbols:
        try:
            candles = fetcher.fetch_candles(sym, start, end)
            if candles:
                candles_by_sym[sym] = candles
                print(f"  Fetched {len(candles)} candles for {sym}")
        except Exception as exc:
            print(f"  Failed to fetch {sym}: {exc}")
    return candles_by_sym


def _fetch_moex_benchmark(segment_id: str) -> list[Candle] | None:
    """Fetch IMOEX benchmark for MOEX segments via TinkoffFetcher.

    Returns None if token is not set or fetch fails.
    Token is never logged or printed (T-40-01 mitigation).
    """
    token = os.environ.get("FINALAYZE_TINKOFF_TOKEN")
    if not token:
        print(f"  [{segment_id}] FINALAYZE_TINKOFF_TOKEN not set, skipping MOEX benchmark.")
        return None
    from finalayze.data.fetchers.tinkoff_data import TinkoffFetcher  # noqa: PLC0415
    from finalayze.markets.instruments import build_default_registry  # noqa: PLC0415

    registry = build_default_registry()
    fetcher = TinkoffFetcher(token=token, registry=registry, sandbox=False)
    end = datetime.now(tz=UTC)
    start = end - timedelta(days=_MOEX_LOOKBACK_DAYS)
    try:
        candles = fetcher.fetch_candles(_MOEX_BENCHMARK, start, end)
        if candles:
            print(f"  Benchmark ({_MOEX_BENCHMARK}): {len(candles)} candles")
            return candles
    except Exception as exc:
        print(f"  Benchmark fetch failed: {exc}")
    return None


def _fetch_moex_macro_data() -> MoexMarketData | None:
    """Fetch MOEX macro data: CBR key rate, USDRUB, Brent, turnover.

    Fetched once at script start and reused across all MOEX segments.
    Returns None if FINALAYZE_TINKOFF_TOKEN is not set (macro data requires API
    access for turnover).
    All 4 sources are fetched independently — individual failures fall back to
    empty data rather than aborting the whole fetch.
    """
    token = os.environ.get("FINALAYZE_TINKOFF_TOKEN")
    if not token:
        print("  MOEX macro data unavailable (no FINALAYZE_TINKOFF_TOKEN)")
        return None

    end_dt = datetime.now(tz=UTC)
    start_dt = end_dt - timedelta(days=_MOEX_LOOKBACK_DAYS + 90)  # extra 90d for z-score warmup

    # --- CBR key rate and FX rates (synchronous) ---
    key_rates: list[KeyRateRecord] = []
    fx_rates: list[FXRate] = []
    try:
        from finalayze.data.fetchers.cbr import CBRFetcher  # noqa: PLC0415

        with CBRFetcher() as cbr:
            key_rates = cbr.fetch_key_rate(start_dt, end_dt)
            fx_rates = cbr.fetch_fx_rates("USD", start_dt, end_dt)
        print(f"  CBR: {len(key_rates)} key rate records, {len(fx_rates)} FX rate records")
    except Exception as exc:
        print(f"  CBR fetch failed: {exc}")

    # --- MOEX turnover (synchronous) ---
    turnover: list[TurnoverRecord] = []
    try:
        from finalayze.data.fetchers.moex_iss import MoexISSFetcher  # noqa: PLC0415

        with MoexISSFetcher() as iss:
            turnover = iss.fetch_market_turnover(start_dt, end_dt)
        print(f"  Turnover: {len(turnover)} records")
    except Exception as exc:
        print(f"  Turnover fetch failed: {exc}")

    # --- Brent crude via yfinance (sync, BZ=F is not a MOEX ticker) ---
    brent_candles: list[Candle] = []
    try:
        yf = YFinanceFetcher(market_id="us")
        brent_candles = yf.fetch_candles("BZ=F", start_dt, end_dt)
        print(f"  Brent (BZ=F): {len(brent_candles)} candles")
    except Exception as exc:
        print(f"  Brent fetch failed: {exc}")

    return MoexMarketData(
        fx_rates=tuple(fx_rates) if fx_rates else None,
        key_rates=tuple(key_rates) if key_rates else None,
        commodity_candles={"BZ=F": tuple(brent_candles)} if brent_candles else None,
        turnover=tuple(turnover) if turnover else None,
    )


def _align_benchmark(stock_candles: list[Candle], bench_candles: list[Candle]) -> list[Candle]:
    """Align benchmark to stock candles by date (forward-fill)."""
    if not bench_candles or not stock_candles:
        return []
    bench_by_date: dict[datetime, Candle] = {}
    for c in bench_candles:
        key = c.timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
        bench_by_date[key] = c
    sorted_dates = sorted(bench_by_date.keys())
    if not sorted_dates:
        return []

    # Forward-fill
    min_d = min(
        sorted_dates[0],
        stock_candles[0].timestamp.replace(hour=0, minute=0, second=0, microsecond=0),
    )
    max_d = max(
        sorted_dates[-1],
        stock_candles[-1].timestamp.replace(hour=0, minute=0, second=0, microsecond=0),
    )
    ffill: dict[datetime, Candle] = {}
    cur = min_d
    cur_bench = bench_by_date[sorted_dates[0]]
    day = timedelta(days=1)
    while cur <= max_d:
        if cur in bench_by_date:
            cur_bench = bench_by_date[cur]
        ffill[cur] = cur_bench
        cur += day

    aligned: list[Candle] = []
    for sc in stock_candles:
        sd = sc.timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
        aligned.append(ffill.get(sd, cur_bench))
    return aligned


def build_full_dataset(
    _segment_id: str,
    candles_by_sym: dict[str, list[Candle]],
    benchmark_candles: list[Candle] | None,
    vix_candles: list[Candle] | None,
    moex_data: MoexMarketData | None = None,
) -> tuple[list[dict[str, float]], list[int], _np.ndarray | None, list[int] | None, list[datetime]]:
    """Build triple-barrier dataset from pre-fetched candles."""
    min_candles = _WINDOW_SIZE + _TB_MAX_HOLD + 1
    rows: list[tuple[datetime, dict[str, float], int, float, int]] = []

    # Apply ATR uplift for MOEX segments (wider barriers for higher volatility)
    is_moex = _segment_id.startswith("ru_")
    upper_mult = _TB_UPPER_ATR_MULT * (_MOEX_ATR_UPLIFT if is_moex else 1.0)
    lower_mult = _TB_LOWER_ATR_MULT * (_MOEX_ATR_UPLIFT if is_moex else 1.0)

    for candles in candles_by_sym.values():
        if len(candles) < min_candles:
            continue
        aligned_bench: list[Candle] | None = None
        if benchmark_candles:
            aligned_bench = _align_benchmark(candles, benchmark_candles)
            if len(aligned_bench) != len(candles):
                aligned_bench = None

        market_ctx = MarketContext(
            benchmark_candles=benchmark_candles,
            vix_candles=vix_candles,
            moex_data=moex_data,
        )

        x, y, w, ts, hb = build_triple_barrier_dataset(
            candles,
            window_size=_WINDOW_SIZE,
            upper_atr_mult=upper_mult,
            lower_atr_mult=lower_mult,
            max_hold=_TB_MAX_HOLD,
            atr_period=_TB_ATR_PERIOD,
            atr_scale=True,
            benchmark_candles=aligned_bench,
            vix_candles=vix_candles,
            market_context=market_ctx,
        )
        for t, feat, lbl, wt, h in zip(ts, x, y, w, hb, strict=True):
            rows.append((t, feat, lbl, wt, h))

    rows.sort(key=lambda r: r[0])
    features = [r[1] for r in rows]
    labels = [r[2] for r in rows]
    weights = _np.array([r[3] for r in rows], dtype=float) if rows else None
    hold_bars = [r[4] for r in rows] if rows else None
    timestamps = [r[0] for r in rows]

    n_total = len(labels)
    n_pos = sum(labels)
    pos_pct = n_pos / n_total if n_total > 0 else 0.0
    print(f"\nDataset: {len(features)} samples, {n_pos}/{n_total} positive ({pos_pct:.1%})")
    return features, labels, weights, hold_bars, timestamps


# ---------------------------------------------------------------------------
# Walk-forward folds
# ---------------------------------------------------------------------------


def generate_folds(
    timestamps: list[datetime],
    *,
    train_months: int = _WF_TRAIN_MONTHS,
    cal_months: int = _WF_CAL_MONTHS,
    test_months: int = _WF_TEST_MONTHS,
    step_months: int = _WF_STEP_MONTHS,
    purge_gap: int = _PURGE_GAP,
) -> list[tuple[list[int], list[int], list[int]]]:
    """Generate walk-forward train/cal/test folds by calendar date."""
    if not timestamps:
        return []
    start = timestamps[0]
    end = timestamps[-1]
    folds: list[tuple[list[int], list[int], list[int]]] = []
    fold_start = start

    while True:
        train_end = fold_start + timedelta(days=train_months * 30)
        purge1_end = train_end + timedelta(days=purge_gap)
        cal_end = purge1_end + timedelta(days=cal_months * 30)
        purge2_end = cal_end + timedelta(days=purge_gap)
        test_end = purge2_end + timedelta(days=test_months * 30)

        if test_end > end + timedelta(days=1):
            break
        train_idx = [i for i, t in enumerate(timestamps) if fold_start <= t < train_end]
        cal_idx = [i for i, t in enumerate(timestamps) if purge1_end <= t < cal_end]
        test_idx = [i for i, t in enumerate(timestamps) if purge2_end <= t < test_end]
        if train_idx and test_idx:
            folds.append((train_idx, cal_idx, test_idx))
        fold_start += timedelta(days=step_months * 30)

    return folds


# ---------------------------------------------------------------------------
# Single experiment execution
# ---------------------------------------------------------------------------


def _evaluate_models(
    models: list[XGBoostModel | LightGBMModel | CatBoostModel],
    test_f: list[dict[str, float]],
    test_l: list[int],
    mean_uniqueness: float,
    avg_hold_bars: float,
    weights: list[float] | None = None,
) -> FoldMetrics:
    """Evaluate ensemble on test fold → FoldMetrics."""
    probas: list[float] = []
    for feat in test_f:
        probs = []
        for m in models:
            trained = getattr(m, "_trained", None) or getattr(m, "_model", None)
            if trained is None:
                continue
            try:
                probs.append(m.predict_proba(feat))
            except Exception:
                continue
        if weights and len(probs) == len(weights):
            probas.append(sum(p * w for p, w in zip(probs, weights, strict=True)))
        else:
            probas.append(sum(probs) / len(probs) if probs else 0.5)

    preds = [round(p) for p in probas]
    n = len(test_l)
    n_pos = sum(test_l)
    n_neg = n - n_pos
    acc = float(accuracy_score(test_l, preds)) if n > 0 else 0.5
    brier = float(brier_score_loss(test_l, probas)) if n > 0 else 0.25

    tp = sum(1 for p, y in zip(preds, test_l, strict=True) if p == 1 and y == 1)
    tn = sum(1 for p, y in zip(preds, test_l, strict=True) if p == 0 and y == 0)
    sensitivity = tp / n_pos if n_pos > 0 else 0.0
    specificity = tn / n_neg if n_neg > 0 else 0.0
    buy_ratio = sum(preds) / n if n > 0 else 0.5

    pf_threshold = 0.55
    gross_profit = sum(
        1.0 for prob, lbl in zip(probas, test_l, strict=True) if prob >= pf_threshold and lbl == 1
    )
    gross_loss = sum(
        1.0 for prob, lbl in zip(probas, test_l, strict=True) if prob >= pf_threshold and lbl == 0
    )
    pf = gross_profit / gross_loss if gross_loss > 0 else (2.0 if gross_profit > 0 else 1.0)

    return FoldMetrics(
        accuracy=acc,
        brier_score=brier,
        log_loss=0.0,
        n_test=n,
        mean_uniqueness=mean_uniqueness,
        buy_ratio=buy_ratio,
        sensitivity=sensitivity,
        specificity=specificity,
        profit_factor=pf,
        signal_count=n,
        avg_hold_bars=avg_hold_bars,
    )


def _run_fold(
    train_idx: list[int],
    test_idx: list[int],
    all_features: list[dict[str, float]],
    labels: list[int],
    hold_bars: list[int] | None,
    config: ExperimentConfig,
    segment_id: str,
    min_signals: int = _US_MIN_SIGNALS,
) -> tuple[list, FoldMetrics, list[str]] | None:
    """Train and evaluate a single fold.  Returns None if fold is skipped."""
    train_f = [all_features[i] for i in train_idx]
    train_l = [labels[i] for i in train_idx]
    test_f = [all_features[i] for i in test_idx]
    test_l = [labels[i] for i in test_idx]

    if len(train_f) < _WINDOW_SIZE:
        return None

    if config.feature_subset is not None:
        selected = config.feature_subset
    else:
        train_df = pd.DataFrame(train_f)
        train_s = pd.Series(train_l)
        selected = select_features_efficient(
            train_df,
            train_s,
            max_features=config.max_features,
        )

    if selected:
        train_f = [{k: row.get(k, 0.0) for k in selected} for row in train_f]
        test_f = [{k: row.get(k, 0.0) for k in selected} for row in test_f]

    sw = compute_decay_weights(len(train_f))

    hp = config.hparams
    xgb_model = XGBoostModel(
        segment_id=segment_id,
        max_depth=int(hp.get("xgb_max_depth", 5)),
    )
    lgbm_model = LightGBMModel(segment_id=segment_id)
    cat_model = CatBoostModel(
        segment_id=segment_id,
        depth=int(hp.get("cat_depth", 4)),
    )
    xgb_model.fit(train_f, train_l, sample_weight=sw)
    lgbm_model.fit(train_f, train_l, sample_weight=sw)
    cat_model.fit(train_f, train_l, sample_weight=sw)
    models = [xgb_model, lgbm_model, cat_model]

    fold_avg_hold = 1.0
    if hold_bars is not None:
        test_hb = [hold_bars[i] for i in test_idx if i < len(hold_bars)]
        fold_avg_hold = float(_np.mean(test_hb)) if test_hb else 1.0

    hp = config.hparams
    w_keys = ("xgb_weight", "lgbm_weight", "cat_weight")
    fold_weights = [float(hp[k]) for k in w_keys] if all(k in hp for k in w_keys) else None
    fold_metrics = _evaluate_models(
        models, test_f, test_l, 1.0, fold_avg_hold, weights=fold_weights
    )
    gate_results = evaluate_fold(fold_metrics, min_signals=min_signals)
    return gate_results, fold_metrics, list(selected) if selected else []


def run_experiment(
    config: ExperimentConfig,
    all_features: list[dict[str, float]],
    labels: list[int],
    hold_bars: list[int] | None,
    folds: list[tuple[list[int], list[int], list[int]]],
    segment_id: str,
) -> ExperimentResult:
    """Run one experiment: select features → train → evaluate → score."""
    t0 = time.monotonic()
    result = ExperimentResult(
        config=config,
        timestamp=datetime.now(tz=UTC).isoformat(),
    )

    # Small-fold guard: skip weight optimization when too few folds
    if config.strategy == "ensemble_weights" and len(folds) < _ENSEMBLE_WEIGHTS_MIN_FOLDS:
        logger.warning(
            "ensemble_weights: fewer than 4 folds, using equal weights",
            n_folds=len(folds),
        )
        config = ExperimentConfig(
            name=config.name,
            description=config.description + " (equal weights — insufficient folds)",
            strategy=config.strategy,
            feature_subset=config.feature_subset,
            max_features=config.max_features,
            hparams={
                **config.hparams,
                "xgb_weight": 1 / 3,
                "lgbm_weight": 1 / 3,
                "cat_weight": 1 / 3,
            },
        )

    try:
        all_fold_results: list[list] = []
        fold_accs: list[float] = []
        fold_briers: list[float] = []
        fold_pfs: list[float] = []
        features_used: list[str] = []

        min_signals = _MOEX_MIN_SIGNALS if _is_moex_segment(segment_id) else _US_MIN_SIGNALS
        for fold_idx, (train_idx, _cal_idx, test_idx) in enumerate(folds):
            fold_out = _run_fold(
                train_idx,
                test_idx,
                all_features,
                labels,
                hold_bars,
                config,
                segment_id,
                min_signals=min_signals,
            )
            if fold_out is None:
                continue
            gate_results, fold_metrics, selected = fold_out
            all_fold_results.append(gate_results)
            fold_accs.append(fold_metrics.accuracy)
            fold_briers.append(fold_metrics.brier_score)
            fold_pfs.append(fold_metrics.profit_factor)
            if fold_idx == 0:
                features_used = selected

        if not all_fold_results:
            result.error = "no valid folds"
            result.duration_seconds = time.monotonic() - t0
            return result

        _fill_result(result, all_fold_results, fold_accs, fold_briers, fold_pfs, features_used)

    except Exception as exc:
        result.error = str(exc)
        result.status = "crash"

    result.duration_seconds = round(time.monotonic() - t0, 1)
    return result


# ---------------------------------------------------------------------------
def _fill_result(
    result: ExperimentResult,
    all_fold_results: list[list],
    fold_accs: list[float],
    fold_briers: list[float],
    fold_pfs: list[float],
    features_used: list[str],
) -> None:
    """Populate result fields from fold evaluations."""
    overall_passed, gate_pass_rates = evaluate_walk_forward(all_fold_results)

    result.n_folds = len(all_fold_results)
    result.gate_pass_rates = gate_pass_rates
    result.overall_passed = overall_passed
    result.avg_accuracy = float(_np.mean(fold_accs))
    result.avg_brier = float(_np.mean(fold_briers))
    result.avg_profit_factor = float(_np.mean(fold_pfs))
    result.feature_count = len(features_used)
    result.features_used = features_used
    result.complexity_summary = summarize_complexity(features_used)

    complexity_penalty = result.complexity_summary.get("mean", 0.5)
    result.score = (
        0.4 * result.avg_accuracy
        + 0.3 * (1.0 - result.avg_brier)
        + 0.2 * sum(gate_pass_rates.values()) / max(len(gate_pass_rates), 1)
        + 0.1 * (1.0 - complexity_penalty)
    )
    result.status = "keep" if overall_passed else "discard"


# ---------------------------------------------------------------------------
# Hypothesis generation strategies
# ---------------------------------------------------------------------------


def generate_ablation_experiments(
    baseline_features: list[str],
) -> list[ExperimentConfig]:
    """Drop each feature one at a time to measure its marginal contribution."""
    experiments: list[ExperimentConfig] = []
    for feat in baseline_features:
        subset = [f for f in baseline_features if f != feat]
        experiments.append(
            ExperimentConfig(
                name=f"ablate-{feat}",
                description=f"Drop {feat}, test if quality holds (simplification check)",
                strategy="ablation",
                feature_subset=subset,
            )
        )
    return experiments


def generate_efficiency_experiments() -> list[ExperimentConfig]:
    """Try efficiency-weighted selection with varying budgets."""
    return [
        ExperimentConfig(
            name=f"efficient-top{max_f}",
            description=f"Efficiency-weighted selection, max {max_f} features",
            strategy="efficiency",
            max_features=max_f,
        )
        for max_f in [5, 8, 10, 12, 15]
    ]


def generate_hyperparameter_experiments(
    baseline_features: list[str],
) -> list[ExperimentConfig]:
    """Perturb model hyperparameters one at a time."""
    experiments: list[ExperimentConfig] = []

    perturbations = [
        ("xgb_max_depth", [3, 4, 6, 7]),
        ("xgb_learning_rate", [0.01, 0.03, 0.08, 0.10]),
        ("lgbm_num_leaves", [15, 20, 40, 63]),
        ("cat_depth", [3, 5, 6]),
    ]
    for param, values in perturbations:
        for val in values:
            if val == _DEFAULT_HPARAMS.get(param):
                continue
            hp = dict(_DEFAULT_HPARAMS)
            hp[param] = val
            experiments.append(
                ExperimentConfig(
                    name=f"hp-{param}={val}",
                    description=f"Perturb {param} to {val}",
                    strategy="hyperparameter",
                    feature_subset=baseline_features,
                    hparams=hp,
                )
            )
    return experiments


def generate_random_subset_experiments(
    all_feature_names: list[str],
    n_experiments: int = 10,
) -> list[ExperimentConfig]:
    """Random feature subsets of varying sizes."""
    experiments: list[ExperimentConfig] = []
    for i in range(n_experiments):
        size = random.randint(5, min(15, len(all_feature_names)))  # noqa: S311
        subset = sorted(random.sample(all_feature_names, size))
        name = f"random-{i + 1}-n{size}"
        experiments.append(
            ExperimentConfig(
                name=name,
                description=f"Random subset of {size} features",
                strategy="random_subset",
                feature_subset=subset,
            )
        )
    return experiments


def generate_transfer_experiments(segment_id: str) -> list[ExperimentConfig]:
    """Transfer best US "keep" experiment features to a MOEX segment.

    Reads the best "keep" entry (highest score) from the US tech JSONL log,
    filters out market-specific features (VIX, USDRUB, Brent, CBR, IMOEX, turnover),
    and returns a single ExperimentConfig with the surviving market-neutral features.

    Only applies to ru_* segments (MOEX). US segments return an empty list.
    """
    if not segment_id.startswith("ru_"):
        return []

    source_log = _RESULTS_DIR / "us_tech_experiment_log.jsonl"
    if not source_log.exists():
        logger.warning(
            "Cross-segment transfer skipped: US JSONL log not found",
            path=str(source_log),
        )
        return []

    keep_entries: list[dict] = []
    with source_log.open() as f:
        for raw_line in f:
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                entry = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if entry.get("status") == "keep":
                keep_entries.append(entry)

    if not keep_entries:
        logger.warning(
            "Cross-segment transfer skipped: no 'keep' entries in US JSONL log",
            path=str(source_log),
        )
        return []

    best = max(keep_entries, key=lambda e: e.get("score", 0.0))
    raw_features: list[str] = best.get("features_used", [])

    filtered = [
        feat
        for feat in raw_features
        if not any(kw in feat.lower() for kw in _MARKET_SPECIFIC_KEYWORDS)
    ]

    if not filtered:
        logger.warning(
            "Cross-segment transfer skipped: all features are market-specific after filtering",
            segment_id=segment_id,
        )
        return []

    return [
        ExperimentConfig(
            name=f"transfer-us-to-{segment_id}",
            description=(
                f"Transfer US market-neutral features from best keep experiment "
                f"(score={best.get('score', 0):.4f}) to {segment_id}"
            ),
            strategy="cross_segment_transfer",
            feature_subset=filtered,
        )
    ]


def generate_ensemble_weight_experiments() -> list[ExperimentConfig]:
    """Explore XGB/LGBM/CatBoost weight simplex with step 0.1, cap 0.7."""
    experiments: list[ExperimentConfig] = []
    step = 10  # work in integers to avoid float drift
    max_single = 7  # 0.7 cap
    for i in range(step + 1):
        for j in range(step + 1 - i):
            k = step - i - j
            if i > max_single or j > max_single or k > max_single:
                continue
            if i == 0 or j == 0 or k == 0:
                continue  # require all three models present
            w_xgb = i / step
            w_lgbm = j / step
            w_cat = k / step
            hp = dict(_DEFAULT_HPARAMS)
            hp["xgb_weight"] = w_xgb
            hp["lgbm_weight"] = w_lgbm
            hp["cat_weight"] = w_cat
            experiments.append(
                ExperimentConfig(
                    name=f"ew-{w_xgb:.1f}-{w_lgbm:.1f}-{w_cat:.1f}",
                    description=(
                        f"Ensemble weights: XGB={w_xgb:.1f} LGBM={w_lgbm:.1f} Cat={w_cat:.1f}"
                    ),
                    strategy="ensemble_weights",
                    hparams=hp,
                )
            )
    return experiments


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def _log_result(result: ExperimentResult, log_path: Path) -> None:
    """Append experiment result to JSONL log."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "name": result.config.name,
        "description": result.config.description,
        "strategy": result.config.strategy,
        "timestamp": result.timestamp,
        "status": result.status,
        "score": round(result.score, 6),
        "avg_accuracy": round(result.avg_accuracy, 4),
        "avg_brier": round(result.avg_brier, 4),
        "avg_profit_factor": round(result.avg_profit_factor, 4),
        "feature_count": result.feature_count,
        "features_used": result.features_used,
        "complexity": result.complexity_summary,
        "gate_pass_rates": {k: round(v, 3) for k, v in result.gate_pass_rates.items()},
        "overall_passed": result.overall_passed,
        "n_folds": result.n_folds,
        "duration_seconds": result.duration_seconds,
        "hparams": result.config.hparams,
        "error": result.error,
    }
    with log_path.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def _init_experiment_manager(
    experiment_id: str,
    strategy: str,
    segment_id: str,
) -> object | None:
    """Create and start an ExperimentManager entry (non-crashing).

    Returns the ExperimentManager instance on success, or None on failure.
    """
    try:
        from finalayze.core.experiment_manager import ExperimentManager  # noqa: PLC0415
        from finalayze.core.schemas import SuccessCriteria as SuccessCriteriaSchema  # noqa: PLC0415

        mgr = ExperimentManager()
        mgr.create_experiment(
            experiment_id=experiment_id,
            hypothesis=f"AutoML research: {strategy} on {segment_id}",
            success_criteria=SuccessCriteriaSchema(
                metric="composite_score", threshold=0.0, operator=">="
            ),
        )
        mgr.update_status(experiment_id, "running")
        return mgr
    except Exception:
        logger.warning(
            "ExperimentManager init failed — research loop will continue without tracking"
        )
        return None


def _link_to_experiment_manager(
    mgr: object,
    experiment_id: str,
    result: ExperimentResult,
    segment_id: str,
) -> None:
    """Link a single ExperimentResult to the ExperimentManager (non-crashing)."""
    try:
        from finalayze.core.schemas import (  # noqa: PLC0415
            ExperimentResult as ExperimentResultSchema,
        )

        schema_result = ExperimentResultSchema(
            run_name=result.config.name,
            iteration_name=f"{segment_id}_{result.config.strategy}",
            metrics={
                "score": result.score,
                "accuracy": result.avg_accuracy,
                "brier": result.avg_brier,
                "profit_factor": result.avg_profit_factor,
                "feature_count": result.feature_count,
                "status": result.status,
            },
        )
        mgr.link_result(experiment_id, schema_result)  # type: ignore[union-attr]
    except Exception:
        logger.warning("ExperimentManager link_result failed — continuing")


def _print_result(result: ExperimentResult, baseline_score: float) -> None:
    """Print experiment result to console."""
    delta = result.score - baseline_score
    icon = "+" if delta > 0 else ("-" if delta < 0 else "=")
    status_icons = {"keep": "KEEP", "discard": "DISC", "crash": "FAIL"}
    print(
        f"  [{status_icons.get(result.status, '????')}] {result.config.name:40s} "
        f"score={result.score:.4f} ({icon}{abs(delta):.4f}) "
        f"acc={result.avg_accuracy:.3f} brier={result.avg_brier:.3f} "
        f"pf={result.avg_profit_factor:.2f} feats={result.feature_count} "
        f"({result.duration_seconds:.0f}s)"
    )
    if result.error:
        print(f"         error: {result.error}")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def _generate_experiments(
    strategy: str,
    baseline_features: list[str],
    all_feature_names: list[str],
    max_experiments: int,
    segment_id: str = "us_tech",
) -> list[ExperimentConfig]:
    """Generate experiment configs for the chosen strategy."""
    experiments: list[ExperimentConfig] = []
    if strategy in ("ablation", "all"):
        experiments.extend(generate_ablation_experiments(baseline_features))
    if strategy in ("efficiency", "all"):
        experiments.extend(generate_efficiency_experiments())
    if strategy in ("hyperparameter", "all"):
        experiments.extend(generate_hyperparameter_experiments(baseline_features))
    if strategy in ("random_subset", "all"):
        experiments.extend(generate_random_subset_experiments(all_feature_names))
    if strategy in ("ensemble_weights", "all"):
        experiments.extend(generate_ensemble_weight_experiments())
    if strategy in ("cross_segment_transfer", "all"):
        experiments.extend(generate_transfer_experiments(segment_id))
    return experiments[:max_experiments]


def _print_summary(
    log_path: Path,
    n_experiments: int,
    improvements: int,
    best_name: str,
    best_score: float,
    baseline_score: float,
    total_time: float,
) -> None:
    """Print final summary and top experiments."""
    print(f"\n{'=' * 70}")
    print("  RESEARCH COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Experiments run: {n_experiments + 1} (incl. baseline)")
    print(f"  Improvements found: {improvements}")
    print(f"  Best experiment: {best_name} (score={best_score:.4f})")
    print(f"  Baseline score: {baseline_score:.4f}")
    print(f"  Total time: {total_time:.0f}s")
    print(f"  Results logged to: {log_path}")

    print("\n  Top experiments by score:")
    all_results: list[dict] = []
    if log_path.exists():
        with log_path.open() as f:
            all_results.extend(json.loads(line) for line in f if line.strip())
    all_results.sort(key=lambda r: r.get("score", 0), reverse=True)
    for i, r in enumerate(all_results[:5], 1):
        print(
            f"    {i}. {r['name']:35s} score={r['score']:.4f} "
            f"feats={r['feature_count']} status={r['status']}"
        )


def _prepare_data(
    segment_id: str,
    moex_data: MoexMarketData | None = None,
) -> (
    tuple[
        list[dict[str, float]],
        list[int],
        list[int] | None,
        list[tuple[list[int], list[int], list[int]]],
    ]
    | None
):
    """Fetch data, build dataset, generate folds. Returns None on failure."""
    symbols = _SEGMENT_SYMBOLS.get(segment_id)
    if not symbols:
        print(f"Unknown segment: {segment_id}")
        return None

    is_moex = _is_moex_segment(segment_id)

    print("Step 1: Fetching data...")
    if is_moex:
        candles_by_sym = _fetch_moex_candles(segment_id, symbols)
        benchmark = _fetch_moex_benchmark(segment_id)
        vix = None  # VIX is US-specific
    else:
        candles_by_sym = _fetch_us_candles(segment_id, symbols)
        benchmark = _fetch_benchmark(segment_id)
        vix = _fetch_vix(segment_id)

    if not candles_by_sym:
        print("No data fetched, aborting.")
        return None

    print("\nStep 2: Building dataset...")
    features, labels, _weights, hold_bars, timestamps = build_full_dataset(
        segment_id,
        candles_by_sym,
        benchmark,
        vix,
        moex_data=moex_data if is_moex else None,
    )

    if not features:
        print("Empty dataset, aborting.")
        return None

    print("\nStep 3: Generating walk-forward folds...")
    fold_kwargs: dict[str, int] = {}
    if is_moex:
        fold_kwargs = {
            "train_months": _MOEX_WF_TRAIN_MONTHS,
            "cal_months": _MOEX_WF_CAL_MONTHS,
            "test_months": _MOEX_WF_TEST_MONTHS,
            "step_months": _MOEX_WF_STEP_MONTHS,
            "purge_gap": _MOEX_PURGE_GAP,
        }
    folds = generate_folds(timestamps, **fold_kwargs)
    print(f"  {len(folds)} walk-forward folds")

    if not folds:
        print("No valid folds, aborting.")
        return None

    return features, labels, hold_bars, folds


def run_research_loop(
    segment_id: str,
    strategy: str = "all",
    max_experiments: int = 100,
    experiment_id: str | None = None,
) -> None:
    """Run the autonomous experiment loop."""
    print(f"\n{'=' * 70}")
    print(f"  AUTO-ML RESEARCH — segment={segment_id}, strategy={strategy}")
    print(f"{'=' * 70}\n")

    # ExperimentManager (opt-in via --experiment-id) -------------------------
    _exp_mgr: object | None = (
        _init_experiment_manager(experiment_id, strategy, segment_id)
        if experiment_id is not None
        else None
    )

    # Fetch MOEX macro data once (reused across segments if called multiple times)
    moex_data: MoexMarketData | None = None
    if _is_moex_segment(segment_id):
        print("\nFetching MOEX macro data (CBR, USDRUB, Brent, turnover)...")
        moex_data = _fetch_moex_macro_data()

    prepared = _prepare_data(segment_id, moex_data=moex_data)
    if prepared is None:
        return
    features, labels, hold_bars, folds = prepared

    # Step 4: Baseline experiment
    print("\nStep 4: Running baseline...")
    all_feature_names = sorted(features[0].keys())
    baseline_config = ExperimentConfig(
        name="baseline",
        description="Baseline with standard MI feature selection",
        strategy="baseline",
        max_features=_get_max_features(segment_id),
    )
    baseline_result = run_experiment(
        baseline_config,
        features,
        labels,
        hold_bars,
        folds,
        segment_id,
    )

    log_path = _RESULTS_DIR / f"{segment_id}_experiment_log.jsonl"
    _log_result(baseline_result, log_path)
    if _exp_mgr is not None and experiment_id is not None:
        _link_to_experiment_manager(_exp_mgr, experiment_id, baseline_result, segment_id)
    print(
        f"\n  Baseline: score={baseline_result.score:.4f} "
        f"acc={baseline_result.avg_accuracy:.3f} "
        f"brier={baseline_result.avg_brier:.3f} "
        f"pf={baseline_result.avg_profit_factor:.2f} "
        f"feats={baseline_result.feature_count}"
    )

    baseline_features = baseline_result.features_used
    baseline_score = baseline_result.score

    # Step 5: Generate experiments
    print(f"\nStep 5: Generating experiments (strategy={strategy})...")
    experiments = _generate_experiments(
        strategy,
        baseline_features,
        all_feature_names,
        max_experiments,
        segment_id=segment_id,
    )
    print(f"  {len(experiments)} experiments queued")

    # Step 6: Run experiments
    print("\nStep 6: Running experiments...")
    best_score = baseline_score
    best_name = "baseline"
    improvements = 0
    total_time = 0.0

    for idx, config in enumerate(experiments, 1):
        print(f"\n[{idx}/{len(experiments)}] {config.name}: {config.description}")
        result = run_experiment(config, features, labels, hold_bars, folds, segment_id)
        _log_result(result, log_path)
        if _exp_mgr is not None and experiment_id is not None:
            _link_to_experiment_manager(_exp_mgr, experiment_id, result, segment_id)
        _print_result(result, baseline_score)
        total_time += result.duration_seconds

        if result.score > best_score:
            best_score = result.score
            best_name = result.config.name
            improvements += 1

    _print_summary(
        log_path,
        len(experiments),
        improvements,
        best_name,
        best_score,
        baseline_score,
        total_time,
    )

    if _exp_mgr is not None and experiment_id is not None:
        try:
            _exp_mgr.record_verdict(experiment_id, best_score)  # type: ignore[union-attr]
            print(f"  ExperimentManager verdict recorded for {experiment_id}")
        except Exception:
            logger.warning("ExperimentManager record_verdict failed — continuing")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Autonomous ML experiment loop (autoresearch-inspired)"
    )
    parser.add_argument(
        "--segment",
        required=True,
        choices=list(_SEGMENT_SYMBOLS.keys()),
        help="Market segment to experiment on",
    )
    parser.add_argument(
        "--strategy",
        default="all",
        choices=[
            "ablation",
            "efficiency",
            "hyperparameter",
            "random_subset",
            "ensemble_weights",
            "cross_segment_transfer",
            "all",
        ],
        help="Experiment strategy (default: all)",
    )
    parser.add_argument(
        "--max-experiments",
        type=int,
        default=100,
        help="Maximum number of experiments to run (default: 100)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    import re as _re  # noqa: PLC0415

    def _valid_experiment_id(value: str) -> str:
        if not _re.match(r"^[a-zA-Z0-9_-]+$", value):
            parser.error(
                f"--experiment-id '{value}' is invalid: only [a-zA-Z0-9_-] characters are allowed"
            )
        return value

    parser.add_argument(
        "--experiment-id",
        type=_valid_experiment_id,
        default=None,
        metavar="ID",
        help=(
            "Optional experiment ID for ExperimentManager tracking. "
            "Only [a-zA-Z0-9_-] characters allowed. "
            "Creates a named experiment with hypothesis lifecycle and verdict."
        ),
    )
    args = parser.parse_args()
    random.seed(args.seed)
    _np.random.seed(args.seed)

    run_research_loop(
        segment_id=args.segment,
        strategy=args.strategy,
        max_experiments=args.max_experiments,
        experiment_id=args.experiment_id,
    )


if __name__ == "__main__":
    main()
