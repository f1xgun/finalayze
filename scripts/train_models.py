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
        "AAPL", "MSFT", "GOOGL", "NVDA", "META", "AMZN", "TSLA",
        "CRM", "ADBE", "INTC", "AMD", "AVGO", "CSCO", "ORCL", "QCOM",
    ],
    "us_healthcare": [
        "JNJ", "PFE", "UNH", "ABBV", "MRK", "LLY", "TMO", "ABT", "BMY", "AMGN", "GILD", "MDT",
    ],
    "us_finance": [
        "JPM", "BAC", "GS", "MS", "WFC", "C", "BLK", "SCHW", "AXP", "USB", "PNC", "TFC",
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
) -> tuple[list[dict[str, float]], list[int], np.ndarray | None, list[int] | None]:
    """Build (features, labels, barrier_weights, hold_bars) per symbol.

    Collects windows from all symbols and sorts by timestamp to maintain
    proper temporal ordering for train/test splits (no future leakage).

    Args:
        segment_id: Segment identifier (e.g. "us_tech", "ru_blue_chips").
        symbols: List of ticker symbols.
        settings: Application settings.
        label_mode: "triple_barrier" (default) or "direction".

    Returns:
        Tuple of (features, labels, barrier_weights, hold_bars).
        barrier_weights is non-None only in triple_barrier mode (abs(pnl_pct)).
        hold_bars is non-None only in triple_barrier mode.
    """
    if settings is None:
        settings = Settings()
    market_id = segment_id.split("_", maxsplit=1)[0]

    if label_mode == LABEL_MODE_TRIPLE_BARRIER:
        return _build_dataset_triple_barrier(segment_id, symbols, market_id, settings)
    return _build_dataset_direction(segment_id, symbols, market_id, settings)


def _build_dataset_direction(
    segment_id: str,
    symbols: list[str],
    market_id: str,
    settings: Settings,
) -> tuple[list[dict[str, float]], list[int], np.ndarray | None, list[int] | None]:
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
    return features_out, labels_out, None, None


def _build_dataset_triple_barrier(
    segment_id: str,
    symbols: list[str],
    market_id: str,
    settings: Settings,
) -> tuple[list[dict[str, float]], list[int], np.ndarray | None, list[int] | None]:
    """Build dataset with triple barrier labels."""
    import numpy as _np  # noqa: PLC0415

    tb_params = _get_triple_barrier_params(segment_id)
    min_candles_tb = _WINDOW_SIZE + int(tb_params["max_hold"]) + 1
    rows: list[tuple[datetime, dict[str, float], int, float, int]] = []

    for symbol in symbols:
        candles = _fetch_symbol_candles(symbol, market_id, settings, segment_id=segment_id)
        if len(candles) < min_candles_tb:
            print(
                f"  [{segment_id}] {symbol}: only {len(candles)} candles, "
                f"need {min_candles_tb}+ for triple barrier -- skipping."
            )
            continue
        x_sym, y_sym, w_sym, ts_sym, hb_sym = build_triple_barrier_dataset(
            candles,
            window_size=_WINDOW_SIZE,
            upper_atr_mult=float(tb_params["upper_atr_mult"]),
            lower_atr_mult=float(tb_params["lower_atr_mult"]),
            max_hold=int(tb_params["max_hold"]),
            atr_period=int(tb_params["atr_period"]),
            atr_scale=bool(tb_params["atr_scale"]),
        )
        print(
            f"  [{segment_id}] {symbol}: {len(x_sym)} triple barrier samples "
            f"(label balance: {sum(y_sym)}/{len(y_sym)} positive)"
        )
        for ts, feat, lbl, wt, hb in zip(ts_sym, x_sym, y_sym, w_sym, hb_sym, strict=True):
            rows.append((ts, feat, lbl, wt, hb))

    rows.sort(key=lambda r: r[0])
    features_out = [r[1] for r in rows]
    labels_out = [r[2] for r in rows]
    weights_out = _np.array([r[3] for r in rows], dtype=float) if rows else None
    hold_bars_out = [r[4] for r in rows] if rows else None
    return features_out, labels_out, weights_out, hold_bars_out


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


def train_one_segment(  # noqa: PLR0915
    segment_id: str,
    symbols: list[str],
    output_dir: Path,
    settings: Settings | None = None,
    label_mode: str = LABEL_MODE_TRIPLE_BARRIER,
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
        description="Train XGBoost + LightGBM + LSTM models per segment"
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
    return parser.parse_args(argv)


def main() -> None:
    """Entry point."""
    args = _parse_args()
    output_dir = Path(args.output_dir)
    label_mode: str = args.label_mode

    if args.segment:
        segments = {args.segment: _SEGMENT_SYMBOLS.get(args.segment, [])}
    else:
        segments = _SEGMENT_SYMBOLS

    print(f"Label mode: {label_mode}")
    for segment_id, symbols in segments.items():
        try:
            train_one_segment(
                segment_id=segment_id,
                symbols=symbols,
                output_dir=output_dir,
                label_mode=label_mode,
            )
        except FileNotFoundError as exc:
            print(f"[{segment_id}] FileNotFoundError -- {exc}, skipping.")
        except Exception as exc:
            print(f"[{segment_id}] Unexpected error -- {exc}, skipping.")


if __name__ == "__main__":
    main()
