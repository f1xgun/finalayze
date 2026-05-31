"""Shared training utilities for ML models (Layer 3).

Extracted from ``scripts/train_models.py`` so that both the CLI script and
the automated retrain cycle in ``TradingLoop`` use the same window-building
logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime  # noqa: TC003  — used at runtime in _slice_market_context
from typing import TYPE_CHECKING

from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

from finalayze.core.schemas import MarketContext, MoexMarketData
from finalayze.ml.features.technical import compute_features

if TYPE_CHECKING:
    from finalayze.core.schemas import Candle
    from finalayze.ml.models.ensemble import EnsembleModel

DEFAULT_WINDOW_SIZE = 80

# Validation gate thresholds (6C.7)
_MIN_ACCURACY = 0.54  # 2 SE above coin-flip for n=500
_MAX_BRIER_SCORE = 0.235  # meaningfully below coin-flip (0.250)
_MAX_LOG_LOSS = 0.680  # below ln(2) ≈ 0.693


def _slice_market_context(ctx: MarketContext, max_ts: datetime) -> MarketContext:
    """Return a copy of ctx with all time-series data filtered to timestamp <= max_ts.

    This prevents look-ahead bias when building training windows: each window at
    time T must only see ambient market data (FX rates, key rates, commodity candles,
    turnover, benchmark/VIX candles) that was available at or before T.

    Args:
        ctx: The full MarketContext for the entire training period.
        max_ts: The maximum timestamp (inclusive) for this training window.

    Returns:
        A new MarketContext containing only data with timestamp <= max_ts.
    """
    sliced_benchmark = (
        [c for c in ctx.benchmark_candles if c.timestamp <= max_ts]
        if ctx.benchmark_candles is not None
        else None
    )
    sliced_vix = (
        [c for c in ctx.vix_candles if c.timestamp <= max_ts]
        if ctx.vix_candles is not None
        else None
    )

    sliced_moex: MoexMarketData | None = None
    if ctx.moex_data is not None:
        md = ctx.moex_data
        sliced_fx = (
            tuple(r for r in md.fx_rates if r.timestamp <= max_ts)
            if md.fx_rates is not None
            else None
        )
        sliced_kr = (
            tuple(r for r in md.key_rates if r.timestamp <= max_ts)
            if md.key_rates is not None
            else None
        )
        sliced_commodities: dict[str, tuple[Candle, ...]] | None = None
        if md.commodity_candles is not None:
            sliced_commodities = {
                sym: tuple(c for c in candles if c.timestamp <= max_ts)
                for sym, candles in md.commodity_candles.items()
            }
        sliced_turnover = (
            tuple(r for r in md.turnover if r.timestamp <= max_ts)
            if md.turnover is not None
            else None
        )
        sliced_fund = (
            tuple(s for s in md.fundamentals if s.as_of <= max_ts)
            if md.fundamentals is not None
            else None
        )
        sliced_moex = MoexMarketData(
            fx_rates=sliced_fx,
            key_rates=sliced_kr,
            commodity_candles=sliced_commodities,
            turnover=sliced_turnover,
            fundamentals=sliced_fund,
        )

    return MarketContext(
        benchmark_candles=sliced_benchmark,
        vix_candles=sliced_vix,
        moex_data=sliced_moex,
    )


@dataclass
class ValidationResult:
    """Result of ensemble validation with multiple metrics."""

    accuracy: float
    brier_score: float
    log_loss_val: float
    n_samples: int
    passed: bool


def validate_ensemble(
    ensemble: EnsembleModel,
    val_features: list[dict[str, float]],
    val_labels: list[int],
) -> ValidationResult:
    """Evaluate an ensemble on validation data and return metrics + pass/fail."""
    probas = [ensemble.predict_proba(f) for f in val_features]
    preds = [round(p) for p in probas]
    acc = float(accuracy_score(val_labels, preds))
    brier = float(brier_score_loss(val_labels, probas))
    ll = float(log_loss(val_labels, probas, labels=[0, 1]))
    passed = acc >= _MIN_ACCURACY and brier <= _MAX_BRIER_SCORE and ll <= _MAX_LOG_LOSS
    return ValidationResult(
        accuracy=acc,
        brier_score=brier,
        log_loss_val=ll,
        n_samples=len(val_labels),
        passed=passed,
    )


def build_windows(
    candles: list[Candle],
    window_size: int = DEFAULT_WINDOW_SIZE,
    *,
    skip_split_windows: bool = True,
    market_context: MarketContext | None = None,
) -> tuple[list[dict[str, float]], list[int], list[datetime]]:
    """Build (features, labels, timestamps) from a single contiguous candle series.

    For each position *i* the feature window is ``candles[i:i+window_size]``
    and the label is ``sign(candles[i+window_size].close - candles[i+window_size-1].close)``.
    The label bar is **strictly outside** the feature window (no look-ahead).

    When ``skip_split_windows`` is True, windows spanning a detected stock
    split are excluded to avoid poisoning indicators (6C.8).

    When ``market_context`` is provided, each window receives a time-sliced copy of
    the context via ``_slice_market_context`` so that no future ambient data leaks
    into the feature computation (no look-ahead bias for MOEX/cross-asset features).

    Returns:
        Tuple of (feature_dicts, binary_labels, timestamps).  Empty lists when
        there are fewer than ``window_size + 1`` candles.  The timestamp for
        each sample is the timestamp of the label bar (candles[i+window_size]).
    """
    from finalayze.ml.features.corporate_actions import detect_splits  # noqa: PLC0415

    features_list: list[dict[str, float]] = []
    label_list: list[int] = []
    ts_list: list[datetime] = []
    sorted_candles = sorted(candles, key=lambda c: c.timestamp)

    # Detect split indices for filtering (6C.8)
    split_indices: set[int] = set()
    if skip_split_windows:
        split_indices = set(detect_splits(sorted_candles))

    for i in range(len(sorted_candles) - window_size):
        # Skip windows that contain a split index (6C.8)
        if skip_split_windows and split_indices:
            window_range = range(i, i + window_size + 1)
            if any(si in window_range for si in split_indices):
                continue

        window = sorted_candles[i : i + window_size]

        # Slice ambient market data to this window's max timestamp (no look-ahead)
        window_ctx: MarketContext | None = None
        if market_context is not None:
            window_max_ts = sorted_candles[i + window_size - 1].timestamp
            window_ctx = _slice_market_context(market_context, window_max_ts)

        try:
            row_features = compute_features(window, market_context=window_ctx)
        except Exception:  # noqa: S112
            continue
        next_close = float(sorted_candles[i + window_size].close)
        cur_close = float(sorted_candles[i + window_size - 1].close)
        label = 1 if next_close > cur_close else 0
        features_list.append(row_features)
        label_list.append(label)
        ts_list.append(sorted_candles[i + window_size].timestamp)

    return features_list, label_list, ts_list


def build_dataset(
    candles_by_symbol: dict[str, list[Candle]],
    window_size: int = DEFAULT_WINDOW_SIZE,
    *,
    market_context: MarketContext | None = None,
) -> tuple[list[dict[str, float]], list[int], list[datetime]]:
    """Build (features, labels, timestamps) aggregated across multiple symbols.

    Collects windows from all symbols and sorts by timestamp to maintain
    proper temporal ordering for train/test splits.

    When ``market_context`` is provided it is threaded through to each call of
    ``build_windows``, which in turn time-slices it per window to prevent
    look-ahead bias in MOEX/cross-asset features.

    Args:
        candles_by_symbol: Mapping of symbol → sorted candle list.
        window_size: Number of bars per feature window.
        market_context: Optional ambient market data for the full training period.

    Returns:
        Tuple of (feature_dicts, binary_labels, timestamps) sorted by time.
    """
    rows: list[tuple[datetime, dict[str, float], int]] = []
    min_candles = window_size + 1

    for candles in candles_by_symbol.values():
        if len(candles) < min_candles:
            continue
        x_sym, y_sym, ts_sym = build_windows(candles, window_size, market_context=market_context)
        for ts, feat, lbl in zip(ts_sym, x_sym, y_sym, strict=True):
            rows.append((ts, feat, lbl))

    rows.sort(key=lambda r: r[0])

    features_out = [r[1] for r in rows]
    labels_out = [r[2] for r in rows]
    ts_out = [r[0] for r in rows]
    return features_out, labels_out, ts_out
