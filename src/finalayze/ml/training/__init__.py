"""Shared training utilities for ML models (Layer 3).

Extracted from ``scripts/train_models.py`` so that both the CLI script and
the automated retrain cycle in ``TradingLoop`` use the same window-building
logic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from finalayze.ml.features.technical import compute_features

if TYPE_CHECKING:
    from datetime import datetime

    from finalayze.core.schemas import Candle

DEFAULT_WINDOW_SIZE = 60


def build_windows(
    candles: list[Candle],
    window_size: int = DEFAULT_WINDOW_SIZE,
) -> tuple[list[dict[str, float]], list[int], list[datetime]]:
    """Build (features, labels, timestamps) from a single contiguous candle series.

    For each position *i* the feature window is ``candles[i:i+window_size]``
    and the label is ``sign(candles[i+window_size].close - candles[i+window_size-1].close)``.
    The label bar is **strictly outside** the feature window (no look-ahead).

    Returns:
        Tuple of (feature_dicts, binary_labels, timestamps).  Empty lists when
        there are fewer than ``window_size + 1`` candles.  The timestamp for
        each sample is the timestamp of the label bar (candles[i+window_size]).
    """
    features_list: list[dict[str, float]] = []
    label_list: list[int] = []
    ts_list: list[datetime] = []
    sorted_candles = sorted(candles, key=lambda c: c.timestamp)

    for i in range(len(sorted_candles) - window_size):
        window = sorted_candles[i : i + window_size]
        try:
            row_features = compute_features(window)
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
) -> tuple[list[dict[str, float]], list[int], list[datetime]]:
    """Build (features, labels, timestamps) aggregated across multiple symbols.

    Collects windows from all symbols and sorts by timestamp to maintain
    proper temporal ordering for train/test splits.

    Args:
        candles_by_symbol: Mapping of symbol → sorted candle list.
        window_size: Number of bars per feature window.

    Returns:
        Tuple of (feature_dicts, binary_labels, timestamps) sorted by time.
    """
    rows: list[tuple[datetime, dict[str, float], int]] = []
    min_candles = window_size + 1

    for candles in candles_by_symbol.values():
        if len(candles) < min_candles:
            continue
        x_sym, y_sym, ts_sym = build_windows(candles, window_size)
        for ts, feat, lbl in zip(ts_sym, x_sym, y_sym, strict=True):
            rows.append((ts, feat, lbl))

    rows.sort(key=lambda r: r[0])

    features_out = [r[1] for r in rows]
    labels_out = [r[2] for r in rows]
    ts_out = [r[0] for r in rows]
    return features_out, labels_out, ts_out
