"""Shared training utilities for ML models (Layer 3).

Extracted from ``scripts/train_models.py`` so that both the CLI script and
the automated retrain cycle in ``TradingLoop`` use the same window-building
logic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from finalayze.ml.features.technical import compute_features

if TYPE_CHECKING:
    from finalayze.core.schemas import Candle

DEFAULT_WINDOW_SIZE = 60


def build_windows(
    candles: list[Candle],
    window_size: int = DEFAULT_WINDOW_SIZE,
) -> tuple[list[dict[str, float]], list[int]]:
    """Build (features, labels) from a single contiguous candle series.

    For each position *i* the feature window is ``candles[i:i+window_size]``
    and the label is ``sign(candles[i+window_size].close - candles[i+window_size-1].close)``.
    The label bar is **strictly outside** the feature window (no look-ahead).

    Returns:
        Tuple of (feature_dicts, binary_labels).  Empty lists when there are
        fewer than ``window_size + 1`` candles.
    """
    features_list: list[dict[str, float]] = []
    label_list: list[int] = []
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

    return features_list, label_list


def build_dataset(
    candles_by_symbol: dict[str, list[Candle]],
    window_size: int = DEFAULT_WINDOW_SIZE,
) -> tuple[list[dict[str, float]], list[int]]:
    """Build (features, labels) aggregated across multiple symbols.

    Args:
        candles_by_symbol: Mapping of symbol → sorted candle list.
        window_size: Number of bars per feature window.

    Returns:
        Tuple of (feature_dicts, binary_labels) aggregated over all symbols.
    """
    features_out: list[dict[str, float]] = []
    labels_out: list[int] = []
    min_candles = window_size + 1

    for candles in candles_by_symbol.values():
        if len(candles) < min_candles:
            continue
        x_sym, y_sym = build_windows(candles, window_size)
        features_out.extend(x_sym)
        labels_out.extend(y_sym)

    return features_out, labels_out
