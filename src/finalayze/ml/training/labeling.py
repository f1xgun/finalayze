"""Triple barrier labeling for ML training (Layer 3).

Replaces simple binary (up/down) labels with three-barrier labels:
- Upper barrier: profit target hit -> label = 1
- Lower barrier: stop loss hit -> label = 0
- Vertical barrier: timeout -> label based on final return sign

See docs/plans/2026-03-02-enhanced-improvement-plan.md, task B.3.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pandas_ta as ta

from finalayze.ml.features.technical import compute_features

if TYPE_CHECKING:
    from finalayze.core.schemas import Candle


@dataclass(frozen=True)
class TripleBarrierResult:
    """Result of triple barrier labeling for a single entry point."""

    label: int  # 1 = profit, 0 = loss
    pnl_pct: float  # actual PnL percentage
    barrier_type: str  # "upper", "lower", "vertical"
    hold_bars: int  # bars held


def _compute_atr(candles: list[Candle], period: int = 14) -> float | None:
    """Compute ATR from candles. Returns None if insufficient data."""
    if len(candles) < period + 1:
        return None
    highs = pd.Series([float(c.high) for c in candles], dtype=float)
    lows = pd.Series([float(c.low) for c in candles], dtype=float)
    closes = pd.Series([float(c.close) for c in candles], dtype=float)
    atr = ta.atr(highs, lows, closes, length=period)
    if atr is None or atr.empty:
        return None
    val = float(atr.iloc[-1])
    if np.isnan(val):
        return None
    return val


def triple_barrier_label(
    candles: list[Candle],
    entry_index: int,
    upper_pct: float = 0.03,
    lower_pct: float = 0.03,
    max_hold: int = 20,
    atr_scale: bool = True,
    atr_period: int = 14,
) -> TripleBarrierResult | None:
    """Apply triple barrier labeling at a given entry index.

    Args:
        candles: Full sorted candle list.
        entry_index: Index of the entry bar (must have enough history for ATR).
        upper_pct: Profit target as fraction (e.g. 0.03 = 3%).
        lower_pct: Stop loss as fraction (e.g. 0.03 = 3%).
        max_hold: Maximum bars to hold before vertical barrier.
        atr_scale: If True, scale barriers using ATR instead of fixed pct.
        atr_period: Period for ATR computation.

    Returns:
        TripleBarrierResult or None if the label should be discarded (noise).
    """
    if entry_index < 0 or entry_index >= len(candles) or float(candles[entry_index].close) <= 0:
        return None

    entry_price = float(candles[entry_index].close)

    # Compute ATR-scaled barriers if requested
    effective_upper = upper_pct
    effective_lower = lower_pct
    atr_pct: float | None = None

    if atr_scale:
        # Use candles up to and including entry_index for ATR (no look-ahead)
        history = candles[max(0, entry_index - atr_period - 1) : entry_index + 1]
        atr_val = _compute_atr(history, atr_period)
        if atr_val is not None and atr_val > 0:
            atr_pct = atr_val / entry_price
            effective_upper = 2.0 * atr_pct
            effective_lower = 2.0 * atr_pct
        # If ATR not computable, fall back to fixed pct

    upper_barrier = entry_price * (1.0 + effective_upper)
    lower_barrier = entry_price * (1.0 - effective_lower)

    # Scan forward up to max_hold bars
    end_index = min(entry_index + max_hold, len(candles) - 1)

    for bar_offset in range(1, end_index - entry_index + 1):
        bar_idx = entry_index + bar_offset
        bar_high = float(candles[bar_idx].high)
        bar_low = float(candles[bar_idx].low)

        # Check upper barrier (profit target)
        if bar_high >= upper_barrier:
            pnl_pct = (upper_barrier - entry_price) / entry_price
            return TripleBarrierResult(
                label=1,
                pnl_pct=pnl_pct,
                barrier_type="upper",
                hold_bars=bar_offset,
            )

        # Check lower barrier (stop loss)
        if bar_low <= lower_barrier:
            pnl_pct = (lower_barrier - entry_price) / entry_price
            return TripleBarrierResult(
                label=0,
                pnl_pct=pnl_pct,
                barrier_type="lower",
                hold_bars=bar_offset,
            )

    # Vertical barrier: timeout
    exit_idx = end_index
    if exit_idx <= entry_index:
        return None

    exit_price = float(candles[exit_idx].close)
    pnl_pct = (exit_price - entry_price) / entry_price
    hold_bars = exit_idx - entry_index

    # Filter noise: discard vertical hits with tiny PnL
    noise_threshold = 0.005  # default 0.5%
    if atr_pct is not None and atr_pct > 0:
        noise_threshold = 0.5 * atr_pct
    if abs(pnl_pct) < noise_threshold:
        return None

    label = 1 if pnl_pct > 0 else 0
    return TripleBarrierResult(
        label=label,
        pnl_pct=pnl_pct,
        barrier_type="vertical",
        hold_bars=hold_bars,
    )


def build_triple_barrier_dataset(
    candles: list[Candle],
    window_size: int = 60,
    upper_pct: float = 0.03,
    lower_pct: float = 0.03,
    max_hold: int = 20,
    atr_scale: bool = True,
    atr_period: int = 14,
) -> tuple[list[dict[str, float]], list[int], list[float]]:
    """Build a dataset using triple barrier labels.

    Returns:
        Tuple of (features, labels, sample_weights).
        sample_weights are abs(pnl_pct) for weighting in training.
    """
    sorted_candles = sorted(candles, key=lambda c: c.timestamp)

    features_list: list[dict[str, float]] = []
    label_list: list[int] = []
    weight_list: list[float] = []

    # Need window_size bars for features + max_hold bars for label
    for i in range(len(sorted_candles) - window_size - max_hold):
        window = sorted_candles[i : i + window_size]
        entry_index = i + window_size - 1  # last bar of the feature window

        try:
            row_features = compute_features(window)
        except Exception:  # noqa: S112
            continue

        result = triple_barrier_label(
            sorted_candles,
            entry_index,
            upper_pct=upper_pct,
            lower_pct=lower_pct,
            max_hold=max_hold,
            atr_scale=atr_scale,
            atr_period=atr_period,
        )

        if result is None:
            continue

        features_list.append(row_features)
        label_list.append(result.label)
        weight_list.append(abs(result.pnl_pct))

    return features_list, label_list, weight_list
