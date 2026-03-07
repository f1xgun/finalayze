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
    from datetime import datetime

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
    lower_atr_mult: float = 2.0,
    upper_atr_mult: float = 2.0,
    benchmark_candles: list[Candle] | None = None,
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
        lower_atr_mult: ATR multiplier for the lower (stop-loss) barrier.
        upper_atr_mult: ATR multiplier for the upper (profit-target) barrier.
        benchmark_candles: Optional benchmark candles aligned by index. When provided,
            barriers are checked against excess return (stock - benchmark) instead of
            raw return. This produces market-neutral labels.

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
            effective_upper = upper_atr_mult * atr_pct
            effective_lower = lower_atr_mult * atr_pct
        # If ATR not computable, fall back to fixed pct

    bench_entry_price = _get_bench_entry_price(benchmark_candles, entry_index)

    return _scan_barriers(
        candles,
        entry_index,
        entry_price,
        effective_upper,
        effective_lower,
        max_hold,
        atr_pct,
        benchmark_candles,
        bench_entry_price,
    )


def _get_bench_entry_price(
    benchmark_candles: list[Candle] | None,
    entry_index: int,
) -> float | None:
    """Get benchmark entry price, returning None if unavailable."""
    if benchmark_candles is None or entry_index >= len(benchmark_candles):
        return None
    val = float(benchmark_candles[entry_index].close)
    return val if val > 0 else None


def _scan_barriers(
    candles: list[Candle],
    entry_index: int,
    entry_price: float,
    effective_upper: float,
    effective_lower: float,
    max_hold: int,
    atr_pct: float | None,
    benchmark_candles: list[Candle] | None,
    bench_entry_price: float | None,
) -> TripleBarrierResult | None:
    """Scan forward bars checking barriers. Returns result or None."""
    use_excess = bench_entry_price is not None
    end_index = min(entry_index + max_hold, len(candles) - 1)

    # Check upper/lower barriers on each forward bar
    for bar_offset in range(1, end_index - entry_index + 1):
        bar_idx = entry_index + bar_offset
        hit = _check_bar_barriers(
            candles,
            bar_idx,
            entry_price,
            effective_upper,
            effective_lower,
            bar_offset,
            use_excess,
            benchmark_candles,
            bench_entry_price,
        )
        if hit is not None:
            return hit

    # Vertical barrier: timeout
    if end_index <= entry_index:
        return None

    pnl_pct = _compute_exit_pnl(
        candles,
        end_index,
        entry_price,
        use_excess,
        benchmark_candles,
        bench_entry_price,
    )
    hold_bars = end_index - entry_index

    # Filter noise: discard vertical hits with tiny PnL
    noise_threshold = 0.5 * atr_pct if (atr_pct is not None and atr_pct > 0) else 0.005
    if abs(pnl_pct) < noise_threshold:
        return None

    return TripleBarrierResult(
        label=1 if pnl_pct > 0 else 0,
        pnl_pct=pnl_pct,
        barrier_type="vertical",
        hold_bars=hold_bars,
    )


def _check_bar_barriers(
    candles: list[Candle],
    bar_idx: int,
    entry_price: float,
    effective_upper: float,
    effective_lower: float,
    bar_offset: int,
    use_excess: bool,
    benchmark_candles: list[Candle] | None,
    bench_entry_price: float | None,
) -> TripleBarrierResult | None:
    """Check if upper or lower barrier is hit at a single bar."""
    if use_excess:
        assert bench_entry_price is not None
        stock_ret = (float(candles[bar_idx].close) - entry_price) / entry_price
        bench_ret = (
            (float(benchmark_candles[bar_idx].close) - bench_entry_price)  # type: ignore[index]
            / bench_entry_price
        )
        excess = stock_ret - bench_ret
        if excess >= effective_upper:
            return TripleBarrierResult(1, excess, "upper", bar_offset)
        if excess <= -effective_lower:
            return TripleBarrierResult(0, excess, "lower", bar_offset)
    else:
        upper_barrier = entry_price * (1.0 + effective_upper)
        lower_barrier = entry_price * (1.0 - effective_lower)
        if float(candles[bar_idx].high) >= upper_barrier:
            return TripleBarrierResult(1, effective_upper, "upper", bar_offset)
        if float(candles[bar_idx].low) <= lower_barrier:
            return TripleBarrierResult(0, -effective_lower, "lower", bar_offset)
    return None


def _compute_exit_pnl(
    candles: list[Candle],
    exit_idx: int,
    entry_price: float,
    use_excess: bool,
    benchmark_candles: list[Candle] | None,
    bench_entry_price: float | None,
) -> float:
    """Compute PnL at vertical barrier exit."""
    if use_excess:
        assert bench_entry_price is not None
        stock_ret = (float(candles[exit_idx].close) - entry_price) / entry_price
        bench_ret = (
            (float(benchmark_candles[exit_idx].close) - bench_entry_price)  # type: ignore[index]
            / bench_entry_price
        )
        return stock_ret - bench_ret
    return (float(candles[exit_idx].close) - entry_price) / entry_price


def build_triple_barrier_dataset(
    candles: list[Candle],
    window_size: int = 60,
    upper_pct: float = 0.03,
    lower_pct: float = 0.03,
    max_hold: int = 20,
    atr_scale: bool = True,
    atr_period: int = 14,
    lower_atr_mult: float = 2.0,
    upper_atr_mult: float = 2.0,
    benchmark_candles: list[Candle] | None = None,
) -> tuple[list[dict[str, float]], list[int], list[float], list[datetime], list[int]]:
    """Build a dataset using triple barrier labels.

    Args:
        candles: Stock candles (will be sorted by timestamp).
        window_size: Number of bars for feature computation.
        upper_pct: Profit target as fraction.
        lower_pct: Stop loss as fraction.
        max_hold: Maximum bars to hold.
        atr_scale: Scale barriers using ATR.
        atr_period: Period for ATR computation.
        lower_atr_mult: ATR multiplier for lower barrier.
        upper_atr_mult: ATR multiplier for upper barrier.
        benchmark_candles: Optional benchmark candles for market-neutral labels.

    Returns:
        Tuple of (features, labels, sample_weights, timestamps, hold_bars).
        sample_weights are abs(pnl_pct) for weighting in training.
        timestamps are the entry bar timestamps for temporal ordering.
        hold_bars are the number of bars held for each sample.
    """
    from finalayze.ml.features.corporate_actions import detect_splits  # noqa: PLC0415

    sorted_candles = sorted(candles, key=lambda c: c.timestamp)

    # Detect split indices for filtering (A4)
    split_indices = set(detect_splits(sorted_candles))

    features_list: list[dict[str, float]] = []
    label_list: list[int] = []
    weight_list: list[float] = []
    ts_list: list[datetime] = []
    hold_bars_list: list[int] = []

    # Need window_size bars for features + max_hold bars for label
    for i in range(len(sorted_candles) - window_size - max_hold):
        # Skip if a split occurs in the label period (A4)
        entry_index = i + window_size - 1
        label_range = range(entry_index, entry_index + max_hold + 1)
        if any(si in label_range for si in split_indices):
            continue

        window = sorted_candles[i : i + window_size]

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
            lower_atr_mult=lower_atr_mult,
            upper_atr_mult=upper_atr_mult,
            benchmark_candles=benchmark_candles,
        )

        if result is None:
            continue

        features_list.append(row_features)
        label_list.append(result.label)
        weight_list.append(abs(result.pnl_pct))
        ts_list.append(sorted_candles[entry_index].timestamp)
        hold_bars_list.append(result.hold_bars)

    return features_list, label_list, weight_list, ts_list, hold_bars_list
