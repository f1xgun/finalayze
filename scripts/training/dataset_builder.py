"""Dataset building functions for the training pipeline.

Constructs feature/label datasets from candles using different labeling
modes: direction, triple barrier, and trend scanning.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from scripts.training.data_loader import (
    align_benchmark_candles,
    fetch_benchmark_candles,
    fetch_symbol_candles,
    fetch_vix_candles,
    is_moex_segment,
)

from finalayze.core.schemas import Candle, MarketContext
from finalayze.ml.features.technical import compute_features
from finalayze.ml.training import DEFAULT_WINDOW_SIZE, _slice_market_context, build_windows
from finalayze.ml.training.labeling import build_triple_barrier_dataset
from finalayze.ml.training.trend_scanning import trend_scan_labels

if TYPE_CHECKING:
    from datetime import datetime

    import numpy as np
    from config.settings import Settings

_WINDOW_SIZE = DEFAULT_WINDOW_SIZE
_MIN_HISTORY_DAYS = 500  # symbols with fewer trading days produce degenerate ML predictions
_MIN_CANDLES = _WINDOW_SIZE + 1  # need at least WINDOW_SIZE + 1 for one sample

# Triple barrier parameters (match engine execution params)
TB_UPPER_ATR_MULT = 2.0  # match ml_ensemble ATR stop
TB_LOWER_ATR_MULT = 2.0  # symmetric barriers
TB_MAX_HOLD = 20  # match DEFAULT_STRATEGY_HOLD_BARS["ml_ensemble"]
TB_ATR_PERIOD = 14  # standard
MOEX_ATR_UPLIFT = 1.2  # MOEX 1.2x uplift for wider barriers
# Pre-uplift barrier multipliers per segment. MOEX uplift applied in get_barrier_params().
# Must stay in sync with auto_ml_research.py _SEGMENT_BARRIER_CONFIG.
SEGMENT_BARRIER_CONFIG: dict[str, tuple[float, float]] = {
    "ru_energy": (1.5, 2.0),  # (upper, lower) -- wider downside for commodity-linked volatility
}

# Label mode choices
LABEL_MODE_TRIPLE_BARRIER = "triple_barrier"
LABEL_MODE_DIRECTION = "direction"
LABEL_MODE_TREND_SCANNING = "trend_scanning"

# Purge gap between train / calibration / test splits.
# S2.2: name carries the unit ("BARS") to prevent the latent footgun that
# walk_forward.py was applying this value as `timedelta(days=...)`. Sample-
# index callers (model_trainer.py) keep using it; day-based callers
# (walk_forward.py) now have their own day-typed constants.
PURGE_GAP_BARS = _WINDOW_SIZE + TB_MAX_HOLD  # 80 bars: feature window + label horizon
# Back-compat alias (kept until external scripts are migrated).
PURGE_GAP = PURGE_GAP_BARS


def get_barrier_params(segment_id: str) -> tuple[float, float]:
    """Return (upper_atr_mult, lower_atr_mult) with MOEX uplift applied."""
    base_upper, base_lower = SEGMENT_BARRIER_CONFIG.get(
        segment_id, (TB_UPPER_ATR_MULT, TB_LOWER_ATR_MULT)
    )
    if is_moex_segment(segment_id):
        return base_upper * MOEX_ATR_UPLIFT, base_lower * MOEX_ATR_UPLIFT
    return base_upper, base_lower


def get_triple_barrier_params(segment_id: str) -> dict[str, float | int | bool]:
    """Return triple barrier parameters for a segment.

    MOEX segments get 1.2x ATR uplift. ru_energy gets asymmetric barriers.
    """
    upper, lower = get_barrier_params(segment_id)
    return {
        "upper_atr_mult": upper,
        "lower_atr_mult": lower,
        "max_hold": TB_MAX_HOLD,
        "atr_period": TB_ATR_PERIOD,
        "atr_scale": True,
    }


def compute_uniqueness_from_hold_bars(hold_bars: list[int]) -> np.ndarray:  # type: ignore[type-arg]
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


def build_dataset(
    segment_id: str,
    symbols: list[str],
    settings: Settings | None = None,
    label_mode: str = LABEL_MODE_TRIPLE_BARRIER,
    *,
    excess_returns: bool = False,
    market_context: MarketContext | None = None,
) -> tuple[list[dict[str, float]], list[int], np.ndarray | None, list[int] | None]:
    """Build (features, labels, barrier_weights, hold_bars) per symbol.

    Collects windows from all symbols and sorts by timestamp to maintain
    proper temporal ordering for train/test splits (no future leakage).

    Returns:
        Tuple of (features, labels, barrier_weights, hold_bars).
        barrier_weights is non-None only in triple_barrier mode (abs(pnl_pct)).
        hold_bars is non-None only in triple_barrier mode.
    """
    features, labels, weights, hold_bars, _timestamps = build_dataset_with_timestamps(
        segment_id,
        symbols,
        settings,
        label_mode,
        excess_returns=excess_returns,
        market_context=market_context,
    )
    return features, labels, weights, hold_bars


def build_dataset_with_timestamps(
    segment_id: str,
    symbols: list[str],
    settings: Settings | None = None,
    label_mode: str = LABEL_MODE_TRIPLE_BARRIER,
    *,
    excess_returns: bool = False,
    market_context: MarketContext | None = None,
) -> tuple[list[dict[str, float]], list[int], np.ndarray | None, list[int] | None, list[datetime]]:
    """Build dataset with timestamps for calendar-date splitting (D4).

    Returns:
        Tuple of (features, labels, barrier_weights, hold_bars, timestamps).
    """
    from config.settings import Settings as _Settings  # noqa: PLC0415

    if settings is None:
        settings = _Settings()
    market_id = segment_id.split("_", maxsplit=1)[0]

    if label_mode == LABEL_MODE_TRIPLE_BARRIER:
        return _build_dataset_triple_barrier(
            segment_id,
            symbols,
            market_id,
            settings,
            excess_returns=excess_returns,
            market_context=market_context,
        )
    if label_mode == LABEL_MODE_TREND_SCANNING:
        return _build_dataset_trend_scanning(
            segment_id, symbols, market_id, settings, market_context=market_context
        )
    return _build_dataset_direction(
        segment_id, symbols, market_id, settings, market_context=market_context
    )


def _build_dataset_direction(
    segment_id: str,
    symbols: list[str],
    market_id: str,
    settings: Settings,
    market_context: MarketContext | None = None,
) -> tuple[list[dict[str, float]], list[int], np.ndarray | None, list[int] | None, list[datetime]]:
    """Build dataset with simple next-bar direction labels (old behavior)."""
    rows: list[tuple[datetime, dict[str, float], int]] = []
    for symbol in symbols:
        candles = fetch_symbol_candles(symbol, market_id, settings, segment_id=segment_id)
        if len(candles) < _MIN_CANDLES:
            continue
        x_sym, y_sym, ts_sym = build_windows(candles, _WINDOW_SIZE, market_context=market_context)
        for ts, feat, lbl in zip(ts_sym, x_sym, y_sym, strict=True):
            rows.append((ts, feat, lbl))
    rows.sort(key=lambda r: r[0])
    features_out = [r[1] for r in rows]
    labels_out = [r[2] for r in rows]
    timestamps_out = [r[0] for r in rows]
    return features_out, labels_out, None, None, timestamps_out


def _build_dataset_trend_scanning(  # noqa: PLR0915
    segment_id: str,
    symbols: list[str],
    market_id: str,
    settings: Settings,
    market_context: MarketContext | None = None,
) -> tuple[list[dict[str, float]], list[int], np.ndarray | None, list[int] | None, list[datetime]]:
    """Build dataset with trend-scanning labels (Prado 2020).

    For each symbol, computes features via the standard windowed approach, then
    uses trend_scan_labels on close prices to assign labels and t-value weights.
    The selected horizon L* for each bar is used as hold_bars (for n_eff).
    """
    import numpy as _np  # noqa: PLC0415

    from finalayze.core.exceptions import InsufficientDataError  # noqa: PLC0415
    from finalayze.ml.features.corporate_actions import detect_splits  # noqa: PLC0415

    ts_max_horizon = TB_MAX_HOLD  # reuse triple barrier max hold as scan horizon
    ts_min_horizon = 3
    min_candles_ts = _WINDOW_SIZE + ts_max_horizon + 1

    rows: list[tuple[datetime, dict[str, float], int, float, int]] = []

    for symbol in symbols:
        candles = fetch_symbol_candles(symbol, market_id, settings, segment_id=segment_id)
        if len(candles) < min_candles_ts:
            print(
                f"  [{segment_id}] {symbol}: only {len(candles)} candles, "
                f"need {min_candles_ts}+ for trend scanning -- skipping."
            )
            continue

        sorted_candles = sorted(candles, key=lambda c: c.timestamp)
        split_indices = set(detect_splits(sorted_candles))

        # Extract close prices for trend scanning
        close_prices = _np.array([float(c.close) for c in sorted_candles], dtype=_np.float64)
        ts_labels, ts_t_values = trend_scan_labels(
            close_prices, max_horizon=ts_max_horizon, min_horizon=ts_min_horizon
        )

        # Build features for each bar and pair with trend-scanning labels
        for i in range(len(sorted_candles) - _WINDOW_SIZE - ts_max_horizon):
            entry_index = i + _WINDOW_SIZE - 1

            # Skip if a split occurs in the label horizon
            label_range = range(entry_index, entry_index + ts_max_horizon + 1)
            if any(si in label_range for si in split_indices):
                continue

            # Skip bars where trend scanning produced NaN
            if _np.isnan(ts_labels[entry_index]) or _np.isnan(ts_t_values[entry_index]):
                continue

            # Compute features using history up to entry bar (no look-ahead)
            window = sorted_candles[: entry_index + 1]
            entry_ctx: MarketContext | None = None
            if market_context is not None:
                entry_ctx = _slice_market_context(
                    market_context, sorted_candles[entry_index].timestamp
                )
            try:
                row_features = compute_features(window, market_context=entry_ctx)
            except (InsufficientDataError, ValueError):
                continue
            except Exception:
                continue

            label = int(ts_labels[entry_index])
            t_value_weight = float(ts_t_values[entry_index])
            # Use a default hold estimate (the max_horizon / 2) since trend scanning
            # selects variable horizons; the exact L* is internal to trend_scan_labels
            hold_bars_est = ts_max_horizon // 2

            rows.append(
                (
                    sorted_candles[entry_index].timestamp,
                    row_features,
                    label,
                    t_value_weight,
                    hold_bars_est,
                )
            )

        pos_rate = "N/A"
        sym_rows = [r for r in rows if True]  # all rows so far (accumulating)
        if sym_rows:
            sym_labels = [r[2] for r in sym_rows]
            pos_rate = f"{sum(sym_labels) / len(sym_labels):.1%}"
        print(
            f"  [{segment_id}] {symbol}: {len(rows)} trend-scanning samples ({pos_rate} positive)"
        )

    rows.sort(key=lambda r: r[0])
    features_out = [r[1] for r in rows]
    labels_out = [r[2] for r in rows]
    weights_out = _np.array([r[3] for r in rows], dtype=float) if rows else None
    hold_bars_out = [r[4] for r in rows] if rows else None
    timestamps_out = [r[0] for r in rows]

    if labels_out:
        pos_count = sum(labels_out)
        total = len(labels_out)
        print(
            f"  [{segment_id}] Trend-scanning labels: "
            f"{pos_count / total:.1%} positive ({pos_count}/{total})"
        )

    return features_out, labels_out, weights_out, hold_bars_out, timestamps_out


def _build_dataset_triple_barrier(
    segment_id: str,
    symbols: list[str],
    market_id: str,
    settings: Settings,
    *,
    excess_returns: bool = False,
    market_context: MarketContext | None = None,
) -> tuple[list[dict[str, float]], list[int], np.ndarray | None, list[int] | None, list[datetime]]:
    """Build dataset with triple barrier labels.

    When excess_returns=True, fetches benchmark candles (SPY for US, IMOEX
    for MOEX) and aligns them per-symbol to produce market-neutral labels.
    Also fetches VIX candles for US segments to provide regime features.
    When market_context is provided, it is threaded into build_triple_barrier_dataset
    so that MOEX/cross-asset features are sliced per entry bar (no look-ahead).
    """
    import numpy as _np  # noqa: PLC0415

    tb_params = get_triple_barrier_params(segment_id)
    min_candles_tb = _WINDOW_SIZE + int(tb_params["max_hold"]) + 1
    rows: list[tuple[datetime, dict[str, float], int, float, int]] = []

    # Fetch benchmark candles once if excess returns requested
    raw_benchmark: list[Candle] | None = None
    if excess_returns:
        raw_benchmark = fetch_benchmark_candles(segment_id)
        if raw_benchmark is None:
            print(f"  [{segment_id}] Could not fetch benchmark, falling back to absolute returns.")

    # Fetch VIX candles once for US segments (None for MOEX)
    vix_candles = fetch_vix_candles(segment_id)

    for symbol in symbols:
        candles = fetch_symbol_candles(symbol, market_id, settings, segment_id=segment_id)
        if len(candles) < _MIN_HISTORY_DAYS:
            print(
                f"  [{segment_id}] Skipping {symbol}: {len(candles)} trading days "
                f"< {_MIN_HISTORY_DAYS} minimum"
            )
            continue
        if len(candles) < min_candles_tb:
            print(
                f"  [{segment_id}] {symbol}: only {len(candles)} candles, "
                f"need {min_candles_tb}+ for triple barrier -- skipping."
            )
            continue

        # Align benchmark to this symbol's candles
        aligned_bench: list[Candle] | None = None
        if raw_benchmark:
            aligned_bench = align_benchmark_candles(candles, raw_benchmark)
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
            vix_candles=vix_candles,
            market_context=market_context,
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
