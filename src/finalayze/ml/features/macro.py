"""Macro feature computation -- CBR key rate and CPI (Layer 3)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from finalayze.ml.features.zscore import rolling_zscore_clipped

if TYPE_CHECKING:
    from datetime import datetime

    from finalayze.core.schemas import MoexMarketData

# MOEX-specific feature constants
EXTERNAL_DATA_LAG_BARS = 2  # All external data lagged by 2 bars to avoid look-ahead
MOEX_MACRO_ZSCORE_WINDOW = 252  # 252 trading days (~1 year)

# CBR rate comparison epsilon (avoid float equality issues)
_CBR_RATE_EPSILON = 1e-10

# Trailing 12-month CPI (Росстат), annualized as decimal fraction.
# 6-month fallback in compute_macro_features if exact month missing.
TRAILING_CPI: dict[tuple[int, int], float] = {
    (2023, 1): 0.1184,
    (2023, 2): 0.1002,
    (2023, 3): 0.0360,
    (2023, 4): 0.0253,
    (2023, 5): 0.0234,
    (2023, 6): 0.0329,
    (2023, 7): 0.0400,
    (2023, 8): 0.0513,
    (2023, 9): 0.0600,
    (2023, 10): 0.0672,
    (2023, 11): 0.0748,
    (2023, 12): 0.0736,
    (2024, 1): 0.0744,
    (2024, 2): 0.0769,
    (2024, 3): 0.0772,
    (2024, 4): 0.0784,
    (2024, 5): 0.0824,
    (2024, 6): 0.0858,
    (2024, 7): 0.0913,
    (2024, 8): 0.0909,
    (2024, 9): 0.0863,
    (2024, 10): 0.0834,
    (2024, 11): 0.0874,
    (2024, 12): 0.0972,
    (2025, 1): 0.1001,
    (2025, 2): 0.1003,
    (2025, 3): 0.1005,
}


def compute_macro_features(
    moex_data: MoexMarketData | None,
    candle_timestamps: list[datetime] | None = None,
) -> dict[str, float]:
    """Compute real interest rate z-score (key_rate - CPI).

    Builds a sparse real_rate series, forward-fills to daily, applies lag,
    then z-scores over 252d window. Uses a 6-month CPI fallback if exact month
    is missing from the static table.

    Per S19-H1: daily_index is the union of candle_timestamps and sparse_dates
    so pre-window key rates survive forward-fill reindex.
    """
    _default: dict[str, float] = {"real_rate_zscore": 0.0}

    if moex_data is None or not moex_data.key_rates:
        return _default

    # Build sparse real_rate series: key_rate - CPI
    sparse: dict[datetime, float] = {}
    for record in moex_data.key_rates:
        yr = record.timestamp.year
        mo = record.timestamp.month

        # 6-month fallback: try exact month, then up to 6 months back
        cpi: float | None = None
        for offset in range(7):  # 0 = exact, 1-6 = fallback months back
            cpi_mo = mo - offset
            cpi_yr = yr
            while cpi_mo < 1:
                cpi_mo += 12
                cpi_yr -= 1
            cpi = TRAILING_CPI.get((cpi_yr, cpi_mo))
            if cpi is not None:
                break

        if cpi is None:
            continue

        real_rate = float(record.rate) - cpi
        sparse[record.timestamp] = real_rate

    if not sparse:
        return _default

    sparse_dates = list(sparse.keys())

    # S19-H1: union of candle_timestamps and sparse_dates ensures pre-window
    # key rates survive reindex (forward-fill reaches candle window start)
    all_timestamps = set(candle_timestamps or []) | set(sparse_dates)
    daily_index = pd.DatetimeIndex(sorted(all_timestamps))

    sparse_series = pd.Series(sparse)
    # Forward-fill to daily granularity (handles gaps between rate changes)
    daily = sparse_series.reindex(daily_index).ffill().dropna()

    if daily.empty:
        return _default

    min_required = EXTERNAL_DATA_LAG_BARS + 1
    if len(daily) < min_required:
        return _default

    # Apply lag on the daily series
    lagged = daily.iloc[:-EXTERNAL_DATA_LAG_BARS]

    if lagged.empty:
        return _default

    window = min(len(lagged), MOEX_MACRO_ZSCORE_WINDOW)
    return {"real_rate_zscore": rolling_zscore_clipped(lagged, window)}


def compute_cbr_features(
    moex_data: MoexMarketData | None,
    candle_timestamps: list[datetime] | None = None,
) -> dict[str, float]:
    """Compute CBR key rate features: level, delta, direction one-hot.

    Returns 4 features:
    - cbr_rate_level: forward-filled key rate value (already decimal fraction, e.g. 0.16)
    - cbr_rate_delta: change between last two distinct rate values
    - cbr_direction_cut: 1.0 if rate was cut (delta < 0), else 0.0
    - cbr_direction_hike: 1.0 if rate was hiked (delta > 0), else 0.0

    All values are lagged by EXTERNAL_DATA_LAG_BARS to avoid look-ahead bias.
    Rates in KeyRateRecord are already decimal fractions (0.16 = 16%).
    """
    _default: dict[str, float] = {
        "cbr_rate_level": 0.0,
        "cbr_rate_delta": 0.0,
        "cbr_direction_cut": 0.0,
        "cbr_direction_hike": 0.0,
    }

    if moex_data is None or not moex_data.key_rates or len(moex_data.key_rates) < 2:  # noqa: PLR2004
        return _default

    # Build sparse rate series and forward-fill to daily using candle_timestamps union
    sparse: dict[datetime, float] = {
        record.timestamp: float(record.rate) for record in moex_data.key_rates
    }
    sparse_dates = list(sparse.keys())
    all_timestamps = set(candle_timestamps or []) | set(sparse_dates)
    daily_index = pd.DatetimeIndex(sorted(all_timestamps))

    sparse_series = pd.Series(sparse)
    daily = sparse_series.reindex(daily_index).ffill().dropna()

    min_required = EXTERNAL_DATA_LAG_BARS + 2  # need at least 2 values after lag
    if daily.empty or len(daily) < min_required:
        return _default

    # Apply lag
    lagged = daily.iloc[:-EXTERNAL_DATA_LAG_BARS] if EXTERNAL_DATA_LAG_BARS > 0 else daily
    if len(lagged) < 2:  # noqa: PLR2004
        return _default

    rate_level = float(lagged.iloc[-1])

    # Find last two distinct rate values for delta
    # Walk backward through lagged series to find the previous distinct value
    current_rate = rate_level
    prev_rate = current_rate  # default: no change
    for i in range(len(lagged) - 2, -1, -1):
        val = float(lagged.iloc[i])
        if abs(val - current_rate) > _CBR_RATE_EPSILON:
            prev_rate = val
            break

    rate_delta = current_rate - prev_rate

    direction_cut = 1.0 if rate_delta < -_CBR_RATE_EPSILON else 0.0
    direction_hike = 1.0 if rate_delta > _CBR_RATE_EPSILON else 0.0

    return {
        "cbr_rate_level": rate_level,
        "cbr_rate_delta": rate_delta,
        "cbr_direction_cut": direction_cut,
        "cbr_direction_hike": direction_hike,
    }
