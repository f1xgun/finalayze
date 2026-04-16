"""Z-score feature computation utilities (Layer 3)."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import pandas_ta as ta

if TYPE_CHECKING:
    import pandas as pd

# Z-score window lengths
ZSCORE_WINDOW = 60
VOLUME_ZSCORE_WINDOW = 20

# Rolling z-score parameters
MOEX_ZSCORE_CLIP = 3.0
MIN_ZSCORE_OBSERVATIONS = 20


def safe_zscore(value: float, mean: float, std: float) -> float:
    """Compute z-score, returning 0.0 when std is zero or non-finite."""
    if std <= 0.0 or not math.isfinite(std):
        return 0.0
    z = (value - mean) / std
    return z if math.isfinite(z) else 0.0


def rolling_zscore_clipped(
    values: pd.Series,
    window: int,
    clip: float = MOEX_ZSCORE_CLIP,
) -> float:
    """Compute z-score of the last value in *values* using a rolling window.

    Returns 0.0 when:
    - Fewer than max(window, MIN_ZSCORE_OBSERVATIONS) data points are available.
    - Standard deviation is zero or non-finite.

    The result is clipped to [-clip, clip].
    """
    required = max(window, MIN_ZSCORE_OBSERVATIONS)
    if len(values) < required:
        return 0.0

    windowed = values.iloc[-window:]
    mean = float(windowed.mean())
    std = float(windowed.std())

    if std <= 0.0 or not math.isfinite(std):
        return 0.0

    last = float(values.iloc[-1])
    z = (last - mean) / std

    if not math.isfinite(z):
        return 0.0

    return float(np.clip(z, -clip, clip))


def compute_zscore_features(
    close_s: pd.Series,
    high_s: pd.Series,
    low_s: pd.Series,
    volume_s: pd.Series,
    rsi_lookback: int = 14,
) -> dict[str, float]:
    """Compute z-score normalized features for relative strength analysis.

    All windows use min_periods=1 so short series degrade gracefully.
    No look-ahead bias: rolling windows use only past data.
    """
    # Price z-score: (close - SMA60) / std60
    price_mean = float(close_s.rolling(ZSCORE_WINDOW, min_periods=1).mean().iloc[-1])
    price_std = float(close_s.rolling(ZSCORE_WINDOW, min_periods=1).std().iloc[-1])
    price_zscore = safe_zscore(float(close_s.iloc[-1]), price_mean, price_std)

    # Volume z-score: (volume - vol_mean_20) / vol_std_20
    vol_mean = float(volume_s.rolling(VOLUME_ZSCORE_WINDOW, min_periods=1).mean().iloc[-1])
    vol_std = float(volume_s.rolling(VOLUME_ZSCORE_WINDOW, min_periods=1).std().iloc[-1])
    vol_zscore = safe_zscore(float(volume_s.iloc[-1]), vol_mean, vol_std)

    # RSI z-score: (RSI14 - mean_RSI14_60d) / std_RSI14_60d
    rsi_series = ta.rsi(close_s, length=rsi_lookback)
    rsi_zscore = 0.0
    if rsi_series is not None and not rsi_series.empty:
        rsi_mean = float(rsi_series.rolling(ZSCORE_WINDOW, min_periods=1).mean().iloc[-1])
        rsi_std = float(rsi_series.rolling(ZSCORE_WINDOW, min_periods=1).std().iloc[-1])
        rsi_val = float(rsi_series.iloc[-1])
        if math.isfinite(rsi_val):
            rsi_zscore = safe_zscore(rsi_val, rsi_mean, rsi_std)

    # ATR z-score: (ATR14 - mean_ATR_60d) / std_ATR_60d
    atr_series = ta.atr(high_s, low_s, close_s, length=rsi_lookback)
    atr_zscore = 0.0
    if atr_series is not None and not atr_series.empty:
        atr_mean = float(atr_series.rolling(ZSCORE_WINDOW, min_periods=1).mean().iloc[-1])
        atr_std = float(atr_series.rolling(ZSCORE_WINDOW, min_periods=1).std().iloc[-1])
        atr_val = float(atr_series.iloc[-1])
        if math.isfinite(atr_val):
            atr_zscore = safe_zscore(atr_val, atr_mean, atr_std)

    return {
        "price_zscore_60d": price_zscore,
        "volume_zscore_20d": vol_zscore,
        "rsi_zscore_60d": rsi_zscore,
        "atr_zscore_60d": atr_zscore,
    }
