"""ADX (Average Directional Index) regime detection helper (Layer 4)."""

from __future__ import annotations

import pandas as pd
import pandas_ta as ta


def compute_adx(
    closes: list[float],
    highs: list[float],
    lows: list[float],
    period: int = 14,
) -> float | None:
    """Compute ADX from price data.

    Args:
        closes: List of closing prices (oldest first).
        highs: List of high prices (oldest first).
        lows: List of low prices (oldest first).
        period: ADX period (default 14).

    Returns:
        Latest ADX value as float, or None if insufficient data.
        Needs at least 2 * period bars for a valid computation.
    """
    min_bars = 2 * period
    if len(closes) < min_bars or len(highs) < min_bars or len(lows) < min_bars:
        return None

    high_series = pd.Series(highs)
    low_series = pd.Series(lows)
    close_series = pd.Series(closes)

    adx_df = ta.adx(high_series, low_series, close_series, length=period)
    if adx_df is None:
        return None

    adx_col = f"ADX_{period}"
    if adx_col not in adx_df.columns:
        return None

    adx_val = adx_df[adx_col].iloc[-1]
    if pd.isna(adx_val):
        return None

    return float(adx_val)
