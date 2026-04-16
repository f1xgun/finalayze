"""Regime / VIX feature computation (Layer 3)."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from finalayze.core.schemas import Candle

# Regime / VIX feature constants
_VIX_PERCENTILE_WINDOW = 252
_VIX_MIN_PERIODS = 63
_VIX_CHANGE_WINDOW = 5
_SHORT_VOL_WINDOW = 20
_LONG_VOL_WINDOW = 60


def compute_regime_features(
    close_s: pd.Series,
    vix_candles: list[Candle] | None,
) -> dict[str, float]:
    """Compute regime/VIX features and realized volatility ratio.

    VIX features use lagged values (no look-ahead bias).
    realized_vol_ratio uses the stock's own close prices (works for all markets).
    When vix_candles is None (e.g., MOEX), VIX features default to 0.0.
    """
    # --- VIX features ---
    vix_level = 0.0
    vix_percentile = 0.0
    vix_change = 0.0

    if vix_candles and len(vix_candles) >= 2:  # noqa: PLR2004
        # Lagged VIX: use [-2] to avoid look-ahead (current bar not yet closed)
        vix_level = float(vix_candles[-2].close)

        vix_closes = pd.Series(
            [float(c.close) for c in vix_candles],
            dtype=float,
        )

        # Percentile rank over 252 trading days (min 63 for warmup)
        current_vix = float(vix_closes.iloc[-2])
        window = min(_VIX_PERCENTILE_WINDOW, len(vix_closes) - 1)
        if window >= 1:
            # Use all bars except last (lagged) for the ranking window
            lookback = vix_closes.iloc[:-1].iloc[-window:]
            if len(lookback) >= _VIX_MIN_PERIODS:
                vix_percentile = float((lookback <= current_vix).mean())

        # 5-day VIX change (percentage)
        lag_offset = _VIX_CHANGE_WINDOW + 1  # +1 for lag
        if len(vix_closes) > lag_offset:
            vix_prev = float(vix_closes.iloc[-lag_offset - 1])
            if vix_prev > 0:
                vix_change = (current_vix - vix_prev) / vix_prev

    # --- Realized volatility ratio (works for all markets) ---
    returns = close_s.pct_change()
    short_vol = returns.rolling(_SHORT_VOL_WINDOW, min_periods=1).std()
    long_vol = returns.rolling(_LONG_VOL_WINDOW, min_periods=1).std()

    short_val = float(short_vol.iloc[-1])
    long_val = float(long_vol.iloc[-1])

    if long_val > 0 and math.isfinite(long_val) and math.isfinite(short_val):
        realized_vol_ratio = short_val / long_val
    else:
        realized_vol_ratio = 0.0

    return {
        "vix_level": vix_level,
        "vix_percentile_252d": vix_percentile,
        "vix_change_5d": vix_change,
        "realized_vol_ratio": realized_vol_ratio,
    }
