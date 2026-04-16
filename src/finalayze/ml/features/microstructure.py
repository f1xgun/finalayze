"""Microstructure feature computation (Layer 3)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

# Microstructure constants
_PROXIMITY_WINDOW = 252
_AMIHUD_WINDOW = 20
_AMIHUD_RANK_LOOKBACK = 252
_AMIHUD_MIN_RANK_PERIODS = 20
_AMIHUD_NEUTRAL_DEFAULT = 0.5
_MIN_CS_BARS = 2


def compute_amihud_series(
    close_s: pd.Series,
    volume_s: pd.Series,
    window: int = _AMIHUD_WINDOW,
) -> pd.Series:
    """Compute rolling average Amihud illiquidity ratio.

    Amihud = mean(|return| / dollar_volume) over *window* bars.
    No look-ahead bias: uses only past data via rolling window.
    """
    abs_returns = close_s.pct_change().abs()
    dollar_volume = (close_s * volume_s).replace(0, np.nan)
    raw_amihud = abs_returns / dollar_volume
    raw_amihud = raw_amihud.replace([np.inf, -np.inf], np.nan)
    return raw_amihud.rolling(window, min_periods=max(1, window // 2)).mean()


def compute_microstructure_features(
    close_s: pd.Series,
    high_s: pd.Series,
    low_s: pd.Series,
    volume_s: pd.Series,
    last_close: float,
) -> dict[str, float]:
    """Compute microstructure features: rolling high proximity, Amihud, Corwin-Schultz."""
    # 52-week high proximity: close / rolling_max(close, 252)
    rolling_max_252 = close_s.rolling(min(_PROXIMITY_WINDOW, len(close_s)), min_periods=1).max()
    rm_val = float(rolling_max_252.iloc[-1])
    proximity_52wk = last_close / rm_val if rm_val > 0 and math.isfinite(rm_val) else 1.0

    # Amihud illiquidity: percentile rank over 252-bar lookback, producing [0, 1]
    amihud_series = compute_amihud_series(close_s, volume_s, window=_AMIHUD_WINDOW)
    valid_amihud = amihud_series.dropna()
    if len(valid_amihud) >= _AMIHUD_MIN_RANK_PERIODS:
        current_val = float(amihud_series.iloc[-1])
        if math.isfinite(current_val):
            lookback = valid_amihud.iloc[-min(_AMIHUD_RANK_LOOKBACK, len(valid_amihud)) :]
            amihud_20d = float((lookback < current_val).mean())
        else:
            amihud_20d = _AMIHUD_NEUTRAL_DEFAULT
    else:
        amihud_20d = _AMIHUD_NEUTRAL_DEFAULT

    return {
        "proximity_rolling_high": proximity_52wk,
        "amihud_20d": amihud_20d,
        "corwin_schultz_spread": corwin_schultz(high_s, low_s),
    }


def corwin_schultz(high_s: pd.Series, low_s: pd.Series) -> float:
    """Compute Corwin-Schultz (2012) bid-ask spread estimator from high/low prices.

    Returns the last available spread estimate, clamped to [0, 1].
    If insufficient data (< 2 bars), returns 0.0.
    """
    if len(high_s) < _MIN_CS_BARS:
        return 0.0

    _sqrt2 = math.sqrt(2)
    _denom = 3 - 2 * _sqrt2

    # ln(H/L)^2 for each bar
    hl_log2 = pd.Series(np.log(high_s / low_s) ** 2, index=high_s.index)

    # beta: sum of ln(H_t/L_t)^2 for consecutive pairs
    beta = hl_log2 + hl_log2.shift(1)

    # gamma: ln(max(H_t, H_{t-1}) / min(L_t, L_{t-1}))^2
    h_max = pd.concat([high_s, high_s.shift(1)], axis=1).max(axis=1)
    l_min = pd.concat([low_s, low_s.shift(1)], axis=1).min(axis=1)
    gamma = np.log(h_max / l_min) ** 2

    # alpha
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / _denom - np.sqrt(gamma / _denom)

    # spread = 2 * (exp(alpha) - 1) / (1 + exp(alpha))
    exp_alpha = np.exp(alpha)
    spread = 2 * (exp_alpha - 1) / (1 + exp_alpha)

    # Take the last valid value
    last_spread = float(spread.iloc[-1])
    if not math.isfinite(last_spread):
        return 0.0
    # Clamp to [0, 1]
    return max(0.0, min(1.0, last_spread))


def garman_klass_vol(
    open_s: pd.Series,
    high_s: pd.Series,
    low_s: pd.Series,
    close_s: pd.Series,
    length: int = 20,
) -> float:
    """Compute Garman-Klass volatility over *length* bars.

    Formula per bar: 0.5 * ln(H/L)^2 - (2*ln(2) - 1) * ln(C/O)^2
    Returns the mean over the last *length* bars, clamped to >= 0.
    """
    hl_ratio = high_s / low_s
    co_ratio = close_s / open_s

    # Guard against zero/negative values
    hl_ratio = hl_ratio.clip(lower=1e-10)
    co_ratio = co_ratio.clip(lower=1e-10)

    hl_log2 = pd.Series(np.log(hl_ratio) ** 2, dtype=float)
    co_log2 = pd.Series(np.log(co_ratio) ** 2, dtype=float)

    gk_per_bar = 0.5 * hl_log2 - (2 * math.log(2) - 1) * co_log2
    gk_rolling = gk_per_bar.rolling(length).mean()

    if gk_rolling is not None and not gk_rolling.empty:
        val = float(gk_rolling.iloc[-1])
        if math.isfinite(val):
            return max(val, 0.0)
    return 0.0
