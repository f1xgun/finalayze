"""MOEX external data features -- FX, commodity, turnover (Layer 3)."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from finalayze.data.moex_calendar import trading_days_gap
from finalayze.ml.features.macro import EXTERNAL_DATA_LAG_BARS
from finalayze.ml.features.zscore import rolling_zscore_clipped

if TYPE_CHECKING:
    from finalayze.core.schemas import MoexMarketData

# MOEX-specific feature constants
MOEX_ZSCORE_WINDOW = 60
_FX_STD_BREAKPOINT_PCT = 0.20  # If std > 20% of mean, suppress (structural break)
_BRENT_HOLIDAY_SUPPRESS_BARS = 2  # Suppress z-score for this many bars after MOEX reopening
_BRENT_HOLIDAY_MIN_GAP = 3  # Trigger only if gap > this many non-trading days (>weekend)


def compute_fx_features(moex_data: MoexMarketData | None) -> dict[str, float]:
    """Compute FX z-score feature from USD/RUB daily rates.

    Returns usdrub_zscore_60d: z-score of lagged 60d rolling window.
    Lag of EXTERNAL_DATA_LAG_BARS is applied to avoid look-ahead bias.
    Circuit-breaker: if rolling std > 20% of mean, returns 0.0 (structural break).
    """
    _default: dict[str, float] = {"usdrub_zscore_60d": 0.0}

    if moex_data is None or not moex_data.fx_rates:
        return _default

    rates = moex_data.fx_rates
    min_required = MOEX_ZSCORE_WINDOW + EXTERNAL_DATA_LAG_BARS
    if len(rates) < min_required:
        return _default

    # Apply lag: exclude the last EXTERNAL_DATA_LAG_BARS records
    lagged = rates[:-EXTERNAL_DATA_LAG_BARS]
    values = pd.Series([float(r.rate) for r in lagged], dtype=float)

    # Circuit-breaker: structural break if std > 20% of mean in 60d window
    window_vals = values.iloc[-MOEX_ZSCORE_WINDOW:]
    mean_val = float(window_vals.mean())
    std_val = float(window_vals.std())
    if mean_val > 0 and std_val / mean_val > _FX_STD_BREAKPOINT_PCT:
        return _default

    return {"usdrub_zscore_60d": rolling_zscore_clipped(values, MOEX_ZSCORE_WINDOW)}


def compute_commodity_features(moex_data: MoexMarketData | None) -> dict[str, float]:
    """Compute Brent crude z-score feature with 2-bar holiday suppression.

    Returns brent_zscore_60d: z-score of lagged 60d rolling window of Brent close prices.
    Lag of EXTERNAL_DATA_LAG_BARS is applied to avoid look-ahead bias.

    Suppression: if any of the last _BRENT_HOLIDAY_SUPPRESS_BARS consecutive pairs in the
    lagged sequence have a gap > _BRENT_HOLIDAY_MIN_GAP non-trading days, the z-score is
    suppressed to 0.0.  This prevents catch-up moves after MOEX extended closures (e.g.
    New Year Jan 1-8 or May holidays) from polluting the feature signal.
    """
    _default: dict[str, float] = {"brent_zscore_60d": 0.0}

    if moex_data is None or not moex_data.commodity_candles:
        return _default

    brent = moex_data.commodity_candles.get("BZ=F")
    if not brent:
        return _default

    min_required = MOEX_ZSCORE_WINDOW + EXTERNAL_DATA_LAG_BARS
    if len(brent) < min_required:
        return _default

    # Apply lag: exclude the last EXTERNAL_DATA_LAG_BARS candles
    lagged = brent[:-EXTERNAL_DATA_LAG_BARS]

    # Holiday suppression: check the last _BRENT_HOLIDAY_SUPPRESS_BARS pairs for extended gaps
    if len(lagged) >= _BRENT_HOLIDAY_SUPPRESS_BARS + 1:
        for i in range(-_BRENT_HOLIDAY_SUPPRESS_BARS, 0):
            gap = trading_days_gap(
                lagged[i - 1].timestamp.date(),
                lagged[i].timestamp.date(),
            )
            if gap > _BRENT_HOLIDAY_MIN_GAP:
                return _default

    values = pd.Series([float(c.close) for c in lagged], dtype=float)
    return {"brent_zscore_60d": rolling_zscore_clipped(values, MOEX_ZSCORE_WINDOW)}


def compute_fx_return_features(moex_data: MoexMarketData | None) -> dict[str, float]:
    """Compute FX return features from USD/RUB daily rates.

    Returns 2 features:
    - usdrub_return: log return of USDRUB over 1 bar, lagged by EXTERNAL_DATA_LAG_BARS.
      Clipped to [-0.15, 0.15].
    - usdrub_vol: 20-day rolling std of USDRUB log returns, lagged.
      Clipped to [0, 0.10].
    """
    _default: dict[str, float] = {"usdrub_return": 0.0, "usdrub_vol": 0.0}

    if moex_data is None or not moex_data.fx_rates:
        return _default

    rates = moex_data.fx_rates
    lag = EXTERNAL_DATA_LAG_BARS
    # Need at least lag + 2 rates for 1-bar return + lag
    min_required = lag + 2
    if len(rates) < min_required:
        return _default

    # Compute lagged 1-bar log return
    rate_prev = float(rates[-lag - 2].rate)
    rate_curr = float(rates[-lag - 1].rate)

    if rate_prev <= 0 or rate_curr <= 0:
        return _default

    usdrub_return = float(np.clip(np.log(rate_curr / rate_prev), -0.15, 0.15))

    # Compute rolling vol: need enough data for 20-day window
    _vol_window = 20
    usdrub_vol = 0.0
    if len(rates) >= lag + _vol_window + 1:
        # Use lagged series for vol computation
        lagged_rates = rates[:-lag] if lag > 0 else rates
        rate_values = pd.Series([float(r.rate) for r in lagged_rates], dtype=float)
        log_returns = pd.Series(np.log(rate_values / rate_values.shift(1))).dropna()
        if len(log_returns) >= _vol_window:
            rolling_std = log_returns.rolling(_vol_window).std()
            last_std = float(rolling_std.iloc[-1])
            if math.isfinite(last_std):
                usdrub_vol = float(np.clip(last_std, 0.0, 0.10))

    return {"usdrub_return": usdrub_return, "usdrub_vol": usdrub_vol}


def compute_brent_return_features(moex_data: MoexMarketData | None) -> dict[str, float]:
    """Compute Brent crude log return features.

    Returns 3 features: brent_return (1-bar), brent_ret_5d (5-bar), brent_ret_21d (21-bar).
    Each is a log return lagged by EXTERNAL_DATA_LAG_BARS, clipped.
    Each feature falls back to 0.0 independently.
    """
    _default: dict[str, float] = {
        "brent_return": 0.0,
        "brent_ret_5d": 0.0,
        "brent_ret_21d": 0.0,
    }

    if moex_data is None or not moex_data.commodity_candles:
        return _default

    brent = moex_data.commodity_candles.get("BZ=F")
    if not brent:
        return _default

    lag = EXTERNAL_DATA_LAG_BARS
    result = dict(_default)

    # 1-bar return (existing logic, unchanged behavior)
    if len(brent) >= lag + 2:
        close_prev = float(brent[-lag - 2].close)
        close_curr = float(brent[-lag - 1].close)
        if close_prev > 0 and close_curr > 0:
            result["brent_return"] = float(np.clip(np.log(close_curr / close_prev), -0.15, 0.15))

    # 5-bar return
    if len(brent) >= lag + 6:
        c0 = float(brent[-lag - 6].close)
        c1 = float(brent[-lag - 1].close)
        if c0 > 0 and c1 > 0:
            result["brent_ret_5d"] = float(np.clip(np.log(c1 / c0), -0.30, 0.30))

    # 21-bar return
    if len(brent) >= lag + 22:
        c0 = float(brent[-lag - 22].close)
        c1 = float(brent[-lag - 1].close)
        if c0 > 0 and c1 > 0:
            result["brent_ret_21d"] = float(np.clip(np.log(c1 / c0), -0.50, 0.50))

    return result


def compute_turnover_features(moex_data: MoexMarketData | None) -> dict[str, float]:
    """Compute MOEX aggregate market turnover z-score.

    Returns market_turnover_zscore: z-score of lagged 60d rolling window.
    Lag of EXTERNAL_DATA_LAG_BARS is applied to avoid look-ahead bias.
    """
    _default: dict[str, float] = {"market_turnover_zscore": 0.0}

    if moex_data is None or not moex_data.turnover:
        return _default

    records = moex_data.turnover
    min_required = MOEX_ZSCORE_WINDOW + EXTERNAL_DATA_LAG_BARS
    if len(records) < min_required:
        return _default

    # Apply lag: exclude the last EXTERNAL_DATA_LAG_BARS records
    lagged = records[:-EXTERNAL_DATA_LAG_BARS]
    values = pd.Series([float(r.volume_rub) for r in lagged], dtype=float).ffill()

    return {"market_turnover_zscore": rolling_zscore_clipped(values, MOEX_ZSCORE_WINDOW)}
