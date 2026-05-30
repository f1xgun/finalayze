"""Macro feature computation -- CBR key rate and CPI (Layer 3)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import structlog

from finalayze.data.fetchers.cbr import (
    cpi_data_staleness_months,
    get_cpi_yoy_fraction,
)
from finalayze.ml.features.zscore import rolling_zscore_clipped

if TYPE_CHECKING:
    from datetime import datetime

    from finalayze.core.schemas import MoexMarketData

_log = structlog.get_logger()

# MOEX-specific feature constants
EXTERNAL_DATA_LAG_BARS = 2  # All external data lagged by 2 bars to avoid look-ahead
MOEX_MACRO_ZSCORE_WINDOW = 252  # 252 trading days (~1 year)

# CBR rate comparison epsilon (avoid float equality issues)
_CBR_RATE_EPSILON = 1e-10

# Months of missing CPI before we warn that the static table has silently rotted.
# CPI now comes from the single source of truth in data/fetchers/cbr.py
# (get_cpi_yoy_fraction); macro.py no longer keeps its own table.
_CPI_STALE_WARN_MONTHS = 3


def _warn_if_cpi_stale(
    moex_data: MoexMarketData,
    candle_timestamps: list[datetime] | None,
) -> None:
    """Emit a structured warning if the static CPI table lags the scored data.

    Picks the most recent reference date available (candle timestamps preferred,
    else the latest key-rate record) and checks it against the CPI coverage.
    """
    ref: datetime | None = None
    if candle_timestamps:
        ref = max(candle_timestamps)
    elif moex_data.key_rates:
        ref = max(r.timestamp for r in moex_data.key_rates)
    if ref is None:
        return

    stale_months = cpi_data_staleness_months(ref.date())
    if stale_months >= _CPI_STALE_WARN_MONTHS:
        _log.warning(
            "cpi_data_stale",
            stale_months=stale_months,
            reference_date=ref.date().isoformat(),
            hint="extend _CPI_DATA in data/fetchers/cbr.py or wire a live CPI feed",
        )


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

    # Observability: warn (don't fail) if the static CPI table has fallen behind
    # the data we're scoring against. Without this the feature silently collapses
    # to 0.0 once CPI lookups start missing (the May-2026 stale-table incident).
    _warn_if_cpi_stale(moex_data, candle_timestamps)

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
            cpi = get_cpi_yoy_fraction(cpi_yr, cpi_mo)
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
