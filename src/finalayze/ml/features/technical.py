"""Technical feature engineering for ML models (Layer 3)."""

from __future__ import annotations

import logging
import math
import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pandas_ta as ta

from finalayze.core.exceptions import InsufficientDataError
from finalayze.core.schemas import MarketContext, MoexMarketData
from finalayze.data.moex_calendar import trading_days_gap

if TYPE_CHECKING:
    from datetime import datetime

    from finalayze.core.schemas import Candle

_MIN_CANDLES = 80
_MAX_FEATURE_LOOKBACK = 252
_SPLIT_WARNING_THRESHOLD = 0.40

# Lookback lengths for indicators
_ROC_LENGTH = 10
_OBV_SLOPE_LENGTH = 10
_RSI_LOOKBACK = 14
_PROXIMITY_WINDOW = 252
_AMIHUD_WINDOW = 20
_AMIHUD_RANK_LOOKBACK = 252
_AMIHUD_MIN_RANK_PERIODS = 20
_AMIHUD_NEUTRAL_DEFAULT = 0.5
_MIN_CS_BARS = 2
_WAVELET_LEVEL = 3
_MIN_WAVELET_SAMPLES = 16  # pywt.wavedec('db4', level=3) needs at least 2^level samples

# Lagged return lookback thresholds (minimum bars needed)
_RET_1D_MIN = 2
_RET_5D_MIN = 6
_RET_21D_MIN = 22
_RET_63D_MIN = 64
_RET_126D_MIN = 127

# Momentum reversal ratio epsilon (avoid division by zero)
_MOM_RATIO_EPSILON = 1e-8

# Return distribution constants
_RECENT_RETURN_WINDOW = 20
_MIN_SKEW_BARS = 5

# Z-score window lengths
_ZSCORE_WINDOW = 60
_VOLUME_ZSCORE_WINDOW = 20

# Calendar cyclical encoding constants
_MONTHS_PER_YEAR = 12

# Regime / VIX feature constants
_VIX_PERCENTILE_WINDOW = 252
_VIX_MIN_PERIODS = 63
_VIX_CHANGE_WINDOW = 5
_SHORT_VOL_WINDOW = 20
_LONG_VOL_WINDOW = 60

# Cross-asset feature constants
_CROSS_ASSET_LOOKBACK = 63
_RELATIVE_STRENGTH_WINDOW = 21
_VOL_FLOOR = 0.01
_DEFAULT_BETA = 1.0
_DEFAULT_CORR = 0.5

_OLS_INDICES_DTYPE = float  # np.arange dtype for OLS slope computation

# MOEX-specific feature constants
_EXTERNAL_DATA_LAG_BARS = 2  # All external data lagged by 2 bars to avoid look-ahead
_MOEX_ZSCORE_WINDOW = 60
_MOEX_MACRO_ZSCORE_WINDOW = 252  # 252 trading days (~1 year)
_MOEX_ZSCORE_CLIP = 3.0
_FX_STD_BREAKPOINT_PCT = 0.20  # If std > 20% of mean, suppress (structural break)
_MIN_ZSCORE_OBSERVATIONS = 20  # Minimum for meaningful z-score
_BRENT_HOLIDAY_SUPPRESS_BARS = 2  # Suppress z-score for this many bars after MOEX reopening
_BRENT_HOLIDAY_MIN_GAP = 3  # Trigger only if gap > this many non-trading days (>weekend)

# CBR rate comparison epsilon (avoid float equality issues)
_CBR_RATE_EPSILON = 1e-10

# Trailing 12-month CPI (Росстат), annualized as decimal fraction.
# 6-month fallback in _compute_macro_features if exact month missing.
_TRAILING_CPI: dict[tuple[int, int], float] = {
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

_log = logging.getLogger(__name__)

try:
    import pywt as _pywt

    _HAS_PYWT = True
except ImportError:  # pragma: no cover
    _pywt = None
    _HAS_PYWT = False


def _compute_rsi_divergence(price_slope: float, rsi_slope: float) -> float:
    """Compute RSI divergence from OLS slopes of price returns and RSI changes.

    Divergence = price_slope - rsi_slope (both pre-normalized to z-scores).
      - Positive => bearish divergence (price trending up, RSI trending down)
      - Negative => bullish divergence (price trending down, RSI trending up)
      - Near-zero => no divergence (price and RSI agree)

    Args:
        price_slope: z-score normalized OLS slope of price returns over lookback.
        rsi_slope: z-score normalized OLS slope of RSI changes over lookback.

    Returns:
        Divergence score (float). Positive = bearish, negative = bullish.
    """
    return price_slope - rsi_slope


def _compute_wavelet_features(log_returns: list[float]) -> dict[str, float]:
    """Compute wavelet energy features from log returns via Daubechies-4 decomposition.

    Decomposes the signal into 1 approximation level and 3 detail levels.
    Returns the fraction of total energy in each level (normalized so they sum to ~1.0).

    If pywt is not available or there is insufficient data, returns 0.0 for all 4 features.
    """
    _zero = {
        "wavelet_approx_energy": 0.0,
        "wavelet_detail1_energy": 0.0,
        "wavelet_detail2_energy": 0.0,
        "wavelet_detail3_energy": 0.0,
    }

    if not _HAS_PYWT or len(log_returns) < _MIN_WAVELET_SAMPLES:
        return _zero

    try:
        coeffs = _pywt.wavedec(log_returns, "db4", level=_WAVELET_LEVEL)
    except Exception:
        _log.debug("Wavelet decomposition failed, returning zeros")
        return _zero

    # coeffs = [cA3, cD3, cD2, cD1]
    energies = [float(np.sum(np.square(c))) for c in coeffs]
    total_energy = sum(energies)

    if total_energy <= 0.0:
        return _zero

    return {
        "wavelet_approx_energy": energies[0] / total_energy,
        "wavelet_detail3_energy": energies[1] / total_energy,
        "wavelet_detail2_energy": energies[2] / total_energy,
        "wavelet_detail1_energy": energies[3] / total_energy,
    }


def _compute_calendar_features(
    last_timestamp: datetime,
) -> dict[str, float]:
    """Compute cyclical calendar encoding from the last candle's timestamp.

    No look-ahead bias: uses only the timestamp of the most recent candle.
    Encodes month as sin/cos pair for cyclical continuity.
    Day-of-week encoding removed (negligible effect post-2000, Sullivan et al. 2001).
    """
    month = last_timestamp.month  # 1-12

    two_pi = 2.0 * math.pi
    return {
        "month_sin": math.sin(two_pi * month / _MONTHS_PER_YEAR),
        "month_cos": math.cos(two_pi * month / _MONTHS_PER_YEAR),
    }


def _compute_regime_features(
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


def _compute_cross_asset_features(
    close_s: pd.Series,
    benchmark_close_s: pd.Series | None,
) -> dict[str, float]:
    """Compute cross-asset features: relative strength, beta, correlation, excess momentum.

    All features compare the stock to a benchmark (e.g., SPY).
    When benchmark_close_s is None or has insufficient data, returns domain-aware defaults.
    No look-ahead bias: uses only past data via rolling windows.
    """
    defaults = {
        "relative_strength_21d": 0.0,
        "rolling_beta_63d": _DEFAULT_BETA,
        "rolling_corr_63d": _DEFAULT_CORR,
        "excess_momentum_score": 0.0,
    }

    if benchmark_close_s is None or len(benchmark_close_s) < _RELATIVE_STRENGTH_WINDOW:
        return defaults

    # Align lengths: use the shorter of the two series (from the end)
    min_len = min(len(close_s), len(benchmark_close_s))
    stock_close = close_s.iloc[-min_len:].reset_index(drop=True)
    bench_close = benchmark_close_s.iloc[-min_len:].reset_index(drop=True)

    stock_returns = stock_close.pct_change()
    bench_returns = bench_close.pct_change()

    # --- relative_strength_21d: stock 21d return minus benchmark 21d return ---
    relative_strength = 0.0
    if min_len >= _RELATIVE_STRENGTH_WINDOW + 1:
        stock_ret_21d = float(
            stock_close.iloc[-1] / stock_close.iloc[-_RELATIVE_STRENGTH_WINDOW - 1] - 1,
        )
        bench_ret_21d = float(
            bench_close.iloc[-1] / bench_close.iloc[-_RELATIVE_STRENGTH_WINDOW - 1] - 1,
        )
        relative_strength = stock_ret_21d - bench_ret_21d

    # --- rolling_beta_63d: cov(stock, bench) / var(bench) over 63d window ---
    rolling_beta = _DEFAULT_BETA
    if min_len >= _CROSS_ASSET_LOOKBACK + 1:
        cov = stock_returns.rolling(_CROSS_ASSET_LOOKBACK, min_periods=_CROSS_ASSET_LOOKBACK).cov(
            bench_returns,
        )
        var = bench_returns.rolling(
            _CROSS_ASSET_LOOKBACK,
            min_periods=_CROSS_ASSET_LOOKBACK,
        ).var()
        last_cov = float(cov.iloc[-1])
        last_var = float(var.iloc[-1])
        if last_var > 0 and math.isfinite(last_cov) and math.isfinite(last_var):
            rolling_beta = last_cov / last_var
        if not math.isfinite(rolling_beta):
            rolling_beta = _DEFAULT_BETA

    # --- rolling_corr_63d: rolling correlation over 63d window ---
    rolling_corr = _DEFAULT_CORR
    if min_len >= _CROSS_ASSET_LOOKBACK + 1:
        corr = stock_returns.rolling(
            _CROSS_ASSET_LOOKBACK,
            min_periods=_CROSS_ASSET_LOOKBACK,
        ).corr(bench_returns)
        last_corr = float(corr.iloc[-1])
        if math.isfinite(last_corr):
            rolling_corr = last_corr

    # --- excess_momentum_score: (stock_ret_63d - bench_ret_63d) / max(stock_vol_63d, floor) ---
    excess_momentum = 0.0
    if min_len >= _CROSS_ASSET_LOOKBACK + 1:
        stock_ret_63d = float(
            stock_close.iloc[-1] / stock_close.iloc[-_CROSS_ASSET_LOOKBACK - 1] - 1,
        )
        bench_ret_63d = float(
            bench_close.iloc[-1] / bench_close.iloc[-_CROSS_ASSET_LOOKBACK - 1] - 1,
        )
        stock_vol_63d = float(
            stock_returns.iloc[-_CROSS_ASSET_LOOKBACK:].std(),
        )
        denom = max(stock_vol_63d, _VOL_FLOOR) if math.isfinite(stock_vol_63d) else _VOL_FLOOR
        excess_momentum = (stock_ret_63d - bench_ret_63d) / denom
        if not math.isfinite(excess_momentum):
            excess_momentum = 0.0

    return {
        "relative_strength_21d": relative_strength,
        "rolling_beta_63d": rolling_beta,
        "rolling_corr_63d": rolling_corr,
        "excess_momentum_score": excess_momentum,
    }


def _rolling_zscore_clipped(
    values: pd.Series,
    window: int,
    clip: float = _MOEX_ZSCORE_CLIP,
) -> float:
    """Compute z-score of the last value in *values* using a rolling window.

    Returns 0.0 when:
    - Fewer than max(window, _MIN_ZSCORE_OBSERVATIONS) data points are available.
    - Standard deviation is zero or non-finite.

    The result is clipped to [-clip, clip].
    """
    required = max(window, _MIN_ZSCORE_OBSERVATIONS)
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


def _compute_fx_features(moex_data: MoexMarketData | None) -> dict[str, float]:
    """Compute FX z-score feature from USD/RUB daily rates.

    Returns usdrub_zscore_60d: z-score of lagged 60d rolling window.
    Lag of _EXTERNAL_DATA_LAG_BARS is applied to avoid look-ahead bias.
    Circuit-breaker: if rolling std > 20% of mean, returns 0.0 (structural break).
    """
    _default: dict[str, float] = {"usdrub_zscore_60d": 0.0}

    if moex_data is None or not moex_data.fx_rates:
        return _default

    rates = moex_data.fx_rates
    min_required = _MOEX_ZSCORE_WINDOW + _EXTERNAL_DATA_LAG_BARS
    if len(rates) < min_required:
        return _default

    # Apply lag: exclude the last _EXTERNAL_DATA_LAG_BARS records
    lagged = rates[:-_EXTERNAL_DATA_LAG_BARS]
    values = pd.Series([float(r.rate) for r in lagged], dtype=float)

    # Circuit-breaker: structural break if std > 20% of mean in 60d window
    window_vals = values.iloc[-_MOEX_ZSCORE_WINDOW:]
    mean_val = float(window_vals.mean())
    std_val = float(window_vals.std())
    if mean_val > 0 and std_val / mean_val > _FX_STD_BREAKPOINT_PCT:
        return _default

    return {"usdrub_zscore_60d": _rolling_zscore_clipped(values, _MOEX_ZSCORE_WINDOW)}


def _compute_commodity_features(moex_data: MoexMarketData | None) -> dict[str, float]:
    """Compute Brent crude z-score feature with 2-bar holiday suppression.

    Returns brent_zscore_60d: z-score of lagged 60d rolling window of Brent close prices.
    Lag of _EXTERNAL_DATA_LAG_BARS is applied to avoid look-ahead bias.

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

    min_required = _MOEX_ZSCORE_WINDOW + _EXTERNAL_DATA_LAG_BARS
    if len(brent) < min_required:
        return _default

    # Apply lag: exclude the last _EXTERNAL_DATA_LAG_BARS candles
    lagged = brent[:-_EXTERNAL_DATA_LAG_BARS]

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
    return {"brent_zscore_60d": _rolling_zscore_clipped(values, _MOEX_ZSCORE_WINDOW)}


def _compute_macro_features(
    moex_data: MoexMarketData | None,
    candle_timestamps: list[datetime] | None = None,
) -> dict[str, float]:
    """Compute real interest rate z-score (key_rate - CPI).

    Builds a sparse real_rate series, forward-fills to daily, applies lag,
    then z-scores over 252d window. Uses a 6-month CPI fallback if exact month
    is missing from the static table.

    Per §19-H1: daily_index is the union of candle_timestamps and sparse_dates
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
            cpi = _TRAILING_CPI.get((cpi_yr, cpi_mo))
            if cpi is not None:
                break

        if cpi is None:
            continue

        real_rate = float(record.rate) - cpi
        sparse[record.timestamp] = real_rate

    if not sparse:
        return _default

    sparse_dates = list(sparse.keys())

    # §19-H1: union of candle_timestamps and sparse_dates ensures pre-window
    # key rates survive reindex (forward-fill reaches candle window start)
    all_timestamps = set(candle_timestamps or []) | set(sparse_dates)
    daily_index = pd.DatetimeIndex(sorted(all_timestamps))

    sparse_series = pd.Series(sparse)
    # Forward-fill to daily granularity (handles gaps between rate changes)
    daily = sparse_series.reindex(daily_index).ffill().dropna()

    if daily.empty:
        return _default

    min_required = _EXTERNAL_DATA_LAG_BARS + 1
    if len(daily) < min_required:
        return _default

    # Apply lag on the daily series
    lagged = daily.iloc[:-_EXTERNAL_DATA_LAG_BARS]

    if lagged.empty:
        return _default

    window = min(len(lagged), _MOEX_MACRO_ZSCORE_WINDOW)
    return {"real_rate_zscore": _rolling_zscore_clipped(lagged, window)}


def _compute_turnover_features(moex_data: MoexMarketData | None) -> dict[str, float]:
    """Compute MOEX aggregate market turnover z-score.

    Returns market_turnover_zscore: z-score of lagged 60d rolling window.
    Lag of _EXTERNAL_DATA_LAG_BARS is applied to avoid look-ahead bias.
    """
    _default: dict[str, float] = {"market_turnover_zscore": 0.0}

    if moex_data is None or not moex_data.turnover:
        return _default

    records = moex_data.turnover
    min_required = _MOEX_ZSCORE_WINDOW + _EXTERNAL_DATA_LAG_BARS
    if len(records) < min_required:
        return _default

    # Apply lag: exclude the last _EXTERNAL_DATA_LAG_BARS records
    lagged = records[:-_EXTERNAL_DATA_LAG_BARS]
    values = pd.Series([float(r.volume_rub) for r in lagged], dtype=float).ffill()

    return {"market_turnover_zscore": _rolling_zscore_clipped(values, _MOEX_ZSCORE_WINDOW)}


def _compute_cbr_features(
    moex_data: MoexMarketData | None,
    candle_timestamps: list[datetime] | None = None,
) -> dict[str, float]:
    """Compute CBR key rate features: level, delta, direction one-hot.

    Returns 4 features:
    - cbr_rate_level: forward-filled key rate value (already decimal fraction, e.g. 0.16)
    - cbr_rate_delta: change between last two distinct rate values
    - cbr_direction_cut: 1.0 if rate was cut (delta < 0), else 0.0
    - cbr_direction_hike: 1.0 if rate was hiked (delta > 0), else 0.0

    All values are lagged by _EXTERNAL_DATA_LAG_BARS to avoid look-ahead bias.
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

    min_required = _EXTERNAL_DATA_LAG_BARS + 2  # need at least 2 values after lag
    if daily.empty or len(daily) < min_required:
        return _default

    # Apply lag
    lagged = daily.iloc[:-_EXTERNAL_DATA_LAG_BARS] if _EXTERNAL_DATA_LAG_BARS > 0 else daily
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


def _compute_fx_return_features(moex_data: MoexMarketData | None) -> dict[str, float]:
    """Compute FX return features from USD/RUB daily rates.

    Returns 2 features:
    - usdrub_return: log return of USDRUB over 1 bar, lagged by _EXTERNAL_DATA_LAG_BARS.
      Clipped to [-0.15, 0.15].
    - usdrub_vol: 20-day rolling std of USDRUB log returns, lagged.
      Clipped to [0, 0.10].
    """
    _default: dict[str, float] = {"usdrub_return": 0.0, "usdrub_vol": 0.0}

    if moex_data is None or not moex_data.fx_rates:
        return _default

    rates = moex_data.fx_rates
    lag = _EXTERNAL_DATA_LAG_BARS
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


def _compute_brent_return_features(moex_data: MoexMarketData | None) -> dict[str, float]:
    """Compute Brent crude log return features.

    Returns 3 features: brent_return (1-bar), brent_ret_5d (5-bar), brent_ret_21d (21-bar).
    Each is a log return lagged by _EXTERNAL_DATA_LAG_BARS, clipped.
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

    lag = _EXTERNAL_DATA_LAG_BARS
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


def compute_features(
    candles: list[Candle],
    sentiment_score: float = 0.0,  # noqa: ARG001 — kept for backward compatibility
    market_context: MarketContext | None = None,
    # Deprecated: pass market_context=MarketContext(benchmark_candles=...) instead
    benchmark_candles: list[Candle] | None = None,
    vix_candles: list[Candle] | None = None,
) -> dict[str, float]:
    """Compute technical features from a list of candles.

    Args:
        candles: OHLCV candles sorted ascending by timestamp.
        sentiment_score: External sentiment score in [-1.0, 1.0].
        market_context: Optional ambient market data (benchmark, VIX, MOEX).
        benchmark_candles: Deprecated. Use market_context instead.
        vix_candles: Deprecated. Use market_context instead.

    Returns:
        Dict of feature name -> float value.

    Raises:
        InsufficientDataError: When fewer than _MIN_CANDLES candles are provided.
    """
    # Deprecation shim: convert old kwargs to MarketContext
    if benchmark_candles is not None or vix_candles is not None:
        warnings.warn(
            "benchmark_candles/vix_candles kwargs are deprecated. "
            "Use market_context=MarketContext(benchmark_candles=..., vix_candles=...) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if market_context is None:
            market_context = MarketContext(
                benchmark_candles=benchmark_candles,
                vix_candles=vix_candles,
            )

    # Extract components from market_context
    _benchmark = market_context.benchmark_candles if market_context else None
    _vix = market_context.vix_candles if market_context else None
    _moex = market_context.moex_data if market_context else None

    if len(candles) < _MIN_CANDLES:
        msg = f"Need at least {_MIN_CANDLES} candles, got {len(candles)}"
        raise InsufficientDataError(msg)

    closes = [float(c.close) for c in candles]
    highs = [float(c.high) for c in candles]
    lows = [float(c.low) for c in candles]
    opens = [float(c.open) for c in candles]
    volumes = [float(c.volume) for c in candles]

    close_s = pd.Series(closes, dtype=float)
    high_s = pd.Series(highs, dtype=float)
    low_s = pd.Series(lows, dtype=float)
    open_s = pd.Series(opens, dtype=float)
    volume_s = pd.Series(volumes, dtype=float)

    last_close = closes[-1]

    _warn_if_split_suspected(close_s)

    core = _compute_core_features(close_s, high_s, low_s, volume_s, last_close)
    extra = _compute_extra_features(
        close_s,
        high_s,
        low_s,
        open_s,
        volume_s,
        candles,
        last_close,
    )

    # Wavelet energy features from log returns (no look-ahead: uses only past data)
    log_returns = list(np.diff(np.log(np.array(closes, dtype=float))))
    wavelet = _compute_wavelet_features(log_returns)

    # Z-score features (relative strength / normalized indicators)
    zscore = _compute_zscore_features(close_s, high_s, low_s, volume_s)

    # Calendar features (cyclical encoding of day-of-week and month)
    calendar = _compute_calendar_features(candles[-1].timestamp)

    # Regime features (VIX + realized volatility ratio)
    regime = _compute_regime_features(close_s, _vix)

    # Cross-asset features (relative strength vs benchmark)
    benchmark_close_s = None
    if _benchmark:
        benchmark_close_s = pd.Series(
            [float(c.close) for c in _benchmark],
            dtype=float,
        )
    cross_asset = _compute_cross_asset_features(close_s, benchmark_close_s)

    # MOEX-specific features (FX, commodity, macro, turnover)
    candle_timestamps = [c.timestamp for c in candles]
    fx_features = _compute_fx_features(_moex)
    commodity_features = _compute_commodity_features(_moex)
    macro_features = _compute_macro_features(_moex, candle_timestamps=candle_timestamps)
    turnover_features = _compute_turnover_features(_moex)
    cbr_features = _compute_cbr_features(_moex, candle_timestamps=candle_timestamps)
    fx_return_features = _compute_fx_return_features(_moex)
    brent_return_features = _compute_brent_return_features(_moex)

    all_features = {
        **core,
        **extra,
        **wavelet,
        **zscore,
        **calendar,
        **regime,
        **cross_asset,
        **fx_features,
        **commodity_features,
        **macro_features,
        **turnover_features,
        **cbr_features,
        **fx_return_features,
        **brent_return_features,
    }

    feature_df = pd.DataFrame({k: [v] for k, v in all_features.items()})
    # Safety net: replace any remaining NaN/inf with 0 (feature-specific defaults above)
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan).fillna(0)

    return {col: float(feature_df[col].iloc[0]) for col in feature_df.columns}


def _warn_if_split_suspected(close_s: pd.Series) -> None:
    """Log a warning if a single-bar return exceeds the split threshold (6C.8)."""
    pct_changes = close_s.pct_change().abs()
    max_pct_change = float(pct_changes.max()) if not pct_changes.empty else 0.0
    if max_pct_change > _SPLIT_WARNING_THRESHOLD:
        _log.warning(
            "Suspicious single-bar return %.1f%% detected in candle window "
            "(possible stock split or corporate action)",
            max_pct_change * 100,
        )


def _compute_core_features(
    close_s: pd.Series,
    high_s: pd.Series,
    low_s: pd.Series,
    volume_s: pd.Series,
    last_close: float,
) -> dict[str, float]:
    """Compute the original 5 core features (RSI, MACD, BB, vol ratio, ATR)."""
    # RSI-14
    rsi = ta.rsi(close_s, length=14)
    rsi_val = float(rsi.iloc[-1]) if rsi is not None and not rsi.empty else 50.0

    # MACD histogram (6C.2: normalized by price)
    macd_df = ta.macd(close_s, fast=12, slow=26, signal=9)
    macd_hist_raw = 0.0
    if macd_df is not None and not macd_df.empty:
        hist_col = [c for c in macd_df.columns if "h" in c.lower()]
        if hist_col:
            macd_hist_raw = float(macd_df[hist_col[0]].iloc[-1])
    macd_hist_pct = macd_hist_raw / last_close if last_close > 0 else 0.0

    # Bollinger %B
    bb = ta.bbands(close_s, length=20, std=2.0)  # type: ignore[arg-type]
    bb_pct_b = 0.5
    if bb is not None and not bb.empty:
        pct_cols = [c for c in bb.columns if "P" in c]
        if pct_cols:
            bb_pct_b = float(bb[pct_cols[0]].iloc[-1])

    # Volume ratio (current vs 20-day average excluding current bar -- no look-ahead).
    prior_vol_mean = volume_s.shift(1).rolling(20).mean()
    last_prior_mean = float(prior_vol_mean.iloc[-1])
    volume_ratio = float(volume_s.iloc[-1] / last_prior_mean) if last_prior_mean > 0 else 1.0

    # ATR-14 (6C.2: normalized by price)
    atr = ta.atr(high_s, low_s, close_s, length=14)
    atr_val = float(atr.iloc[-1]) if atr is not None and not atr.empty else 0.0
    atr_pct = atr_val / last_close if last_close > 0 else 0.0

    return {
        "rsi_14": rsi_val,
        "macd_hist_pct": macd_hist_pct,
        "bb_pct_b": bb_pct_b,
        "volume_ratio_20d": volume_ratio,
        "atr_14_pct": atr_pct,
    }


def _compute_extra_features(
    close_s: pd.Series,
    high_s: pd.Series,
    low_s: pd.Series,
    open_s: pd.Series,
    volume_s: pd.Series,
    candles: list[Candle],
    last_close: float,
) -> dict[str, float]:
    """Compute extra features beyond the 5 core indicators."""
    closes = [float(c.close) for c in candles]

    # ROC(10)
    roc = ta.roc(close_s, length=_ROC_LENGTH)
    roc_val = float(roc.iloc[-1]) if roc is not None and not roc.empty else 0.0

    # Williams %R(14)
    willr = ta.willr(high_s, low_s, close_s, length=14)
    willr_val = float(willr.iloc[-1]) if willr is not None and not willr.empty else -50.0

    # ADX(14)
    adx_df = ta.adx(high_s, low_s, close_s, length=14)
    adx_val = 0.0
    if adx_df is not None and not adx_df.empty:
        adx_cols = [c for c in adx_df.columns if "ADX" in c and "DM" not in c]
        if adx_cols:
            adx_val = float(adx_df[adx_cols[0]].iloc[-1])

    # Historical volatility (20): stdev of returns
    returns = close_s.pct_change()
    hist_vol = ta.stdev(returns, length=20)
    hist_vol_val = float(hist_vol.iloc[-1]) if hist_vol is not None and not hist_vol.empty else 0.0

    # Garman-Klass volatility (20)
    gk_vol_val = _garman_klass_vol(open_s, high_s, low_s, close_s, length=20)

    # OBV slope (10), normalized by volume mean
    obv = ta.obv(close_s, volume_s)
    obv_slope_val = 0.0
    if obv is not None and len(obv) >= _OBV_SLOPE_LENGTH:
        obv_recent = obv.iloc[-_OBV_SLOPE_LENGTH:]
        slope = float(obv_recent.iloc[-1] - obv_recent.iloc[0])
        vol_mean = float(volume_s.mean())
        obv_slope_val = slope / vol_mean if vol_mean > 0 else 0.0

    # RSI divergence: OLS-slope based detection of price-RSI directional disagreement
    rsi = ta.rsi(close_s, length=14)
    rsi_divergence = 0.0
    if rsi is not None and len(rsi) >= _RSI_LOOKBACK and len(closes) >= _RSI_LOOKBACK:
        price_returns = close_s.pct_change().iloc[-_RSI_LOOKBACK:].dropna()
        rsi_changes = rsi.diff().iloc[-_RSI_LOOKBACK:].dropna()
        if len(price_returns) >= 2 and len(rsi_changes) >= 2:  # noqa: PLR2004
            # OLS slope via np.polyfit (degree=1) over the lookback window
            pr_vals = price_returns.to_numpy(dtype=float)
            rc_vals = rsi_changes.to_numpy(dtype=float)
            pr_x = np.arange(len(pr_vals), dtype=_OLS_INDICES_DTYPE)
            rc_x = np.arange(len(rc_vals), dtype=_OLS_INDICES_DTYPE)
            price_slope = float(np.polyfit(pr_x, pr_vals, 1)[0])
            rsi_slope_raw = float(np.polyfit(rc_x, rc_vals, 1)[0])
            # Normalize slopes to z-scores for comparability
            pr_std = float(price_returns.std())
            rc_std = float(rsi_changes.std())
            if (
                pr_std > 0
                and rc_std > 0
                and math.isfinite(pr_std)
                and math.isfinite(rc_std)
                and math.isfinite(price_slope)
                and math.isfinite(rsi_slope_raw)
            ):
                price_z = price_slope / pr_std
                rsi_z = rsi_slope_raw / rc_std
                rsi_divergence = _compute_rsi_divergence(price_z, rsi_z)

    predictive = _compute_predictive_features(close_s, closes, returns)
    microstructure = _compute_microstructure_features(close_s, high_s, low_s, volume_s, last_close)

    return {
        "roc_10": roc_val,
        "willr_14": willr_val,
        "adx_14": adx_val,
        "hist_vol_20": hist_vol_val,
        "gk_vol_20": gk_vol_val,
        "obv_slope_10": obv_slope_val,
        "rsi_divergence": rsi_divergence,
        **predictive,
        **microstructure,
    }


def _compute_predictive_features(
    close_s: pd.Series,
    closes: list[float],
    returns: pd.Series,
) -> dict[str, float]:
    """Compute lagged returns, return distribution, and short-period RSI features."""
    # Lagged returns (most predictive feature class in financial ML)
    ret_1d = closes[-1] / closes[-2] - 1 if len(closes) >= _RET_1D_MIN else 0.0
    ret_5d = closes[-1] / closes[-6] - 1 if len(closes) >= _RET_5D_MIN else 0.0
    ret_21d = closes[-1] / closes[-22] - 1 if len(closes) >= _RET_21D_MIN else 0.0
    # Medium-term momentum (Gu, Kelly & Xiu 2020: top predictor class)
    ret_63d = closes[-1] / closes[-63] - 1 if len(closes) >= _RET_63D_MIN else 0.0
    ret_126d = closes[-1] / closes[-126] - 1 if len(closes) >= _RET_126D_MIN else 0.0
    # Short-term reversal relative to monthly momentum (mean-reversion within trends)
    mom_reversal_ratio = ret_5d / ret_21d if abs(ret_21d) > _MOM_RATIO_EPSILON else 0.0

    # Return distribution (Harvey & Siddique 2000)
    recent_returns = returns.iloc[-_RECENT_RETURN_WINDOW:].dropna()
    skew_20d = float(recent_returns.skew()) if len(recent_returns) >= _MIN_SKEW_BARS else 0.0  # type: ignore[arg-type]
    kurt_20d = float(recent_returns.kurtosis()) if len(recent_returns) >= _MIN_SKEW_BARS else 0.0  # type: ignore[arg-type]
    max_ret_20d = float(recent_returns.max()) if len(recent_returns) >= 1 else 0.0
    min_ret_20d = float(recent_returns.min()) if len(recent_returns) >= 1 else 0.0
    if not math.isfinite(skew_20d):
        skew_20d = 0.0
    if not math.isfinite(kurt_20d):
        kurt_20d = 0.0

    # Short-period RSI (Connors RSI family)
    rsi_2 = ta.rsi(close_s, length=2)
    rsi_2_val = float(rsi_2.iloc[-1]) if rsi_2 is not None and not rsi_2.empty else 50.0
    if not math.isfinite(rsi_2_val):
        rsi_2_val = 50.0
    rsi_5 = ta.rsi(close_s, length=5)
    rsi_5_val = float(rsi_5.iloc[-1]) if rsi_5 is not None and not rsi_5.empty else 50.0
    if not math.isfinite(rsi_5_val):
        rsi_5_val = 50.0

    return {
        "ret_1d": ret_1d,
        "ret_5d": ret_5d,
        "ret_21d": ret_21d,
        "ret_63d": ret_63d,
        "ret_126d": ret_126d,
        "mom_reversal_ratio": mom_reversal_ratio,
        "skew_20d": skew_20d,
        "kurt_20d": kurt_20d,
        "max_ret_20d": max_ret_20d,
        "min_ret_20d": min_ret_20d,
        "rsi_2": rsi_2_val,
        "rsi_5": rsi_5_val,
    }


def _compute_amihud_series(
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


def _compute_microstructure_features(
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
    amihud_series = _compute_amihud_series(close_s, volume_s, window=_AMIHUD_WINDOW)
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
        "corwin_schultz_spread": _corwin_schultz(high_s, low_s),
    }


def _safe_zscore(value: float, mean: float, std: float) -> float:
    """Compute z-score, returning 0.0 when std is zero or non-finite."""
    if std <= 0.0 or not math.isfinite(std):
        return 0.0
    z = (value - mean) / std
    return z if math.isfinite(z) else 0.0


def _compute_zscore_features(
    close_s: pd.Series,
    high_s: pd.Series,
    low_s: pd.Series,
    volume_s: pd.Series,
) -> dict[str, float]:
    """Compute z-score normalized features for relative strength analysis.

    All windows use min_periods=1 so short series degrade gracefully.
    No look-ahead bias: rolling windows use only past data.
    """
    # Price z-score: (close - SMA60) / std60
    price_mean = float(close_s.rolling(_ZSCORE_WINDOW, min_periods=1).mean().iloc[-1])
    price_std = float(close_s.rolling(_ZSCORE_WINDOW, min_periods=1).std().iloc[-1])
    price_zscore = _safe_zscore(float(close_s.iloc[-1]), price_mean, price_std)

    # Volume z-score: (volume - vol_mean_20) / vol_std_20
    vol_mean = float(volume_s.rolling(_VOLUME_ZSCORE_WINDOW, min_periods=1).mean().iloc[-1])
    vol_std = float(volume_s.rolling(_VOLUME_ZSCORE_WINDOW, min_periods=1).std().iloc[-1])
    vol_zscore = _safe_zscore(float(volume_s.iloc[-1]), vol_mean, vol_std)

    # RSI z-score: (RSI14 - mean_RSI14_60d) / std_RSI14_60d
    rsi_series = ta.rsi(close_s, length=_RSI_LOOKBACK)
    rsi_zscore = 0.0
    if rsi_series is not None and not rsi_series.empty:
        rsi_mean = float(rsi_series.rolling(_ZSCORE_WINDOW, min_periods=1).mean().iloc[-1])
        rsi_std = float(rsi_series.rolling(_ZSCORE_WINDOW, min_periods=1).std().iloc[-1])
        rsi_val = float(rsi_series.iloc[-1])
        if math.isfinite(rsi_val):
            rsi_zscore = _safe_zscore(rsi_val, rsi_mean, rsi_std)

    # ATR z-score: (ATR14 - mean_ATR_60d) / std_ATR_60d
    atr_series = ta.atr(high_s, low_s, close_s, length=_RSI_LOOKBACK)
    atr_zscore = 0.0
    if atr_series is not None and not atr_series.empty:
        atr_mean = float(atr_series.rolling(_ZSCORE_WINDOW, min_periods=1).mean().iloc[-1])
        atr_std = float(atr_series.rolling(_ZSCORE_WINDOW, min_periods=1).std().iloc[-1])
        atr_val = float(atr_series.iloc[-1])
        if math.isfinite(atr_val):
            atr_zscore = _safe_zscore(atr_val, atr_mean, atr_std)

    return {
        "price_zscore_60d": price_zscore,
        "volume_zscore_20d": vol_zscore,
        "rsi_zscore_60d": rsi_zscore,
        "atr_zscore_60d": atr_zscore,
    }


def _corwin_schultz(high_s: pd.Series, low_s: pd.Series) -> float:
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


def _garman_klass_vol(
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
