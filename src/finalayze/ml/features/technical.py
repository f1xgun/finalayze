"""Technical feature engineering for ML models (Layer 3)."""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pandas_ta as ta

from finalayze.core.exceptions import InsufficientDataError

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
_MIN_CS_BARS = 2
_WAVELET_LEVEL = 3
_MIN_WAVELET_SAMPLES = 16  # pywt.wavedec('db4', level=3) needs at least 2^level samples

# Lagged return lookback thresholds (minimum bars needed)
_RET_1D_MIN = 2
_RET_5D_MIN = 6
_RET_21D_MIN = 22

# Return distribution constants
_RECENT_RETURN_WINDOW = 20
_MIN_SKEW_BARS = 5

# Z-score window lengths
_ZSCORE_WINDOW = 60
_VOLUME_ZSCORE_WINDOW = 20

# Calendar cyclical encoding constants
_TRADING_DAYS_PER_WEEK = 5
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

_log = logging.getLogger(__name__)

try:
    import pywt as _pywt

    _HAS_PYWT = True
except ImportError:  # pragma: no cover
    _pywt = None
    _HAS_PYWT = False


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
    Encodes day-of-week and month as sin/cos pairs for cyclical continuity.
    """
    dow = last_timestamp.weekday()  # 0=Monday, 4=Friday
    month = last_timestamp.month  # 1-12

    two_pi = 2.0 * math.pi
    return {
        "dow_sin": math.sin(two_pi * dow / _TRADING_DAYS_PER_WEEK),
        "dow_cos": math.cos(two_pi * dow / _TRADING_DAYS_PER_WEEK),
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


def compute_features(
    candles: list[Candle],
    sentiment_score: float = 0.0,  # noqa: ARG001 — kept for backward compatibility
    benchmark_candles: list[Candle] | None = None,
    vix_candles: list[Candle] | None = None,
) -> dict[str, float]:
    """Compute technical features from a list of candles.

    Args:
        candles: OHLCV candles sorted ascending by timestamp.
        sentiment_score: External sentiment score in [-1.0, 1.0].

    Returns:
        Dict of feature name -> float value.

    Raises:
        InsufficientDataError: When fewer than 30 candles are provided.
    """
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
    regime = _compute_regime_features(close_s, vix_candles)

    # Cross-asset features (relative strength vs benchmark)
    benchmark_close_s = None
    if benchmark_candles:
        benchmark_close_s = pd.Series(
            [float(c.close) for c in benchmark_candles],
            dtype=float,
        )
    cross_asset = _compute_cross_asset_features(close_s, benchmark_close_s)

    all_features = {
        **core,
        **extra,
        **wavelet,
        **zscore,
        **calendar,
        **regime,
        **cross_asset,
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

    # RSI divergence: z-score normalized price vs RSI changes over 14 bars
    rsi = ta.rsi(close_s, length=14)
    rsi_divergence = 0.0
    if rsi is not None and len(rsi) >= _RSI_LOOKBACK and len(closes) >= _RSI_LOOKBACK:
        price_returns = close_s.pct_change().iloc[-_RSI_LOOKBACK:]
        rsi_changes = rsi.diff().iloc[-_RSI_LOOKBACK:]
        # Z-score normalize
        pr_std = float(price_returns.std())
        rc_std = float(rsi_changes.std())
        if pr_std > 0 and rc_std > 0 and math.isfinite(pr_std) and math.isfinite(rc_std):
            price_z = float(price_returns.iloc[-1]) / pr_std
            rsi_z = float(rsi_changes.iloc[-1]) / rc_std
            rsi_divergence = price_z - rsi_z

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

    # Return distribution (Harvey & Siddique 2000)
    recent_returns = returns.iloc[-_RECENT_RETURN_WINDOW:].dropna()
    skew_20d = float(recent_returns.skew()) if len(recent_returns) >= _MIN_SKEW_BARS else 0.0
    kurt_20d = float(recent_returns.kurtosis()) if len(recent_returns) >= _MIN_SKEW_BARS else 0.0
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
        "skew_20d": skew_20d,
        "kurt_20d": kurt_20d,
        "max_ret_20d": max_ret_20d,
        "min_ret_20d": min_ret_20d,
        "rsi_2": rsi_2_val,
        "rsi_5": rsi_5_val,
    }


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

    # Amihud illiquidity ratio (20-day rolling mean), log-transformed for normalization
    dollar_volume = close_s * volume_s
    abs_returns = close_s.pct_change().abs()
    illiq_per_bar = abs_returns / dollar_volume
    illiq_per_bar = illiq_per_bar.replace([np.inf, -np.inf], np.nan)
    amihud_rolling = illiq_per_bar.rolling(_AMIHUD_WINDOW, min_periods=1).mean()
    amihud_val = float(amihud_rolling.iloc[-1])
    amihud_20d = math.log1p(amihud_val * 1e6) if math.isfinite(amihud_val) else 0.0

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
