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
from finalayze.core.schemas import MarketContext
from finalayze.ml.features.calendar import compute_calendar_features
from finalayze.ml.features.cross_asset import compute_cross_asset_features
from finalayze.ml.features.fundamental import compute_fundamental_features
from finalayze.ml.features.macro import (
    EXTERNAL_DATA_LAG_BARS as _EXTERNAL_DATA_LAG_BARS,  # noqa: F401  # re-export
)
from finalayze.ml.features.macro import (
    compute_cbr_features,
    compute_macro_features,
)
from finalayze.ml.features.microstructure import (
    compute_microstructure_features,
    garman_klass_vol,
)
from finalayze.ml.features.moex_external import (
    compute_brent_return_features,
    compute_commodity_features,
    compute_fx_features,
    compute_fx_return_features,
    compute_turnover_features,
)
from finalayze.ml.features.regime import compute_regime_features
from finalayze.ml.features.wavelet import compute_wavelet_features
from finalayze.ml.features.zscore import compute_zscore_features

if TYPE_CHECKING:
    from finalayze.core.schemas import Candle

_MIN_CANDLES = 80
_MAX_FEATURE_LOOKBACK = 252
_SPLIT_WARNING_THRESHOLD = 0.40

# Lookback lengths for indicators
_ROC_LENGTH = 10
_OBV_SLOPE_LENGTH = 10
_RSI_LOOKBACK = 14

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

_OLS_INDICES_DTYPE = float  # np.arange dtype for OLS slope computation

_log = logging.getLogger(__name__)


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
    wavelet = compute_wavelet_features(log_returns)

    # Z-score features (relative strength / normalized indicators)
    zscore = compute_zscore_features(close_s, high_s, low_s, volume_s)

    # Calendar features (cyclical encoding of day-of-week and month)
    calendar = compute_calendar_features(candles[-1].timestamp)

    # Regime features (VIX + realized volatility ratio)
    regime = compute_regime_features(close_s, _vix)

    # Cross-asset features (relative strength vs benchmark)
    benchmark_close_s = None
    if _benchmark:
        benchmark_close_s = pd.Series(
            [float(c.close) for c in _benchmark],
            dtype=float,
        )
    cross_asset = compute_cross_asset_features(close_s, benchmark_close_s)

    # MOEX-specific features (FX, commodity, macro, turnover)
    candle_timestamps = [c.timestamp for c in candles]
    fx_features = compute_fx_features(_moex)
    commodity_features = compute_commodity_features(_moex)
    macro_features = compute_macro_features(_moex, candle_timestamps=candle_timestamps)
    # Pass the symbol being scored so segment-wide fundamentals are attributed to
    # the right ticker, not the globally-latest snapshot (audit 2026-06-28).
    fundamental_features = compute_fundamental_features(
        _moex, as_of=candles[-1].timestamp, symbol=candles[-1].symbol
    )
    turnover_features = compute_turnover_features(_moex)
    cbr_features = compute_cbr_features(_moex, candle_timestamps=candle_timestamps)
    fx_return_features = compute_fx_return_features(_moex)
    brent_return_features = compute_brent_return_features(_moex)

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
        **fundamental_features,
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
    gk_vol_val = garman_klass_vol(open_s, high_s, low_s, close_s, length=20)

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
    microstructure = compute_microstructure_features(close_s, high_s, low_s, volume_s, last_close)

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
