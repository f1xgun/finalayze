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
    from finalayze.core.schemas import Candle

_MIN_CANDLES = 30
_SPLIT_WARNING_THRESHOLD = 0.40

# Lookback lengths for new indicators
_ROC_LENGTH = 10
_OBV_SLOPE_LENGTH = 10
_RSI_LOOKBACK = 14
_DOW_DIVISOR = 5
_MIN_SMA_POINTS = 2
_RSI_SCALE = 100.0

_log = logging.getLogger(__name__)


def compute_features(candles: list[Candle], sentiment_score: float = 0.0) -> dict[str, float]:
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
    extra = _compute_extra_features(close_s, high_s, low_s, open_s, volume_s, candles, last_close)

    all_features = {**core, **extra, "sentiment": sentiment_score}

    feature_df = pd.DataFrame({k: [v] for k, v in all_features.items()})
    feature_df = feature_df.ffill().bfill().fillna(0)

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
    """Compute the 10 new features added in 6C.1."""
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

    # MA slope (20-bar SMA), normalized by price
    sma_20 = ta.sma(close_s, length=20)
    ma_slope = 0.0
    if sma_20 is not None and len(sma_20) >= _MIN_SMA_POINTS:
        sma_curr = float(sma_20.iloc[-1])
        sma_prev = float(sma_20.iloc[-2])
        ma_slope = (sma_curr - sma_prev) / last_close if last_close > 0 else 0.0

    # Historical volatility (20): stdev of returns
    returns = close_s.pct_change()
    hist_vol = ta.stdev(returns, length=20)
    hist_vol_val = float(hist_vol.iloc[-1]) if hist_vol is not None and not hist_vol.empty else 0.0

    # Garman-Klass volatility (20)
    gk_vol_val = _garman_klass_vol(open_s, high_s, low_s, close_s, length=20)

    # Day-of-week cyclical encoding
    last_ts = candles[-1].timestamp
    dow = last_ts.weekday()  # 0=Monday, 4=Friday
    dow_sin = math.sin(2 * math.pi * dow / _DOW_DIVISOR)
    dow_cos = math.cos(2 * math.pi * dow / _DOW_DIVISOR)

    # OBV slope (10), normalized by volume mean
    obv = ta.obv(close_s, volume_s)
    obv_slope_val = 0.0
    if obv is not None and len(obv) >= _OBV_SLOPE_LENGTH:
        obv_recent = obv.iloc[-_OBV_SLOPE_LENGTH:]
        slope = float(obv_recent.iloc[-1] - obv_recent.iloc[0])
        vol_mean = float(volume_s.mean())
        obv_slope_val = slope / vol_mean if vol_mean > 0 else 0.0

    # RSI divergence: difference between price ROC and RSI ROC over 14 bars
    rsi = ta.rsi(close_s, length=14)
    rsi_divergence = 0.0
    if rsi is not None and len(rsi) >= _RSI_LOOKBACK:
        prev_close = closes[-_RSI_LOOKBACK]
        price_roc_14 = (closes[-1] - prev_close) / prev_close if prev_close != 0 else 0.0
        rsi_roc_14 = (float(rsi.iloc[-1]) - float(rsi.iloc[-_RSI_LOOKBACK])) / _RSI_SCALE
        rsi_divergence = price_roc_14 - rsi_roc_14

    return {
        "roc_10": roc_val,
        "willr_14": willr_val,
        "adx_14": adx_val,
        "ma_slope_20": ma_slope,
        "hist_vol_20": hist_vol_val,
        "gk_vol_20": gk_vol_val,
        "dow_sin": dow_sin,
        "dow_cos": dow_cos,
        "obv_slope_10": obv_slope_val,
        "rsi_divergence": rsi_divergence,
    }


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
