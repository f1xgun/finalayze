"""Technical feature engineering for ML models (Layer 3)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import pandas_ta as ta

from finalayze.core.exceptions import InsufficientDataError

if TYPE_CHECKING:
    from finalayze.core.schemas import Candle

_MIN_CANDLES = 30


def compute_features(candles: list[Candle], sentiment_score: float = 0.0) -> dict[str, float]:
    """Compute technical features from a list of candles.

    Args:
        candles: OHLCV candles sorted ascending by timestamp.
        sentiment_score: External sentiment score in [-1.0, 1.0].

    Returns:
        Dict of feature name → float value.

    Raises:
        InsufficientDataError: When fewer than 30 candles are provided.
    """
    if len(candles) < _MIN_CANDLES:
        msg = f"Need at least {_MIN_CANDLES} candles, got {len(candles)}"
        raise InsufficientDataError(msg)

    closes = [float(c.close) for c in candles]
    highs = [float(c.high) for c in candles]
    lows = [float(c.low) for c in candles]
    volumes = [float(c.volume) for c in candles]

    close_s = pd.Series(closes, dtype=float)
    high_s = pd.Series(highs, dtype=float)
    low_s = pd.Series(lows, dtype=float)
    volume_s = pd.Series(volumes, dtype=float)

    # RSI-14
    rsi = ta.rsi(close_s, length=14)
    rsi_val = float(rsi.iloc[-1]) if rsi is not None and not rsi.empty else 50.0

    # MACD histogram
    macd_df = ta.macd(close_s, fast=12, slow=26, signal=9)
    macd_hist = 0.0
    if macd_df is not None and not macd_df.empty:
        hist_col = [c for c in macd_df.columns if "h" in c.lower()]
        if hist_col:
            macd_hist = float(macd_df[hist_col[0]].iloc[-1])

    # Bollinger %B
    bb = ta.bbands(close_s, length=20, std=2.0)  # type: ignore[arg-type]
    bb_pct_b = 0.5
    if bb is not None and not bb.empty:
        pct_cols = [c for c in bb.columns if "P" in c]
        if pct_cols:
            bb_pct_b = float(bb[pct_cols[0]].iloc[-1])

    # Volume ratio (current vs 20-day average)
    vol_mean = volume_s.tail(20).mean()
    volume_ratio = float(volume_s.iloc[-1] / vol_mean) if vol_mean > 0 else 1.0

    # ATR-14
    atr = ta.atr(high_s, low_s, close_s, length=14)
    atr_val = float(atr.iloc[-1]) if atr is not None and not atr.empty else 0.0

    return {
        "rsi_14": rsi_val,
        "macd_hist": macd_hist,
        "bb_pct_b": bb_pct_b,
        "volume_ratio_20d": volume_ratio,
        "atr_14": atr_val,
        "sentiment": sentiment_score,
    }
