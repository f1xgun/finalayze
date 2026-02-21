"""ATR-based stop-loss calculation (Layer 4).

See docs/architecture/DEPENDENCY_LAYERS.md for layering rules.
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

import pandas as pd  # type: ignore[import-untyped]

if TYPE_CHECKING:
    from finalayze.core.schemas import Candle


def compute_atr_stop_loss(
    entry_price: Decimal,
    candles: list[Candle],
    atr_period: int = 14,
    atr_multiplier: Decimal = Decimal("2.0"),
) -> Decimal | None:
    """Compute ATR-based stop-loss price.

    Formula:
        stop_loss = entry_price - ATR(period) * multiplier

    Args:
        entry_price: The entry price for the position.
        candles: Historical OHLCV candles (oldest first).
        atr_period: Number of periods for ATR calculation.
        atr_multiplier: Multiplier applied to ATR for the stop distance.

    Returns:
        Stop-loss price, or ``None`` if insufficient data.
    """
    if len(candles) < atr_period + 1:
        return None

    highs = pd.Series([float(c.high) for c in candles])
    lows = pd.Series([float(c.low) for c in candles])
    closes = pd.Series([float(c.close) for c in candles])

    tr1 = highs - lows
    tr2 = (highs - closes.shift(1)).abs()
    tr3 = (lows - closes.shift(1)).abs()

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=atr_period).mean().iloc[-1]

    if pd.isna(atr):
        return None

    return entry_price - Decimal(str(atr)) * atr_multiplier
