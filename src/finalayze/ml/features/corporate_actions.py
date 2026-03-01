"""Corporate action detection for candle data (Layer 3).

Detects suspected stock splits and reverse-splits based on price gaps,
and provides backward adjustment to smooth the price series.
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from finalayze.core.schemas import Candle

_SPLIT_THRESHOLD = 0.40  # 40% single-bar move likely indicates split/reverse-split


def detect_splits(candles: list[Candle]) -> list[int]:
    """Return indices where a suspected split/reverse-split occurred.

    A split is detected when the close-to-close return exceeds the threshold
    AND the high-low range of the suspect bar is small relative to the gap
    (ruling out genuine crash/rally bars).
    """
    suspect_indices: list[int] = []
    for i in range(1, len(candles)):
        prev_close = float(candles[i - 1].close)
        if prev_close == 0:
            continue
        ret = abs(float(candles[i].close) - prev_close) / prev_close
        bar_range = abs(float(candles[i].high) - float(candles[i].low))
        gap = abs(float(candles[i].close) - prev_close)
        # Split: large gap but small intraday range (price just shifted)
        if ret > _SPLIT_THRESHOLD and bar_range < gap * 0.5:
            suspect_indices.append(i)
    return suspect_indices


def adjust_for_splits(candles: list[Candle]) -> list[Candle]:
    """Return candles with suspected splits adjusted via ratio.

    Uses backward adjustment: multiply all bars before the split by
    the ratio new_close / old_close. Iterates splits in reverse order
    so that adjustments compound correctly.
    """
    splits = detect_splits(candles)
    if not splits:
        return list(candles)

    # Work with mutable copies
    adjusted = list(candles)

    # Process splits in reverse order (rightmost first)
    for split_idx in sorted(splits, reverse=True):
        prev_close = float(adjusted[split_idx - 1].close)
        if prev_close == 0:
            continue
        new_close = float(adjusted[split_idx].close)
        ratio = new_close / prev_close

        # Adjust all bars before the split
        for i in range(split_idx):
            c = adjusted[i]
            adjusted[i] = type(c).model_validate(
                {
                    "symbol": c.symbol,
                    "market_id": c.market_id,
                    "timeframe": c.timeframe,
                    "timestamp": c.timestamp,
                    "open": Decimal(str(float(c.open) * ratio)),
                    "high": Decimal(str(float(c.high) * ratio)),
                    "low": Decimal(str(float(c.low) * ratio)),
                    "close": Decimal(str(float(c.close) * ratio)),
                    "volume": c.volume,
                    "source": c.source,
                }
            )

    return adjusted
