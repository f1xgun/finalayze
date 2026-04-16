"""ATR-based stop-loss calculation (Layer 4).

See docs/architecture/DEPENDENCY_LAYERS.md for layering rules.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from finalayze.core.schemas import Candle


def filter_candles_by_exclusion(
    candles: list[Candle],
    exclude_periods: tuple[tuple[str, str], ...],
) -> list[Candle]:
    """Remove candles whose date falls within any excluded period.

    Args:
        candles: List of OHLCV candles.
        exclude_periods: Tuple of (start_date_iso, end_date_iso) inclusive ranges.

    Returns:
        Filtered list with excluded-period candles removed.
    """
    if not exclude_periods:
        return candles

    parsed_ranges = [
        (date.fromisoformat(start), date.fromisoformat(end)) for start, end in exclude_periods
    ]

    result: list[Candle] = []
    for c in candles:
        candle_date = c.timestamp.date()
        excluded = any(start <= candle_date <= end for start, end in parsed_ranges)
        if not excluded:
            result.append(c)
    return result


def compute_atr_stop_loss(
    entry_price: Decimal,
    candles: list[Candle],
    atr_period: int = 14,
    atr_multiplier: Decimal = Decimal("2.0"),
    exclude_periods: tuple[tuple[str, str], ...] = (),
) -> Decimal | None:
    """Compute ATR-based stop-loss price.

    Formula:
        stop_loss = entry_price - ATR(period) * multiplier

    Args:
        entry_price: The entry price for the position.
        candles: Historical OHLCV candles (oldest first).
        atr_period: Number of periods for ATR calculation.
        atr_multiplier: Multiplier applied to ATR for the stop distance.
        exclude_periods: Date ranges to exclude from ATR computation.

    Returns:
        Stop-loss price, or ``None`` if insufficient data.
    """
    filtered = filter_candles_by_exclusion(candles, exclude_periods) if exclude_periods else candles

    if len(filtered) < atr_period + 1:
        return None

    recent = filtered[-(atr_period + 1) :]
    true_ranges: list[Decimal] = []
    for i in range(1, len(recent)):
        prev_close = recent[i - 1].close
        high = recent[i].high
        low = recent[i].low
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        true_ranges.append(tr)

    atr = sum(true_ranges, Decimal(0)) / Decimal(len(true_ranges))
    stop = entry_price - atr * atr_multiplier
    return max(stop, Decimal(0))
