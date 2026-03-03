"""Commodity-currency premium signal for MOEX energy exporters (Layer 4)."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from finalayze.core.schemas import Candle


_DEFAULT_WINDOW = 20  # trading days
_DEFAULT_THRESHOLD = 0.05  # 5% premium triggers signal
_MIN_CANDLES = 2


def compute_commodity_currency_premium(
    oil_candles: list[Candle],
    rub_candles: list[Candle],
    window: int = _DEFAULT_WINDOW,
) -> float:
    """Compute commodity-currency premium over a rolling window.

    premium = oil_return - rub_depreciation
    (rub_depreciation is positive when RUB weakens = USDRUB rises)

    When premium > 0: oil gained more than RUB weakened -> exporters undervalued.
    When premium < 0: RUB weakened more than oil gained -> exporters overvalued.

    Args:
        oil_candles: Brent oil candles, chronological.
        rub_candles: USDRUB candles, chronological. Rising = RUB weakening.
        window: Number of bars for return computation.

    Returns:
        Premium value (positive = exporters undervalued).
        Returns 0.0 if insufficient data.
    """
    # Need window+1 candles to compute return over window
    if len(oil_candles) < window + 1 or len(rub_candles) < window + 1:
        return 0.0

    oil_start = float(oil_candles[-(window + 1)].close)
    oil_end = float(oil_candles[-1].close)
    rub_start = float(rub_candles[-(window + 1)].close)
    rub_end = float(rub_candles[-1].close)

    if oil_start <= 0 or rub_start <= 0:
        return 0.0

    oil_return = (oil_end - oil_start) / oil_start
    # RUB depreciation: positive when USDRUB rises (RUB weakens)
    rub_depreciation = (rub_end - rub_start) / rub_start

    premium = oil_return - rub_depreciation

    if not math.isfinite(premium):
        return 0.0

    return premium


def is_energy_undervalued(
    oil_candles: list[Candle],
    rub_candles: list[Candle],
    window: int = _DEFAULT_WINDOW,
    threshold: float = _DEFAULT_THRESHOLD,
) -> bool:
    """Check if MOEX energy exporters appear undervalued.

    Returns True when commodity-currency premium exceeds threshold.
    """
    premium = compute_commodity_currency_premium(oil_candles, rub_candles, window)
    return premium > threshold


def compute_premium_confidence_boost(
    oil_candles: list[Candle],
    rub_candles: list[Candle],
    window: int = _DEFAULT_WINDOW,
    threshold: float = _DEFAULT_THRESHOLD,
    max_boost: float = 0.10,
) -> float:
    """Compute a confidence boost for energy BUY signals based on premium.

    Returns a boost in [0, max_boost] proportional to how far premium exceeds threshold.
    Returns 0.0 if premium is below threshold.
    """
    premium = compute_commodity_currency_premium(oil_candles, rub_candles, window)
    if premium <= threshold:
        return 0.0

    excess = premium - threshold
    # Linear scaling: at 1x threshold excess, reach max_boost
    scale = min(1.0, excess / threshold) if threshold > 0 else 1.0
    return scale * max_boost
