"""Volatility-targeting utility for momentum strategies (Layer 4).

Scales signal confidence by target_vol / realized_vol so that
high-volatility regimes reduce position sizing and low-volatility
regimes increase it.
"""

from __future__ import annotations

import math

_CLAMP_MIN = 0.0
_CLAMP_MAX = 2.0


def compute_vol_scale(
    closes: list[float],
    lookback: int = 126,
    target_vol: float = 0.15,
    annualization: float = 252.0,
) -> float:
    """Compute volatility scaling factor: target_vol / realized_vol.

    Args:
        closes: List of closing prices (oldest first).
        lookback: Number of returns to use for realized vol calculation.
        target_vol: Annualized target volatility (e.g. 0.15 = 15%).
        annualization: Trading days per year for annualization.

    Returns:
        Scale factor clamped to [0.0, 2.0].
        Returns 1.0 if insufficient data (< lookback + 1 prices).
    """
    if len(closes) < lookback + 1:
        return 1.0

    # Compute daily log returns
    returns = [
        math.log(closes[i] / closes[i - 1])
        for i in range(len(closes) - lookback, len(closes))
        if closes[i - 1] > 0 and closes[i] > 0
    ]

    if len(returns) < 2:  # noqa: PLR2004
        return 1.0

    # Realized vol = std(returns) * sqrt(annualization)
    mean_ret = sum(returns) / len(returns)
    variance = sum((r - mean_ret) ** 2 for r in returns) / (len(returns) - 1)
    realized_vol = math.sqrt(variance) * math.sqrt(annualization)

    if realized_vol <= 0:
        return _CLAMP_MAX

    scale = target_vol / realized_vol
    return max(_CLAMP_MIN, min(_CLAMP_MAX, scale))
