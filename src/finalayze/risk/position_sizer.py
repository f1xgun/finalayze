"""Half-Kelly position sizing (Layer 4).

See docs/architecture/DEPENDENCY_LAYERS.md for layering rules.
"""

from __future__ import annotations

from decimal import Decimal


def compute_position_size(
    win_rate: float,
    avg_win_ratio: Decimal,
    equity: Decimal,
    kelly_fraction: float = 0.5,
    max_position_pct: float = 0.20,
) -> Decimal:
    """Compute position size using Half-Kelly criterion.

    Formula:
        f* = (win_rate * avg_win_ratio - (1 - win_rate)) / avg_win_ratio
        position = equity * f* * kelly_fraction, capped at max_position_pct.

    Args:
        win_rate: Historical win rate (0..1).
        avg_win_ratio: Average win / average loss ratio.
        equity: Current portfolio equity.
        kelly_fraction: Fraction of full Kelly to use (default 0.5 = half-Kelly).
        max_position_pct: Maximum position as fraction of equity (default 20%).

    Returns:
        Position size in currency units, or zero if Kelly is non-positive.
    """
    if equity <= 0 or avg_win_ratio <= 0:
        return Decimal(0)

    loss_rate = 1.0 - win_rate
    kelly_f = (win_rate * float(avg_win_ratio) - loss_rate) / float(avg_win_ratio)

    if kelly_f <= 0:
        return Decimal(0)

    half_kelly = kelly_f * kelly_fraction
    position = equity * Decimal(str(half_kelly))
    max_position = equity * Decimal(str(max_position_pct))

    return min(position, max_position)
