"""Half-Kelly position sizing (Layer 4).

See docs/architecture/DEPENDENCY_LAYERS.md for layering rules.
"""

from __future__ import annotations

from decimal import Decimal


def compute_position_size(
    win_rate: Decimal,
    avg_win_ratio: Decimal,
    equity: Decimal,
    kelly_fraction: Decimal = Decimal("0.5"),
    max_position_pct: Decimal = Decimal("0.20"),
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
    if avg_win_ratio <= 0 or win_rate <= 0:
        return Decimal(0)

    b = avg_win_ratio
    f_star = (win_rate * b - (Decimal(1) - win_rate)) / b

    half_kelly = f_star * kelly_fraction
    if half_kelly <= 0:
        return Decimal(0)

    return min(equity * half_kelly, equity * max_position_pct)
