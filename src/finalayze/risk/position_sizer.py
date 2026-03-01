"""Half-Kelly position sizing (Layer 4).

See docs/architecture/DEPENDENCY_LAYERS.md for layering rules.
"""

from __future__ import annotations

from decimal import Decimal

# Maximum win rate cap to prevent Kelly = infinity when win_rate = 1.0
_MAX_WIN_RATE = Decimal("0.99")


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

    # Cap win_rate at 0.99 to prevent Kelly = infinity when win_rate = 1.0
    win_rate = min(win_rate, _MAX_WIN_RATE)

    b = avg_win_ratio
    f_star = (win_rate * b - (Decimal(1) - win_rate)) / b

    half_kelly = f_star * kelly_fraction
    if half_kelly <= 0:
        return Decimal(0)

    return min(equity * half_kelly, equity * max_position_pct)


def compute_vol_adjusted_position_size(
    base_position: Decimal,
    target_vol: Decimal,
    asset_vol: Decimal,
    min_scale: Decimal = Decimal("0.25"),
    max_scale: Decimal = Decimal("2.0"),
) -> Decimal:
    """Scale position size by target_vol / asset_vol.

    Args:
        base_position: Position size from Kelly or other sizer (currency units).
        target_vol: Target annualized portfolio volatility (e.g., 0.15 for 15%).
        asset_vol: Realized annualized volatility of the asset.
        min_scale: Minimum scaling factor (floor).
        max_scale: Maximum scaling factor (cap).

    Returns:
        Adjusted position size.
    """
    if asset_vol <= 0:
        return base_position
    scale = target_vol / asset_vol
    scale = max(min_scale, min(max_scale, scale))
    return base_position * scale


def compute_realized_vol(candles: list, lookback: int = 20) -> Decimal | None:
    """Compute annualized realized volatility from daily log returns.

    Uses the last ``lookback`` candles. Returns None if insufficient data.

    Args:
        candles: List of Candle objects with a ``close`` attribute.
        lookback: Number of periods for volatility calculation.

    Returns:
        Annualized volatility as Decimal, or None if not enough data.
    """
    min_candles = lookback + 1
    if len(candles) < min_candles:
        return None
    import math
    import statistics as stats

    closes = [float(c.close) for c in candles[-(lookback + 1) :]]
    log_returns = [
        math.log(closes[i] / closes[i - 1])
        for i in range(1, len(closes))
        if closes[i - 1] > 0
    ]
    if len(log_returns) < 2:  # noqa: PLR2004
        return None
    daily_vol = stats.stdev(log_returns)
    annualized = daily_vol * math.sqrt(252)
    return Decimal(str(annualized))
