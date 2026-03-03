"""Square-root market impact model (Layer 5).

Implements the Almgren-style square-root impact model:
    slippage = daily_vol * sqrt(shares / adv) * impact_coeff

See docs/architecture/DEPENDENCY_LAYERS.md for layering rules.
"""

from __future__ import annotations

import math


def compute_market_impact(
    daily_vol: float,
    shares: float,
    adv: float,
    impact_coeff: float = 0.1,
) -> float:
    """Compute market impact slippage as a fraction of price.

    Args:
        daily_vol: Daily volatility as a fraction (e.g. 0.02 = 2%).
        shares: Number of shares in the order.
        adv: Average daily volume for the symbol.
        impact_coeff: Scaling coefficient (default 0.1).

    Returns:
        Slippage as a fraction (e.g. 0.001 = 10bps). Returns 0.0 if
        adv <= 0 or shares <= 0.
    """
    if adv <= 0 or shares <= 0:
        return 0.0
    return daily_vol * math.sqrt(shares / adv) * impact_coeff


def should_reject_trade(
    daily_vol: float,
    shares: float,
    adv: float,
    impact_coeff: float = 0.1,
    max_impact_bps: float = 50.0,
) -> bool:
    """Return True if estimated market impact exceeds the threshold.

    Args:
        daily_vol: Daily volatility as a fraction.
        shares: Number of shares in the order.
        adv: Average daily volume for the symbol.
        impact_coeff: Scaling coefficient.
        max_impact_bps: Maximum allowed impact in basis points (50bps = 0.005).

    Returns:
        True if the trade should be rejected due to excessive impact.
    """
    impact = compute_market_impact(daily_vol, shares, adv, impact_coeff)
    max_impact_fraction = max_impact_bps / 10_000.0
    return impact > max_impact_fraction
