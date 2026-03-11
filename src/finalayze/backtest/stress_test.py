"""Bond portfolio stress testing utilities.

Simulates the impact of sudden rate changes on an OFZ portfolio
using duration/convexity approximation.

Price change formula:
    delta_P / P ≈ -mod_duration * delta_y + 0.5 * convexity * (delta_y)^2

where delta_y = rate_change_bps / 10_000.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StressScenario:
    """A rate shock scenario."""

    name: str
    rate_change_bps: int  # positive = rates up (losses for bond holders)
    description: str


@dataclass(frozen=True)
class StressResult:
    """Result of applying a stress scenario to a portfolio."""

    scenario: StressScenario
    portfolio_value_before: Decimal
    portfolio_value_after: Decimal
    pnl: Decimal
    pnl_pct: float
    dd_pct: float  # drawdown from peak (assuming peak = before)
    breaches_portfolio_limit: bool  # True if DD > portfolio_dd_limit

    # Per-layer breakdown
    layer_pnl: dict[str, Decimal] = field(default_factory=dict)
    layer_pnl_pct: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Standard scenarios from the multi-asset plan
# ---------------------------------------------------------------------------

STRESS_SCENARIOS: list[StressScenario] = [
    StressScenario("moderate_hike", 300, "CBR hikes +300bps (e.g., 15.5% -> 18.5%)"),
    StressScenario("severe_hike", 500, "CBR hikes +500bps (e.g., 15.5% -> 20.5%)"),
    StressScenario("extreme_hike", 1000, "CBR hikes +1000bps (2022-style: 15.5% -> 25.5%)"),
    StressScenario("moderate_cut", -300, "CBR cuts -300bps (easing: 15.5% -> 12.5%)"),
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BPS_DIVISOR = Decimal(10000)
_HALF = Decimal("0.5")


# ---------------------------------------------------------------------------
# Single-position PnL estimator
# ---------------------------------------------------------------------------


def estimate_bond_pnl_from_rate_change(
    face_value: Decimal,
    quantity: int,
    mod_duration: float,
    convexity_val: float,
    rate_change_bps: int,
) -> Decimal:
    """Estimate bond PnL from a rate change using duration + convexity.

    PnL approx  -mod_duration * dy + 0.5 * convexity * dy**2
    where dy = rate_change_bps / 10_000.

    Returns PnL in RUB (can be negative for rate hikes).
    """
    dy = Decimal(rate_change_bps) / _BPS_DIVISOR
    # Price change per unit face value
    price_change_pct = (
        -Decimal(str(mod_duration)) * dy + _HALF * Decimal(str(convexity_val)) * dy * dy
    )
    position_value = face_value * quantity
    return position_value * price_change_pct


# ---------------------------------------------------------------------------
# Portfolio-level stress runner
# ---------------------------------------------------------------------------


def run_portfolio_stress(
    layer_positions: dict[str, list[dict[str, Any]]],
    layer_cash: dict[str, Decimal],
    scenario: StressScenario,
    portfolio_dd_limit: float = 0.10,
) -> StressResult:
    """Run a stress scenario against a layered bond portfolio.

    Args:
        layer_positions: per-layer bond positions.  Each position dict must
            contain keys ``face_value`` (:class:`Decimal`), ``quantity``
            (:class:`int`), ``mod_duration`` (:class:`float`), and
            ``convexity`` (:class:`float`).
        layer_cash: per-layer cash balances (unaffected by rate shocks).
        scenario: the rate shock to apply.
        portfolio_dd_limit: maximum acceptable drawdown (default 10%).

    Returns:
        :class:`StressResult` with per-layer and portfolio-level impact.
    """
    layer_pnl: dict[str, Decimal] = {}
    layer_pnl_pct: dict[str, float] = {}
    total_value_before = Decimal(0)
    total_pnl = Decimal(0)

    all_layer_ids = set(layer_positions.keys()) | set(layer_cash.keys())

    for layer_id in sorted(all_layer_ids):
        bonds = layer_positions.get(layer_id, [])
        cash = layer_cash.get(layer_id, Decimal(0))

        # Layer value before shock = sum of bond face * qty + cash
        layer_bond_value = Decimal(0)
        layer_bond_pnl = Decimal(0)
        for bond in bonds:
            pos_value = bond["face_value"] * bond["quantity"]
            layer_bond_value += pos_value
            pnl = estimate_bond_pnl_from_rate_change(
                face_value=bond["face_value"],
                quantity=bond["quantity"],
                mod_duration=bond["mod_duration"],
                convexity_val=bond["convexity"],
                rate_change_bps=scenario.rate_change_bps,
            )
            layer_bond_pnl += pnl

        layer_value = layer_bond_value + cash
        total_value_before += layer_value
        total_pnl += layer_bond_pnl

        layer_pnl[layer_id] = layer_bond_pnl
        if layer_value > 0:
            layer_pnl_pct[layer_id] = float(layer_bond_pnl / layer_value)
        else:
            layer_pnl_pct[layer_id] = 0.0

    # Portfolio-level metrics
    portfolio_value_after = total_value_before + total_pnl
    pnl_pct = float(total_pnl / total_value_before) if total_value_before > 0 else 0.0
    # Drawdown = magnitude of loss (positive number)
    dd_pct = abs(pnl_pct) if pnl_pct < 0 else 0.0
    breaches = dd_pct > portfolio_dd_limit

    return StressResult(
        scenario=scenario,
        portfolio_value_before=total_value_before,
        portfolio_value_after=portfolio_value_after,
        pnl=total_pnl,
        pnl_pct=pnl_pct,
        dd_pct=dd_pct,
        breaches_portfolio_limit=breaches,
        layer_pnl=layer_pnl,
        layer_pnl_pct=layer_pnl_pct,
    )
