"""Rollout phase risk limits (Layer 4).

Maps each RolloutPhase to a frozen set of risk parameters.
FULL phase limits MUST match existing Settings/PreTradeChecker defaults
for backward compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from finalayze.core.modes import RolloutPhase


@dataclass(frozen=True)
class RolloutLimits:
    """Risk limits for a rollout phase."""

    max_position_pct: Decimal
    max_positions_per_market: int
    daily_loss_limit_pct: float
    circuit_breaker_l1: float
    circuit_breaker_l2: float
    circuit_breaker_l3: float
    max_sector_concentration_pct: Decimal
    min_cash_reserve_pct: Decimal


ROLLOUT_LIMITS: dict[RolloutPhase, RolloutLimits] = {
    RolloutPhase.MINIMAL: RolloutLimits(
        max_position_pct=Decimal("0.03"),
        max_positions_per_market=5,
        daily_loss_limit_pct=0.01,
        circuit_breaker_l1=0.01,
        circuit_breaker_l2=0.02,
        circuit_breaker_l3=0.03,
        max_sector_concentration_pct=Decimal("0.20"),
        min_cash_reserve_pct=Decimal("0.40"),
    ),
    RolloutPhase.STANDARD: RolloutLimits(
        max_position_pct=Decimal("0.10"),
        max_positions_per_market=8,
        daily_loss_limit_pct=0.03,
        circuit_breaker_l1=0.03,
        circuit_breaker_l2=0.05,
        circuit_breaker_l3=0.10,
        max_sector_concentration_pct=Decimal("0.30"),
        min_cash_reserve_pct=Decimal("0.30"),
    ),
    RolloutPhase.FULL: RolloutLimits(
        max_position_pct=Decimal("0.20"),
        max_positions_per_market=10,
        daily_loss_limit_pct=0.02,
        circuit_breaker_l1=0.05,
        circuit_breaker_l2=0.10,
        circuit_breaker_l3=0.15,
        max_sector_concentration_pct=Decimal("0.40"),
        min_cash_reserve_pct=Decimal("0.20"),
    ),
}
