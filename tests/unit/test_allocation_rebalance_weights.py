"""Phase 79 P79-10: additive AllocationOrchestrator.get_rebalance_weights(as_of) accessor.

The live weights-to-orders planner must rebalance to the SAME regime-tilted weights the FROZEN
analytics path uses, so the plan and the analytics curve agree on the high_rate/easing tilt. This
accessor is ADDITIVE -- it reuses ``_target_weights`` / ``rate_regime_as_of`` without modifying
``run()`` or any existing method (L-04).
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

from finalayze.config.allocation_profiles import load_allocation_profiles
from finalayze.core.schemas import (
    RATE_REGIME_EASING,
    RATE_REGIME_HIGH_RATE,
    AssetClass,
    RiskProfile,
)
from finalayze.orchestration.allocation import AllocationOrchestrator

# rate_regime_as_of flips at the 2025-06-06 first cut (REGIME_SPLIT_BOUNDARY).
_HIGH_RATE_DATE = date(2025, 1, 1)  # before the first cut -> high_rate
_EASING_DATE = date(2026, 1, 1)  # after the first cut -> easing


def _expected(regime: str) -> dict[AssetClass, Decimal]:
    profile = load_allocation_profiles()[RiskProfile.BALANCED]
    assert profile.regime_weights is not None
    vec = profile.regime_weights[regime]
    return {
        AssetClass.DEPOSIT: vec[AssetClass.DEPOSIT],
        AssetClass.OFZ_PK: vec[AssetClass.OFZ_PK],
        AssetClass.EQUITY: vec[AssetClass.EQUITY],
    }


def test_high_rate_date_returns_high_rate_vector() -> None:
    """A high-rate as_of returns the profile's high_rate regime vector."""
    orch = AllocationOrchestrator(risk_profile=RiskProfile.BALANCED)
    assert orch.get_rebalance_weights(_HIGH_RATE_DATE) == _expected(RATE_REGIME_HIGH_RATE)


def test_easing_date_returns_easing_vector() -> None:
    """An easing as_of returns the profile's easing regime vector."""
    orch = AllocationOrchestrator(risk_profile=RiskProfile.BALANCED)
    assert orch.get_rebalance_weights(_EASING_DATE) == _expected(RATE_REGIME_EASING)


def test_regimes_actually_differ() -> None:
    """The tilt genuinely changes between regimes (the accessor is not a constant)."""
    orch = AllocationOrchestrator(risk_profile=RiskProfile.BALANCED)
    assert orch.get_rebalance_weights(_HIGH_RATE_DATE) != orch.get_rebalance_weights(_EASING_DATE)


def test_weights_sum_to_one() -> None:
    """The returned weight vector is a valid allocation (sums to exactly 1)."""
    orch = AllocationOrchestrator(risk_profile=RiskProfile.BALANCED)
    weights = orch.get_rebalance_weights(_EASING_DATE)
    assert sum(weights.values()) == Decimal(1)


def test_accessor_keys_are_the_three_asset_classes() -> None:
    """The accessor returns exactly the three SAA asset classes."""
    orch = AllocationOrchestrator(risk_profile=RiskProfile.GROWTH)
    weights = orch.get_rebalance_weights(_HIGH_RATE_DATE)
    assert set(weights) == {AssetClass.DEPOSIT, AssetClass.OFZ_PK, AssetClass.EQUITY}
