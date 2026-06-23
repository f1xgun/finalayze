"""Budget-driven rescaling of SAA curves (Phase 78 P3-05).

The budget_driver reads the active portfolio (budget, risk_profile) from the database,
loads the FIXED allocation profile weights via config, and rescales each pre-computed
total-return curve to the opening notional = budget * profile.weights[leg].

The REAL cost/NDFL arise from the per-leg rescale delta through the actual
AllocationOrchestrator.run() path — NO forced-delta test-only hooks (Phase 72/73
anti-hollow lesson). Netting is preserved: deposit+OFZ legs net through the one
shared YtdTaxAccumulator; equity NEVER routes through NDFL.

KEY DESIGN: each leg is rescaled using its OWN curve[0], not a shared base.
The MCFTR equity leg is a real index level (~2.5-3k) and is rescaled multiplicatively,
never substituted for a base constant.
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

import structlog

from finalayze.config.allocation_profiles import load_allocation_profiles
from finalayze.core.schemas import AssetClass, RiskProfile
from finalayze.execution.saa_portfolio_writer import get_active_portfolio

if TYPE_CHECKING:
    from datetime import date

    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from finalayze.orchestration.allocation import AllocationOrchestrator, AllocationResult

_log = structlog.get_logger()

# Constants
_ZERO = Decimal(0)


async def run_with_active_budget(
    orchestrator: AllocationOrchestrator,
    deposit_curve: list[tuple[date, Decimal]],
    ofz_pk_curve: list[tuple[date, Decimal]],
    equity_curve: list[tuple[date, Decimal]],
    session_factory: async_sessionmaker[AsyncSession],
    *,
    legacy_monthly_drift_cadence: bool = False,
    zero_cost: bool = False,
) -> AllocationResult:
    """Run the allocator with budget-driven rescaled curves.

    Fetches the active portfolio (budget_rub, risk_profile) from the database,
    loads the FIXED allocation profile weights, rescales each leg to
    budget * weight / leg[0] (the leg's OWN starting value), and runs the
    orchestrator on the rescaled curves.

    The rescaling is multiplicative (not additive): for each leg, the rescale
    factor is computed as (budget * weight) / leg_curve[0], so the MCFTR equity
    leg is a real index level (~2.5-3k) and rescales multiplicatively within the
    real run() path. This ensures cost/NDFL arise from the REAL per-leg delta,
    not a test-only hook (Phase 72/73 anti-hollow lesson).

    Netting is preserved (no change from orchestrator):
    - deposit+OFZ legs net through the one shared YtdTaxAccumulator.
    - equity NEVER routes through NDFL (SAA-04, D-07).

    Args:
        orchestrator: The AllocationOrchestrator (frozen W2 allocator).
        deposit_curve: Pre-computed deposit TR curve.
        ofz_pk_curve: Pre-computed OFZ-PK TR curve.
        equity_curve: Pre-computed equity(MCFTR) TR curve.
        session_factory: For reading active portfolio.
        legacy_monthly_drift_cadence: Pass through to orchestrator.run().
        zero_cost: Pass through to orchestrator.run().

    Returns:
        AllocationResult from the orchestrator with the budget-rescaled curves.

    Raises:
        ValueError: If no active portfolio exists in the database.
        ConfigurationError: If the profile is not in allocation_profiles.yaml.
    """
    # Fetch the active portfolio
    active = await get_active_portfolio(session_factory)
    if active is None:
        msg = "no active portfolio found in database"
        raise ValueError(msg)

    portfolio_id, profile_str, budget_rub = active

    # Resolve profile and load weights
    risk_profile = RiskProfile(profile_str)
    profiles = load_allocation_profiles()
    profile = profiles[risk_profile]
    weights = profile.weights

    # Rescale each leg multiplicatively to budget * weight
    deposit_scaled = _rescale_curve(deposit_curve, budget_rub * weights[AssetClass.DEPOSIT])
    ofz_pk_scaled = _rescale_curve(ofz_pk_curve, budget_rub * weights[AssetClass.OFZ_PK])
    equity_scaled = _rescale_curve(equity_curve, budget_rub * weights[AssetClass.EQUITY])

    _log.info(
        "budget_driver_rescaled_curves",
        portfolio_id=str(portfolio_id),
        budget_rub=str(budget_rub),
        risk_profile=profile_str,
        deposit_target=str(budget_rub * weights[AssetClass.DEPOSIT]),
        ofz_pk_target=str(budget_rub * weights[AssetClass.OFZ_PK]),
        equity_target=str(budget_rub * weights[AssetClass.EQUITY]),
    )

    # Run the orchestrator on the rescaled curves
    return orchestrator.run(
        deposit_scaled,
        ofz_pk_scaled,
        equity_scaled,
        legacy_monthly_drift_cadence=legacy_monthly_drift_cadence,
        zero_cost=zero_cost,
    )


def _rescale_curve(
    curve: list[tuple[date, Decimal]],
    target_notional: Decimal,
) -> list[tuple[date, Decimal]]:
    """Rescale a TR curve multiplicatively to a target notional.

    Each leg uses its OWN curve[0] as the base. The rescale factor is
    target_notional / curve[0], applied multiplicatively to every point.

    If the curve is empty or curve[0] is zero, returns the curve unchanged
    (edge case; should not occur in production).

    Args:
        curve: List of (date, value) tuples (TR index levels).
        target_notional: The desired starting value (budget * weight).

    Returns:
        Rescaled curve with the same dates but multiplied values.
    """
    if not curve or curve[0][1] == _ZERO:
        _log.warning(
            "rescale_curve_edge_case",
            curve_len=len(curve),
            first_value=str(curve[0][1]) if curve else "empty",
        )
        return curve

    base_value = curve[0][1]
    scale_factor = target_notional / base_value

    return [(d, v * scale_factor) for d, v in curve]
