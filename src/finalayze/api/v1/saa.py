"""SAA target-allocation endpoint (Layer 6) -- read-only, token-free (Phase 81).

Surfaces the operator's Strategic Asset Allocation: the active portfolio (budget, risk profile), the
regime-tilted target weights for today, the per-leg target notionals (``budget * weight``), and the
deposit mark. Entirely token-free (DB read + pure compute + committed snapshot). It constructs no
broker, fetches no live positions/prices, and places NO orders -- the live rebalance is the CLI
(``scripts/run_rebalance.py``), and real-money go-live is a hard stop.
"""

from __future__ import annotations

from decimal import Decimal

import structlog
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict

from finalayze.api.v1.auth import api_key_auth
from finalayze.core.clock import RealClock
from finalayze.core.schemas import RiskProfile
from finalayze.execution.deposit_loader import load_deposit_broker_from_db
from finalayze.execution.saa_portfolio_writer import get_active_portfolio
from finalayze.markets.instruments import build_default_registry
from finalayze.orchestration.allocation import AllocationOrchestrator
from finalayze.orchestration.rebalance_execution import resolve_leg_instruments

_log = structlog.get_logger()

router = APIRouter(prefix="/saa", tags=["saa"], dependencies=[Depends(api_key_auth)])

_HTTP_NOT_FOUND = 404


class LegTarget(BaseModel):
    """One SAA leg's target: weight, target notional (RUB), and tradeable symbol (None=deposit)."""

    model_config = ConfigDict(frozen=True)

    weight: str
    target_notional_rub: str
    symbol: str | None = None


class SaaTargetAllocation(BaseModel):
    """The active portfolio's SAA target allocation (read-only)."""

    model_config = ConfigDict(frozen=True)

    portfolio_id: str
    risk_profile: str
    budget_rub: str
    as_of: str
    deposit_current_notional_rub: str
    legs: dict[str, LegTarget]


@router.get("/target-allocation", response_model=SaaTargetAllocation)
async def target_allocation() -> SaaTargetAllocation:
    """Return the active portfolio's SAA target allocation (read-only, no Tinkoff token).

    Shows the budget, the regime-tilted target weights for today, the per-leg target notionals
    (``budget * weight``), and the deposit mark. Does NOT fetch live positions/prices or place
    orders. Returns 404 when there is no active SAA portfolio.
    """
    from finalayze.core.db import get_async_session_factory  # noqa: PLC0415

    session_factory = get_async_session_factory()

    active = await get_active_portfolio(session_factory)
    if active is None:
        raise HTTPException(status_code=_HTTP_NOT_FOUND, detail="no active SAA portfolio")
    portfolio_id, risk_profile_str, budget_rub = active

    as_of = RealClock().now().date()
    weights = AllocationOrchestrator(
        risk_profile=RiskProfile(risk_profile_str)
    ).get_rebalance_weights(as_of)
    leg_instruments = resolve_leg_instruments(build_default_registry())

    deposit_broker = await load_deposit_broker_from_db(portfolio_id, as_of, session_factory)
    deposit_value = deposit_broker.deposit_value() if deposit_broker is not None else Decimal(0)

    legs: dict[str, LegTarget] = {}
    for asset_class, weight in weights.items():
        instrument = leg_instruments.get(asset_class)
        legs[asset_class.value] = LegTarget(
            weight=str(weight),
            target_notional_rub=str(budget_rub * weight),
            symbol=instrument.symbol if instrument is not None else None,
        )

    return SaaTargetAllocation(
        portfolio_id=str(portfolio_id),
        risk_profile=risk_profile_str,
        budget_rub=str(budget_rub),
        as_of=as_of.isoformat(),
        deposit_current_notional_rub=str(deposit_value),
        legs=legs,
    )
