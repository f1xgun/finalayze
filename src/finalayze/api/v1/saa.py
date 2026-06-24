"""SAA target-allocation endpoint (Layer 6) -- read-only, token-free (Phase 81).

Surfaces the operator's Strategic Asset Allocation: the active portfolio (budget, risk profile), the
regime-tilted target weights for today, the per-leg target notionals (``budget * weight``), and the
deposit mark. Entirely token-free (DB read + pure compute + committed snapshot). It constructs no
broker, fetches no live positions/prices, and places NO orders -- the live rebalance is the CLI
(``scripts/run_rebalance.py``), and real-money go-live is a hard stop.
"""

from __future__ import annotations

from dataclasses import asdict
from decimal import Decimal

import structlog
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict

from finalayze.api.v1.auth import api_key_auth
from finalayze.core.clock import RealClock
from finalayze.core.schemas import RiskProfile
from finalayze.execution.deposit_loader import load_deposit_broker_from_db
from finalayze.execution.rebalance_reader import list_rebalance_runs
from finalayze.execution.saa_portfolio_writer import get_active_portfolio
from finalayze.markets.instruments import build_default_registry
from finalayze.orchestration.allocation import AllocationOrchestrator
from finalayze.orchestration.rebalance_execution import resolve_leg_instruments

_log = structlog.get_logger()

router = APIRouter(prefix="/saa", tags=["saa"], dependencies=[Depends(api_key_auth)])

_HTTP_NOT_FOUND = 404
_HTTP_NO_CERT = 503  # no committed allocation-gate cert -> fail-closed (Phase 87)


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

    NOTE: ``target_notional_rub`` is the STRATEGIC EXPOSURE (``budget * weight``), NOT the
    execution-layer funded CASH. For the leveraged equity FUTURE the actually-committed cash is only
    margin + a drawdown reserve (Phase 86 fully-funded synthetic equity), and the freed cash is
    swept into the deposit -- so the deposit's funded plug computed by ``scripts/run_rebalance.py``
    is larger than the deposit's strategic ``budget * weight`` shown here. Do not cross-read the two
    as the same cash figure; the rebalance CLI/preview is the source of truth for the cash to move.
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


class RebalanceOrderOut(BaseModel):
    """One persisted per-leg order outcome (read-only)."""

    model_config = ConfigDict(frozen=True)

    asset_class: str
    symbol: str
    side: str
    requested_qty: str
    filled_qty: str
    status: str
    reason: str | None = None


class RebalanceRunOut(BaseModel):
    """One persisted rebalance run + its per-leg orders (read-only)."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    plan_id: str
    as_of: str
    mode: str
    status: str
    fill_rate: str
    created_at: str
    orders: list[RebalanceOrderOut]


class RebalanceRunsResponse(BaseModel):
    """The active portfolio's recent rebalance runs (newest first)."""

    model_config = ConfigDict(frozen=True)

    portfolio_id: str
    runs: list[RebalanceRunOut]


@router.get("/rebalance-runs", response_model=RebalanceRunsResponse)
async def rebalance_runs(limit: int = 20) -> RebalanceRunsResponse:
    """Return the active portfolio's recent rebalance runs (read-only, no Tinkoff token).

    Reads the persisted audit trail (``saa_rebalance_runs`` / ``saa_rebalance_orders``), newest
    first. ``limit`` is clamped to [1, 100]. Returns 404 when there is no active SAA portfolio; an
    empty ``runs`` list when the portfolio has no runs yet.
    """
    from finalayze.core.db import get_async_session_factory  # noqa: PLC0415

    session_factory = get_async_session_factory()
    active = await get_active_portfolio(session_factory)
    if active is None:
        raise HTTPException(status_code=_HTTP_NOT_FOUND, detail="no active SAA portfolio")
    portfolio_id = active[0]

    records = await list_rebalance_runs(session_factory, portfolio_id, limit=limit)
    runs = [
        RebalanceRunOut(
            run_id=str(record.run_id),
            plan_id=record.plan_id,
            as_of=record.as_of.isoformat(),
            mode=record.mode,
            status=record.status,
            fill_rate=str(record.fill_rate),
            created_at=record.created_at.isoformat(),
            orders=[
                RebalanceOrderOut(
                    asset_class=order.asset_class,
                    symbol=order.symbol,
                    side=order.side,
                    requested_qty=str(order.requested_qty),
                    filled_qty=str(order.filled_qty),
                    status=order.status,
                    reason=order.reason,
                )
                for order in record.orders
            ],
        )
        for record in records
    ]
    return RebalanceRunsResponse(portfolio_id=str(portfolio_id), runs=runs)


class RegimeStoryOut(BaseModel):
    """One rate-regime sub-window's allocation-vs-best-naive benchmark row (read-only)."""

    model_config = ConfigDict(frozen=True)

    unit_key: str
    unit_label: str
    window_start: str
    window_end: str
    allocation_sharpe: float
    best_naive_sharpe: float
    allocation_sortino: float
    best_naive_sortino: float
    unit_verdict: str


class CertDecisionResponse(BaseModel):
    """The latest binding cert verdict + per-regime benchmark stories (read-only)."""

    model_config = ConfigDict(frozen=True)

    cert_path: str
    cert_timestamp: str
    git_sha: str
    staleness_days: int
    phase_verdict: str
    escalation: str | None
    n1_caveat: bool
    alloc_sharpe_full: float
    best_naive_sharpe_full: float
    full_verdict: str
    high_rate_caveat: str
    headline: str
    when_framing: str
    regime_stories: list[RegimeStoryOut]


@router.get("/cert-decision", response_model=CertDecisionResponse)
async def cert_decision() -> CertDecisionResponse:
    """Return the latest binding allocation-gate cert verdict (read-only, no Tinkoff token).

    Surfaces the FROZEN allocator's honest binding verdict -- measured on real net-of-tax curves --
    ALONGSIDE the deposit-anchor benchmark, so the operator sees the honest truth: in a 16-21%
    regime the deposit wins, and in the single observed easing cycle so far all sleeves are negative
    (N=1). Every number + verdict is DERIVED from the committed cert summary.json -- no pre-baked
    literal, no softened HARD_FAIL. Returns 503 when no committed cert exists (fail-closed). The
    sole side-effect is a filesystem read of results/iterations/ (no DB, no network, no token).
    """
    from finalayze.backtest.cert_reader import load_latest_cert  # noqa: PLC0415
    from finalayze.core.exceptions import CertNotFoundError  # noqa: PLC0415

    try:
        decision = load_latest_cert()
    except CertNotFoundError as exc:
        raise HTTPException(status_code=_HTTP_NO_CERT, detail=str(exc)) from exc

    return CertDecisionResponse(
        cert_path=decision.cert_path,
        cert_timestamp=decision.cert_timestamp,
        git_sha=decision.git_sha,
        staleness_days=decision.staleness_days,
        phase_verdict=decision.phase_verdict,
        escalation=decision.escalation,
        n1_caveat=decision.n1_caveat,
        alloc_sharpe_full=decision.alloc_sharpe_full,
        best_naive_sharpe_full=decision.best_naive_sharpe_full,
        full_verdict=decision.full_verdict,
        high_rate_caveat=decision.high_rate_caveat,
        headline=decision.headline,
        when_framing=decision.when_framing,
        regime_stories=[RegimeStoryOut(**asdict(s)) for s in decision.regime_stories],
    )
