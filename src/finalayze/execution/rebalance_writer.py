"""Persist a rebalance run + per-leg order outcomes to the DB (Phase 82).

An audit trail of every real (submit) rebalance: one ``saa_rebalance_runs`` row + N
``saa_rebalance_orders`` rows, written in one transaction (Phase 77 pattern). The pure row-builders
(``_run_row`` / ``_order_rows``) are unit-testable without a DB; ``persist_rebalance_run`` does the
transaction (integration-tested, gated on a DB).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from finalayze.core.models import SaaRebalanceOrderModel, SaaRebalanceRunModel

if TYPE_CHECKING:
    from collections.abc import Sequence
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from finalayze.orchestration.rebalance_planner import LegOutcome, RebalancePlan
    from finalayze.orchestration.rebalance_reconcile import RebalanceReconciliation

_log = structlog.get_logger()


def _run_row(plan: RebalancePlan, reconciliation: RebalanceReconciliation) -> SaaRebalanceRunModel:
    """Build the run row from the plan + reconciliation rollup (pure, P82-R3)."""
    return SaaRebalanceRunModel(
        portfolio_id=plan.portfolio_id,
        plan_id=plan.plan_id,
        as_of=plan.created_at.date(),
        mode=plan.mode,
        budget_rub=plan.budget_rub,
        status=reconciliation.status,
        fill_rate=reconciliation.fill_rate,
    )


def _order_rows(
    run_id: UUID, plan: RebalancePlan, outcomes: Sequence[LegOutcome]
) -> list[SaaRebalanceOrderModel]:
    """Build the per-leg order rows (pure, P82-R3).

    The symbol / side / client_order_id come from the PLAN leg (reliable even when a FAILED
    ``OrderResult`` left them blank); qty/status/reason come from the outcome.
    """
    plan_legs = {leg.asset_class: leg for leg in plan.auto_legs}
    rows: list[SaaRebalanceOrderModel] = []
    for outcome in outcomes:
        leg = plan_legs.get(outcome.asset_class)
        symbol = leg.order.symbol if leg is not None else outcome.result.symbol
        side = leg.side if leg is not None else str(outcome.result.side)
        client_order_id = leg.order.client_order_id if leg is not None else outcome.result.order_id
        rows.append(
            SaaRebalanceOrderModel(
                run_id=run_id,
                asset_class=outcome.asset_class.value,
                symbol=symbol,
                side=side,
                requested_qty=outcome.requested_qty,
                filled_qty=outcome.result.quantity,
                status=outcome.status,
                client_order_id=client_order_id or "",
                reason=outcome.result.reason or None,
            )
        )
    return rows


async def persist_rebalance_run(
    session_factory: async_sessionmaker[AsyncSession],
    plan: RebalancePlan,
    outcomes: Sequence[LegOutcome],
    reconciliation: RebalanceReconciliation,
) -> UUID:
    """Persist one run + its order rows in one transaction; return the run id (P82-R4)."""
    run = _run_row(plan, reconciliation)
    async with session_factory() as session, session.begin():
        session.add(run)
        await session.flush()  # populate run.id (FK for the order rows)
        run_id = run.id
        for order in _order_rows(run_id, plan, outcomes):
            session.add(order)

    _log.info(
        "rebalance_run_persisted",
        run_id=str(run_id),
        plan_id=plan.plan_id,
        status=reconciliation.status,
        fill_rate=str(reconciliation.fill_rate),
        order_count=len(outcomes),
    )
    return run_id
