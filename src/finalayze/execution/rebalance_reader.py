"""Read the rebalance audit trail (Phase 83) -- token-free DB reads of the Phase 82 tables.

Surfaces ``saa_rebalance_runs`` (+ their ``saa_rebalance_orders``) as frozen records for the
read-only API / dashboard. No broker, no Tinkoff token, no order placement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import date, datetime
    from decimal import Decimal
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from finalayze.core.models import SaaRebalanceRunModel

_DEFAULT_LIMIT = 20
_MAX_LIMIT = 100


@dataclass(frozen=True)
class OrderRecord:
    """One persisted per-leg order outcome."""

    asset_class: str
    symbol: str
    side: str
    requested_qty: Decimal
    filled_qty: Decimal
    status: str
    reason: str | None


@dataclass(frozen=True)
class RebalanceRunRecord:
    """One persisted rebalance run + its per-leg orders."""

    run_id: UUID
    plan_id: str
    as_of: date
    mode: str
    status: str
    fill_rate: Decimal
    created_at: datetime
    orders: tuple[OrderRecord, ...]


def _to_record(run: SaaRebalanceRunModel) -> RebalanceRunRecord:
    """Map an ORM run (+ its eager-loaded orders) to a frozen record (pure; P83-R1)."""
    orders = tuple(
        OrderRecord(
            asset_class=order.asset_class,
            symbol=order.symbol,
            side=order.side,
            requested_qty=order.requested_qty,
            filled_qty=order.filled_qty,
            status=order.status,
            reason=order.reason,
        )
        for order in sorted(run.orders, key=lambda o: o.asset_class)
    )
    return RebalanceRunRecord(
        run_id=run.id,
        plan_id=run.plan_id,
        as_of=run.as_of,
        mode=run.mode,
        status=run.status,
        fill_rate=run.fill_rate,
        created_at=run.created_at,
        orders=orders,
    )


def _clamp_limit(limit: int) -> int:
    """Bound the requested limit to [1, _MAX_LIMIT]."""
    return max(1, min(limit, _MAX_LIMIT))


async def list_rebalance_runs(
    session_factory: async_sessionmaker[AsyncSession],
    portfolio_id: UUID,
    *,
    limit: int = _DEFAULT_LIMIT,
) -> list[RebalanceRunRecord]:
    """Return a portfolio's rebalance runs (+ orders), newest first (token-free; P83-R2/R6)."""
    from sqlalchemy import select  # noqa: PLC0415
    from sqlalchemy.orm import selectinload  # noqa: PLC0415

    from finalayze.core.models import SaaRebalanceRunModel  # noqa: PLC0415

    async with session_factory() as session:
        runs = (
            (
                await session.execute(
                    select(SaaRebalanceRunModel)
                    .where(SaaRebalanceRunModel.portfolio_id == portfolio_id)
                    .order_by(SaaRebalanceRunModel.created_at.desc())
                    .limit(_clamp_limit(limit))
                    .options(selectinload(SaaRebalanceRunModel.orders))
                )
            )
            .scalars()
            .all()
        )
    return [_to_record(run) for run in runs]
