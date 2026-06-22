"""Load a DepositSimulatedBroker from persisted SAA portfolio state (Phase 77 P2-06).

Rehydrates an active portfolio + its non-broken unmatured tranches from the DB, then
RECONSTRUCTS the broker by replaying ``accrue()`` over each tranche's life. The replay
mirrors live operation -- each tranche is added to the broker ON its ``open_date`` and
accrues every bar through ``current_date`` -- so the year-scoped NDFL/floor accumulators
(which reset at Jan-1) AND the accrued marks are rebuilt deterministically from the tranche
IDENTITY (principal/term/rate/dates) + the committed-stable CBR calendar. This is why the
accumulators are NOT persisted (the replay-equivalence test is the binding gate).

CRITICAL: the reconstruction rebuilds accrued marks FROM ZERO; it does NOT read the persisted
``accrued_net``/``accrued_gross`` back onto the tranche before replaying (doing so would
DOUBLE-COUNT -- the persisted mark plus a full replayed accrual). The persisted marks are an
authoritative checkpoint the reconstruction is verified against (a mismatch is logged).
"""

from __future__ import annotations

from dataclasses import replace
from datetime import date, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING

import structlog

from finalayze.core.schemas import DepositTranche
from finalayze.execution.deposit_broker import DepositSimulatedBroker

if TYPE_CHECKING:
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

_log = structlog.get_logger()


def reconstruct_deposit_broker(
    tranches: list[DepositTranche], current_date: date
) -> DepositSimulatedBroker:
    """Rebuild a DepositSimulatedBroker by replaying ``accrue()`` over each tranche's life.

    Mirrors live operation: each tranche is added to the broker ON its ``open_date`` and then
    accrues every bar through ``current_date``, so a tranche never accrues before it opens and
    the shared YTD/floor accumulators reset correctly at each Jan-1. The input tranches supply
    only the IDENTITY (principal/term/annual_rate/open_date/maturity_date); accrued marks are
    rebuilt FROM ZERO (the input ``accrued_*`` is ignored), so there is NO double-counting.
    """
    broker = DepositSimulatedBroker(initial_cash=Decimal(0), tranches=[])
    if not tranches:
        return broker
    # Fresh identity-only copies (accrued zeroed) so the replay reconstructs the marks; sorted
    # by open_date so each is added exactly when it opens during the bar loop.
    pending = sorted(
        (
            replace(t, accrued_net=Decimal(0), accrued_gross=Decimal(0), broken=False)
            for t in tranches
        ),
        key=lambda t: t.open_date,
    )
    for replay_date in _date_range(pending[0].open_date, current_date):
        while pending and pending[0].open_date == replay_date:
            broker._tranches.append(pending.pop(0))
        broker.accrue(replay_date)
    return broker


async def load_deposit_broker_from_db(
    portfolio_id: UUID,
    current_date: date,
    session_factory: async_sessionmaker[AsyncSession],
) -> DepositSimulatedBroker | None:
    """Reload a DepositSimulatedBroker from persisted portfolio state (Phase 77 P2-06).

    Queries ``saa_portfolios`` + ``deposit_tranches``; filters to the active portfolio and its
    non-broken unmatured tranches (``maturity_date >= current_date``). Returns ``None`` if the
    portfolio is missing or inactive. Reconstructs the broker via :func:`reconstruct_deposit_broker`
    (replay from each tranche's open_date) and verifies the reconstructed accrued marks against the
    persisted checkpoint (a mismatch -- e.g. a CBR-calendar revision -- is logged, not fatal).
    """
    from sqlalchemy import select  # noqa: PLC0415

    from finalayze.core.models import (  # noqa: PLC0415
        DepositTrancheModel,
        SaaPortfolioModel,
    )

    async with session_factory() as session:
        portfolio = (
            await session.execute(
                select(SaaPortfolioModel).where(SaaPortfolioModel.id == portfolio_id)
            )
        ).scalar_one_or_none()
        if portfolio is None or not portfolio.is_active:
            _log.warning("deposit_loader_portfolio_not_found", portfolio_id=str(portfolio_id))
            return None

        rows = (
            (
                await session.execute(
                    select(DepositTrancheModel).where(
                        DepositTrancheModel.portfolio_id == portfolio_id,
                        DepositTrancheModel.broken.is_(False),
                        DepositTrancheModel.maturity_date >= current_date,
                    )
                )
            )
            .scalars()
            .all()
        )

    db_tranches = [
        DepositTranche(
            principal=tr.principal,
            term_months=tr.term_months,
            annual_rate=tr.annual_rate,
            open_date=tr.open_date,
            maturity_date=tr.maturity_date,
            accrued_net=tr.accrued_net,
            accrued_gross=tr.accrued_gross,
            broken=tr.broken,
        )
        for tr in rows
    ]
    broker = reconstruct_deposit_broker(db_tranches, current_date)

    # Verify the reconstruction against the persisted checkpoint (sum-of-marks; identity-only
    # reconstruction is authoritative, so a mismatch is a data/calendar drift signal, not fatal).
    persisted_net = sum((t.accrued_net for t in db_tranches), Decimal(0))
    reconstructed_net = sum((t.accrued_net for t in broker._tranches), Decimal(0))
    if persisted_net != reconstructed_net:
        _log.warning(
            "deposit_loader_accrued_checkpoint_mismatch",
            portfolio_id=str(portfolio_id),
            persisted_net=str(persisted_net),
            reconstructed_net=str(reconstructed_net),
        )
    _log.info(
        "deposit_loader_loaded_portfolio",
        portfolio_id=str(portfolio_id),
        tranche_count=len(db_tranches),
    )
    return broker


def _date_range(start: date, end: date) -> list[date]:
    """All dates in ``[start, end]`` inclusive."""
    out: list[date] = []
    current = start
    while current <= end:
        out.append(current)
        current += timedelta(days=1)
    return out
