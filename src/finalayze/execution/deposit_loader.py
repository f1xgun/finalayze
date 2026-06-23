"""Reload a DepositSimulatedBroker from persisted SAA portfolio state (Phase 77 P2-06).

Reload is a DIRECT load, NOT a replay. The deposit ladder accrues on TRADING days only
(``backtest/engine.py`` iterates candle timestamps) and ``accrue()`` compounds
``(1+annual)^(1/252)`` per call, so a calendar-day replay would over-compound the mark
(the bug review CR-01 caught). Instead we persist the broker's mutable state:
- per-tranche accrued marks (on ``deposit_tranches``), and
- the broker-level year-scoped accumulators (``_ytd_deposit_gross`` /
  ``_running_max_key_rate`` / ``_current_year``) + totals + the last accrual date
  (on ``saa_portfolios.deposit_accumulators`` JSONB),
and restore them verbatim. The binding gate is that the NEXT ``accrue()`` after a
restore behaves bit-identically to a never-restarted broker (no cadence assumption).
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import TYPE_CHECKING

import structlog

from finalayze.core.schemas import DepositTranche
from finalayze.execution.deposit_broker import DepositSimulatedBroker

if TYPE_CHECKING:
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

_log = structlog.get_logger()


def serialize_deposit_accumulators(broker: DepositSimulatedBroker) -> dict[str, object]:
    """Extract the broker's mutable accumulator state to a JSON-safe dict (Decimals as strings).

    Captures the year-scoped NDFL/floor accumulators (which drive the NEXT bar's tax) + the
    running totals + the last accrued calendar date (to seed the WR-04 same-day idempotency
    guard). Tranche accrued marks are persisted separately on ``deposit_tranches``.
    """
    dates = broker._processed_accrual_dates
    return {
        "ytd_deposit_gross": str(broker._ytd_deposit_gross),
        "running_max_key_rate": str(broker._running_max_key_rate),
        "current_year": broker._current_year,
        "total_interest_gross": str(broker._total_interest_gross),
        "total_interest_net": str(broker._total_interest_net),
        "total_tax_paid": str(broker._total_tax_paid),
        "last_accrual_date": max(dates).isoformat() if dates else None,
    }


def restore_deposit_accumulators(broker: DepositSimulatedBroker, data: dict[str, object]) -> None:
    """Restore broker accumulator state from :func:`serialize_deposit_accumulators` output.

    Sets the year-scoped accumulators + totals verbatim and seeds the same-day idempotency
    guard with the last accrued date, so the NEXT ``accrue()`` resumes exactly where the live
    broker left off (no replay, cadence-independent).
    """
    broker._ytd_deposit_gross = Decimal(str(data["ytd_deposit_gross"]))
    broker._running_max_key_rate = Decimal(str(data["running_max_key_rate"]))
    broker._current_year = data["current_year"]  # type: ignore[assignment]
    broker._total_interest_gross = Decimal(str(data["total_interest_gross"]))
    broker._total_interest_net = Decimal(str(data["total_interest_net"]))
    broker._total_tax_paid = Decimal(str(data["total_tax_paid"]))
    last = data.get("last_accrual_date")
    broker._processed_accrual_dates = {date.fromisoformat(str(last))} if last else set()


async def load_deposit_broker_from_db(
    portfolio_id: UUID,
    current_date: date,
    session_factory: async_sessionmaker[AsyncSession],
) -> DepositSimulatedBroker | None:
    """Reload a DepositSimulatedBroker from persisted state — a DIRECT load (no replay).

    Loads the active portfolio + its non-broken unmatured tranches (``maturity_date >=
    current_date``) with their persisted accrued marks, then restores the broker-level
    accumulators verbatim from ``saa_portfolios.deposit_accumulators``. Returns ``None`` if the
    portfolio is missing or inactive.
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
        accumulators = portfolio.deposit_accumulators

    tranches = [
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
    broker = DepositSimulatedBroker(initial_cash=Decimal(0), tranches=tranches)
    if accumulators:
        restore_deposit_accumulators(broker, accumulators)
    _log.info(
        "deposit_loader_loaded_portfolio",
        portfolio_id=str(portfolio_id),
        tranche_count=len(tranches),
        accumulators_restored=bool(accumulators),
    )
    return broker
