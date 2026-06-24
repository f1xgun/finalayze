"""Phase 82 P82-05: persist_rebalance_run against a real DB (gated on FINALAYZE_DATABASE_URL).

Runs the real alembic migration (013) then writes a run + its order rows and reads them back.
Skipped when no DB URL is set (CI without a provisioned Postgres).
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from finalayze.core.schemas import AssetClass, RiskProfile
from finalayze.execution.broker_base import OrderRequest, OrderResult
from finalayze.orchestration.rebalance_planner import LegOutcome, PlannedLeg, RebalancePlan
from finalayze.orchestration.rebalance_reconcile import reconcile_rebalance_run


def _db_url() -> str:
    url = os.environ.get("FINALAYZE_DATABASE_URL") or os.environ.get("DATABASE_URL")
    if not url:
        pytest.skip("FINALAYZE_DATABASE_URL not set; integration DB unavailable")
    return url


def _upgrade_head(url: str) -> None:
    from alembic import command  # noqa: PLC0415
    from alembic.config import Config  # noqa: PLC0415

    cfg = Config("alembic/alembic.ini")
    cfg.set_main_option("sqlalchemy.url", url)
    command.upgrade(cfg, "head")


def test_persist_rebalance_run_writes_run_and_orders() -> None:
    import asyncio  # noqa: PLC0415

    import sqlalchemy as sa  # noqa: PLC0415
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine  # noqa: PLC0415

    from finalayze.core.models import (  # noqa: PLC0415
        SaaRebalanceOrderModel,
        SaaRebalanceRunModel,
    )
    from finalayze.execution.rebalance_writer import persist_rebalance_run  # noqa: PLC0415
    from finalayze.execution.saa_portfolio_writer import create_active_portfolio  # noqa: PLC0415

    url = _db_url()
    _upgrade_head(url)

    async def _run() -> None:
        engine = create_async_engine(url)
        try:
            sf = async_sessionmaker(engine, expire_on_commit=False)
            portfolio_id = await create_active_portfolio(
                sf, budget_rub=Decimal(1_000_000), risk_profile=RiskProfile.BALANCED
            )
            plan = RebalancePlan(
                plan_id=f"{portfolio_id}:2026-06-23",
                created_at=datetime(2026, 6, 23, 12, 0, tzinfo=UTC),
                portfolio_id=portfolio_id,  # FK to saa_portfolios
                risk_profile="balanced",
                budget_rub=Decimal(1_000_000),
                mode="SANDBOX",
                auto_legs=(
                    PlannedLeg(
                        asset_class=AssetClass.EQUITY,
                        market_id="moex",
                        order=OrderRequest(
                            symbol="EQMX",
                            side="BUY",
                            quantity=Decimal(100),
                            client_order_id="fnz-eq",
                        ),
                        side="BUY",
                        target_notional=Decimal(10_000),
                        est_price=Decimal(100),
                    ),
                ),
                manual_actions=(),
            )
            outcomes = [
                LegOutcome(
                    asset_class=AssetClass.EQUITY,
                    requested_qty=Decimal(100),
                    result=OrderResult(
                        filled=True, quantity=Decimal(100), symbol="EQMX", side="BUY"
                    ),
                    status="FILLED",
                )
            ]
            rec = reconcile_rebalance_run(plan, outcomes)
            run_id = await persist_rebalance_run(sf, plan, outcomes, rec)

            async with sf() as session:
                run = (
                    await session.execute(
                        sa.select(SaaRebalanceRunModel).where(SaaRebalanceRunModel.id == run_id)
                    )
                ).scalar_one()
                assert run.status == "COMPLETE"
                assert run.fill_rate == Decimal("1.0000")
                assert run.budget_rub == Decimal("1000000.00")

                orders = (
                    (
                        await session.execute(
                            sa.select(SaaRebalanceOrderModel).where(
                                SaaRebalanceOrderModel.run_id == run_id
                            )
                        )
                    )
                    .scalars()
                    .all()
                )
                assert len(orders) == 1
                assert orders[0].symbol == "EQMX"
                assert orders[0].filled_qty == Decimal("100.00000000")
                assert orders[0].status == "FILLED"
        finally:
            await engine.dispose()

    asyncio.run(_run())
