"""Phase 83 P83-R2: list_rebalance_runs against a real DB (gated on FINALAYZE_DATABASE_URL).

Persists two runs then reads them back (with their orders eager-loaded) and checks the limit.
Skipped when no DB URL is set.
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


def _plan(portfolio_id: object, plan_id: str) -> RebalancePlan:
    return RebalancePlan(
        plan_id=plan_id,
        created_at=datetime(2026, 6, 23, 12, 0, tzinfo=UTC),
        portfolio_id=portfolio_id,  # type: ignore[arg-type]
        risk_profile="balanced",
        budget_rub=Decimal(1_000_000),
        mode="SANDBOX",
        auto_legs=(
            PlannedLeg(
                asset_class=AssetClass.EQUITY,
                market_id="moex",
                order=OrderRequest(
                    symbol="EQMX", side="BUY", quantity=Decimal(100), client_order_id="fnz-eq"
                ),
                side="BUY",
                target_notional=Decimal(10_000),
                est_price=Decimal(100),
            ),
        ),
        manual_actions=(),
    )


def _outcomes() -> list[LegOutcome]:
    return [
        LegOutcome(
            asset_class=AssetClass.EQUITY,
            requested_qty=Decimal(100),
            result=OrderResult(filled=True, quantity=Decimal(100), symbol="EQMX", side="BUY"),
            status="FILLED",
        )
    ]


def test_list_rebalance_runs_reads_persisted_runs() -> None:
    import asyncio  # noqa: PLC0415

    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine  # noqa: PLC0415

    from finalayze.execution.rebalance_reader import list_rebalance_runs  # noqa: PLC0415
    from finalayze.execution.rebalance_writer import persist_rebalance_run  # noqa: PLC0415
    from finalayze.execution.saa_portfolio_writer import create_active_portfolio  # noqa: PLC0415

    url = _db_url()
    _upgrade_head(url)

    async def _run() -> None:
        engine = create_async_engine(url)
        try:
            sf = async_sessionmaker(engine, expire_on_commit=False)
            pid = await create_active_portfolio(
                sf, budget_rub=Decimal(1_000_000), risk_profile=RiskProfile.BALANCED
            )
            # Persist run-a then run-b with a gap so created_at is distinct (locks "newest first").
            for plan_id in ("run-a", "run-b"):
                plan = _plan(pid, plan_id)
                outcomes = _outcomes()
                await persist_rebalance_run(
                    sf, plan, outcomes, reconcile_rebalance_run(plan, outcomes)
                )
                await asyncio.sleep(0.01)

            records = await list_rebalance_runs(sf, pid, limit=10)
            assert len(records) == 2
            assert {r.plan_id for r in records} == {"run-a", "run-b"}
            assert all(len(r.orders) == 1 for r in records)
            assert records[0].orders[0].symbol == "EQMX"
            # newest first (CORR-01 / AH-02): run-b was persisted last.
            assert records[0].plan_id == "run-b"
            assert records[1].plan_id == "run-a"

            # limit=1 returns only the newest run.
            limited = await list_rebalance_runs(sf, pid, limit=1)
            assert len(limited) == 1
            assert limited[0].plan_id == "run-b"
        finally:
            await engine.dispose()

    asyncio.run(_run())
