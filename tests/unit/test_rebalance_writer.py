"""Phase 82 P82-04: pure rebalance-run row builders (no DB)."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from uuid import uuid4

from finalayze.core.schemas import AssetClass
from finalayze.execution.broker_base import OrderRequest, OrderResult
from finalayze.execution.rebalance_writer import _order_rows, _run_row
from finalayze.orchestration.rebalance_planner import LegOutcome, PlannedLeg, RebalancePlan
from finalayze.orchestration.rebalance_reconcile import reconcile_rebalance_run

_CREATED = datetime(2026, 6, 23, 12, 0, tzinfo=UTC)


def _leg(asset_class: AssetClass, symbol: str, qty: int, price: int, coid: str) -> PlannedLeg:
    return PlannedLeg(
        asset_class=asset_class,
        market_id="moex",
        order=OrderRequest(symbol=symbol, side="BUY", quantity=Decimal(qty), client_order_id=coid),
        side="BUY",
        target_notional=Decimal(qty) * Decimal(price),
        est_price=Decimal(price),
    )


def _plan(legs: list[PlannedLeg]) -> RebalancePlan:
    return RebalancePlan(
        plan_id="pid:2026-06-23",
        created_at=_CREATED,
        portfolio_id=uuid4(),
        risk_profile="balanced",
        budget_rub=Decimal(1_000_000),
        mode="SANDBOX",
        auto_legs=tuple(legs),
        manual_actions=(),
    )


def _outcome(
    asset_class: AssetClass, requested: int, filled: int, status: str, reason: str = ""
) -> LegOutcome:
    return LegOutcome(
        asset_class=asset_class,
        requested_qty=Decimal(requested),
        result=OrderResult(
            filled=filled > 0, quantity=Decimal(filled), reason=reason, symbol="", side="BUY"
        ),
        status=status,
    )


def test_run_row_carries_plan_and_reconciliation() -> None:
    plan = _plan([_leg(AssetClass.EQUITY, "EQMX", 100, 100, "fnz-eq")])
    outcomes = [_outcome(AssetClass.EQUITY, 100, 100, "FILLED")]
    row = _run_row(plan, reconcile_rebalance_run(plan, outcomes))
    assert row.portfolio_id == plan.portfolio_id
    assert row.plan_id == "pid:2026-06-23"
    assert row.as_of == _CREATED.date()
    assert row.mode == "SANDBOX"
    assert row.budget_rub == Decimal(1_000_000)
    assert row.status == "COMPLETE"
    assert row.fill_rate == Decimal("1.0000")


def test_order_rows_use_plan_symbol_and_client_order_id() -> None:
    """A FAILED OrderResult left symbol blank -> the row must take it from the plan leg."""
    plan = _plan([_leg(AssetClass.OFZ_PK, "SU29024RMFS5", 50, 955, "fnz-ofz")])
    outcomes = [_outcome(AssetClass.OFZ_PK, 50, 0, "FAILED", "rejected by exchange")]
    run_id = uuid4()
    rows = _order_rows(run_id, plan, outcomes)
    assert len(rows) == 1
    row = rows[0]
    assert row.run_id == run_id
    assert row.asset_class == "ofz_pk"
    assert row.symbol == "SU29024RMFS5"  # from the plan leg, not the blank result.symbol
    assert row.client_order_id == "fnz-ofz"  # from the plan leg
    assert row.side == "BUY"
    assert row.requested_qty == Decimal(50)
    assert row.filled_qty == Decimal(0)
    assert row.status == "FAILED"
    assert row.reason == "rejected by exchange"
