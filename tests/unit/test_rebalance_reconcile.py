"""Phase 82 P82-02: pure post-rebalance reconciliation (planned vs filled)."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from uuid import uuid4

from finalayze.core.schemas import AssetClass
from finalayze.execution.broker_base import OrderRequest, OrderResult
from finalayze.orchestration.rebalance_planner import LegOutcome, PlannedLeg, RebalancePlan
from finalayze.orchestration.rebalance_reconcile import reconcile_rebalance_run


def _leg(asset_class: AssetClass, symbol: str, qty: int, price: int) -> PlannedLeg:
    return PlannedLeg(
        asset_class=asset_class,
        market_id="moex",
        order=OrderRequest(
            symbol=symbol, side="BUY", quantity=Decimal(qty), client_order_id=f"fnz-{symbol}"
        ),
        side="BUY",
        target_notional=Decimal(qty) * Decimal(price),
        est_price=Decimal(price),
    )


def _plan(legs: list[PlannedLeg]) -> RebalancePlan:
    return RebalancePlan(
        plan_id="pid:2026-06-23",
        created_at=datetime(2026, 6, 23, tzinfo=UTC),
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
        result=OrderResult(filled=filled > 0, quantity=Decimal(filled), reason=reason),
        status=status,
    )


def test_all_filled_is_complete() -> None:
    plan = _plan(
        [
            _leg(AssetClass.EQUITY, "EQMX", 100, 100),
            _leg(AssetClass.OFZ_PK, "SU29024RMFS5", 50, 955),
        ]
    )
    rec = reconcile_rebalance_run(
        plan,
        [
            _outcome(AssetClass.EQUITY, 100, 100, "FILLED"),
            _outcome(AssetClass.OFZ_PK, 50, 50, "FILLED"),
        ],
    )
    assert rec.status == "COMPLETE"
    assert rec.fill_rate == Decimal("1.0000")
    assert rec.alerts == ()
    assert len(rec.legs) == 2


def test_partial_fill() -> None:
    plan = _plan([_leg(AssetClass.EQUITY, "EQMX", 100, 100)])
    rec = reconcile_rebalance_run(
        plan, [_outcome(AssetClass.EQUITY, 100, 60, "PARTIAL", "partial fill")]
    )
    assert rec.status == "PARTIAL"
    assert rec.fill_rate == Decimal("0.6000")  # 60*100 / 100*100
    assert len(rec.alerts) == 1
    assert rec.legs[0].shortfall_qty == Decimal(40)


def test_all_failed_is_failed() -> None:
    plan = _plan([_leg(AssetClass.EQUITY, "EQMX", 100, 100)])
    rec = reconcile_rebalance_run(plan, [_outcome(AssetClass.EQUITY, 100, 0, "FAILED", "rejected")])
    assert rec.status == "FAILED"
    assert rec.fill_rate == Decimal("0.0000")
    assert len(rec.alerts) == 1


def test_no_outcomes_is_none() -> None:
    rec = reconcile_rebalance_run(_plan([]), [])
    assert rec.status == "NONE"
    assert rec.fill_rate == Decimal(1)


def test_notional_weighted_fill_rate() -> None:
    """fill_rate is RUB-weighted, not a raw qty sum across instruments."""
    plan = _plan(
        [
            _leg(AssetClass.EQUITY, "EQMX", 100, 100),
            _leg(AssetClass.OFZ_PK, "SU29024RMFS5", 50, 955),
        ]
    )
    rec = reconcile_rebalance_run(
        plan,
        [
            _outcome(AssetClass.EQUITY, 100, 100, "FILLED"),  # 10_000 RUB filled
            _outcome(
                AssetClass.OFZ_PK, 50, 0, "FAILED", "rejected"
            ),  # 47_750 RUB requested, 0 filled
        ],
    )
    assert rec.status == "PARTIAL"
    # 10_000 filled / 57_750 requested = 0.17316.. -> 0.1732
    assert rec.fill_rate == Decimal("0.1732")
