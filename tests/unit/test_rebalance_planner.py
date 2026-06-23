"""Phase 79 P79-01: frozen rebalance-plan dataclasses + deposit-never-an-order invariant.

The plan record is immutable (an audit artifact) and structurally forbids a DEPOSIT auto leg:
the deposit is mark-only with no broker API, so it can only ever surface as a ManualAction
(L-01). Constructing a plan whose auto_legs contains DEPOSIT must raise, not silently allow it.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import UTC, datetime
from decimal import Decimal
from uuid import uuid4

import pytest

from finalayze.core.schemas import AssetClass
from finalayze.execution.broker_base import OrderRequest, OrderResult
from finalayze.orchestration.rebalance_planner import (
    LegOutcome,
    LegSizing,
    ManualAction,
    PlannedLeg,
    RebalancePlan,
    size_auto_leg,
)

_BUDGET = Decimal(1_000_000)

_CREATED = datetime(2026, 1, 1, tzinfo=UTC)


def _equity_leg() -> PlannedLeg:
    return PlannedLeg(
        asset_class=AssetClass.EQUITY,
        market_id="moex",
        order=OrderRequest(
            symbol="EQMX", side="BUY", quantity=Decimal(10), client_order_id="fnz-eq"
        ),
        side="BUY",
        target_notional=Decimal(1000),
        est_price=Decimal(100),
    )


def _deposit_action() -> ManualAction:
    return ManualAction(
        asset_class=AssetClass.DEPOSIT,
        description="place 300000 RUB on deposit",
        target_notional=Decimal(300_000),
        current_notional=Decimal(0),
        funding_advisory=None,
    )


def _make_plan(
    *,
    auto_legs: tuple[PlannedLeg, ...] = (),
    manual_actions: tuple[ManualAction, ...] = (),
) -> RebalancePlan:
    return RebalancePlan(
        plan_id="p1",
        created_at=_CREATED,
        portfolio_id=uuid4(),
        risk_profile="balanced",
        budget_rub=Decimal(1_000_000),
        mode="DRY_RUN",
        auto_legs=auto_legs,
        manual_actions=manual_actions,
    )


def test_plan_constructs_with_auto_leg_and_deposit_action() -> None:
    """A plan holds an EQUITY auto leg and a DEPOSIT manual action."""
    plan = _make_plan(auto_legs=(_equity_leg(),), manual_actions=(_deposit_action(),))
    assert len(plan.auto_legs) == 1
    assert plan.auto_legs[0].asset_class is AssetClass.EQUITY
    assert plan.manual_actions[0].asset_class is AssetClass.DEPOSIT


def test_plan_is_frozen() -> None:
    """RebalancePlan is an immutable audit record; reassigning a field raises."""
    plan = _make_plan(auto_legs=(_equity_leg(),))
    with pytest.raises(FrozenInstanceError):
        plan.budget_rub = Decimal(0)  # type: ignore[misc]


def test_planned_leg_is_frozen() -> None:
    """A PlannedLeg cannot be mutated after construction."""
    leg = _equity_leg()
    with pytest.raises(FrozenInstanceError):
        leg.side = "SELL"  # type: ignore[misc]


def test_manual_action_is_frozen() -> None:
    """A ManualAction cannot be mutated after construction."""
    action = _deposit_action()
    with pytest.raises(FrozenInstanceError):
        action.target_notional = Decimal(0)  # type: ignore[misc]


def test_deposit_cannot_be_an_auto_leg() -> None:
    """The structural invariant: DEPOSIT never produces an order / auto leg (L-01)."""
    deposit_leg = PlannedLeg(
        asset_class=AssetClass.DEPOSIT,
        market_id="moex",
        order=OrderRequest(
            symbol="DEP", side="BUY", quantity=Decimal(1), client_order_id="fnz-dep"
        ),
        side="BUY",
        target_notional=Decimal(1),
        est_price=None,
    )
    with pytest.raises(ValueError, match="DEPOSIT"):
        _make_plan(auto_legs=(deposit_leg,))


def test_leg_outcome_constructs() -> None:
    """LegOutcome wraps the per-leg submit result with a classification status."""
    outcome = LegOutcome(
        asset_class=AssetClass.EQUITY,
        requested_qty=Decimal(10),
        result=OrderResult(filled=True, quantity=Decimal(10)),
        status="FILLED",
    )
    assert outcome.status == "FILLED"
    assert outcome.result.filled is True


class TestSizeAutoLeg:
    """P79-03/04/05: pure signed delta sizing, no-churn band, lot-size flooring."""

    def test_first_build_is_buy(self) -> None:
        """current=0 -> a full BUY of the lot-floored target qty (P79-R3)."""
        sizing = size_auto_leg(
            target_notional=Decimal(550_000),
            est_price=Decimal(100),
            current_qty=Decimal(0),
            lot_size=1,
            budget_rub=_BUDGET,
        )
        assert sizing is not None
        assert sizing.side == "BUY"
        assert sizing.delta_qty == Decimal(5500)

    def test_overweight_is_sell_of_signed_delta(self) -> None:
        """Holding more than target -> SELL the signed delta, not the absolute target (P79-R3)."""
        sizing = size_auto_leg(
            target_notional=Decimal(550_000),
            est_price=Decimal(100),
            current_qty=Decimal(6000),  # target qty is 5500 -> sell 500
            lot_size=1,
            budget_rub=_BUDGET,
        )
        assert sizing is not None
        assert sizing.side == "SELL"
        assert sizing.delta_qty == Decimal(500)

    def test_dust_below_band_is_suppressed(self) -> None:
        """|delta_notional| < 2% of budget -> no trade (P79-R4)."""
        sizing = size_auto_leg(
            target_notional=Decimal(510_000),  # vs current 500_000 -> delta 10_000 < 20_000 band
            est_price=Decimal(100),
            current_qty=Decimal(5000),
            lot_size=1,
            budget_rub=_BUDGET,
        )
        assert sizing is None

    def test_above_band_emits(self) -> None:
        """|delta_notional| above the band -> a real leg (P79-R4)."""
        sizing = size_auto_leg(
            target_notional=Decimal(530_000),  # vs current 500_000 -> delta 30_000 > 20_000 band
            est_price=Decimal(100),
            current_qty=Decimal(5000),
            lot_size=1,
            budget_rub=_BUDGET,
        )
        assert sizing is not None
        assert sizing.side == "BUY"
        assert sizing.delta_qty == Decimal(300)

    def test_lot_floor_rounds_down(self) -> None:
        """A 185-share target at lot_size 10 floors to 180 (matches the broker rule, P79-R5)."""
        sizing = size_auto_leg(
            target_notional=Decimal(18_500),
            est_price=Decimal(100),
            current_qty=Decimal(0),
            lot_size=10,
            budget_rub=Decimal(100_000),
        )
        assert sizing is not None
        assert sizing.delta_qty == Decimal(180)

    def test_below_one_lot_is_suppressed(self) -> None:
        """A target under one lot (7 shares, lot 10) emits no leg even above the band (P79-R5)."""
        sizing = size_auto_leg(
            target_notional=Decimal(700),
            est_price=Decimal(100),
            current_qty=Decimal(0),
            lot_size=10,
            budget_rub=Decimal(10_000),  # band 200 < delta 700, so band passes; lot floor kills it
        )
        assert sizing is None

    def test_zero_price_raises(self) -> None:
        """A non-positive price cannot size a leg (avoids div-by-zero on a money path)."""
        with pytest.raises(ValueError, match="price"):
            size_auto_leg(
                target_notional=Decimal(1000),
                est_price=Decimal(0),
                current_qty=Decimal(0),
                lot_size=1,
                budget_rub=_BUDGET,
            )

    def test_returns_legsizing_type(self) -> None:
        """The sizing result is the immutable LegSizing value type."""
        sizing = size_auto_leg(
            target_notional=Decimal(550_000),
            est_price=Decimal(100),
            current_qty=Decimal(0),
            lot_size=1,
            budget_rub=_BUDGET,
        )
        assert isinstance(sizing, LegSizing)
