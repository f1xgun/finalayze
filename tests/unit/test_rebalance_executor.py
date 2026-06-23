"""Phase 79 P79-11/12: dry-run executor, per-leg outcome classification, live triple gate.

The executor must (a) classify each broker OrderResult correctly, (b) isolate a failing leg so it
never aborts the others, (c) never submit the deposit ManualAction, and (d) refuse a LIVE
submission unless mode==LIVE AND ModeManager==REAL AND confirm=True (the real-money hard stop).
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from uuid import uuid4

import pytest

from finalayze.core.modes import ModeManager, WorkMode
from finalayze.core.schemas import AssetClass
from finalayze.execution.broker_base import OrderRequest, OrderResult
from finalayze.execution.broker_router import BrokerRouter
from finalayze.orchestration.rebalance_executor import submit_rebalance_plan
from finalayze.orchestration.rebalance_planner import (
    ManualAction,
    Mode,
    PlannedLeg,
    RebalancePlan,
)


class _ProgrammedBroker:
    """A fake broker returning a pre-programmed OrderResult per symbol; records submitted orders."""

    def __init__(self, by_symbol: dict[str, OrderResult]) -> None:
        self._by_symbol = by_symbol
        self.submitted: list[OrderRequest] = []

    def submit_order(self, order: OrderRequest, fill_candle: object = None) -> OrderResult:
        self.submitted.append(order)
        if order.symbol not in self._by_symbol:
            msg = f"unprogrammed symbol {order.symbol}"
            raise KeyError(msg)
        return self._by_symbol[order.symbol]


class _RaisingBroker:
    """A fake broker that raises on submit (to test per-leg isolation)."""

    def submit_order(self, order: OrderRequest, fill_candle: object = None) -> OrderResult:
        msg = "gRPC channel down"
        raise RuntimeError(msg)


def _leg(asset_class: AssetClass, symbol: str, qty: int) -> PlannedLeg:
    return PlannedLeg(
        asset_class=asset_class,
        market_id="moex",
        order=OrderRequest(
            symbol=symbol, side="BUY", quantity=Decimal(qty), client_order_id=f"fnz-{symbol}"
        ),
        side="BUY",
        target_notional=Decimal(qty),
        est_price=Decimal(1),
    )


def _plan(
    legs: tuple[PlannedLeg, ...],
    *,
    mode: Mode = "DRY_RUN",
    manual: tuple[ManualAction, ...] = (),
) -> RebalancePlan:
    return RebalancePlan(
        plan_id="p",
        created_at=datetime(2026, 6, 23, tzinfo=UTC),
        portfolio_id=uuid4(),
        risk_profile="balanced",
        budget_rub=Decimal(1_000_000),
        mode=mode,
        auto_legs=legs,
        manual_actions=manual,
    )


def _router(broker: object) -> BrokerRouter:
    return BrokerRouter({"moex": broker})  # type: ignore[dict-item]


def test_filled_and_partial_classified() -> None:
    """A full fill -> FILLED; a fill below the requested quantity -> PARTIAL."""
    broker = _ProgrammedBroker(
        {
            "EQMX": OrderResult(filled=True, symbol="EQMX", quantity=Decimal(100)),
            "SU29024RMFS5": OrderResult(
                filled=True, symbol="SU29024RMFS5", quantity=Decimal(50), reason="partial fill"
            ),
        }
    )
    plan = _plan(
        (_leg(AssetClass.EQUITY, "EQMX", 100), _leg(AssetClass.OFZ_PK, "SU29024RMFS5", 100))
    )
    outcomes = submit_rebalance_plan(plan, _router(broker), ModeManager())
    by_class = {o.asset_class: o.status for o in outcomes}
    assert by_class[AssetClass.EQUITY] == "FILLED"
    assert by_class[AssetClass.OFZ_PK] == "PARTIAL"


def test_failed_and_below_lot_classified() -> None:
    """A non-fill reject -> FAILED; a below-lot reject (reason mentions lot size) -> SKIPPED."""
    broker = _ProgrammedBroker(
        {
            "EQMX": OrderResult(
                filled=False, symbol="EQMX", quantity=Decimal(0), reason="rejected by exchange"
            ),
            "SU29024RMFS5": OrderResult(
                filled=False,
                symbol="SU29024RMFS5",
                quantity=Decimal(0),
                reason="Quantity 0.5 is less than lot size 1",
            ),
        }
    )
    plan = _plan(
        (_leg(AssetClass.EQUITY, "EQMX", 100), _leg(AssetClass.OFZ_PK, "SU29024RMFS5", 100))
    )
    outcomes = submit_rebalance_plan(plan, _router(broker), ModeManager())
    by_class = {o.asset_class: o.status for o in outcomes}
    assert by_class[AssetClass.EQUITY] == "FAILED"
    assert by_class[AssetClass.OFZ_PK] == "SKIPPED_BELOW_LOT"


def test_failed_reason_mentioning_lot_size_is_not_misclassified_skipped() -> None:
    """A FAILED whose reason merely mentions a lot size stays FAILED, not SKIPPED (WR-01)."""
    broker = _ProgrammedBroker(
        {
            "EQMX": OrderResult(
                filled=False,
                symbol="EQMX",
                quantity=Decimal(0),
                reason="order rejected: lot size mismatch on instrument",
            ),
        }
    )
    plan = _plan((_leg(AssetClass.EQUITY, "EQMX", 100),))
    outcomes = submit_rebalance_plan(plan, _router(broker), ModeManager())
    assert outcomes[0].status == "FAILED"


def test_failed_leg_does_not_abort_others() -> None:
    """A broker that RAISES on a leg isolates that failure; other legs still execute."""
    # The raising broker hits both legs; assert both come back as FAILED outcomes (not an abort).
    plan = _plan(
        (_leg(AssetClass.EQUITY, "EQMX", 100), _leg(AssetClass.OFZ_PK, "SU29024RMFS5", 50))
    )
    outcomes = submit_rebalance_plan(plan, _router(_RaisingBroker()), ModeManager())
    assert len(outcomes) == 2
    assert all(o.status == "FAILED" for o in outcomes)
    assert all("submit error" in o.result.reason for o in outcomes)


def test_manual_action_is_not_submitted() -> None:
    """The deposit ManualAction is surfaced, never sent to the broker."""
    broker = _ProgrammedBroker(
        {"EQMX": OrderResult(filled=True, symbol="EQMX", quantity=Decimal(10))}
    )
    deposit = ManualAction(
        asset_class=AssetClass.DEPOSIT,
        description="place 450000 RUB on deposit",
        target_notional=Decimal(450_000),
        current_notional=Decimal(0),
    )
    plan = _plan((_leg(AssetClass.EQUITY, "EQMX", 10),), manual=(deposit,))
    submit_rebalance_plan(plan, _router(broker), ModeManager())
    assert [o.symbol for o in broker.submitted] == ["EQMX"]  # only the equity leg, no deposit


def test_dry_run_default_proceeds_without_confirm() -> None:
    """A DRY_RUN plan submits without confirm (no live gate)."""
    broker = _ProgrammedBroker(
        {"EQMX": OrderResult(filled=True, symbol="EQMX", quantity=Decimal(10))}
    )
    plan = _plan((_leg(AssetClass.EQUITY, "EQMX", 10),), mode="DRY_RUN")
    outcomes = submit_rebalance_plan(plan, _router(broker), ModeManager())
    assert outcomes[0].status == "FILLED"


def test_live_without_confirm_raises() -> None:
    """A LIVE plan without confirm=True is refused (hard stop, P79-R10)."""
    broker = _ProgrammedBroker(
        {"EQMX": OrderResult(filled=True, symbol="EQMX", quantity=Decimal(10))}
    )
    plan = _plan((_leg(AssetClass.EQUITY, "EQMX", 10),), mode="LIVE")
    with pytest.raises(PermissionError, match="confirm"):
        submit_rebalance_plan(plan, _router(broker), ModeManager(), confirm=False)
    assert broker.submitted == []  # nothing submitted


def test_live_confirm_but_not_real_mode_raises() -> None:
    """A LIVE plan with confirm but a non-REAL ModeManager is refused (P79-R10)."""
    broker = _ProgrammedBroker(
        {"EQMX": OrderResult(filled=True, symbol="EQMX", quantity=Decimal(10))}
    )
    plan = _plan((_leg(AssetClass.EQUITY, "EQMX", 10),), mode="LIVE")
    with pytest.raises(PermissionError, match="REAL"):
        submit_rebalance_plan(plan, _router(broker), ModeManager(WorkMode.DEBUG), confirm=True)
    assert broker.submitted == []


def test_live_full_triple_gate_proceeds(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only the full triple (mode==LIVE, confirm=True, ModeManager==REAL) reaches submission."""
    monkeypatch.setenv("FINALAYZE_REAL_CONFIRMED", "true")
    broker = _ProgrammedBroker(
        {"EQMX": OrderResult(filled=True, symbol="EQMX", quantity=Decimal(10))}
    )
    plan = _plan((_leg(AssetClass.EQUITY, "EQMX", 10),), mode="LIVE")
    outcomes = submit_rebalance_plan(
        plan, _router(broker), ModeManager(WorkMode.REAL), confirm=True
    )
    assert outcomes[0].status == "FILLED"
    assert [o.symbol for o in broker.submitted] == ["EQMX"]
