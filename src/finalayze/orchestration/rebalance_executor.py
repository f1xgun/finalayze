"""Rebalance executor -- dispatch a RebalancePlan's AUTO legs through the broker router (Phase 79).

DRY_RUN by default; LIVE is TRIPLE-gated (L-03 hard stop, P79-R10): a live submission requires
``plan.mode == "LIVE"`` AND ``mode_manager.current_mode == WorkMode.REAL`` (which itself needs
``FINALAYZE_REAL_CONFIRMED=true``) AND an explicit per-call ``confirm=True``. No code path defaults
to LIVE. The deposit ``ManualAction`` is surfaced/logged for the operator, never submitted.

The executor receives an ALREADY-WIRED ``BrokerRouter`` -- it constructs no brokers and never
re-runs an engine. It is SYNCHRONOUS (``TinkoffBroker.submit_order`` + ``RetryPolicy`` block on
gRPC); an async caller must offload via ``asyncio.to_thread``. A failing leg does NOT abort the
others -- each leg returns its own classified ``LegOutcome`` (P79-R11).
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

import structlog

from finalayze.core.modes import WorkMode
from finalayze.execution.broker_base import OrderResult
from finalayze.orchestration.rebalance_planner import LegOutcome

if TYPE_CHECKING:
    from finalayze.core.modes import ModeManager
    from finalayze.execution.broker_router import BrokerRouter
    from finalayze.orchestration.rebalance_planner import LegStatus, RebalancePlan

_log = structlog.get_logger()

_ZERO = Decimal(0)
_LOT_REJECT_MARKER = "lot size"  # TinkoffBroker's below-lot reject: "... less than lot size N"


def _classify_outcome(result: OrderResult, requested_qty: Decimal) -> LegStatus:
    """Map a broker ``OrderResult`` to a per-leg status (P79-R11).

    - filled, full quantity        -> FILLED
    - filled, less than requested  -> PARTIAL
    - not filled, below-lot reject  -> SKIPPED_BELOW_LOT
    - not filled, anything else    -> FAILED
    """
    if result.filled:
        return "PARTIAL" if result.quantity < requested_qty else "FILLED"
    if _LOT_REJECT_MARKER in result.reason.lower():
        return "SKIPPED_BELOW_LOT"
    return "FAILED"


def _enforce_live_gate(plan: RebalancePlan, mode_manager: ModeManager, *, confirm: bool) -> None:
    """Enforce the LIVE triple gate (L-03 hard stop). No-op for DRY_RUN / SANDBOX.

    Raises:
        PermissionError: If a LIVE plan is missing ``confirm=True`` or the ModeManager is not in
            ``WorkMode.REAL``. Refusal is explicit -- a LIVE request never silently downgrades.
    """
    if plan.mode != "LIVE":
        return
    if not confirm:
        msg = "LIVE rebalance requires an explicit confirm=True (real-money hard stop)"
        raise PermissionError(msg)
    if mode_manager.current_mode != WorkMode.REAL:
        msg = (
            "LIVE rebalance requires the ModeManager in WorkMode.REAL "
            f"(needs FINALAYZE_REAL_CONFIRMED=true); current mode is {mode_manager.current_mode}"
        )
        raise PermissionError(msg)


def submit_rebalance_plan(
    plan: RebalancePlan,
    broker_router: BrokerRouter,
    mode_manager: ModeManager,
    *,
    confirm: bool = False,
) -> list[LegOutcome]:
    """Dispatch the plan's AUTO legs through the router and classify each outcome (P79-R9/R11).

    DRY_RUN by default (the caller wires a SimulatedBroker into the router so no live channel is
    touched). LIVE is triple-gated (``_enforce_live_gate``). The deposit ``ManualAction`` is
    logged for the operator, never submitted. A failing leg is isolated -- it does not abort the
    remaining legs; every leg returns its own ``LegOutcome``.

    Raises:
        PermissionError: If the LIVE triple gate is not satisfied (before any submission).
    """
    _enforce_live_gate(plan, mode_manager, confirm=confirm)

    for action in plan.manual_actions:
        _log.info(
            "rebalance_manual_action",
            asset_class=action.asset_class.value,
            description=action.description,
            target_notional=str(action.target_notional),
        )

    outcomes: list[LegOutcome] = []
    for leg in plan.auto_legs:
        requested_qty = leg.order.quantity
        try:
            result = broker_router.submit(leg.order, market_id=leg.market_id)
        except Exception as exc:  # intentional broad catch: one leg's failure must not abort others
            _log.error(
                "rebalance_leg_submit_error",
                asset_class=leg.asset_class.value,
                symbol=leg.order.symbol,
                error=str(exc),
            )
            result = OrderResult(
                filled=False,
                symbol=leg.order.symbol,
                side=leg.side,
                quantity=_ZERO,
                reason=f"submit error: {exc}",
            )
        status = _classify_outcome(result, requested_qty)
        outcomes.append(
            LegOutcome(
                asset_class=leg.asset_class,
                requested_qty=requested_qty,
                result=result,
                status=status,
            )
        )
        _log.info(
            "rebalance_leg_outcome",
            asset_class=leg.asset_class.value,
            status=status,
            mode=plan.mode,
            requested=str(requested_qty),
            filled=str(result.quantity),
        )
    return outcomes
