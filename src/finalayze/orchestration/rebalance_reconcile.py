"""Post-rebalance reconciliation -- planned vs actually-filled (Phase 82).

Pure: compares a ``RebalancePlan``'s AUTO legs against the executor's ``LegOutcome``s and produces a
per-leg + overall report (status rollup, RUB-weighted fill rate, alerts). No DB, no broker -- the
inputs (requested qty, filled qty, status, the plan's est_price) are all already in memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from finalayze.orchestration.rebalance_planner import LegOutcome, RebalancePlan

_ZERO = Decimal(0)
_RATE_QUANT = Decimal("0.0001")


@dataclass(frozen=True)
class LegReconciliation:
    """One AUTO leg's planned-vs-filled reconciliation."""

    asset_class: str
    requested_qty: Decimal
    filled_qty: Decimal
    status: str
    shortfall_qty: Decimal  # max(0, requested - filled)


@dataclass(frozen=True)
class RebalanceReconciliation:
    """The whole run's reconciliation: per-leg detail + an overall rollup."""

    plan_id: str
    status: str  # COMPLETE | PARTIAL | FAILED | NONE
    fill_rate: Decimal  # RUB-weighted filled/requested notional (4dp); 1 when nothing to trade
    legs: tuple[LegReconciliation, ...]
    alerts: tuple[str, ...]


def _rollup_status(statuses: set[str]) -> str:
    """Roll per-leg statuses up to a run status."""
    if not statuses:
        return "NONE"
    if statuses == {"FILLED"}:
        return "COMPLETE"
    if "FILLED" not in statuses and "PARTIAL" not in statuses:
        return "FAILED"  # nothing filled at all (all FAILED / SKIPPED_BELOW_LOT)
    return "PARTIAL"


def reconcile_rebalance_run(
    plan: RebalancePlan, outcomes: Sequence[LegOutcome]
) -> RebalanceReconciliation:
    """Reconcile executor outcomes against the plan (pure, P82-R5).

    Per leg: requested vs filled qty, status, and the shortfall. Overall: a status rollup
    (COMPLETE/PARTIAL/FAILED/NONE), a RUB-weighted fill rate (filled notional / requested notional
    using the plan leg's ``est_price``), and an alert for every non-FILLED leg.
    """
    plan_legs = {leg.asset_class: leg for leg in plan.auto_legs}

    legs: list[LegReconciliation] = []
    alerts: list[str] = []
    total_requested_notional = _ZERO
    total_filled_notional = _ZERO

    for outcome in outcomes:
        requested = outcome.requested_qty
        filled = outcome.result.quantity
        plan_leg = plan_legs.get(outcome.asset_class)
        price = plan_leg.est_price if plan_leg is not None and plan_leg.est_price else _ZERO
        total_requested_notional += requested * price
        total_filled_notional += filled * price

        legs.append(
            LegReconciliation(
                asset_class=outcome.asset_class.value,
                requested_qty=requested,
                filled_qty=filled,
                status=outcome.status,
                shortfall_qty=max(_ZERO, requested - filled),
            )
        )
        if outcome.status != "FILLED":
            alerts.append(
                f"{outcome.asset_class.value}: {outcome.status} "
                f"(filled {filled} of {requested}; {outcome.result.reason or 'no reason'})"
            )

    fill_rate = (
        (total_filled_notional / total_requested_notional).quantize(_RATE_QUANT)
        if total_requested_notional > _ZERO
        else Decimal(1)
    )
    return RebalanceReconciliation(
        plan_id=plan.plan_id,
        status=_rollup_status({o.status for o in outcomes}),
        fill_rate=fill_rate,
        legs=tuple(legs),
        alerts=tuple(alerts),
    )
