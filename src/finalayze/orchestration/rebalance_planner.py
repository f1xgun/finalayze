"""Weights-to-orders planner -- turns SAA target weights into a REBALANCE PLAN (Phase 79).

The frozen ``AllocationOrchestrator`` is analytics-only; this module is the missing path from a
per-leg target (``budget * weight``) to a concrete broker order. It is a PURE, broker-free,
deterministic planner: it owns no broker handle, performs no I/O, and constructs no live channel
(P79-R1). The result is a ``RebalancePlan`` -- an immutable audit record holding:

- ``auto_legs``: real ``OrderRequest`` legs for EQUITY + OFZ_PK (routed to Tinkoff via the broker
  router by the executor), and
- ``manual_actions``: a DEPOSIT operator action item only (the deposit is mark-only, no T-Bank
  deposit API -- it NEVER produces an order, enforced structurally in ``RebalancePlan``).

This file currently defines the plan dataclasses (P79-01); the ``plan_rebalance`` sizing logic
lands in subsequent subtasks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from finalayze.core.schemas import AssetClass

if TYPE_CHECKING:
    from datetime import datetime
    from decimal import Decimal
    from uuid import UUID

    from finalayze.execution.broker_base import OrderRequest, OrderResult
    from finalayze.orchestration.allocation import FundingBreakdown

Mode = Literal["DRY_RUN", "SANDBOX", "LIVE"]
Side = Literal["BUY", "SELL"]
LegStatus = Literal["FILLED", "PARTIAL", "FAILED", "SKIPPED_BELOW_LOT"]


@dataclass(frozen=True)
class PlannedLeg:
    """One AUTO (broker-routed) leg of a rebalance plan: a concrete order + its market."""

    asset_class: AssetClass
    market_id: str
    order: OrderRequest
    side: Side
    target_notional: Decimal
    est_price: Decimal | None = None


@dataclass(frozen=True)
class ManualAction:
    """A MANUAL operator action item (DEPOSIT only) -- never a broker order.

    ``funding_advisory`` is a READ-ONLY breakdown of where the cash should come from when the
    deposit must be raised from the sleeve (negative delta); the engine never executes the move.
    """

    asset_class: AssetClass
    description: str
    target_notional: Decimal
    current_notional: Decimal
    funding_advisory: FundingBreakdown | None = None


@dataclass(frozen=True)
class LegOutcome:
    """The executor's per-leg result: the submit ``OrderResult`` + a classification status."""

    asset_class: AssetClass
    requested_qty: Decimal
    result: OrderResult
    status: LegStatus


@dataclass(frozen=True)
class RebalancePlan:
    """Immutable rebalance plan: AUTO order legs + MANUAL action items for one active portfolio.

    Structural invariant (L-01): DEPOSIT may NEVER appear in ``auto_legs`` -- it is a mark-only
    manual action with no broker API. Constructing such a plan raises ``ValueError``.
    """

    plan_id: str
    created_at: datetime
    portfolio_id: UUID
    risk_profile: str
    budget_rub: Decimal
    mode: Mode
    auto_legs: tuple[PlannedLeg, ...]
    manual_actions: tuple[ManualAction, ...]

    def __post_init__(self) -> None:
        for leg in self.auto_legs:
            if leg.asset_class is AssetClass.DEPOSIT:
                msg = (
                    "DEPOSIT cannot be an auto leg / broker order -- it is a mark-only "
                    "manual action (L-01)"
                )
                raise ValueError(msg)
