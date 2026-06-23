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
from decimal import Decimal
from typing import TYPE_CHECKING, Literal

from finalayze.config.rebalance_config import SAA_REBALANCE_BAND_PCT
from finalayze.core.schemas import AssetClass

if TYPE_CHECKING:
    from datetime import datetime
    from uuid import UUID

    from finalayze.execution.broker_base import OrderRequest, OrderResult
    from finalayze.orchestration.allocation import FundingBreakdown

Mode = Literal["DRY_RUN", "SANDBOX", "LIVE"]
Side = Literal["BUY", "SELL"]
LegStatus = Literal["FILLED", "PARTIAL", "FAILED", "SKIPPED_BELOW_LOT"]

_ZERO = Decimal(0)


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


@dataclass(frozen=True)
class LegSizing:
    """Pure signed sizing result for one AUTO leg (no order/instrument context yet).

    ``delta_qty`` is the POSITIVE number of units to trade in direction ``side``, already
    floored to the instrument lot size. ``delta_notional`` is the signed RUB delta
    (``target - current``) used for the no-churn band decision.
    """

    side: Side
    delta_qty: Decimal
    delta_notional: Decimal
    target_notional: Decimal


def size_auto_leg(
    *,
    target_notional: Decimal,
    est_price: Decimal,
    current_qty: Decimal,
    lot_size: int,
    budget_rub: Decimal,
    band_pct: Decimal = SAA_REBALANCE_BAND_PCT,
) -> LegSizing | None:
    """Size one AUTO leg from its target notional and current holding (pure, P79-R3/R4/R5).

    Returns ``None`` when the leg should NOT trade:
    - the signed RUB delta is within the no-churn band (``|delta| < band_pct * budget``), or
    - the lot-floored trade quantity is below one lot.

    Otherwise returns a ``LegSizing`` with a positive, lot-floored ``delta_qty`` and a BUY/SELL
    ``side`` derived from the signed delta (not the absolute target).

    Raises:
        ValueError: If ``est_price`` is not positive (cannot size / would divide by zero).
    """
    if est_price <= _ZERO:
        msg = f"est_price must be positive to size a leg; got {est_price}"
        raise ValueError(msg)

    current_notional = current_qty * est_price
    delta_notional = target_notional - current_notional

    # No-churn / dust band on the signed RUB delta -- dust must not churn the book.
    if abs(delta_notional) < band_pct * budget_rub:
        return None

    target_qty = target_notional / est_price
    delta_qty = target_qty - current_qty
    side: Side = "BUY" if delta_qty > _ZERO else "SELL"

    # Floor the ABSOLUTE delta down to the instrument lot size, mirroring the broker's
    # floor(qty / lot) * lot rule so the planned qty equals the qty the broker will accept.
    lot = Decimal(lot_size)
    floored = (abs(delta_qty) // lot) * lot
    if floored <= _ZERO:
        return None  # below one lot -> no order (SKIPPED_BELOW_LOT at plan time)

    return LegSizing(
        side=side,
        delta_qty=floored,
        delta_notional=delta_notional,
        target_notional=target_notional,
    )
