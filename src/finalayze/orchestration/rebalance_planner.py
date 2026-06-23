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

import copy
import hashlib
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, Literal

from finalayze.config.rebalance_config import SAA_REBALANCE_BAND_PCT
from finalayze.core.exceptions import InstrumentNotFoundError
from finalayze.core.schemas import AssetClass
from finalayze.execution.broker_base import OrderRequest

if TYPE_CHECKING:
    from collections.abc import Mapping
    from datetime import date, datetime
    from uuid import UUID

    from finalayze.execution.broker_base import OrderResult
    from finalayze.execution.deposit_broker import DepositSimulatedBroker
    from finalayze.markets.instruments import Instrument
    from finalayze.orchestration.allocation import FundingBreakdown

Mode = Literal["DRY_RUN", "SANDBOX", "LIVE"]
Side = Literal["BUY", "SELL"]
LegStatus = Literal["FILLED", "PARTIAL", "FAILED", "SKIPPED_BELOW_LOT"]

_ZERO = Decimal(0)
# Notional sanity tolerance: leg targets must sum to the budget within this RUB epsilon (P79-R12).
_NOTIONAL_TOLERANCE = Decimal("0.01")

# The two AUTO (broker-routed) legs, in deterministic order. DEPOSIT is intentionally absent --
# it is a MANUAL action item only (L-01), handled separately by plan_rebalance.
_AUTO_CLASSES: tuple[AssetClass, ...] = (AssetClass.EQUITY, AssetClass.OFZ_PK)


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

    # No-churn / dust band on the signed RUB delta -- dust must not churn the book. Strict ``<``:
    # a delta EXACTLY at the band threshold trades (band = the suppress-below boundary, INFO-03).
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


def _deterministic_client_order_id(plan_id: str, asset_class: AssetClass, side: Side) -> str:
    """Derive a stable, replay-safe client_order_id from (plan_id, asset_class, side) (P79-R6).

    Re-running the planner for the same plan_id yields a byte-identical id, so the broker's
    idempotent ``post_order(order_id=...)`` collapses accidental duplicates on the money path.
    SHA-256 (usedforsecurity=False -- this is an identifier digest, not a security primitive),
    truncated to a fixed 24 hex chars under the broker's ``fnz-`` convention.
    """
    raw = f"{plan_id}|{asset_class.value}|{side}"
    digest = hashlib.sha256(raw.encode("utf-8"), usedforsecurity=False).hexdigest()[:24]
    return f"fnz-{digest}"


def _deposit_description(delta_notional: Decimal) -> str:
    """Human-readable operator instruction for the DEPOSIT leg, by delta sign."""
    if delta_notional > _ZERO:
        return f"DEPOSIT: place {delta_notional} RUB on a bank deposit (underweight by this amount)"
    if delta_notional < _ZERO:
        return (
            f"DEPOSIT: withdraw {abs(delta_notional)} RUB from the deposit sleeve "
            "to fund the other legs"
        )
    return "DEPOSIT: on target, no action"


def compute_funding_advisory(
    deposit_broker: DepositSimulatedBroker,
    need: Decimal,
    as_of: date,
) -> FundingBreakdown:
    """Compute a READ-ONLY deposit funding breakdown WITHOUT mutating the real broker (P79-R8).

    The strict lockup-respecting order (matured -> income -> cash -> last-resort break) is the
    same one the analytics path uses, but here it runs on a ``deepcopy`` SHADOW of the broker so
    the real deposit sleeve is never broken/withdrawn. The returned ``FundingBreakdown`` tells the
    operator how much to raise and from where; the operator performs the actual deposit move.
    """
    from finalayze.orchestration.allocation import (  # noqa: PLC0415 -- avoid load-time coupling
        _fund_underweight_breakdown,
    )

    shadow = copy.deepcopy(deposit_broker)
    return _fund_underweight_breakdown(shadow, need, as_of)


def _require_resolved_instruments(leg_instruments: Mapping[AssetClass, Instrument]) -> None:
    """Fail loud if any AUTO leg is missing an instrument or a FIGI (P79-R13).

    Validated UPFRONT across ALL auto legs so a missing FIGI aborts the WHOLE plan -- there is
    never a half-rebalance where one leg's order is built and another's silently dropped.
    """
    for asset_class in _AUTO_CLASSES:
        instrument = leg_instruments.get(asset_class)
        if instrument is None or not instrument.figi:
            msg = (
                f"no resolved FIGI for the {asset_class.value} leg; aborting the whole plan "
                "(no half-rebalance)"
            )
            raise InstrumentNotFoundError(msg)


def _assert_notional_sane(
    budget_rub: Decimal, target_weights: Mapping[AssetClass, Decimal]
) -> None:
    """Guard the weight vector: complete, every leg target >= 0, none > budget, sum == budget.

    A missing leg is an unsound weight vector and raises a clear ``ValueError`` (P79-R12) rather
    than an opaque ``KeyError`` from the target computation below (WR-02).
    """
    missing = [ac.value for ac in AssetClass if ac not in target_weights]
    if missing:
        msg = f"weight vector is missing legs: {missing} (must define all of deposit/ofz_pk/equity)"
        raise ValueError(msg)
    targets = {ac: budget_rub * target_weights[ac] for ac in AssetClass}
    for asset_class, target in targets.items():
        if target < _ZERO:
            msg = f"leg {asset_class.value} target notional is negative: {target}"
            raise ValueError(msg)
        if target > budget_rub:
            msg = f"leg {asset_class.value} target {target} exceeds the budget {budget_rub}"
            raise ValueError(msg)
    total = sum(targets.values(), _ZERO)
    if abs(total - budget_rub) >= _NOTIONAL_TOLERANCE:
        msg = f"leg targets sum to {total}, which != budget {budget_rub} (weights must sum to 1)"
        raise ValueError(msg)


def plan_rebalance(
    *,
    active_portfolio: tuple[UUID, str, Decimal],
    target_weights: Mapping[AssetClass, Decimal],
    current_positions: Mapping[str, Decimal],
    last_prices: Mapping[str, Decimal],
    leg_instruments: Mapping[AssetClass, Instrument],
    deposit_current_notional: Decimal,
    plan_id: str,
    created_at: datetime,
    mode: Mode = "DRY_RUN",
    deposit_broker: DepositSimulatedBroker | None = None,
    as_of: date | None = None,
) -> RebalancePlan:
    """Turn target weights + current holdings into a REBALANCE PLAN (pure, P79-R1/R6/R7/R8).

    For each AUTO class (EQUITY, OFZ_PK): ``target = budget * weight``, size the signed delta
    against the current holding (``size_auto_leg``), and emit a ``PlannedLeg`` with a
    deterministic ``OrderRequest`` (or nothing, when within-band / below-one-lot). The DEPOSIT
    class becomes a single ``ManualAction`` (never an order); when the deposit is overweight
    (negative delta) and a ``deposit_broker`` + ``as_of`` are supplied, a READ-ONLY funding
    advisory is attached. The planner owns no broker handle and performs no I/O.

    Raises:
        InstrumentNotFoundError: If an AUTO leg has no resolved instrument/FIGI (whole-plan abort).
        ValueError: If the weight vector is unsound (negative / over-budget / not summing to 1).
    """
    portfolio_id, risk_profile, budget_rub = active_portfolio

    # Validate UPFRONT (fail fast, no half-plan) before constructing any order.
    _require_resolved_instruments(leg_instruments)
    _assert_notional_sane(budget_rub, target_weights)

    auto_legs: list[PlannedLeg] = []
    for asset_class in _AUTO_CLASSES:
        instrument = leg_instruments[asset_class]
        symbol = instrument.symbol
        target_notional = budget_rub * target_weights[asset_class]
        if symbol not in last_prices:
            # Fail loud like the FIGI guard (P79-R13) rather than an opaque KeyError (INFO-01).
            msg = f"no est_price for the {asset_class.value} leg symbol {symbol!r}"
            raise ValueError(msg)
        est_price = last_prices[symbol]
        current_qty = current_positions.get(symbol, _ZERO)
        sizing = size_auto_leg(
            target_notional=target_notional,
            est_price=est_price,
            current_qty=current_qty,
            lot_size=instrument.lot_size,
            budget_rub=budget_rub,
        )
        if sizing is None:
            continue
        order = OrderRequest(
            symbol=symbol,
            side=sizing.side,
            quantity=sizing.delta_qty,
            client_order_id=_deterministic_client_order_id(plan_id, asset_class, sizing.side),
        )
        auto_legs.append(
            PlannedLeg(
                asset_class=asset_class,
                market_id=instrument.market_id,
                order=order,
                side=sizing.side,
                target_notional=target_notional,
                est_price=est_price,
            )
        )

    # DEPOSIT -> MANUAL action item only (mark-only, no broker API).
    deposit_target = budget_rub * target_weights[AssetClass.DEPOSIT]
    deposit_delta = deposit_target - deposit_current_notional
    advisory: FundingBreakdown | None = None
    if deposit_delta < _ZERO and deposit_broker is not None and as_of is not None:
        advisory = compute_funding_advisory(deposit_broker, abs(deposit_delta), as_of)
    manual_actions = (
        ManualAction(
            asset_class=AssetClass.DEPOSIT,
            description=_deposit_description(deposit_delta),
            target_notional=deposit_target,
            current_notional=deposit_current_notional,
            funding_advisory=advisory,
        ),
    )

    return RebalancePlan(
        plan_id=plan_id,
        created_at=created_at,
        portfolio_id=portfolio_id,
        risk_profile=risk_profile,
        budget_rub=budget_rub,
        mode=mode,
        auto_legs=tuple(auto_legs),
        manual_actions=manual_actions,
    )
