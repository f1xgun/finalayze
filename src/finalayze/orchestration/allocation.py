"""L5 strategic-allocation orchestrator (Phase 72 Wave-2, SAA-02/03/04).

Sibling of ``apply_ofz_rotation`` (``orchestration/bond_cycle.py``). Builds the
keystone ``AllocationOrchestrator``: a FRESH generalization of the 2-way
``PortfolioBacktestOrchestrator`` arithmetic spine to a 3-way (deposit / OFZ-PK /
equity-MCFTR) merge driven by a pure quarterly calendar.

The orchestrator NEVER re-runs a backtest engine (D-12). It consumes three
already-computed total-return curves and merges them via the proven
pre-fund -> forward-fill-align -> sum -> scale-at-boundary pattern. At each
quarter boundary it:

- moves each leg to its exact L1 target weight (no drift band, D-08);
- charges the explicit ``MOEX_RETAIL_COSTS`` (Investor ~1.10%) round-trip cost on
  the traded equity/OFZ legs only (the deposit leg is cost-free, D-09);
- accrues realized-gains NDFL on profitable eq/OFZ sells via a FIFO
  ``CostBasisLedger`` fed into the W1 ``YtdTaxAccumulator`` 13/15% band (D-07);
- funds the underweight class in the lockup-respecting order
  matured -> accrued-income -> liquid-cash -> last-resort break-with-penalty
  (D-06, via ``fund_underweight``).

The equity sleeve is the passive MCFTR series only -- no StrategyCombiner /
momentum / ML import re-enters the path (SAA-04, the closed-alpha invariant).

A deposit=0 / legacy monthly+drift / zero-cost run of the same spine reproduces
the legacy ``PortfolioBacktestOrchestrator`` 60/40 merged curve (D-13 structural
equivalence -- NOT a byte-match of the live quarterly+cost path).

Layer 5: imports L0 (schemas / ndfl / constants), L4 (backtest costs), and the
L5 ``DepositSimulatedBroker``. Never imports L6 (api / dashboard / monitoring),
never imports the dormant ``tighten`` (W3 wires the freeze, R-4), never imports a
StrategyCombiner / ml module (SAA-04).

See docs/architecture/DEPENDENCY_LAYERS.md for layering rules.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import date

    from finalayze.core.schemas import AssetClass, DepositTranche
    from finalayze.execution.deposit_broker import DepositSimulatedBroker

# -- Named constants (no magic numbers, ruff PLR2004) -------------------------

_ZERO = Decimal(0)


# ---------------------------------------------------------------------------
# FIFO cost-basis ledger (D-07 / R-3) -- allocator-owned, net-new.
# ---------------------------------------------------------------------------


class CostBasisLedger:
    """Per-asset-class FIFO lot ledger for realized-gains NDFL on rebalance sells (D-07/R-3).

    RU broker default = FIFO (A2). Net-new: ``SimulatedBroker`` tracks net-qty only
    (simulated_broker.py:62), so this ledger is genuinely net-new. Owned by
    ``AllocationOrchestrator``; the broker is untouched.
    """

    def __init__(self) -> None:
        self._lots: dict[AssetClass, deque[tuple[Decimal, Decimal]]] = {}

    def buy(self, asset_class: AssetClass, qty: Decimal, unit_cost: Decimal) -> None:
        """Append a FIFO lot (qty bought at unit_cost) for ``asset_class``."""
        self._lots.setdefault(asset_class, deque()).append((qty, unit_cost))

    def sell(self, asset_class: AssetClass, qty: Decimal, unit_price: Decimal) -> Decimal:
        """Pop earliest lots until ``qty`` covered; return realized gain (may be negative).

        ``gain = sum(sold_qty_i * (unit_price - unit_cost_i))`` across the popped
        FIFO lots (R-3). A partial last lot leaves its residual qty open. NO
        holding-period argument -- there is no 3-year LDV exemption branch (D-07).
        """
        remaining = qty
        gain = _ZERO
        lots = self._lots.get(asset_class, deque())
        while remaining > _ZERO and lots:
            lot_qty, unit_cost = lots[0]
            take = min(remaining, lot_qty)
            gain += take * (unit_price - unit_cost)
            remaining -= take
            if take == lot_qty:
                lots.popleft()
            else:
                lots[0] = (lot_qty - take, unit_cost)
        return gain


# ---------------------------------------------------------------------------
# Lockup-respecting funding-order (D-06 / R-6).
# ---------------------------------------------------------------------------


@dataclass
class FundingBreakdown:
    """Where a rebalance's underweight delta was sourced (D-06 / R-6).

    Strict-order bookkeeping so the orchestrator can fold the result into the
    deposit-class value and the line items. ``total`` is the amount actually
    raised; ``broke_tranche`` / ``break_penalty`` flag a forced last-resort break.
    """

    from_matured: Decimal = _ZERO
    from_income: Decimal = _ZERO
    from_cash: Decimal = _ZERO
    from_break: Decimal = _ZERO
    broke_tranche: bool = False
    break_penalty: Decimal = _ZERO

    @property
    def total(self) -> Decimal:
        """Total amount raised across all sources."""
        return self.from_matured + self.from_income + self.from_cash + self.from_break


def fund_underweight(
    broker: DepositSimulatedBroker,
    need: Decimal,
    as_of: date,
) -> Decimal:
    """Fund ``need`` in the lockup-respecting order (D-06, Pitfall 7).

    Strict order:
      1. matured deposit tranches (``maturity_date <= as_of``, principal +
         accrued, NO break penalty -- they matured);
      2. accrued income (net deposit interest credited so far);
      3. liquid / demand cash on the broker;
      4. last-resort: break a LOCKED tranche (``break_tranche`` -- forfeit accrued
         to ``DEPOSIT_DEMAND_RATE`` and set ``broken=True``).

    The degraded path (no matured tranche this quarter) falls through gracefully
    (R-6): a cheaper source (income/cash) covers the need without breaking a
    locked rung; a locked rung is broken ONLY when matured+income+cash are
    insufficient (never a silent full-rate locked sale). A mid-quarter tranche is
    NOT matured at the boundary (``maturity_date > as_of``), so it is never a
    step-1 source.

    Returns the total amount raised (``>= need`` when sources suffice).
    """
    breakdown = _fund_underweight_breakdown(broker, need, as_of)
    return breakdown.total


def _fund_underweight_breakdown(
    broker: DepositSimulatedBroker,
    need: Decimal,
    as_of: date,
) -> FundingBreakdown:
    """Run the strict funding-order and return the structured breakdown (D-06)."""
    breakdown = FundingBreakdown()
    remaining = need

    tranches = list(getattr(broker, "_tranches", []))

    # Step 1: matured tranches (no penalty) -- maturity_date <= as_of, not broken.
    # The drawn amount leaves the deposit sleeve (the matured principal/accrued is
    # withdrawn to fund the rebalance), so it must reduce the tranche mark.
    for tranche in tranches:
        if remaining <= _ZERO:
            break
        if tranche.broken or tranche.maturity_date > as_of:
            continue
        available = tranche.principal + tranche.accrued_net
        draw = min(remaining, available)
        _withdraw_matured(tranche, draw)
        breakdown.from_matured += draw
        remaining -= draw

    # Step 2: accrued income (net deposit interest credited so far).
    if remaining > _ZERO:
        income = max(_ZERO, broker.interest_income_net)
        draw = min(remaining, income)
        breakdown.from_income += draw
        remaining -= draw

    # Step 3: liquid / demand cash on the broker.
    if remaining > _ZERO:
        cash = max(_ZERO, getattr(broker, "_cash", _ZERO))
        draw = min(remaining, cash)
        breakdown.from_cash += draw
        remaining -= draw

    # Step 4: last-resort break a LOCKED tranche (Pitfall 7 -- never a silent
    # full-rate sale). ``break_tranche`` forfeits accrued to the demand rate (D-03)
    # but keeps the principal in the W1 mark (CR-01 reconciliation). The W2
    # funding-order is what WITHDRAWS that liquidated value -- so the broken rung is
    # removed from the active deposit sleeve (its principal + demand penalty is now
    # liquid funding, no longer a full-rate deposit asset). This drops
    # ``deposit_value()`` without mutating the broken tranche's W1 invariant
    # (``accrued_net == principal * DEPOSIT_DEMAND_RATE``).
    if remaining > _ZERO:
        broker_tranches = getattr(broker, "_tranches", None)
        for tranche in tranches:
            if remaining <= _ZERO:
                break
            if tranche.broken or tranche.maturity_date <= as_of:
                continue
            value_before = tranche.principal + tranche.accrued_net
            broker.break_tranche(tranche, as_of)
            # The penalty forfeits accrued down to the demand rate (D-03).
            penalty = value_before - (tranche.principal + tranche.accrued_net)
            breakdown.break_penalty += max(_ZERO, penalty)
            broke_value = tranche.principal + tranche.accrued_net
            draw = min(remaining, broke_value)
            breakdown.from_break += draw
            breakdown.broke_tranche = True
            remaining -= draw
            # Withdraw the liquidated rung from the active sleeve so deposit_value()
            # reflects that its capital now funds the rebalance.
            if broker_tranches is not None and tranche in broker_tranches:
                broker_tranches.remove(tranche)

    return breakdown


def _withdraw_matured(tranche: DepositTranche, amount: Decimal) -> None:
    """Withdraw ``amount`` from a matured tranche, drawing accrued then principal.

    A matured tranche has no broken-relation invariant, so its mark can be reduced
    directly: the withdrawn cash leaves the deposit sleeve to fund the rebalance.
    """
    take_accrued = min(amount, tranche.accrued_net)
    tranche.accrued_net -= take_accrued
    remaining = amount - take_accrued
    tranche.principal -= min(remaining, tranche.principal)
