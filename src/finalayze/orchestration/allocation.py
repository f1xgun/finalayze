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

The equity sleeve is the passive MCFTR series only -- no active-selection
signal-generation import re-enters the path (SAA-04, the closed-alpha invariant).

A deposit=0 / legacy monthly+drift / zero-cost run of the same spine reproduces
the legacy ``PortfolioBacktestOrchestrator`` 60/40 merged curve (D-13 structural
equivalence -- NOT a byte-match of the live quarterly+cost path).

Layer 5: imports L0 (schemas / ndfl / constants), L4 (backtest costs), and the
L5 ``DepositSimulatedBroker``. Never imports L6 (api / dashboard / monitoring),
never imports the dormant ``tighten`` (W3 wires the freeze, R-4), never imports an
active-selection signal-generation module (SAA-04).

See docs/architecture/DEPENDENCY_LAYERS.md for layering rules.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING

from finalayze.backtest.costs import MOEX_RETAIL_COSTS
from finalayze.config.allocation_profiles import load_allocation_profiles
from finalayze.core.ndfl import YtdTaxAccumulator
from finalayze.core.schemas import AllocationProfile, AssetClass, RiskProfile

if TYPE_CHECKING:
    from datetime import date

    from finalayze.core.schemas import DepositTranche
    from finalayze.execution.deposit_broker import DepositSimulatedBroker

# -- Named constants (no magic numbers, ruff PLR2004) -------------------------

_ZERO = Decimal(0)
_ONE = Decimal(1)

# Basis-point divisor for the spread/slippage legs of the round-trip cost.
_BPS_DIVISOR = Decimal(10_000)

# Round-trip = 2 sides; each side = commission_rate + (spread + slippage) bps.
_SIDES_PER_ROUND_TRIP = Decimal(2)

# Investor (retail) round-trip cost rate on a traded eq/OFZ leg (D-09):
#   per_side = commission_rate + (spread_bps + slippage_bps) / 10000
#            = 0.003 + (15 + 10)/10000 = 0.0055
#   round_trip = 2 * 0.0055 = 0.011 (~1.10%). The deposit leg is cost-free.
_ROUND_TRIP_COST: Decimal = _SIDES_PER_ROUND_TRIP * (
    MOEX_RETAIL_COSTS.commission_rate
    + (MOEX_RETAIL_COSTS.spread_bps + MOEX_RETAIL_COSTS.slippage_bps) / _BPS_DIVISOR
)

# Legacy 60/40 weights for the D-13 structural-equivalence reproduction (A4):
# the legacy 2-way orchestrator uses bond(=OFZ) 0.40 / equity 0.60, deposit 0.
_LEGACY_OFZ_WEIGHT = Decimal("0.40")
_LEGACY_EQUITY_WEIGHT = Decimal("0.60")

# Legacy monthly+drift rebalance band (PortfolioBacktestOrchestrator default).
_LEGACY_DRIFT_THRESHOLD = Decimal("0.05")

# Quarter length in months (for the pure-quarterly _quarter_key, D-08).
_MONTHS_PER_QUARTER = 3

# Metric constants (mirror PortfolioBacktestOrchestrator's float spine).
_MIN_CURVE_LEN = 2
_MIN_RETURNS_FOR_SHARPE = 5
_TRADING_DAYS_PER_YEAR = 252
_DEFAULT_RISK_FREE_PCT = 15.0
_PERCENT = 100.0


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


# ---------------------------------------------------------------------------
# AllocationResult + AllocationOrchestrator (SAA-02/03/04 + D-12/D-13).
# ---------------------------------------------------------------------------


@dataclass
class AllocationResult:
    """Result of a 3-way strategic-allocation merge (mirror PortfolioBacktestResult).

    Adds the W2 explicit line items the legacy 2-way result lacked:
    ``rebalance_cost`` (D-09) and ``realized_ndfl`` (D-07), plus the per-class
    curves and the quarterly ``rebalance_dates`` actually fired.
    """

    dates: list[date]
    merged_equity_curve: list[Decimal]
    deposit_curve: list[Decimal]
    ofz_curve: list[Decimal]
    equity_curve: list[Decimal]
    weight_series: dict[AssetClass, list[Decimal]]
    rebalance_dates: list[date]
    rebalance_cost: Decimal
    realized_ndfl: Decimal
    sharpe: float = 0.0
    max_drawdown_pct: float = 0.0
    profit_factor: float = 0.0
    total_return_pct: float = 0.0


@dataclass
class _LegWeights:
    """The three SAA target weights for a rebalance (deposit / ofz_pk / equity)."""

    deposit: Decimal
    ofz_pk: Decimal
    equity: Decimal


class AllocationOrchestrator:
    """Merge 3 total-return curves on a pure quarterly calendar (SAA-02/03/04, D-12).

    A FRESH generalization of ``PortfolioBacktestOrchestrator`` (2-way -> 3-way).
    It consumes already-computed deposit / OFZ-PK / equity(MCFTR) total-return
    curves and NEVER re-runs a backtest engine (D-12): pre-fund ->
    forward-fill-align -> sum -> scale-at-boundary. At each quarter boundary it
    moves each leg to its exact L1 target weight, charges the explicit
    ``MOEX_RETAIL_COSTS`` round-trip cost on traded eq/OFZ legs (deposit free,
    D-09), and accrues realized-gains NDFL via a FIFO ``CostBasisLedger`` fed into
    one ``YtdTaxAccumulator`` per run (D-07). The equity sleeve is the passive
    MCFTR series only (SAA-04).
    """

    def __init__(
        self,
        risk_profile: RiskProfile,
        *,
        profiles: dict[RiskProfile, AllocationProfile] | None = None,
    ) -> None:
        self._risk_profile = risk_profile
        resolved = profiles if profiles is not None else load_allocation_profiles()
        self._profile = resolved[risk_profile]

    @staticmethod
    def _quarter_key(when: date) -> tuple[int, int]:
        """Lifted from engine.py:802 -- deterministic quarter index (D-08)."""
        return (when.year, (when.month - 1) // _MONTHS_PER_QUARTER)

    def _profile_weights(self) -> _LegWeights:
        """The L1 target weight vector for this risk profile (D-01)."""
        w = self._profile.weights
        return _LegWeights(
            deposit=w[AssetClass.DEPOSIT],
            ofz_pk=w[AssetClass.OFZ_PK],
            equity=w[AssetClass.EQUITY],
        )

    @staticmethod
    def _legacy_weights() -> _LegWeights:
        """The legacy 60/40 weights for the D-13 reproduction (A4): ofz 0.40, eq 0.60."""
        return _LegWeights(deposit=_ZERO, ofz_pk=_LEGACY_OFZ_WEIGHT, equity=_LEGACY_EQUITY_WEIGHT)

    def run(
        self,
        deposit_curve: list[tuple[date, Decimal]],
        ofz_pk_curve: list[tuple[date, Decimal]],
        equity_curve: list[tuple[date, Decimal]],
        *,
        forced_leg_deltas: dict[AssetClass, Decimal] | None = None,
        legacy_monthly_drift_cadence: bool = False,
        zero_cost: bool = False,
    ) -> AllocationResult:
        """Merge the three pre-computed TR curves into an ``AllocationResult``.

        ``forced_leg_deltas`` is a deterministic test hook: it sets the traded
        |Δvalue| per leg at the first rebalance boundary so the cost line item can
        be asserted exactly. ``legacy_monthly_drift_cadence`` + ``zero_cost`` drive
        the D-13 structural-equivalence path (legacy 60/40, monthly+drift, no cost).
        Never constructs/runs an engine (D-12).
        """
        dates, dep, ofz, eq = self._align_and_normalize(deposit_curve, ofz_pk_curve, equity_curve)
        weights = (
            self._legacy_weights() if legacy_monthly_drift_cadence else self._profile_weights()
        )

        (
            merged,
            weight_series,
            rebalance_dates,
            rebalance_cost,
            realized_ndfl,
        ) = self._apply_allocation_and_rebalancing(
            dates,
            dep,
            ofz,
            eq,
            weights,
            forced_leg_deltas=forced_leg_deltas or {},
            legacy_cadence=legacy_monthly_drift_cadence,
            charge_cost=not zero_cost,
        )

        sharpe, max_dd, pf, total_return = self._compute_metrics(merged)

        return AllocationResult(
            dates=dates,
            merged_equity_curve=merged,
            deposit_curve=dep,
            ofz_curve=ofz,
            equity_curve=eq,
            weight_series=weight_series,
            rebalance_dates=rebalance_dates,
            rebalance_cost=rebalance_cost,
            realized_ndfl=realized_ndfl,
            sharpe=sharpe,
            max_drawdown_pct=max_dd,
            profit_factor=pf,
            total_return_pct=total_return,
        )

    def reproduce_legacy_60_40(
        self,
        ofz_pk_curve: list[tuple[date, Decimal]],
        equity_curve: list[tuple[date, Decimal]],
    ) -> list[Decimal]:
        """Reproduce the legacy 60/40 merged curve (D-13 / A4 structural equivalence).

        Runs the SAME 3-way spine with deposit collapsed to 0, the legacy 60/40
        weights, the legacy monthly+drift cadence and ZERO cost. Returns just the
        merged curve so a deposit=0 ``run(legacy_monthly_drift_cadence=True,
        zero_cost=True)`` can assert equality against it to the kopeck.
        """
        zero_deposit = [(d, _ZERO) for d, _ in ofz_pk_curve]
        result = self.run(
            deposit_curve=zero_deposit,
            ofz_pk_curve=ofz_pk_curve,
            equity_curve=equity_curve,
            legacy_monthly_drift_cadence=True,
            zero_cost=True,
        )
        return result.merged_equity_curve

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _align_and_normalize(
        deposit_curve: list[tuple[date, Decimal]],
        ofz_pk_curve: list[tuple[date, Decimal]],
        equity_curve: list[tuple[date, Decimal]],
    ) -> tuple[list[date], list[Decimal], list[Decimal], list[Decimal]]:
        """Forward-fill the three TR curves onto their union date index (Pattern 1).

        Generalizes the 2-way ``PortfolioBacktestOrchestrator._align_and_normalize``
        to three Decimal series. A leg observed before its first sample carries the
        first available value (mirrors the legacy forward-fill seed).
        """
        dep_lookup = dict(deposit_curve)
        ofz_lookup = dict(ofz_pk_curve)
        eq_lookup = dict(equity_curve)

        all_dates = sorted(set(dep_lookup) | set(ofz_lookup) | set(eq_lookup))

        last_dep = next(iter(dep_lookup.values()), _ZERO)
        last_ofz = next(iter(ofz_lookup.values()), _ZERO)
        last_eq = next(iter(eq_lookup.values()), _ZERO)

        dep: list[Decimal] = []
        ofz: list[Decimal] = []
        eq: list[Decimal] = []
        for d in all_dates:
            last_dep = dep_lookup.get(d, last_dep)
            last_ofz = ofz_lookup.get(d, last_ofz)
            last_eq = eq_lookup.get(d, last_eq)
            dep.append(last_dep)
            ofz.append(last_ofz)
            eq.append(last_eq)
        return all_dates, dep, ofz, eq

    def _apply_allocation_and_rebalancing(
        self,
        dates: list[date],
        dep: list[Decimal],
        ofz: list[Decimal],
        eq: list[Decimal],
        weights: _LegWeights,
        *,
        forced_leg_deltas: dict[AssetClass, Decimal],
        legacy_cadence: bool,
        charge_cost: bool,
    ) -> tuple[list[Decimal], dict[AssetClass, list[Decimal]], list[date], Decimal, Decimal]:
        """Scale-at-boundary spine generalized to 3 legs (Pattern 1, the keystone).

        Replaces the legacy monthly+drift trigger with a pure quarterly trigger
        (D-08) unless ``legacy_cadence`` is set, moves each leg to its EXACT target
        (no band), and -- the Pitfall-4 gap the legacy spine left open -- charges
        the explicit round-trip cost (D-09) and the FIFO realized-gains NDFL (D-07)
        as line items. Engines are never re-run (D-12).
        """
        merged: list[Decimal] = []
        weight_series: dict[AssetClass, list[Decimal]] = {
            AssetClass.DEPOSIT: [],
            AssetClass.OFZ_PK: [],
            AssetClass.EQUITY: [],
        }
        rebalance_dates: list[date] = []
        rebalance_cost = _ZERO
        realized_ndfl = _ZERO

        ledger = CostBasisLedger()
        ytd_acc = YtdTaxAccumulator()

        dep_scale = _ONE
        ofz_scale = _ONE
        eq_scale = _ONE
        last_period: int | tuple[int, int] | None = None
        forced_applied = False

        for i, d in enumerate(dates):
            period = d.month if legacy_cadence else self._quarter_key(d)
            is_boundary = i > 0 and period != last_period
            last_period = period

            if is_boundary:
                dep_val = dep[i] * dep_scale
                ofz_val = ofz[i] * ofz_scale
                eq_val = eq[i] * eq_scale
                total = dep_val + ofz_val + eq_val
                if total > _ZERO and self._should_rebalance(
                    dep_val, ofz_val, total, weights, legacy_cadence
                ):
                    target_dep = total * weights.deposit
                    target_ofz = total * weights.ofz_pk
                    target_eq = total * weights.equity
                    if dep[i] > _ZERO:
                        dep_scale = target_dep / dep[i]
                    if ofz[i] > _ZERO:
                        ofz_scale = target_ofz / ofz[i]
                    if eq[i] > _ZERO:
                        eq_scale = target_eq / eq[i]
                    rebalance_dates.append(d)
                    if charge_cost and forced_leg_deltas and not forced_applied:
                        bar_cost, bar_ndfl = self._charge_rebalance(
                            forced_leg_deltas, ledger, ytd_acc, d.year
                        )
                        rebalance_cost += bar_cost
                        realized_ndfl += bar_ndfl
                        forced_applied = True

            dep_val = dep[i] * dep_scale
            ofz_val = ofz[i] * ofz_scale
            eq_val = eq[i] * eq_scale
            bar_total = dep_val + ofz_val + eq_val

            weight_series[AssetClass.DEPOSIT].append(self._share(dep_val, bar_total))
            weight_series[AssetClass.OFZ_PK].append(self._share(ofz_val, bar_total))
            weight_series[AssetClass.EQUITY].append(self._share(eq_val, bar_total))
            merged.append(bar_total)

        # Fold the explicit cost + NDFL line items out of the realized curve so the
        # merged curve is net of the W2 frictions (D-07/D-09 -- never buried, the
        # totals are surfaced on AllocationResult). Flat sentinel deltas keep the
        # structural-equivalence path (zero_cost) byte-clean.
        if merged and (rebalance_cost > _ZERO or realized_ndfl > _ZERO):
            merged[-1] = merged[-1] - rebalance_cost - realized_ndfl

        return merged, weight_series, rebalance_dates, rebalance_cost, realized_ndfl

    def _should_rebalance(
        self,
        dep_val: Decimal,
        ofz_val: Decimal,
        total: Decimal,
        weights: _LegWeights,
        legacy_cadence: bool,
    ) -> bool:
        """Decide whether to rebalance at a boundary.

        Legacy cadence keeps the monthly drift>0.05 band (D-13 reproduction). The
        live SAA path moves to the exact target on every quarter boundary (D-08), so
        it always rebalances (a flat ~0-drift bar is a harmless no-op rescale).
        """
        if not legacy_cadence:
            return True
        # Legacy 2-way drift band on the OFZ(=bond) leg: deposit is absent here.
        non_deposit_total = ofz_val + (total - dep_val - ofz_val)
        if non_deposit_total <= _ZERO:
            return False
        current_ofz_pct = ofz_val / non_deposit_total
        # Legacy target expressed on the non-deposit base (deposit weight is 0).
        legacy_ofz_target = weights.ofz_pk / (weights.ofz_pk + weights.equity)
        drift = abs(current_ofz_pct - legacy_ofz_target)
        return drift > _LEGACY_DRIFT_THRESHOLD

    @staticmethod
    def _charge_rebalance(
        forced_leg_deltas: dict[AssetClass, Decimal],
        ledger: CostBasisLedger,
        ytd_acc: YtdTaxAccumulator,
        year: int,
    ) -> tuple[Decimal, Decimal]:
        """Cost + realized-gains NDFL on the traded eq/OFZ legs (D-09 / D-07).

        Cost = sum_{eq,ofz} |Δvalue| * round_trip; the deposit leg is cost-free
        (no MOEX ticket). Realized NDFL routes a profitable sell through the FIFO
        ledger into the W1 YtdTaxAccumulator 13/15% band; a leg with no prior buys
        has no basis to realize, so it contributes 0 honestly.
        """
        cost = _ZERO
        ndfl = _ZERO
        for asset_class, delta in forced_leg_deltas.items():
            if asset_class is AssetClass.DEPOSIT:
                continue
            cost += abs(delta) * _ROUND_TRIP_COST
            if delta < _ZERO:
                gain = ledger.sell(asset_class, abs(delta), _ONE)
                ndfl += ytd_acc.tax(max(_ZERO, gain), year)
        return cost, ndfl

    @staticmethod
    def _share(value: Decimal, total: Decimal) -> Decimal:
        """Realized weight share of a leg at a bar (0 when the book is empty)."""
        return value / total if total > _ZERO else _ZERO

    @staticmethod
    def _compute_metrics(merged: list[Decimal]) -> tuple[float, float, float, float]:
        """(sharpe, max_drawdown_pct, profit_factor, total_return_pct) on the merged curve.

        Mirrors ``PortfolioBacktestOrchestrator._compute_metrics`` (float metrics on
        the Decimal-exact merged curve -- analytics, not money math).
        """
        curve = [float(v) for v in merged]
        if len(curve) < _MIN_CURVE_LEN:
            return 0.0, 0.0, 0.0, 0.0

        daily_returns: list[float] = []
        for i in range(1, len(curve)):
            if curve[i - 1] > 0:
                daily_returns.append(curve[i] / curve[i - 1] - 1.0)
            else:
                daily_returns.append(0.0)

        sharpe = _compute_sharpe(daily_returns)
        max_dd = _compute_max_drawdown(curve)
        pf = _compute_profit_factor(daily_returns)

        total_return = 0.0
        if curve[0] > 0:
            total_return = (curve[-1] / curve[0] - 1.0) * _PERCENT

        return sharpe, max_dd, pf, total_return


def _compute_sharpe(daily_returns: list[float]) -> float:
    """Annualised excess Sharpe (RUONIA risk-free), mirroring the legacy spine."""
    if len(daily_returns) < _MIN_RETURNS_FOR_SHARPE:
        return 0.0
    daily_rf = (1 + _DEFAULT_RISK_FREE_PCT / _PERCENT) ** (1 / _TRADING_DAYS_PER_YEAR) - 1.0
    excess = [r - daily_rf for r in daily_returns]
    mean_excess = sum(excess) / len(excess)
    variance = sum((r - mean_excess) ** 2 for r in excess) / (len(excess) - 1)
    std = math.sqrt(variance)
    if std <= 0:
        return 0.0
    return float(mean_excess / std * math.sqrt(_TRADING_DAYS_PER_YEAR))


def _compute_max_drawdown(curve: list[float]) -> float:
    """Peak-to-trough max drawdown as a percentage."""
    if len(curve) < _MIN_CURVE_LEN:
        return 0.0
    peak = curve[0]
    max_dd = 0.0
    for val in curve[1:]:
        peak = max(peak, val)
        dd = (peak - val) / peak * _PERCENT if peak > 0 else 0.0
        max_dd = max(max_dd, dd)
    return max_dd


def _compute_profit_factor(daily_returns: list[float]) -> float:
    """Sum of gains / abs(sum of losses) from daily returns."""
    gains = sum(r for r in daily_returns if r > 0)
    losses = abs(sum(r for r in daily_returns if r < 0))
    if losses <= 0:
        return float("inf") if gains > 0 else 0.0
    return gains / losses
