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
  the traded equity/OFZ legs only (the deposit leg is cost-free, D-09), computed
  from the REAL per-leg rescale delta every quarter;
- accrues realized-gains NDFL on profitable eq/OFZ sells via a FIFO
  ``CostBasisLedger`` (seeded with the opening eq/OFZ basis and re-fed on each
  rebalance buy) fed into the W1 ``YtdTaxAccumulator`` 13/15% band (D-07).

The deposit leg's interest is taxed on the W1 deposit accrual path (it is NOT a
capital gain), so the FIFO ledger holds eq/OFZ basis only and the deposit leg is
both cost-free and capital-gains-free here (D-07/D-09).

The lockup-respecting funding-order helper ``fund_underweight``
(matured -> accrued-income -> liquid-cash -> last-resort break-with-penalty,
D-06) ships as a TESTED-BUT-UNWIRED broker-level helper (like the dormant
``tighten`` rule). ``run()`` merges precomputed curves and owns no broker, so it
never calls ``fund_underweight``; W3 wires that helper into a broker-driven
boundary rebalance (WR-03 -- the docstring states this honestly rather than
implying ``run()`` already funds via it).

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
            # Remove the liquidated rung from the active sleeve so deposit_value()
            # reflects that its capital now funds the rebalance.
            if broker_tranches is not None and tranche in broker_tranches:
                broker_tranches.remove(tranche)
            # CR-02 capital conservation: when the broken tranche over-funds the
            # need, the UN-DRAWN liquidated value (``broke_value - draw``) does not
            # vanish -- it becomes liquid cash on the broker. Total broker value
            # (``deposit_value() + _cash``) then drops only by the funded draw plus
            # the demand-rate break penalty, never by the residual principal.
            residual = broke_value - draw
            if residual > _ZERO and broker_tranches is not None:
                broker._cash += residual

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
        legacy_monthly_drift_cadence: bool = False,
        zero_cost: bool = False,
    ) -> AllocationResult:
        """Merge the three pre-computed TR curves into an ``AllocationResult``.

        Cost + realized-gains NDFL are charged from the REAL per-leg rescale delta
        on every quarter that actually rebalances (CR-01/WR-01): there is no
        external delta hook -- the traded notional is ``|target_value -
        current_value|`` per eq/OFZ leg (deposit cost-free, D-09), and the FIFO
        ledger (seeded with the opening eq/OFZ basis) realizes a real gain on a
        boundary sell. ``legacy_monthly_drift_cadence`` + ``zero_cost`` drive the
        D-13 structural-equivalence path (legacy 60/40, monthly+drift, no cost --
        ``zero_cost`` keeps ``cumulative_friction`` at 0 so the merged curve is
        byte-identical to the pre-fix reproduction). Never constructs/runs an
        engine (D-12).
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
        legacy_cadence: bool,
        charge_cost: bool,
    ) -> tuple[list[Decimal], dict[AssetClass, list[Decimal]], list[date], Decimal, Decimal]:
        """Scale-at-boundary spine generalized to 3 legs (Pattern 1, the keystone).

        Replaces the legacy monthly+drift trigger with a pure quarterly trigger
        (D-08) unless ``legacy_cadence`` is set, moves each leg to its EXACT target
        (no band), and -- the Pitfall-4 gap the legacy spine left open -- charges
        the explicit round-trip cost (D-09) and the FIFO realized-gains NDFL (D-07)
        as line items computed from the REAL per-leg rescale delta. Engines are
        never re-run (D-12).

        Curve-scale <-> FIFO lots (the keystone mapping): treat each leg's ``scale``
        as the UNITS held and the unscaled curve value at bar ``i`` as the per-unit
        price, so the leg's market value is ``curve[i] * scale``. A rescale from
        ``old_scale`` to ``new_scale`` at price ``curve[i]`` is a buy (units up) or
        sell (units down) of ``|new_scale - old_scale|`` units at price
        ``curve[i]`` -- traded RUB notional ``|new_scale - old_scale| * curve[i] ==
        |target_value - current_value|`` (CR-01/WR-01). Friction is applied
        cumulatively from the rebalance bar onward (WR-02), never folded into the
        last bar; with ``charge_cost`` False ``cumulative_friction`` stays 0, so the
        zero_cost D-13 reproduction is byte-identical to the pre-fix path.
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
        cumulative_friction = _ZERO

        ledger = CostBasisLedger()
        ytd_acc = YtdTaxAccumulator()

        dep_scale = _ONE
        ofz_scale = _ONE
        eq_scale = _ONE

        # Seed the FIFO ledger with the opening eq/OFZ basis (live cost path only).
        if charge_cost:
            self._seed_opening_basis(ledger, ofz, eq)

        last_period: int | tuple[int, int] | None = None

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
                    # Capture the OLD eq/OFZ scales (pre-rebalance UNITS) BEFORE
                    # overwriting them -- the cost + NDFL charge needs them to
                    # compute the traded delta per capital-gains leg.
                    capital_gains_legs = (
                        (AssetClass.OFZ_PK, ofz[i], target_ofz, ofz_scale),
                        (AssetClass.EQUITY, eq[i], target_eq, eq_scale),
                    )
                    if dep[i] > _ZERO:
                        dep_scale = target_dep / dep[i]
                    if ofz[i] > _ZERO:
                        ofz_scale = target_ofz / ofz[i]
                    if eq[i] > _ZERO:
                        eq_scale = target_eq / eq[i]
                    rebalance_dates.append(d)
                    if charge_cost:
                        bar_cost, bar_ndfl = self._charge_rebalance(
                            capital_gains_legs, ledger, ytd_acc, d.year
                        )
                        rebalance_cost += bar_cost
                        realized_ndfl += bar_ndfl
                        cumulative_friction += bar_cost + bar_ndfl

            dep_val = dep[i] * dep_scale
            ofz_val = ofz[i] * ofz_scale
            eq_val = eq[i] * eq_scale
            gross_total = dep_val + ofz_val + eq_val

            # Weight shares are computed on the GROSS (pre-friction) total so they
            # stay legacy-comparable; the merged curve carries the running friction.
            weight_series[AssetClass.DEPOSIT].append(self._share(dep_val, gross_total))
            weight_series[AssetClass.OFZ_PK].append(self._share(ofz_val, gross_total))
            weight_series[AssetClass.EQUITY].append(self._share(eq_val, gross_total))
            merged.append(gross_total - cumulative_friction)

        return merged, weight_series, rebalance_dates, rebalance_cost, realized_ndfl

    @staticmethod
    def _seed_opening_basis(ledger: CostBasisLedger, ofz: list[Decimal], eq: list[Decimal]) -> None:
        """Seed the FIFO ledger with the opening eq/OFZ basis BEFORE the bar loop (D-07).

        Only called on the live cost path -- NOT the zero_cost D-13 reproduction.
        The deposit leg has no capital-gains basis (D-07/D-09): its interest is
        taxed on the W1 deposit accrual path, never here. Each leg's scale starts
        at 1 unit, so the seeded lot is ``(1 unit @ curve[0])`` per capital-gains
        leg -- a later boundary sell realizes a real gain against this basis.
        """
        if eq and eq[0] > _ZERO:
            ledger.buy(AssetClass.EQUITY, _ONE, eq[0])
        if ofz and ofz[0] > _ZERO:
            ledger.buy(AssetClass.OFZ_PK, _ONE, ofz[0])

    @staticmethod
    def _charge_rebalance(
        legs: tuple[tuple[AssetClass, Decimal, Decimal, Decimal], ...],
        ledger: CostBasisLedger,
        ytd_acc: YtdTaxAccumulator,
        year: int,
    ) -> tuple[Decimal, Decimal]:
        """Real per-leg round-trip cost + FIFO realized-gains NDFL for one rebalance (D-09/D-07).

        ``legs`` is ``(asset_class, price, target_value, old_units)`` for the two
        capital-gains legs only (the deposit leg is cost-free AND has no
        capital-gains basis -- its interest is taxed on the W1 deposit path). For
        each leg the traded delta is ``new_units - old_units`` where
        ``new_units = target_value / price``; the round-trip cost is charged on the
        traded notional ``|d_units| * price`` (CR-01), a buy re-feeds the FIFO
        ledger basis and a sell realizes a real gain taxed via the W1
        ``YtdTaxAccumulator`` 13/15% band (WR-01). Returns ``(bar_cost, bar_ndfl)``.
        """
        bar_cost = _ZERO
        bar_ndfl = _ZERO
        for leg, price, target_value, old_units in legs:
            if price <= _ZERO:
                continue
            new_units = target_value / price
            d_units = new_units - old_units
            bar_cost += abs(d_units) * price * _ROUND_TRIP_COST
            if d_units > _ZERO:
                ledger.buy(leg, d_units, price)
            elif d_units < _ZERO:
                gain = ledger.sell(leg, -d_units, price)
                bar_ndfl += ytd_acc.tax(max(_ZERO, gain), year)
        return bar_cost, bar_ndfl

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
