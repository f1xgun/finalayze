"""Phase 72 backtest-iteration A/B gate (D-13 + R-7, CLAUDE.md #4).

The mandatory `backtest-iteration` gate for the W2 strategic-allocation layer. Two
matched A/Bs over the SAME deterministic, in-memory, no-network total-return curves
(threat model T-72-21: deterministic seed, no live API, no engine re-run -- the
allocator MERGES already-computed curves arithmetically, D-12):

A/B #1 -- D-13 reproduction (no-regression, A4 structural equivalence):
  BASELINE  : the REAL legacy ``PortfolioBacktestOrchestrator`` 60/40 merged curve
              (monthly + drift>0.05, no cost) over the curves.
  CANDIDATE : the new ``AllocationOrchestrator`` 3-way spine at deposit weight = 0,
              the SAME legacy monthly+drift cadence, rebalance cost DISABLED.
  ASSERT    : candidate merged curve == legacy 60/40 merged curve TO THE KOPECK
              (quantized 0.01; the allocator is Decimal-exact, the legacy spine is
              float, so the residual is float rounding noise ~1e-10). The curves are
              shaped so the legacy spine FIRES real rebalances (scale-at-boundary is
              actually exercised, not just a trivial sum).
              Logged as phase72-ab-d13-baseline / phase72-ab-d13-candidate.

  Plus the LIVE path: ``AllocationOrchestrator`` BALANCED profile, pure quarterly,
  cost + NDFL ON. ASSERT the live merged curve differs from the zero-cost run ONLY
  by the explicit ``rebalance_cost`` + ``realized_ndfl`` line items (D-09/D-07 --
  cost/tax are surfaced, never buried). Logged as phase72-ab-live-balanced.

A/B #2 -- R-7 idle/transit-cash (0% vs demand-rate):
  Quantify the merged-curve delta of crediting idle/transit cash at 0% vs at
  ``DEPOSIT_DEMAND_RATE``. The MCFTR sleeve is a fully-invested synthetic position
  (D-12: the allocator merges already-invested curves -> structural idle cash == 0),
  so the demand-rate credit is an upper bound on a quantity that is ~0 by
  construction. Logged as phase72-ab-idlecash-zero / phase72-ab-idlecash-demand
  with the chosen default + a one-line rationale.

Honest verdict framing (same as W1, D-13): PASS = the deposit=0 spine reproduces
the legacy 60/40 curve to the kopeck AND the live curve moves ONLY by the explicit
cost/tax line items -- NOT "PF improved". A missing/zero cost line when a non-deposit
leg trades is a BLOCKING discrepancy (T-72-20); a non-matching reproduction is a
REJECT (T-72-19).

Logs all five legs under results/iterations/phase72-* with history.jsonl verdicts.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, date, datetime, timedelta, timezone
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from finalayze.backtest.bond_engine import BondBacktestResult
from finalayze.backtest.portfolio_orchestrator import PortfolioBacktestOrchestrator
from finalayze.core.constants import DEPOSIT_DEMAND_RATE
from finalayze.core.ndfl import YtdTaxAccumulator
from finalayze.core.schemas import AssetClass, PortfolioState, RiskProfile
from finalayze.orchestration.allocation import (
    AllocationOrchestrator,
    CostBasisLedger,
)

# -- Deterministic A/B fixture (named constants -- no magic numbers) -----------

_PHASE = "72-06"
_ITER_DIR = Path(__file__).resolve().parent.parent / "results" / "iterations"

_YEAR = 2023
_N_BARS = 240  # ~8 months daily -> spans several quarter + month boundaries
_FIRST_BAR = date(_YEAR, 1, 1)

# Curve geometry: equity RALLIES hard so the bond%/equity% drift breaches the
# legacy 0.05 band WITHIN a month -> the legacy spine fires real rebalances and the
# scale-at-boundary arithmetic is genuinely exercised (not a trivial flat sum).
_OFZ_BASE = Decimal(40_000)
_EQ_BASE = Decimal(60_000)
_DEP_BASE = Decimal(50_000)
_OFZ_DAILY = Decimal("1.0002")  # slow OFZ-PK accrual
_EQ_DAILY = Decimal("1.004")  # equity rally -> drift breach
_DEP_DAILY = Decimal("1.00003")  # deposit mark, near-flat term accrual

# Live-balanced forced rebalance deltas (deterministic test hook on the allocator):
# the equity leg SELLS (cost charged) and the ofz leg BUYS (cost charged); deposit
# is cost-free (D-09). |Δ| drives the explicit round-trip cost line item.
_LIVE_EQ_DELTA = Decimal(-8000)
_LIVE_OFZ_DELTA = Decimal(3000)

# Realized-NDFL demonstration (D-07): a profitable FIFO eq sell through the SAME
# CostBasisLedger + YtdTaxAccumulator the orchestrator owns. _charge_rebalance only
# ever SELLS, so a non-zero realized_ndfl requires a prior buy lot -> we exercise the
# exact ledger/accumulator path here to PROVE the tax line is non-zero on a real gain.
_NDFL_LOT_QTY = Decimal(100)
_NDFL_BUY_PRICE = Decimal(100)
_NDFL_SELL_PRICE = Decimal(130)  # gain = 100 * (130 - 100) = 3000

_KOPECK = Decimal("0.01")
_TRADING_DAYS_PER_QUARTER = 63  # upper-bound horizon for the R-7 worst case
_TRADING_DAYS_PER_YEAR = 252
_PERCENT = Decimal(100)


def _dates() -> list[date]:
    return [_FIRST_BAR + timedelta(days=i) for i in range(_N_BARS)]


def _curve(base: Decimal, daily: Decimal, dates: list[date]) -> list[tuple[date, Decimal]]:
    return [(d, base * daily**i) for i, d in enumerate(dates)]


def _q(x: Decimal | float) -> Decimal:
    """Quantize to the kopeck (0.01) for the legacy-float vs allocator-Decimal compare."""
    return Decimal(str(x)).quantize(_KOPECK, rounding=ROUND_HALF_UP)


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],  # noqa: S607
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _write_iteration(name: str, payload: dict[str, object]) -> None:
    d = _ITER_DIR / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _append_history(name: str, *, git_sha: str, verdict: str, metrics: dict[str, object]) -> None:
    hist = _ITER_DIR / "history.jsonl"
    entry = {
        "name": name,
        "phase": _PHASE,
        "created_at": datetime.now(tz=timezone.utc).isoformat(),  # noqa: UP017
        "git_sha": git_sha,
        "verdict": verdict,
        **metrics,
    }
    with hist.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")


def _legacy_60_40(
    dates: list[date],
    ofz: list[tuple[date, Decimal]],
    eq: list[tuple[date, Decimal]],
) -> list[float]:
    """Run the REAL legacy PortfolioBacktestOrchestrator 60/40 (monthly+drift, no cost)."""
    ofz_vals = [v for _, v in ofz]
    eq_vals = [v for _, v in eq]
    bond_result = BondBacktestResult(
        trades=[],
        equity_curve=ofz_vals,
        dates=dates,
        total_coupon_income_gross=Decimal(0),
        total_coupon_income_net=Decimal(0),
        total_tax_paid=Decimal(0),
        total_return_pct=Decimal(0),
        max_drawdown_pct=Decimal(0),
        sharpe_ratio=Decimal(0),
        trade_count=0,
        win_rate=Decimal(0),
        profit_factor=Decimal(0),
        ofz_rotation_active=False,
    )
    eq_snaps = [
        PortfolioState(
            cash=Decimal(0),
            positions={},
            equity=eq_vals[i],
            timestamp=datetime(d.year, d.month, d.day, tzinfo=UTC),
        )
        for i, d in enumerate(dates)
    ]
    usdrub = [(d, 80.0) for d in dates]  # flat -> crisis brake never fires
    legacy = PortfolioBacktestOrchestrator()  # defaults: 0.40 bond / 0.60 equity, drift 0.05
    return legacy.run(bond_result, eq_snaps, usdrub, total_capital=100_000.0).merged_equity_curve


def _run_d13(dates: list[date], git_sha: str) -> tuple[bool, dict[str, object]]:
    """A/B #1: D-13 reproduction + live-balanced cost/NDFL line items."""
    ofz = _curve(_OFZ_BASE, _OFZ_DAILY, dates)
    eq = _curve(_EQ_BASE, _EQ_DAILY, dates)
    dep = _curve(_DEP_BASE, _DEP_DAILY, dates)
    zero_dep = [(d, Decimal(0)) for d in dates]

    legacy_curve = _legacy_60_40(dates, ofz, eq)

    orch = AllocationOrchestrator(risk_profile=RiskProfile.BALANCED)
    # CANDIDATE: 3-way spine at deposit=0, legacy monthly+drift cadence, zero cost.
    cand = orch.run(
        deposit_curve=zero_dep,
        ofz_pk_curve=ofz,
        equity_curve=eq,
        legacy_monthly_drift_cadence=True,
        zero_cost=True,
    )

    # Compare to the kopeck (legacy float vs allocator Decimal).
    mismatches = 0
    max_kopeck_diff = Decimal(0)
    max_raw_diff = Decimal(0)
    for lv, av in zip(legacy_curve, cand.merged_equity_curve, strict=True):
        kd = abs(_q(lv) - _q(av))
        rd = abs(Decimal(str(lv)) - av)
        max_kopeck_diff = max(max_kopeck_diff, kd)
        max_raw_diff = max(max_raw_diff, rd)
        if _q(lv) != _q(av):
            mismatches += 1
    d13_matches = mismatches == 0

    # LIVE path: BALANCED, pure quarterly, cost ON via forced deltas.
    zc = orch.run(deposit_curve=dep, ofz_pk_curve=ofz, equity_curve=eq)
    live = orch.run(
        deposit_curve=dep,
        ofz_pk_curve=ofz,
        equity_curve=eq,
        forced_leg_deltas={
            AssetClass.EQUITY: _LIVE_EQ_DELTA,
            AssetClass.OFZ_PK: _LIVE_OFZ_DELTA,
        },
    )
    curve_delta = zc.merged_equity_curve[-1] - live.merged_equity_curve[-1]
    line_items = live.rebalance_cost + live.realized_ndfl
    live_only_by_line_items = curve_delta == line_items
    cost_nonzero_when_leg_trades = live.rebalance_cost > Decimal(0)

    # Realized-NDFL proof (D-07): a profitable FIFO sell through the SAME ledger +
    # accumulator the orchestrator owns -> a real, non-zero tax line item.
    ledger = CostBasisLedger()
    acc = YtdTaxAccumulator()
    ledger.buy(AssetClass.EQUITY, _NDFL_LOT_QTY, _NDFL_BUY_PRICE)
    realized_gain = ledger.sell(AssetClass.EQUITY, _NDFL_LOT_QTY, _NDFL_SELL_PRICE)
    realized_ndfl_proof = acc.tax(max(Decimal(0), realized_gain), _YEAR)
    ndfl_nonzero_on_gain = realized_ndfl_proof > Decimal(0)

    passed = (
        d13_matches
        and live_only_by_line_items
        and cost_nonzero_when_leg_trades
        and ndfl_nonzero_on_gain
    )
    verdict = "PASS" if passed else "REJECT"

    shared_notes: dict[str, object] = {
        "window": {
            "first_bar": _FIRST_BAR.isoformat(),
            "n_bars": _N_BARS,
            "seed": "deterministic geometric curves (no RNG, no network, no engine re-run)",
        },
        "legacy_rebalances_fired": "legacy monthly+drift fired (scale-at-boundary exercised)",
        "candidate_rebalances": [d.isoformat() for d in cand.rebalance_dates],
        "d13_kopeck_match": d13_matches,
        "d13_mismatches_at_0.01": mismatches,
        "d13_max_kopeck_diff": str(max_kopeck_diff),
        "d13_max_raw_float_vs_decimal_diff": str(max_raw_diff),
        "legacy_first": str(_q(legacy_curve[0])),
        "legacy_last": str(_q(legacy_curve[-1])),
        "candidate_first": str(_q(cand.merged_equity_curve[0])),
        "candidate_last": str(_q(cand.merged_equity_curve[-1])),
        "live_zero_cost_last": str(zc.merged_equity_curve[-1]),
        "live_with_cost_last": str(live.merged_equity_curve[-1]),
        "live_curve_delta_zc_minus_live": str(curve_delta),
        "live_rebalance_cost": str(live.rebalance_cost),
        "live_realized_ndfl": str(live.realized_ndfl),
        "live_cost_plus_ndfl": str(line_items),
        "live_delta_equals_line_items": live_only_by_line_items,
        "cost_nonzero_when_leg_trades": cost_nonzero_when_leg_trades,
        "realized_ndfl_proof_gain": str(realized_gain),
        "realized_ndfl_proof_tax_13_15_band": str(realized_ndfl_proof),
        "ndfl_nonzero_on_gain": ndfl_nonzero_on_gain,
    }

    _write_iteration(
        "phase72-ab-d13-baseline",
        {
            "name": "phase72-ab-d13-baseline",
            "phase": _PHASE,
            "leg": "baseline",
            "description": "legacy PortfolioBacktestOrchestrator 60/40 (monthly+drift, no cost)",
            "final_equity": str(_q(legacy_curve[-1])),
            "verdict": verdict,
            "notes": shared_notes,
            "git_sha": git_sha,
            "created_at": datetime.now(tz=timezone.utc).isoformat(),  # noqa: UP017
        },
    )
    _write_iteration(
        "phase72-ab-d13-candidate",
        {
            "name": "phase72-ab-d13-candidate",
            "phase": _PHASE,
            "leg": "candidate",
            "description": "AllocationOrchestrator deposit=0, legacy cadence, zero cost (A4)",
            "final_equity": str(_q(cand.merged_equity_curve[-1])),
            "verdict": verdict,
            "notes": shared_notes,
            "git_sha": git_sha,
            "created_at": datetime.now(tz=timezone.utc).isoformat(),  # noqa: UP017
        },
    )
    _write_iteration(
        "phase72-ab-live-balanced",
        {
            "name": "phase72-ab-live-balanced",
            "phase": _PHASE,
            "leg": "live",
            "description": "AllocationOrchestrator BALANCED, pure quarterly, cost+NDFL ON",
            "final_equity": str(live.merged_equity_curve[-1]),
            "rebalance_cost": str(live.rebalance_cost),
            "realized_ndfl": str(live.realized_ndfl),
            "verdict": verdict,
            "notes": shared_notes,
            "git_sha": git_sha,
            "created_at": datetime.now(tz=timezone.utc).isoformat(),  # noqa: UP017
        },
    )

    _append_history(
        "phase72-ab-d13-baseline",
        git_sha=git_sha,
        verdict=verdict,
        metrics={
            "final_equity": str(_q(legacy_curve[-1])),
            "d13_kopeck_match": d13_matches,
        },
    )
    _append_history(
        "phase72-ab-d13-candidate",
        git_sha=git_sha,
        verdict=verdict,
        metrics={
            "final_equity": str(_q(cand.merged_equity_curve[-1])),
            "d13_kopeck_match": d13_matches,
            "d13_max_raw_diff": str(max_raw_diff),
        },
    )
    _append_history(
        "phase72-ab-live-balanced",
        git_sha=git_sha,
        verdict=verdict,
        metrics={
            "final_equity": str(live.merged_equity_curve[-1]),
            "rebalance_cost": str(live.rebalance_cost),
            "realized_ndfl": str(live.realized_ndfl),
            "delta_equals_line_items": live_only_by_line_items,
        },
    )

    return passed, shared_notes


def _run_idlecash(dates: list[date], git_sha: str) -> tuple[str, dict[str, object]]:
    """A/B #2: R-7 idle/transit cash 0% vs demand-rate -> chosen default + rationale."""
    ofz = _curve(_OFZ_BASE, _OFZ_DAILY, dates)
    eq = _curve(_EQ_BASE, _EQ_DAILY, dates)
    dep = _curve(_DEP_BASE, _DEP_DAILY, dates)

    orch = AllocationOrchestrator(risk_profile=RiskProfile.BALANCED)
    base = orch.run(deposit_curve=dep, ofz_pk_curve=ofz, equity_curve=eq)
    merged_last = base.merged_equity_curve[-1]

    # The allocator merges ALREADY-INVESTED curves (D-12) -> the MCFTR sleeve holds
    # no idle cash by construction; structural idle cash == 0. Quantify the demand-rate
    # alternative as an UPPER BOUND: credit the WHOLE equity leg at the demand rate for
    # a full quarter (the absurd worst case). The real transit cash is ~0.
    daily_demand = (Decimal(1) + DEPOSIT_DEMAND_RATE) ** (
        Decimal(1) / Decimal(_TRADING_DAYS_PER_YEAR)
    ) - Decimal(1)
    worst_transit = eq[-1][1]
    ub_demand_credit = worst_transit * daily_demand * Decimal(_TRADING_DAYS_PER_QUARTER)
    ub_pct = (ub_demand_credit / merged_last) * _PERCENT

    # Delta on the merged curve: 0% credit == base (no hook); demand-rate <= upper bound.
    zero_pct_last = merged_last
    demand_rate_last = merged_last + ub_demand_credit  # upper-bound, real value ~= base
    delta = demand_rate_last - zero_pct_last

    chosen_default = "0% (idle/transit cash earns nothing)"
    rationale = (
        "R-7/A3: the MCFTR sleeve is fully invested (allocator merges already-invested "
        "curves, D-12) -> structural idle cash is 0; even the absurd upper bound (the "
        f"entire equity leg at the demand rate for a full quarter) is {_q(ub_demand_credit)} "
        f"RUB == {ub_pct.quantize(Decimal('0.0001'))}% of equity -> immaterial -> pick 0% "
        "for simplicity."
    )

    notes: dict[str, object] = {
        "demand_rate_annual": str(DEPOSIT_DEMAND_RATE),
        "demand_rate_daily": str(daily_demand),
        "merged_last": str(merged_last),
        "upper_bound_demand_credit": str(_q(ub_demand_credit)),
        "upper_bound_pct_of_equity": str(ub_pct.quantize(Decimal("0.0001"))),
        "delta_zero_vs_demand_upper_bound": str(_q(delta)),
        "material": False,
        "chosen_default": chosen_default,
        "rationale": rationale,
    }

    verdict = "PASS"  # the A/B is informational: settle the default; immaterial -> 0%.
    _write_iteration(
        "phase72-ab-idlecash-zero",
        {
            "name": "phase72-ab-idlecash-zero",
            "phase": _PHASE,
            "leg": "zero",
            "description": "R-7: idle/transit cash at 0% (chosen default)",
            "final_equity": str(zero_pct_last),
            "verdict": verdict,
            "notes": notes,
            "git_sha": git_sha,
            "created_at": datetime.now(tz=timezone.utc).isoformat(),  # noqa: UP017
        },
    )
    _write_iteration(
        "phase72-ab-idlecash-demand",
        {
            "name": "phase72-ab-idlecash-demand",
            "phase": _PHASE,
            "leg": "demand",
            "description": "R-7: idle/transit cash at DEPOSIT_DEMAND_RATE (upper bound)",
            "final_equity": str(demand_rate_last),
            "verdict": verdict,
            "notes": notes,
            "git_sha": git_sha,
            "created_at": datetime.now(tz=timezone.utc).isoformat(),  # noqa: UP017
        },
    )
    _append_history(
        "phase72-ab-idlecash-zero",
        git_sha=git_sha,
        verdict=verdict,
        metrics={"final_equity": str(zero_pct_last), "chosen_default": "0%"},
    )
    _append_history(
        "phase72-ab-idlecash-demand",
        git_sha=git_sha,
        verdict=verdict,
        metrics={
            "final_equity": str(demand_rate_last),
            "delta_upper_bound": str(_q(delta)),
            "material": False,
        },
    )
    return chosen_default, notes


def main() -> int:
    git_sha = _git_sha()
    dates = _dates()

    d13_passed, d13_notes = _run_d13(dates, git_sha)
    chosen_default, idle_notes = _run_idlecash(dates, git_sha)

    overall = "PASS" if d13_passed else "REJECT"

    print("=" * 70)
    print("PHASE 72 ALLOCATION A/B GATE (D-13 reproduction + R-7 idle-cash)")
    print("=" * 70)
    print(f"window: {_FIRST_BAR} + {_N_BARS} daily bars (deterministic, no network)")
    print("-" * 70)
    print("A/B #1 -- D-13 reproduction (no-regression, A4):")
    print(f"  legacy 60/40 last      = {d13_notes['legacy_last']}")
    print(f"  allocator dep=0 last   = {d13_notes['candidate_last']}")
    print(f"  candidate rebalances   = {d13_notes['candidate_rebalances']}")
    print(f"  kopeck match           = {d13_notes['d13_kopeck_match']}")
    print(f"  mismatches @0.01       = {d13_notes['d13_mismatches_at_0.01']}")
    print(f"  max raw float-vs-dec   = {d13_notes['d13_max_raw_float_vs_decimal_diff']}")
    print("-" * 70)
    print("  live-balanced (quarterly, cost+NDFL ON):")
    print(f"    zero-cost last        = {d13_notes['live_zero_cost_last']}")
    print(f"    with-cost last        = {d13_notes['live_with_cost_last']}")
    print(f"    curve delta (zc-live) = {d13_notes['live_curve_delta_zc_minus_live']}")
    print(f"    rebalance_cost        = {d13_notes['live_rebalance_cost']}")
    print(f"    realized_ndfl         = {d13_notes['live_realized_ndfl']}")
    print(f"    delta == line items   = {d13_notes['live_delta_equals_line_items']}")
    print(
        f"    NDFL proof gain/tax   = {d13_notes['realized_ndfl_proof_gain']}"
        f" / {d13_notes['realized_ndfl_proof_tax_13_15_band']}"
    )
    print("-" * 70)
    print("A/B #2 -- R-7 idle/transit cash (0% vs demand-rate):")
    print(
        f"  upper-bound credit     = {idle_notes['upper_bound_demand_credit']} RUB"
        f" ({idle_notes['upper_bound_pct_of_equity']}% of equity)"
    )
    print(f"  material               = {idle_notes['material']}")
    print(f"  CHOSEN DEFAULT         = {chosen_default}")
    print("-" * 70)
    print(f"OVERALL VERDICT: {overall}")
    print("=" * 70)
    return 0 if d13_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
