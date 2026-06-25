#!/usr/bin/env python
"""Recommend a deposit ladder from the committed offered-rate snapshot -- READ-ONLY.

Token-free, DB-free, order-free. Loads the committed (or a supplied) deposit term-structure
snapshot, runs the recommendation-only optimizer, and prints the ranked ladders + the honest
lock-in verdict + caveats. There is NO ``--mode sandbox/live``, no ``--confirm``, no broker,
and no real-money path: this script can only ever describe a ladder PLAN.

Usage:
    uv run python scripts/recommend_deposit_ladder.py
    uv run python scripts/recommend_deposit_ladder.py --budget 2500000 --snapshot path/to.json
"""

from __future__ import annotations

import argparse
from decimal import Decimal
from pathlib import Path

from finalayze.core.exceptions import ConfigurationError
from finalayze.orchestration.deposit_ladder import (
    OptimizerRequest,
    load_term_structure,
    optimize_deposit_ladder,
)

_DEFAULT_BUDGET = Decimal(2500000)  # operator's ~2.5M RUB deposit sleeve


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Recommend a deposit ladder (read-only, no orders).")
    p.add_argument("--budget", type=Decimal, default=_DEFAULT_BUDGET, help="deposit-sleeve RUB")
    p.add_argument("--snapshot", type=Path, default=None, help="override committed snapshot path")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        ts = load_term_structure(args.snapshot) if args.snapshot else load_term_structure()
    except ConfigurationError as exc:
        print(f"REFUSING: {exc}")
        return 1

    req = OptimizerRequest(
        budget=args.budget,
        start=ts.as_of,
        horizon_months=ts.horizon_months,
        term_structure=ts,
    )
    plan = optimize_deposit_ladder(req)
    r = plan.lockin_report

    print("=" * 78)
    print("DEPOSIT-LADDER RECOMMENDATION (read-only, no orders placed)")
    print("=" * 78)
    print(f"budget           : {plan.budget:,.0f} RUB")
    print(f"window           : {plan.start} + {plan.horizon_months}mo")
    print(f"scenarios        : {', '.join(plan.scenarios_used)}")
    print(f"provenance       : {plan.snapshot_provenance}")
    print()
    rec = plan.recommended
    weights = ", ".join(f"{m}mo={w:.0%}" for m, w in sorted(rec.candidate.weights.items()))
    print(f"RECOMMENDED LADDER ({rec.candidate.archetype}): {weights}")
    print(
        f"  robust after-tax terminal (mean/min/max): {rec.mean_eatv:,.0f} / "
        f"{rec.min_eatv:,.0f} / {rec.max_eatv:,.0f} RUB"
    )
    print(
        f"  banks for full ASV insurance: {rec.banks_needed}  "
        f"(uninsured if held in one bank: {rec.uninsured_at_horizon:,.0f} RUB)"
    )
    if plan.recommendation_caveat:
        # B1: never let the recommendation silently contradict the lock-in verdict.
        print(f"  >> RECONCILIATION: {plan.recommendation_caveat}")
    if rec.path_fragile:
        print("  note: terminal value is path-dependent (high scenario dispersion)")
    print()
    print(f"LOCK-IN VERDICT : {r.verdict.value.upper()}")
    print(f"  {r.honest_message}")
    print(
        f"  per-scenario lock-in bps: { {k: round(v, 0) for k, v in r.per_scenario_bps.items()} }"
    )
    caveats = []
    if r.n1_caveat:
        caveats.append("N=1 (committed CBR calendar is one realized easing cycle)")
    if plan.progressive_band_caveat:
        caveats.append("progressive 13/15% band may apply (deposit-sleeve lower bound)")
    if r.scenario_set_degenerate:
        caveats.append("scenario set does not span rate directions -- no robust edge claim")
    print(f"  caveats: {'; '.join(caveats) if caveats else 'none'}")
    print("=" * 78)
    print("This is decision support, not an instruction. No money was moved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
