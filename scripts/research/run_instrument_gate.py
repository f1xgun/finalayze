"""Instrument Integration Gate — validation run on the committed gold + ZO panels.

Builds the Phase-A gold and Phase-B ZO candidates from their committed snapshots and runs them
through the standard gate. Expected (the gate must reproduce the hand-done results):
  gold -> REJECT   (tail in-window, raised the crash-year drawdown + worsened Sortino)
  ZO   -> PROBATION (FX-linked + uncorrelated, but the 2022 tail is un-backtestable)

    uv run python scripts/research/run_instrument_gate.py
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

from finalayze.backtest.allocation_gate import net_index_returns
from finalayze.backtest.gold_sleeve_lab import apply_ter_drag
from finalayze.backtest.instrument_integration_gate import (
    Candidate,
    IntegrationVerdict,
    run_integration_gate,
)
from finalayze.core.ndfl import YtdTaxAccumulator

_GOLD = Path("results/research/gold/panel_snapshot.json")
_ZO = Path("results/research/zo/panel_snapshot.json")
_OUT = Path("results/research/instrument_gate/validation_summary.json")
# Index legs (MCFTRR/RURPLRUBTR) carry the MSK T-1 convention -> shift +1 to true date; the gold
# currency leg is already true-dated (Phase-A/B lesson).
_INDEX_SHIFT = 1


def _load(path: Path, key: str, shift_days: int) -> list[tuple[date, Decimal]]:
    raw = json.loads(path.read_text(encoding="utf-8"))["legs"][key]
    return [(date.fromisoformat(d) + timedelta(days=shift_days), Decimal(c)) for d, c in raw]


def _row(v: IntegrationVerdict) -> dict[str, object]:
    sc = v.scorecard
    return {
        "name": v.name,
        "tier": v.tier,
        "proposed_weight": str(v.proposed_weight),
        "carved_from": v.carved_from,
        "n1_caveat": v.n1_caveat,
        "scorecard": {
            "window_bars": sc.window_bars,
            "regimes_covered": sc.regimes_covered,
            "tail_backtestable": sc.tail_backtestable,
            "marginal_sharpe_delta": round(sc.marginal_sharpe_delta, 4),
            "marginal_sortino_delta": round(sc.marginal_sortino_delta, 4),
            "marginal_maxdd_delta_pp": round(sc.marginal_maxdd_delta_pp, 4),
            "crash_year_maxdd_delta_pp": round(sc.crash_year_maxdd_delta_pp, 4),
            "toehold_sortino_delta": round(sc.toehold_sortino_delta, 4),
            "corr_to_legs": {k: round(c, 4) for k, c in sc.corr_to_legs.items()},
            "max_corr_to_existing_legs": round(sc.max_corr_to_existing_legs, 4),
        },
        "reasons": v.reasons,
    }


def main() -> None:
    gold = Candidate(
        name="GLDRUB_gold",
        net_curve=apply_ter_drag(
            net_index_returns(_load(_GOLD, "gold_gldrub", 0), tax_acc=YtdTaxAccumulator())
        ),
        risk_tier="high",
        intended_role="hedge",
    )
    gold_verdict = run_integration_gate(gold, _load(_GOLD, "equity_mcftrr", _INDEX_SHIFT))

    zo = Candidate(
        name="RURPLRUBTR_zo",
        net_curve=net_index_returns(
            _load(_ZO, "zo_rurplrubtr", _INDEX_SHIFT), tax_acc=YtdTaxAccumulator()
        ),
        risk_tier="medium",
        intended_role="diversifier",
    )
    zo_verdict = run_integration_gate(zo, _load(_ZO, "equity_mcftrr", _INDEX_SHIFT))

    verdicts = [gold_verdict, zo_verdict]
    for v in verdicts:
        sc = v.scorecard
        print(
            f"{v.name}: {v.tier} w={v.proposed_weight} carved={v.carved_from} | "
            f"dSharpe={sc.marginal_sharpe_delta:.3f} dSortino={sc.marginal_sortino_delta:.3f} "
            f"dMaxDD={sc.marginal_maxdd_delta_pp:.2f}pp "
            f"crashD={sc.crash_year_maxdd_delta_pp:.2f}pp "
            f"toeSortino={sc.toehold_sortino_delta:.3f} "
            f"maxcorr={sc.max_corr_to_existing_legs:.3f} "
            f"tail_bt={sc.tail_backtestable} bars={sc.window_bars}"
        )
        print(f"   reasons: {v.reasons}")

    expected = {"GLDRUB_gold": "REJECT", "RURPLRUBTR_zo": "PROBATION"}
    validation = {v.name: (v.tier == expected[v.name]) for v in verdicts}
    all_ok = all(validation.values())

    _OUT.parent.mkdir(parents=True, exist_ok=True)
    _OUT.write_text(
        json.dumps(
            {
                "verdicts": [_row(v) for v in verdicts],
                "expected": expected,
                "validation_pass": validation,
                "all_ok": all_ok,
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    print(f"VALIDATION {'PASS' if all_ok else 'FAIL'}: {validation}")
    if not all_ok:
        msg = f"gate validation failed: {validation}"
        raise SystemExit(msg)


if __name__ == "__main__":
    main()
