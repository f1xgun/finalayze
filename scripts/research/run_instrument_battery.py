"""Instrument battery (iter 1): run a risk-tiered set of candidates through the Integration Gate.

Each candidate is built from the committed battery snapshot (net of NDFL, true-date aligned) and
run through the standard Instrument Integration Gate against the deposit+equity core. Emits a
verdict table + summary; any INTEGRATE passer would graduate to a real SAA leg (operator-gated,
config-only — never an order).

    uv run python scripts/research/run_instrument_battery.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

from finalayze.backtest.allocation_gate import net_index_returns
from finalayze.backtest.instrument_integration_gate import (
    Candidate,
    IntegrationVerdict,
    run_integration_gate,
)
from finalayze.core.ndfl import YtdTaxAccumulator

_DIR = Path("results/research/instrument_battery")
_SNAP = _DIR / "panel_snapshot.json"
_INDEX_SHIFT = 1  # MSK T-1 index/MCFTRR legs -> +1 to true date
_NO_SHIFT = 0  # LQDT (shares true-date) + CNYRUB (currency true-date)


@dataclass(frozen=True)
class _Spec:
    name: str
    leg_key: str
    shift: int
    risk_tier: str
    role: str


_SPECS = [
    _Spec("RGBITR_ofz_fixed", "rgbitr_ofz_fixed", _INDEX_SHIFT, "medium", "carry"),
    _Spec("RUCBITR_corp_ig", "rucbitr_corp_ig", _INDEX_SHIFT, "medium", "carry"),
    _Spec("RUCBHYTR_corp_hy", "rucbhytr_corp_hy", _INDEX_SHIFT, "high", "carry"),
    _Spec("LQDT_money_market", "lqdt_money_market", _NO_SHIFT, "low", "cash"),
    _Spec("CNYRUB_fx", "cnyrub_fx", _NO_SHIFT, "high", "diversifier"),
]


def _load(legs: dict[str, list[list[str]]], key: str, shift: int) -> list[tuple[date, Decimal]]:
    return [(date.fromisoformat(d) + timedelta(days=shift), Decimal(c)) for d, c in legs[key]]


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
    legs = json.loads(_SNAP.read_text(encoding="utf-8"))["legs"]
    equity = _load(legs, "equity_mcftrr", _INDEX_SHIFT)

    verdicts: list[IntegrationVerdict] = []
    for spec in _SPECS:
        candidate = Candidate(
            name=spec.name,
            net_curve=net_index_returns(
                _load(legs, spec.leg_key, spec.shift), tax_acc=YtdTaxAccumulator()
            ),
            risk_tier=spec.risk_tier,
            intended_role=spec.role,
        )
        verdicts.append(run_integration_gate(candidate, equity))

    for v in verdicts:
        sc = v.scorecard
        print(
            f"{v.name:20s} {v.tier:16s} w={v.proposed_weight} | "
            f"dSharpe={sc.marginal_sharpe_delta:+.3f} dSortino={sc.marginal_sortino_delta:+.3f} "
            f"dMaxDD={sc.marginal_maxdd_delta_pp:+.2f}pp "
            f"crashD={sc.crash_year_maxdd_delta_pp:+.2f}pp "
            f"maxcorr={sc.max_corr_to_existing_legs:.2f} tail_bt={sc.tail_backtestable}"
        )

    tiers = {v.name: v.tier for v in verdicts}
    integrated = [v.name for v in verdicts if v.tier == "INTEGRATE"]
    probation = [v.name for v in verdicts if v.tier == "PROBATION"]
    _DIR.mkdir(parents=True, exist_ok=True)
    (_DIR / "battery_summary.json").write_text(
        json.dumps(
            {
                "verdicts": [_row(v) for v in verdicts],
                "tiers": tiers,
                "integrate": integrated,
                "probation": probation,
                "finding": (
                    f"{len(integrated)} INTEGRATE, {len(probation)} PROBATION, "
                    f"{len(verdicts) - len(integrated) - len(probation)} REJECT/INSUFFICIENT "
                    "against the deposit+equity core"
                ),
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    print(f"INTEGRATE={integrated} PROBATION={probation}")


if __name__ == "__main__":
    main()
