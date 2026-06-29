"""Iter-2: inflation-linker through the gate (Part A) + a multi-hedge basket test (Part B).

Part A — INFLTR (inflation-linked OFZ) through the standard Integration Gate (the last distinct
RUB fixed-income factor). Expected REJECT/limited (RUB bond, correlated with the rate factor,
short series).

Part B — does COMBINING two uncorrelated zero-carry hedges that each REJECTED individually
(gold + CNY, both 2022-covering) produce a crash-window diversification benefit neither gave
alone? Tests the "portfolio of hedges" idea directly via the gold_sleeve_lab blender:
core (deposit 40 / equity 60) vs core + gold 3% + CNY 3% (carved from equity), full window AND
the 2022 crash year. Honest: if it still drags risk-adjusted return, the combination doesn't
rescue what the singles couldn't.

    uv run python scripts/research/run_iter2.py
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

from finalayze.backtest.allocation_gate import accrue_real_risk_free_leg, net_index_returns
from finalayze.backtest.equity_tilt_experiment import _metrics, _slice
from finalayze.backtest.equity_tilt_lab import quarter_end_dates
from finalayze.backtest.gold_sleeve_lab import (
    apply_ter_drag,
    blend_portfolio,
    forward_align_legs,
    master_axis,
)
from finalayze.backtest.instrument_integration_gate import Candidate, run_integration_gate
from finalayze.core.ndfl import YtdTaxAccumulator

_DIR = Path("results/research/iter2")
_LINKER = _DIR / "linker_panel.json"
_GOLD = Path("results/research/gold/panel_snapshot.json")
_BATTERY = Path("results/research/instrument_battery/panel_snapshot.json")

_INDEX_SHIFT = 1
_BASE_DEPOSIT_W = Decimal("0.4")
_BASE_EQUITY_W = Decimal("0.6")
_HEDGE_W = Decimal("0.03")  # each hedge at the probation toe-hold (sum 6% < equity)
_DEPOSIT_SPREAD_PP = Decimal("1.0")
_CRASH_START = date(2022, 1, 1)
_CRASH_END = date(2022, 12, 31)
_BINDING_END = date(2026, 6, 10)


def _load(path: Path, key: str, shift: int) -> list[tuple[date, Decimal]]:
    raw = json.loads(path.read_text(encoding="utf-8"))["legs"][key]
    return [(date.fromisoformat(d) + timedelta(days=shift), Decimal(c)) for d, c in raw]


def _vals(nav: list[tuple[date, Decimal]]) -> list[Decimal]:
    return [v for _, v in nav]


def _part_a() -> dict[str, object]:
    linker = Candidate(
        name="INFLTR_inflation_linker",
        net_curve=net_index_returns(
            _load(_LINKER, "infltr_linker", _INDEX_SHIFT), tax_acc=YtdTaxAccumulator()
        ),
        risk_tier="medium",
        intended_role="diversifier",
    )
    v = run_integration_gate(linker, _load(_LINKER, "equity_mcftrr", _INDEX_SHIFT))
    sc = v.scorecard
    print(
        f"[A] {v.name}: {v.tier} w={v.proposed_weight} | "
        f"dSharpe={sc.marginal_sharpe_delta:+.3f} dSortino={sc.marginal_sortino_delta:+.3f} "
        f"dMaxDD={sc.marginal_maxdd_delta_pp:+.2f}pp maxcorr={sc.max_corr_to_existing_legs:.2f} "
        f"regimes={sc.regimes_covered} bars={sc.window_bars} tail_bt={sc.tail_backtestable}"
    )
    print(f"    reasons: {v.reasons}")
    return {
        "name": v.name,
        "tier": v.tier,
        "reasons": v.reasons,
        "window_bars": sc.window_bars,
        "regimes_covered": sc.regimes_covered,
        "marginal_sortino_delta": round(sc.marginal_sortino_delta, 4),
        "max_corr_to_existing_legs": round(sc.max_corr_to_existing_legs, 4),
    }


def _metrics_window(nav: list[tuple[date, Decimal]], start: date, end: date) -> dict[str, float]:
    m = _metrics(_slice([d for d, _ in nav], _vals(nav), start, end))
    return {
        "sharpe": m.sharpe,
        "sortino": m.sortino,
        "maxdd_pct": m.maxdd_pct,
        "tr_pct": m.total_return_pct,
    }


def _part_b() -> dict[str, object]:
    gold_levels = _load(_GOLD, "gold_gldrub", 0)  # currency true-date
    cny_levels = _load(_BATTERY, "cnyrub_fx", 0)  # currency true-date
    equity = _load(_GOLD, "equity_mcftrr", _INDEX_SHIFT)
    start = max(gold_levels[0][0], cny_levels[0][0], equity[0][0])
    axis = [
        d
        for d in master_axis({"g": gold_levels, "c": cny_levels, "e": equity})
        if start <= d <= _BINDING_END
    ]
    aligned = forward_align_legs({"equity": equity, "gold": gold_levels, "cny": cny_levels}, axis)
    deposit = accrue_real_risk_free_leg(
        axis, Decimal(1), spread_pp=_DEPOSIT_SPREAD_PP, tax_acc=YtdTaxAccumulator()
    )
    gold_net = apply_ter_drag(
        net_index_returns(
            list(zip(axis, aligned["gold"], strict=True)), tax_acc=YtdTaxAccumulator()
        )
    )
    cny_net = net_index_returns(
        list(zip(axis, aligned["cny"], strict=True)), tax_acc=YtdTaxAccumulator()
    )
    legs = {
        "deposit": _vals(deposit),
        "equity": list(aligned["equity"]),
        "gold": _vals(gold_net),
        "cny": _vals(cny_net),
    }
    rebal = sorted({axis[0], *quarter_end_dates(axis)})
    core = blend_portfolio(
        legs={k: legs[k] for k in ("deposit", "equity")},
        dates=axis,
        target_weights={"deposit": _BASE_DEPOSIT_W, "equity": _BASE_EQUITY_W},
        rebalance_dates=rebal,
        free_legs={"deposit"},
    )
    basket = blend_portfolio(
        legs=legs,
        dates=axis,
        target_weights={
            "deposit": _BASE_DEPOSIT_W,
            "equity": _BASE_EQUITY_W - 2 * _HEDGE_W,
            "gold": _HEDGE_W,
            "cny": _HEDGE_W,
        },
        rebalance_dates=rebal,
        free_legs={"deposit"},
    )
    full_core = _metrics_window(core, axis[0], axis[-1])
    full_basket = _metrics_window(basket, axis[0], axis[-1])
    crash_core = _metrics_window(core, _CRASH_START, _CRASH_END)
    crash_basket = _metrics_window(basket, _CRASH_START, _CRASH_END)
    d_sortino = full_basket["sortino"] - full_core["sortino"]
    d_maxdd = full_core["maxdd_pct"] - full_basket["maxdd_pct"]
    crash_d_maxdd = crash_core["maxdd_pct"] - crash_basket["maxdd_pct"]
    crash_d_sortino = crash_basket["sortino"] - crash_core["sortino"]
    # A combination "helps" only if it cuts the crash MaxDD AND does not worsen full-window Sortino.
    helps = crash_d_maxdd >= 3.0 and d_sortino >= 0.0  # noqa: PLR2004 — mirrors gate maxdd bar
    print(
        f"[B] gold+CNY basket vs core | full dMaxDD={d_maxdd:+.2f}pp dSortino={d_sortino:+.3f} | "
        f"crash dMaxDD={crash_d_maxdd:+.2f}pp dSortino={crash_d_sortino:+.3f} | helps={helps}"
    )
    return {
        "full_core": full_core,
        "full_basket": full_basket,
        "crash_core": crash_core,
        "crash_basket": crash_basket,
        "full_maxdd_cut_pp": round(d_maxdd, 4),
        "full_sortino_delta": round(d_sortino, 4),
        "crash_maxdd_cut_pp": round(crash_d_maxdd, 4),
        "crash_sortino_delta": round(crash_d_sortino, 4),
        "combination_helps": helps,
        "finding": (
            "the gold+CNY hedge basket cuts crash MaxDD but still drags full-window "
            "risk-adjusted return (zero-carry) — combining two REJECTED hedges does not "
            "rescue diversification"
            if not helps
            else "the gold+CNY hedge basket clears the crash de-risk bar without a full-window cost"
        ),
    }


def main() -> None:
    part_a = _part_a()
    part_b = _part_b()
    _DIR.mkdir(parents=True, exist_ok=True)
    (_DIR / "iter2_summary.json").write_text(
        json.dumps({"part_a_linker": part_a, "part_b_hedge_basket": part_b}, indent=1),
        encoding="utf-8",
    )
    print(f"DONE: linker={part_a['tier']} basket_helps={part_b['combination_helps']}")


if __name__ == "__main__":
    main()
