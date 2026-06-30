"""Iter-3: regime-CONDITIONAL hedging — hold the gold hedge only under a trailing stress flag.

Every STATIC hedge (gold/CNY/ZO) was REJECTED because of its calm-time carry drag against the
~18% deposit. This tests the idea that attacks that root cause: hold the gold sleeve ONLY when a
look-ahead-safe stress regime is flagged (equity in a trailing drawdown), and sit gold-free
otherwise — so you pay the hedge's drag only when you might need it.

Three arms over the gold panel window (2022-2026), through the SAME net-cost/NDFL blender:
  - core           : deposit 40 / equity 60                       (no hedge)
  - static_gold    : deposit 40 / equity 50 / gold 10  (always)   (the iter-1 REJECT shape)
  - conditional    : deposit 40 / equity 60 baseline; on a stress-flagged quarter rotate to
                     deposit 40 / equity 50 / gold 10 (weight_schedule), else gold 0

Stress flag (strictly trailing, no look-ahead): equity is more than _STRESS_DD below its running
peak using only closes dated <= the rebalance date. Honest expectation: the conditional arm
avoids most of static_gold's calm-time drag, but the 2022 acute gap is too fast for a quarterly
flag to catch — so it likely lands between core and static, not beating the deposit-anchored core.

    uv run python scripts/research/run_regime_hedging.py
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

from finalayze.backtest.allocation_gate import (
    accrue_real_risk_free_leg,
    net_index_returns,
    regime_split,
)
from finalayze.backtest.equity_tilt_experiment import _metrics, _slice
from finalayze.backtest.equity_tilt_lab import quarter_end_dates
from finalayze.backtest.gold_sleeve_lab import (
    apply_ter_drag,
    blend_portfolio,
    forward_align_legs,
    master_axis,
)
from finalayze.core.ndfl import YtdTaxAccumulator

_DIR = Path("results/research/regime_hedging")
_GOLD = Path("results/research/gold/panel_snapshot.json")
_INDEX_SHIFT = 1
_DEPOSIT_W = Decimal("0.4")
_EQUITY_W = Decimal("0.6")
_GOLD_W = Decimal("0.10")  # hedge size when stress is flagged
_DEPOSIT_SPREAD_PP = Decimal("1.0")
_STRESS_DD = Decimal("0.10")  # equity >10% below its trailing peak = stress
_STRESS_MIN_BARS = 60  # need some history before the flag can fire
_CRASH_START = date(2022, 1, 1)
_CRASH_END = date(2022, 12, 31)
_BINDING_END = date(2026, 6, 10)


def _load(path: Path, key: str, shift: int) -> list[tuple[date, Decimal]]:
    raw = json.loads(path.read_text(encoding="utf-8"))["legs"][key]
    return [(date.fromisoformat(d) + timedelta(days=shift), Decimal(c)) for d, c in raw]


def _vals(nav: list[tuple[date, Decimal]]) -> list[Decimal]:
    return [v for _, v in nav]


def _stress_on(d: date, equity: list[tuple[date, Decimal]]) -> bool:
    """Trailing-drawdown stress flag: equity > _STRESS_DD below its running peak, closes <= d."""
    hist = [v for dt, v in equity if dt <= d and v > 0]
    if len(hist) < _STRESS_MIN_BARS:
        return False
    peak = max(hist)
    return peak > 0 and (peak - hist[-1]) / peak > _STRESS_DD


def _metrics_window(nav: list[tuple[date, Decimal]], start: date, end: date) -> dict[str, float]:
    m = _metrics(_slice([d for d, _ in nav], _vals(nav), start, end))
    return {
        "sharpe": round(m.sharpe, 4),
        "sortino": round(m.sortino, 4),
        "maxdd_pct": round(m.maxdd_pct, 4),
        "tr_pct": round(m.total_return_pct, 4),
    }


def main() -> None:
    gold_levels = _load(_GOLD, "gold_gldrub", 0)
    equity_raw = _load(_GOLD, "equity_mcftrr", _INDEX_SHIFT)
    start = max(gold_levels[0][0], equity_raw[0][0])
    axis = [
        d for d in master_axis({"g": gold_levels, "e": equity_raw}) if start <= d <= _BINDING_END
    ]
    aligned = forward_align_legs({"equity": equity_raw, "gold": gold_levels}, axis)
    deposit = accrue_real_risk_free_leg(
        axis, Decimal(1), spread_pp=_DEPOSIT_SPREAD_PP, tax_acc=YtdTaxAccumulator()
    )
    equity = list(zip(axis, aligned["equity"], strict=True))
    gold = apply_ter_drag(
        net_index_returns(
            list(zip(axis, aligned["gold"], strict=True)), tax_acc=YtdTaxAccumulator()
        )
    )
    legs = {"deposit": _vals(deposit), "equity": _vals(equity), "gold": _vals(gold)}
    rebal = sorted({axis[0], *quarter_end_dates(axis)})

    # Conditional schedule: gold ON only on stress-flagged rebalance dates (trailing, as-of).
    stress_dates = [d for d in rebal if _stress_on(d, equity)]
    on_w = {"deposit": _DEPOSIT_W, "equity": _EQUITY_W - _GOLD_W, "gold": _GOLD_W}
    schedule = dict.fromkeys(stress_dates, on_w)

    core = blend_portfolio(
        legs={k: legs[k] for k in ("deposit", "equity")},
        dates=axis,
        target_weights={"deposit": _DEPOSIT_W, "equity": _EQUITY_W},
        rebalance_dates=rebal,
        free_legs={"deposit"},
    )
    static_gold = blend_portfolio(
        legs=legs,
        dates=axis,
        target_weights=on_w,
        rebalance_dates=rebal,
        free_legs={"deposit"},
    )
    conditional = blend_portfolio(
        legs=legs,
        dates=axis,
        target_weights={"deposit": _DEPOSIT_W, "equity": _EQUITY_W, "gold": Decimal(0)},
        rebalance_dates=rebal,
        free_legs={"deposit"},
        weight_schedule=schedule,
    )

    arms = {"core": core, "static_gold": static_gold, "conditional_gold": conditional}
    regions = regime_split(axis)
    windows = {"full": (axis[0], axis[-1]), "crash_2022": (_CRASH_START, _CRASH_END), **regions}
    table = {
        name: {w: _metrics_window(nav, s, e) for w, (s, e) in windows.items()}
        for name, nav in arms.items()
    }

    core_full = table["core"]["full"]
    cond_full = table["conditional_gold"]["full"]
    static_full = table["static_gold"]["full"]
    # The real bar is the no-hedge CORE: conditional hedging "works" only if it beats the core on
    # risk-adjusted return (Sortino) AND does not worsen MaxDD. Beating static_gold is irrelevant if
    # both lose to the core.
    beats_core = (
        cond_full["sortino"] >= core_full["sortino"]
        and cond_full["maxdd_pct"] <= core_full["maxdd_pct"]
    )
    verdict = "WORKS" if beats_core else "NO"
    dominated_by_core = (
        cond_full["sortino"] < core_full["sortino"]
        and cond_full["maxdd_pct"] > core_full["maxdd_pct"]
        and cond_full["tr_pct"] < core_full["tr_pct"]
    )

    finding = (
        f"Conditional hedging verdict: {verdict}. Stress flagged on "
        f"{len(stress_dates)}/{len(rebal)} "
        f"quarters (the trailing-DD flag is ON most of the volatile 2022-2025 window). "
        f"Full-window Sortino core={core_full['sortino']} conditional={cond_full['sortino']} "
        f"static={static_full['sortino']}; MaxDD core={core_full['maxdd_pct']} "
        f"conditional={cond_full['maxdd_pct']} static={static_full['maxdd_pct']}; "
        f"TR core={core_full['tr_pct']} conditional={cond_full['tr_pct']}. "
        + (
            "Conditional hedging is DOMINATED by the core (worse Sortino, MaxDD AND total return): "
            "the lagging quarterly flag mis-times the rotation — it buys gold AFTER drawdowns and "
            "sells AFTER recoveries (buy-high-sell-low) and pays the switching turnover, while the "
            "2022 acute gap is too fast to catch. Even SMART (conditional) use of the hedge does "
            "not beat the deposit-anchored core."
            if dominated_by_core
            else (
                "Conditional hedging does not beat the deposit-anchored core on risk-adjusted."
                if verdict == "NO"
                else "Conditional hedging BEATS the core on risk-adjusted return, no worse MaxDD."
            )
        )
    )

    _DIR.mkdir(parents=True, exist_ok=True)
    (_DIR / "regime_hedging_summary.json").write_text(
        json.dumps(
            {
                "stress_dd_threshold": str(_STRESS_DD),
                "gold_weight_when_on": str(_GOLD_W),
                "n_rebalances": len(rebal),
                "n_stress_quarters": len(stress_dates),
                "stress_dates": [d.isoformat() for d in stress_dates],
                "metrics": table,
                "verdict": verdict,
                "finding": finding,
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    for name in arms:
        f = table[name]["full"]
        print(
            f"{name:18s} full: Sharpe={f['sharpe']:+.3f} Sortino={f['sortino']:+.3f} "
            f"MaxDD={f['maxdd_pct']:.2f} TR={f['tr_pct']:.1f}"
        )
    print(finding)


if __name__ == "__main__":
    main()
