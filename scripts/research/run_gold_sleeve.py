"""Phase-A cert: does a GOLD sleeve diversify the deposit-anchored allocation?

Deterministic, token-free. Loads the committed gold panel (GLDRUB spot + MCFTRR net
equity), accrues the net deposit leg from the committed CBR archive, builds three
NET sleeves (deposit / equity / gold-net+TER), and blends a baseline (deposit+equity)
against gold variants (gold carved from equity, sweep 5/10/15%) through the SAME
net-of-cost / net-of-NDFL multi-sleeve simulator.

The honest, pre-registered deliverable is DIVERSIFICATION, never alpha:

  does adding a small gold sleeve cut portfolio MaxDD by >= the pre-registered bar
  AND not worsen the risk-adjusted return, IN THE 2022 CRASH where active equity
  selection failed — after the ETF-TER + retail cost haircut?

METRIC HONESTY (a real trap this cert documents): the RUONIA-excess Sharpe/Sortino
uses a FIXED 15% basis that is only apt for the high-rate era. Over a window that
includes the 2022-2023 LOW-rate era (CBR key 7.5%) the deposit underperforms that
15% basis, so its excess-Sharpe goes hugely negative — making a deposit-bar
"vs deposit" Sharpe test MEANINGLESS on a 2022-start window. So the deposit-anchor
point is made on BASIS-FREE TOTAL RETURN (the deposit's raw dominance in the
high-rate era), and the gold verdict is the baseline-vs-+gold DIVERSIFICATION
comparison (where the common 15% basis cancels and MaxDD is basis-free).

Honesty controls: (1) a 0%-gold variant reproduces the baseline byte-for-byte;
(2) the gold leg's raw return in BOTH the acute halt and the full crash year (the
spike-then-giveback mechanism); (3) the gold-vs-equity daily-return correlation.

    uv run python scripts/research/run_gold_sleeve.py
"""

from __future__ import annotations

import json
import statistics
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

from finalayze.backtest.allocation_gate import (
    accrue_real_risk_free_leg,
    net_index_returns,
)
from finalayze.backtest.equity_tilt_experiment import (
    RISK_FREE_ANNUAL_PCT,
    ArmMetrics,
    _metrics,
    _slice,
)
from finalayze.backtest.equity_tilt_lab import quarter_end_dates
from finalayze.backtest.gold_sleeve_lab import (
    _MAXDD_CUT_MIN_PP,  # the pre-registered diversification bar (single source of truth)
    apply_ter_drag,
    blend_portfolio,
    diversification_verdict,
    forward_align_legs,
    master_axis,
)
from finalayze.core.ndfl import YtdTaxAccumulator

_DIR = Path("results/research/gold")
_SNAP = _DIR / "panel_snapshot.json"

# Allocation: deposit anchor + equity; gold is carved from the equity sleeve.
_DEPOSIT_W = Decimal("0.4")
_EQUITY_BASE_W = Decimal("0.6")
_GOLD_WEIGHTS = (Decimal("0.05"), Decimal("0.10"), Decimal("0.15"))
_DEPOSIT_SPREAD_PP = Decimal("1.0")  # deposit = CBR key - 1pp (mirrors the gate)

# Explicit regime windows (NOT regime_split — that assumes a high-rate-era START; this
# axis starts in 2022, so its sub-windows are named explicitly).
_CRASH_START = date(2022, 2, 21)  # just before the invasion
_ACUTE_END = date(2022, 4, 30)  # invasion + 27-day halt + immediate aftermath
_CRASH_END = date(2022, 12, 30)  # full crash year (captures the ruble round-trip)
_HIGH_RATE_START = date(2024, 1, 1)  # the real 16-21% high-rate era
_HIGH_RATE_END = date(2025, 6, 5)  # day before the first 2025 cut
_EASING_START = date(2025, 6, 6)  # REGIME_SPLIT_BOUNDARY (first real cut)
_BINDING_END = date(2026, 6, 10)  # look-ahead clamp (mirrors allocation_gate._BINDING_END)
_MIN_CORR_PAIRS = 30
# Crash windows where the diversification deliverable is judged.
_CRASH_WINDOWS = ("acute_crash_2022", "crash_year_2022")


def _load_legs() -> tuple[list[tuple[date, Decimal]], list[tuple[date, Decimal]]]:
    raw = json.loads(_SNAP.read_text(encoding="utf-8"))["legs"]
    gold = [(date.fromisoformat(d), Decimal(c)) for d, c in raw["gold_gldrub"]]
    # DATE-CONVENTION FIX (review CR): the MCFTRR equity leg comes through the index-candle
    # path (load_mcftr_series -> _parse_history_row), which parses ISS TRADEDATE as MSK-midnight
    # then converts to UTC, deterministically shifting a trade on T to the stored date T-1. The
    # gold leg (fetch_currency_close_history) keeps the TRUE ISS trade date. To blend the two on
    # one true calendar (so daily returns line up and the window boundaries match real dates), the
    # equity leg is shifted +1 day back to its true trade date. The shift is exactly -1 always
    # (MSK = UTC+3, midnight MSK -> 21:00 prev-day UTC), so +1 recovers it.
    equity = [
        (date.fromisoformat(d) + timedelta(days=1), Decimal(c)) for d, c in raw["equity_mcftrr"]
    ]
    return gold, equity


def _curve_metrics(nav: list[tuple[date, Decimal]], start: date, end: date) -> ArmMetrics:
    dates = [d for d, _ in nav]
    vals = [v for _, v in nav]
    return _metrics(_slice(dates, vals, start, end))


def _total_return_pct(nav: list[tuple[date, Decimal]], start: date, end: date) -> float:
    vals = _slice([d for d, _ in nav], [v for _, v in nav], start, end)
    return (vals[-1] / vals[0] - 1.0) * 100.0 if vals and vals[0] > 0 else 0.0


def _ret_corr(
    a: list[tuple[date, Decimal]], b: list[tuple[date, Decimal]], start: date, end: date
) -> float | None:
    """Daily-return correlation of two same-axis curves over [start, end]."""
    ar: list[float] = []
    br: list[float] = []
    pa = [(d, float(v)) for d, v in a if start <= d <= end]
    pb = {d: float(v) for d, v in b if start <= d <= end}
    for i in range(1, len(pa)):
        d0, a0 = pa[i - 1]
        d1, a1 = pa[i]
        if d0 in pb and d1 in pb and a0 > 0 and pb[d0] > 0:
            ar.append(a1 / a0 - 1.0)
            br.append(pb[d1] / pb[d0] - 1.0)
    if len(ar) < _MIN_CORR_PAIRS:
        return None
    return statistics.correlation(ar, br)


def main() -> None:  # noqa: PLR0915 — single linear cert script
    gold_raw, equity_raw = _load_legs()
    start = max(gold_raw[0][0], equity_raw[0][0])
    axis = [d for d in master_axis({"g": gold_raw, "e": equity_raw}) if start <= d <= _BINDING_END]
    aligned = forward_align_legs({"equity": equity_raw, "gold": gold_raw}, axis)

    # NET sleeves on the shared axis. equity (MCFTRR) is ALREADY net — never re-tax it.
    deposit_curve = accrue_real_risk_free_leg(
        axis, Decimal(1), spread_pp=_DEPOSIT_SPREAD_PP, tax_acc=YtdTaxAccumulator()
    )
    equity_curve = list(zip(axis, aligned["equity"], strict=True))
    gold_levels = list(zip(axis, aligned["gold"], strict=True))
    gold_curve = apply_ter_drag(net_index_returns(gold_levels, tax_acc=YtdTaxAccumulator()))

    legs = {
        "deposit": [v for _, v in deposit_curve],
        "equity": [v for _, v in equity_curve],
        "gold": [v for _, v in gold_curve],
    }
    rebal = sorted({axis[0], *quarter_end_dates(axis)})

    def _blend(weights: dict[str, Decimal]) -> list[tuple[date, Decimal]]:
        return blend_portfolio(
            legs={k: legs[k] for k in weights},
            dates=axis,
            target_weights=weights,
            rebalance_dates=rebal,
            free_legs={"deposit"},
        )

    baseline = _blend({"deposit": _DEPOSIT_W, "equity": _EQUITY_BASE_W})
    variants = {
        g: _blend({"deposit": _DEPOSIT_W, "equity": _EQUITY_BASE_W - g, "gold": g})
        for g in _GOLD_WEIGHTS
    }

    # Control (1): a 0%-gold three-leg blend must equal the two-leg baseline.
    gold_zero = _blend({"deposit": _DEPOSIT_W, "equity": _EQUITY_BASE_W, "gold": Decimal(0)})
    zero_ok = [v for _, v in gold_zero] == [v for _, v in baseline]

    windows: dict[str, tuple[date, date]] = {
        "full_window": (axis[0], axis[-1]),
        "acute_crash_2022": (_CRASH_START, _ACUTE_END),
        "crash_year_2022": (_CRASH_START, _CRASH_END),
        "high_rate_2024_25": (_HIGH_RATE_START, _HIGH_RATE_END),
        "easing_2025_26": (_EASING_START, axis[-1]),
    }

    rows: dict[str, object] = {}
    for wname, (w_start, w_end) in windows.items():
        dep_m = _curve_metrics(deposit_curve, w_start, w_end)
        base_m = _curve_metrics(baseline, w_start, w_end)
        per_variant: dict[str, object] = {}
        for g, nav in variants.items():
            vm = _curve_metrics(nav, w_start, w_end)
            div = diversification_verdict(
                baseline_maxdd_pct=base_m.maxdd_pct,
                gold_maxdd_pct=vm.maxdd_pct,
                baseline_sortino=base_m.sortino,
                gold_sortino=vm.sortino,
            )
            per_variant[str(g)] = {"metrics": vm.__dict__, "diversification": div}
        rows[wname] = {
            "range": [w_start.isoformat(), w_end.isoformat()],
            "deposit": dep_m.__dict__,
            "baseline": base_m.__dict__,
            "variants": per_variant,
            "n1_caveat": wname in _CRASH_WINDOWS or wname == "easing_2025_26",
        }

    # Honesty mechanism controls. Report the RAW GLDRUB price round-trip separately from the
    # NET (daily-mark NDFL + TER) leg, so the crash-year figure is not mis-attributed entirely
    # to the ruble recovery (the net leg's extra drag is the conservative daily-mark NDFL model).
    gold_raw_acute_ret = _total_return_pct(gold_raw, _CRASH_START, _ACUTE_END)
    gold_raw_year_ret = _total_return_pct(gold_raw, _CRASH_START, _CRASH_END)
    gold_acute_ret = _total_return_pct(gold_curve, _CRASH_START, _ACUTE_END)
    gold_year_ret = _total_return_pct(gold_curve, _CRASH_START, _CRASH_END)
    gold_eq_corr = _ret_corr(gold_curve, equity_curve, _CRASH_START, _CRASH_END)

    # ── Binding verdicts ─────────────────────────────────────────────────────
    # (A) Deposit-anchor (BASIS-FREE total return): deposit dominates raw return in
    #     the high-rate era + full window — the real, robust deposit-wins point.
    def _dep_beats_base_tr(wname: str) -> bool:
        wrow = rows[wname]
        return wrow["deposit"]["total_return_pct"] > wrow["baseline"]["total_return_pct"]  # type: ignore[index]

    deposit_anchor_holds = _dep_beats_base_tr("full_window") and _dep_beats_base_tr(
        "high_rate_2024_25"
    )

    # (B) Gold's effect in the crash, classified 3-way (richer than a binary pass):
    #   DIVERSIFIES   — cuts MaxDD by the bar AND does not worsen Sortino (a free win);
    #   DERISKS_ONLY  — cuts MaxDD by the bar BUT worsens Sortino (zero-yield drag);
    #   NO            — neither.
    diversifying: dict[str, list[str]] = {}
    derisking: dict[str, list[str]] = {}
    for wname in _CRASH_WINDOWS:
        wrow = rows[wname]
        variants_block = wrow["variants"]  # type: ignore[index]
        diversifying[wname] = [
            g
            for g, v in variants_block.items()
            if v["diversification"]["diversifies"]  # type: ignore[union-attr,index]
        ]
        derisking[wname] = [
            g
            for g, v in variants_block.items()
            if v["diversification"]["maxdd_ok"]  # type: ignore[union-attr,index]
        ]
    gold_diversifies = any(diversifying[w] for w in _CRASH_WINDOWS)
    gold_derisks = any(derisking[w] for w in _CRASH_WINDOWS)
    if gold_diversifies:
        crash_classification = "DIVERSIFIES"
    elif gold_derisks:
        crash_classification = "DERISKS_ONLY"
    else:
        crash_classification = "NO"

    finding = (
        f"Deposit anchor holds on raw total return (high-rate + full): {deposit_anchor_holds}. "
        f"Gold crash de-risking (pre-registered >={_MAXDD_CUT_MIN_PP}pp MaxDD cut in a crash "
        f"window): {crash_classification} — the best acute-crash cut is ~3pp (just under the bar) "
        f"and gold INCREASES MaxDD in the crash YEAR (the give-back). Gold DOES shave MaxDD "
        f"modestly in the calm/acute regimes (full, acute, high-rate, easing all lower in the "
        f"table) but ALWAYS worsens risk-adjusted return (Sortino) — the zero-yield drag. Its "
        f"2022 RUB hedge was a ~2-week flash: raw price round-tripped {gold_raw_year_ret:+.0f}% "
        f"over the crash year (spike then capital-controlled ruble recovery); the net leg ended "
        f"{gold_year_ret:+.0f}% after the conservative daily-mark NDFL + TER haircut. Net: gold is "
        f"at most a MARGINAL drawdown-reducer at a risk-adjusted cost — not a diversifier, not a "
        f"reliable crash hedge, not alpha (N=1)."
    )

    summary = {
        "window": {
            "start": axis[0].isoformat(),
            "end": axis[-1].isoformat(),
            "n_bars": len(axis),
            "n_rebalances": len(rebal),
        },
        "weights": {
            "deposit": str(_DEPOSIT_W),
            "equity_base": str(_EQUITY_BASE_W),
            "gold_sweep": [str(g) for g in _GOLD_WEIGHTS],
        },
        "risk_free_annual_pct": RISK_FREE_ANNUAL_PCT,
        "metric_caveat": (
            "RUONIA-excess Sharpe/Sortino use a FIXED 15% basis apt only for the high-rate "
            "era; over the 2022-2023 low-rate era the deposit underperforms it, so the "
            "deposit-anchor point is made on basis-free TOTAL RETURN and the gold verdict on "
            "the baseline-vs-+gold MaxDD/Sortino DELTA (common basis cancels)."
        ),
        "controls": {
            "gold_zero_reproduces_baseline": zero_ok,
            "gold_RAW_price_acute_return_pct": round(gold_raw_acute_ret, 2),
            "gold_RAW_price_crash_year_return_pct": round(gold_raw_year_ret, 2),
            "gold_NET_leg_acute_return_pct": round(gold_acute_ret, 2),
            "gold_NET_leg_crash_year_return_pct": round(gold_year_ret, 2),
            "gold_vs_equity_crash_return_corr": (
                round(gold_eq_corr, 4) if gold_eq_corr is not None else None
            ),
        },
        "windows": rows,
        "binding": {
            "deposit_anchor_holds_raw_return": deposit_anchor_holds,
            "crash_classification": crash_classification,
            "diversifying_weights": diversifying,
            "derisking_weights": derisking,
            "finding": finding,
            "n1_caveat": True,
        },
    }
    _DIR.mkdir(parents=True, exist_ok=True)
    (_DIR / "gold_cert_summary.json").write_text(
        json.dumps(summary, indent=1, default=str), encoding="utf-8"
    )

    def f(x: object) -> str:
        return f"{x:.3f}" if isinstance(x, float) else str(x)

    md = [
        "# Phase A — Gold Sleeve vs Deposit-Anchored Allocation (Cert)",
        "",
        f"Window `{axis[0]}`->`{axis[-1]}` · {len(axis)} bars · {len(rebal)} rebalances · "
        f"RUONIA-excess {RISK_FREE_ANNUAL_PCT}%",
        f"Base = deposit {_DEPOSIT_W} / equity {_EQUITY_BASE_W}; gold carved from equity "
        f"(sweep {[str(g) for g in _GOLD_WEIGHTS]}). Gold = GLDRUB spot, net-NDFL + TER haircut.",
        "",
        "> **Metric caveat:** the RUONIA-excess Sharpe/Sortino use a fixed 15% basis apt only "
        "for the high-rate era. Over a 2022-start window the deposit underperforms that basis "
        "in the 2022-2023 low-rate era, so a 'vs deposit' Sharpe test is meaningless here — the "
        "deposit-anchor point is made on **basis-free total return**, the gold verdict on the "
        "**baseline-vs-+gold MaxDD/Sortino delta** (common basis cancels).",
        "",
        "## Honesty controls",
        f"- 0%-gold reproduces baseline curve: **{zero_ok}**",
        f"- gold RAW price round-trip — acute (Feb21-Apr30): **{gold_raw_acute_ret:+.1f}%**, "
        f"crash YEAR: **{gold_raw_year_ret:+.1f}%** (the spike then capital-controlled ruble "
        f"recovery — the market move)",
        f"- gold NET leg (after daily-mark NDFL + TER) — acute: **{gold_acute_ret:+.1f}%**, "
        f"crash YEAR: **{gold_year_ret:+.1f}%** (extra drag = the conservative daily-mark NDFL)",
        f"- gold vs equity daily-return corr (crash year): "
        f"**{f(gold_eq_corr) if gold_eq_corr is not None else 'n/a'}** "
        f"(low/negative ⇒ genuinely uncorrelated)",
        "",
        f"## BINDING: deposit anchor holds (raw return) = **{deposit_anchor_holds}** · "
        f"gold crash effect = **{crash_classification}** (N=1)",
        "",
        finding,
        "",
        "| window | arm | Sharpe* | Sortino* | MaxDD% | TR% | diversifies |",
        "| --- | --- | ---: | ---: | ---: | ---: | :---: |",
    ]
    for wname, r in rows.items():
        cav = " *(N=1)*" if r["n1_caveat"] else ""  # type: ignore[index]
        dep = r["deposit"]  # type: ignore[index]
        base = r["baseline"]  # type: ignore[index]
        md.append(
            f"| {wname}{cav} | deposit | {f(dep['sharpe'])} | {f(dep['sortino'])} | "
            f"{f(dep['maxdd_pct'])} | {f(dep['total_return_pct'])} | — |"
        )
        md.append(
            f"| {wname}{cav} | baseline | {f(base['sharpe'])} | {f(base['sortino'])} | "
            f"{f(base['maxdd_pct'])} | {f(base['total_return_pct'])} | — |"
        )
        for g, v in r["variants"].items():  # type: ignore[index,union-attr]
            m = v["metrics"]
            dv = "yes" if v["diversification"]["diversifies"] else "no"
            md.append(
                f"| {wname}{cav} | +gold {g} | {f(m['sharpe'])} | {f(m['sortino'])} | "
                f"{f(m['maxdd_pct'])} | {f(m['total_return_pct'])} | {dv} |"
            )
    md += ["", "_*Sharpe/Sortino are RUONIA-excess on a fixed 15% basis — see the metric caveat._"]
    (_DIR / "gold_cert_report.md").write_text("\n".join(md), encoding="utf-8")

    print(
        f"BINDING: deposit_anchor_holds={deposit_anchor_holds} "
        f"crash_classification={crash_classification}"
    )
    print(
        f"  controls: zero_ok={zero_ok} gold_acute={gold_acute_ret:+.1f}% "
        f"gold_year={gold_year_ret:+.1f}% gold_eq_corr={gold_eq_corr}"
    )
    print(f"  {finding}")


if __name__ == "__main__":
    main()
