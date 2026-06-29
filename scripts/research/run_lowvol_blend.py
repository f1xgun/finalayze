"""Step 3 cert: low-vol BLEND vs the REAL IMOEX cap-weight baseline.

Deterministic, token-free. Loads the committed candle panel + IMOEX index-weight
snapshot + IMOEX index level, runs the cap-weight baseline and the low-vol blend
(lambda=0.25) through the SAME net-of-cost/net-of-NDFL basket simulator, and judges
the tilt on the strict Sharpe/Sortino/MaxDD bar per regime. Plus three honesty
controls: (1) lambda=0 reproduces the cap-weight curve byte-for-byte;
(2) per-date index-weight COVERAGE; (3) the cap-weight basket's daily-return
CORRELATION to the real IMOEX index.

    uv run python scripts/research/run_lowvol_blend.py
"""

from __future__ import annotations

import json
import statistics
from datetime import date
from decimal import Decimal
from pathlib import Path

from finalayze.backtest.allocation_gate import regime_split
from finalayze.backtest.costs import MOEX_RETAIL_COSTS
from finalayze.backtest.dividend_schedule import load_dividend_schedule
from finalayze.backtest.equity_tilt_experiment import (
    RISK_FREE_ANNUAL_PCT,
    _metrics,
    _slice,
    _verdict,
)
from finalayze.backtest.equity_tilt_lab import (
    PricePoint,
    _trailing_vols,
    make_index_cap_weight_policy,
    make_low_vol_blend_policy,
    quarter_end_dates,
    simulate_basket,
)

_DIR = Path("results/research/lowvol")
_PANEL = _DIR / "candle_panel.json"
_WEIGHTS = _DIR / "index_weights_snapshot.json"
_IMOEX = _DIR / "imoex_index.json"
_LAMBDA = Decimal("0.25")
_MIN_NAMES = 2  # need at least two names to split a half / correlate
_MIN_CORR_PAIRS = 30


def _load_panel() -> dict[str, list[PricePoint]]:
    raw = json.loads(_PANEL.read_text(encoding="utf-8"))["panel"]
    return {
        s: [(date.fromisoformat(d), Decimal(c), Decimal(v)) for d, c, v in rows]
        for s, rows in raw.items()
    }


def _load_weights() -> dict[date, dict[str, Decimal]]:
    raw = json.loads(_WEIGHTS.read_text(encoding="utf-8"))["weights"]
    return {
        date.fromisoformat(d): {t: Decimal(str(w)) for t, w in per.items()}
        for d, per in raw.items()
    }


def _coverage(
    weights: dict[date, dict[str, Decimal]], panel: dict[str, list[PricePoint]]
) -> list[float]:
    first = {s: min(d for d, _, _ in rows) for s, rows in panel.items()}
    out = []
    for wd, per in sorted(weights.items()):
        covered = sum(float(w) for t, w in per.items() if t in panel and first[t] <= wd)
        out.append(covered)
    return out


def _cosmetic_overlap(
    weights: dict[date, dict[str, Decimal]],
    panel: dict[str, list[PricePoint]],
    rebal: list[date],
) -> float:
    """Avg overlap between the low-vol half and the cap-weight top half (>0.8 = cosmetic)."""
    cap_pol = make_index_cap_weight_policy(weights)
    overlaps = []
    for d in rebal:
        cap = cap_pol(d, panel)
        if len(cap) < _MIN_NAMES:
            continue
        vols = _trailing_vols(d, {s: panel[s] for s in cap if s in panel}, 126)
        if len(vols) < _MIN_NAMES:
            continue
        low_vol_half = set(sorted(vols, key=lambda s: vols[s])[: len(vols) // 2])
        cap_top_half = set(sorted(cap, key=lambda s: cap[s], reverse=True)[: len(cap) // 2])
        if low_vol_half:
            overlaps.append(len(low_vol_half & cap_top_half) / len(low_vol_half))
    return statistics.mean(overlaps) if overlaps else 0.0


def _imoex_corr(cap_dates: list[date], cap_curve: list[Decimal]) -> float | None:
    raw = json.loads(_IMOEX.read_text(encoding="utf-8"))["close"]
    idx = {date.fromisoformat(d): float(c) for d, c in raw.items()}
    pairs = [(d, float(v)) for d, v in zip(cap_dates, cap_curve, strict=True) if d in idx]
    if len(pairs) < _MIN_CORR_PAIRS:
        return None
    cap_r, idx_r = [], []
    for i in range(1, len(pairs)):
        d0, c0 = pairs[i - 1]
        d1, c1 = pairs[i]
        if c0 > 0:
            cap_r.append(c1 / c0 - 1.0)
            idx_r.append(idx[d1] / idx[d0] - 1.0)
    return statistics.correlation(cap_r, idx_r) if len(cap_r) > _MIN_NAMES else None


def main() -> None:
    panel = _load_panel()
    weights = _load_weights()
    sched = load_dividend_schedule()
    all_dates = sorted({d for pts in panel.values() for d, _, _ in pts})
    rebal = sorted({all_dates[0], *quarter_end_dates(all_dates)})

    cap_pol = make_index_cap_weight_policy(weights)
    blend_pol = make_low_vol_blend_policy(weights, lam=_LAMBDA)
    blend0_pol = make_low_vol_blend_policy(weights, lam=Decimal(0))

    def run(pol):
        return simulate_basket(
            panel=panel,
            dividend_schedule=sched,
            weight_policy=pol,
            rebalance_dates=rebal,
            costs=MOEX_RETAIL_COSTS,
        )

    cap = run(cap_pol)
    blend = run(blend_pol)
    blend0 = run(blend0_pol)

    # (1) data-correctness control: lambda=0 reproduces the cap-weight curve
    lam0_ok = blend0.nav_curve == cap.nav_curve

    regions = regime_split(all_dates)
    windows = {"full_window": (all_dates[0], all_dates[-1]), **regions}

    rows = {}
    for name, region in windows.items():
        start, end = region
        cap_m = _metrics(_slice(cap.dates, cap.nav_curve, start, end))
        blend_m = _metrics(_slice(blend.dates, blend.nav_curve, start, end))
        rows[name] = {
            "cap": cap_m.__dict__,
            "blend": blend_m.__dict__,
            "verdict": _verdict(blend_m, cap_m),
            "n1_caveat": name not in ("full_window", "high_rate"),
        }

    cov = _coverage(weights, panel)
    overlap = _cosmetic_overlap(weights, panel, rebal)
    corr = _imoex_corr(cap.dates, cap.nav_curve)

    full = rows["full_window"]["verdict"]
    hr = rows.get("high_rate", {}).get("verdict", {})
    passed = bool(full["passed"] and hr.get("passed"))

    summary = {
        "window": {
            "start": all_dates[0].isoformat(),
            "end": all_dates[-1].isoformat(),
            "n_bars": len(all_dates),
            "n_rebalances": len(rebal),
            "universe": len(panel),
        },
        "lambda": str(_LAMBDA),
        "risk_free_annual_pct": RISK_FREE_ANNUAL_PCT,
        "controls": {
            "lambda0_reproduces_cap": lam0_ok,
            "index_weight_coverage_avg": round(statistics.mean(cov), 4),
            "index_weight_coverage_min": round(min(cov), 4),
            "cap_basket_vs_imoex_return_corr": round(corr, 4) if corr is not None else None,
            "lowvol_vs_capTop_overlap_avg": round(overlap, 4),
        },
        "windows": rows,
        "binding": {
            "verdict": "PASS" if passed else "HARD_FAIL",
            "finding": (
                "low-vol blend beats the REAL cap-weight baseline (full+high_rate)"
                if passed
                else "low-vol blend does NOT beat the real IMOEX cap-weight baseline "
                "(full_window+high_rate) net of cost/tax"
            ),
            "n1_caveat": True,
        },
    }
    (_DIR / "lowvol_cert_summary.json").write_text(
        json.dumps(summary, indent=1, default=str), encoding="utf-8"
    )

    def f(x):
        return f"{x:.3f}" if isinstance(x, float) else str(x)

    md = [
        "# Step 3 — Low-Vol Blend vs REAL IMOEX Cap-Weight (Cert)",
        "",
        f"Window `{all_dates[0]}`->`{all_dates[-1]}` · {len(all_dates)} bars · {len(rebal)} rebalances · {len(panel)} names · RUONIA-excess {RISK_FREE_ANNUAL_PCT}%",  # noqa: E501
        f"Tilt: FINAL = (1-{_LAMBDA})·cap_weight + {_LAMBDA}·inverse_vol(lowest-vol half).",
        "",
        "## Honesty controls",
        f"- lambda=0 reproduces cap-weight curve: **{lam0_ok}**",
        f"- index-weight coverage: avg **{statistics.mean(cov) * 100:.1f}%**, min **{min(cov) * 100:.1f}%**",  # noqa: E501
        f"- cap-basket vs IMOEX return corr: **{f(corr) if corr is not None else 'n/a'}**",
        f"- low-vol-half vs cap-top-half overlap: **{overlap * 100:.1f}%** (>80% ⇒ cosmetic)",
        "",
        f"## BINDING VERDICT: **{summary['binding']['verdict']}** (N=1 caveat)",
        "",
        summary["binding"]["finding"],
        "",
        "| window | armSharpe | armSortino | armMaxDD% | capSharpe | capSortino | capMaxDD% | verdict |",  # noqa: E501
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | :---: |",
    ]
    for name, r in rows.items():
        cm, bm, v = r["cap"], r["blend"], r["verdict"]
        tag = "PASS" if v["passed"] else "FAIL"
        cav = " *(N=1)*" if r["n1_caveat"] else ""
        md.append(
            f"| {name}{cav} | {f(bm['sharpe'])} | {f(bm['sortino'])} | {f(bm['maxdd_pct'])} | "
            f"{f(cm['sharpe'])} | {f(cm['sortino'])} | {f(cm['maxdd_pct'])} | {tag} |"
        )
    (_DIR / "lowvol_cert_report.md").write_text("\n".join(md), encoding="utf-8")
    print(f"BINDING VERDICT: {summary['binding']['verdict']}")
    print(
        f"  lambda0_ok={lam0_ok} coverage_avg={statistics.mean(cov):.3f} "
        f"imoex_corr={corr} cosmetic_overlap={overlap:.3f}"
    )
    print(f"  {summary['binding']['finding']}")


if __name__ == "__main__":
    main()
