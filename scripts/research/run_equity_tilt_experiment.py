"""Run the active-equity-sleeve experiment cert from the committed panel snapshot.

Deterministic, token-free: loads the committed candle snapshot
(``fetch_equity_tilt_panel.py``) + the committed dividend schedule, runs every
low-turnover tilt vs the cap-proxy baseline through the same net-of-cost,
net-of-NDFL basket simulator, and writes a JSON + Markdown cert.

    uv run python scripts/research/run_equity_tilt_experiment.py
"""

from __future__ import annotations

import json
from datetime import date
from decimal import Decimal
from pathlib import Path

from finalayze.backtest.dividend_schedule import load_dividend_schedule
from finalayze.backtest.equity_tilt_experiment import run_experiment
from finalayze.backtest.equity_tilt_lab import PricePoint

_SNAP = Path("results/research/equity_tilt/panel_snapshot.json")
_OUT_JSON = Path("results/research/equity_tilt/cert_summary.json")
_OUT_MD = Path("results/research/equity_tilt/cert_report.md")


def _load_panel(path: Path) -> dict[str, list[PricePoint]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    panel: dict[str, list[PricePoint]] = {}
    for sym, rows in raw["panel"].items():
        panel[sym] = [(date.fromisoformat(d), Decimal(c), Decimal(v)) for d, c, v in rows]
    return panel


def _fmt(x: object) -> str:
    if isinstance(x, float):
        return f"{x:.3f}"
    return str(x)


def _render_md(out: dict[str, object]) -> str:
    w = out["window"]  # type: ignore[index]
    b = out["binding"]  # type: ignore[index]
    lines = [
        "# Active-Equity-Sleeve Experiment — Cert",
        "",
        "**Question:** does routing some of the equity sleeve into a low-turnover "
        "ACTIVE weighting beat just holding the cap-weight index, net of the real "
        "retail 1.10% round-trip cost and net-of-NDFL dividends?",
        "",
        f"- Window: `{w['start']}` → `{w['end']}` ({w['n_bars']} bars, "  # type: ignore[index]
        f"{w['n_rebalances']} quarterly rebalances, {w['universe_size']} names)",  # type: ignore[index]
        f"- Baseline: `{out['baseline']}` (ADV cap-weight proxy, same engine/cost/tax)",  # type: ignore[index]
        f"- Risk-free (RUONIA-excess): {out['risk_free_annual_pct']}%",  # type: ignore[index]
        "",
        f"## BINDING VERDICT: **{b['verdict']}**  (N=1 caveat: {b['n1_caveat']})",  # type: ignore[index]
        "",
        f"{b['finding']}",  # type: ignore[index]
        "",
        "## Arms (full window)",
        "",
        "| arm | Sharpe | Sortino | MaxDD% | TotalRet% | cost drag% | beats base? |",
        "| --- | ---: | ---: | ---: | ---: | ---: | :---: |",
    ]
    arms = out["arms"]  # type: ignore[index]
    for name, data in arms.items():  # type: ignore[union-attr]
        fw = data["windows"]["full_window"]  # type: ignore[index]
        m = fw["metrics"]  # type: ignore[index]
        v = fw["verdict"]  # type: ignore[index]
        verdict = "— (baseline)" if v is None else ("✅ PASS" if v["passed"] else "❌")
        lines.append(
            f"| {name} | {_fmt(m['sharpe'])} | {_fmt(m['sortino'])} | "
            f"{_fmt(m['maxdd_pct'])} | {_fmt(m['total_return_pct'])} | "
            f"{_fmt(round(float(data['cost_drag_pct_of_initial']), 2))} | {verdict} |"  # type: ignore[index]
        )
    lines += ["", "## Per-regime (tilt vs baseline)", ""]
    for name, data in arms.items():  # type: ignore[union-attr]
        if name == out["baseline"]:  # type: ignore[index]
            continue
        lines.append(f"### {name}")
        for region in ("full_window", "high_rate", "early_cut"):
            win = data["windows"].get(region)  # type: ignore[union-attr]
            if not win:
                continue
            m = win["metrics"]  # type: ignore[index]
            v = win["verdict"]  # type: ignore[index]
            tag = "PASS" if (v and v["passed"]) else "FAIL"
            caveat = " *(N=1 caveat)*" if win.get("n1_caveat") else ""  # type: ignore[union-attr]
            lines.append(
                f"- **{region}**{caveat}: Sharpe {_fmt(m['sharpe'])} / "
                f"Sortino {_fmt(m['sortino'])} / MaxDD {_fmt(m['maxdd_pct'])}% → **{tag}**"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    panel = _load_panel(_SNAP)
    sched = load_dividend_schedule()
    out = run_experiment(panel, sched)

    _OUT_JSON.write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    _OUT_MD.write_text(_render_md(out), encoding="utf-8")

    b = out["binding"]  # type: ignore[index]
    print(f"BINDING VERDICT: {b['verdict']}")
    print(f"  {b['finding']}")
    print(f"  wrote {_OUT_JSON} and {_OUT_MD}")


if __name__ == "__main__":
    main()
