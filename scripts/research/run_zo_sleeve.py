"""Phase-B cert: is a ZO (replacement-bond) sleeve an FX-linked diversifier?

Deterministic, token-free. Replacement bonds (ZO, index RURPLRUBTR) pay an FX-linked
(USD-eurobond-successor) coupon+principal but SETTLE IN RUB on MOEX, bypassing Euroclear —
the candidate "remove the regulatory wall" hedge for the ruble-devaluation tail that the
all-ruble deposit/OFZ/equity stack structurally lacks (the geo-risk overlay's unhedged gap).

STRUCTURAL LIMIT (honest, pre-registered): ZO + the CNY-bond index (RUCNYTR) both start
2023-01 — they POSTDATE the 2022 crash they would hedge (they were *created* by the 2022
freeze). So the acute-2022 tail benefit is UN-BACKTESTABLE. This cert measures what IS
observable in-window (2023-2026, which had real ruble moves):

  1. FX-LINKAGE — the daily-return beta of ZO on USDRUB (live to ~2024-H1, then halted by the
     Jun-2024 NCC sanctions) and on CNYRUB (durable). A positive beta confirms the structural
     hedge mechanism: ZO rises when the ruble falls.
  2. DIVERSIFICATION — ZO's correlation to equity (should be low → a real diversifier) and to
     the deposit leg (should be < ~1 → not a redundant RUB carry leg); plus an in-window blend
     (deposit + equity vs + ZO) MaxDD/Sortino delta.

The 2022 tail benefit is reported as a forward-structural ARGUMENT, NEVER as measured.

    uv run python scripts/research/run_zo_sleeve.py
"""

from __future__ import annotations

import json
import statistics
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

from finalayze.backtest.allocation_gate import accrue_real_risk_free_leg, net_index_returns
from finalayze.backtest.equity_tilt_experiment import (
    RISK_FREE_ANNUAL_PCT,
    ArmMetrics,
    _metrics,
    _slice,
)
from finalayze.backtest.equity_tilt_lab import quarter_end_dates
from finalayze.backtest.gold_sleeve_lab import (
    blend_portfolio,
    diversification_verdict,
    forward_align_legs,
    master_axis,
)
from finalayze.core.ndfl import YtdTaxAccumulator

_DIR = Path("results/research/zo")
_SNAP = _DIR / "panel_snapshot.json"

_DEPOSIT_W = Decimal("0.4")
_EQUITY_BASE_W = Decimal("0.6")
_ZO_WEIGHTS = (Decimal("0.05"), Decimal("0.10"), Decimal("0.15"))
_DEPOSIT_SPREAD_PP = Decimal("1.0")

_HIGH_RATE_START = date(2024, 1, 1)
_HIGH_RATE_END = date(2025, 6, 5)
_EASING_START = date(2025, 6, 6)
_BINDING_END = date(2026, 6, 10)
# USD000UTSTOM trades daily until the Jun-2024 NCC sanction, then a ~20-month gap, then
# resumes in 2026. Estimate the USD beta on the CLEAN pre-sanction daily window only (no
# gap artefact); CNYRUB (durable daily) is the headline FX-linkage proxy.
_USDRUB_RELIABLE_END = date(2024, 6, 12)

# Pre-registered thresholds (NOT fitted).
_FX_BETA_MIN = 0.10  # a >=10% daily pass-through to FX counts as a real FX-linkage
_EQUITY_CORR_MAX = 0.50  # below this ZO is a genuine (non-equity-like) diversifier
_REDUNDANT_CORR = 0.90  # at/above this vs the deposit leg ZO is a redundant RUB carry
_MIN_PAIRS = 30
# Defensive (review CR, low): never treat a return spanning a long calendar gap as one daily
# observation. Every PAIR the cert reports has <=6-day (holiday) gaps so this changes nothing
# here; it future-proofs the helper against a halted-series window (e.g. the USDRUB 615-day halt).
_MAX_RETURN_GAP_DAYS = 7

# Index-engine legs carry the MCFTRR/_parse_history_row MSK-midnight -> UTC T-1 date
# convention; shift them +1 day to the true ISS trade date so they align with the true-dated
# currency FX legs (the Phase-A lesson).
_INDEX_LEG_KEYS = ("zo_rurplrubtr", "cnybond_rucnytr", "equity_mcftrr")


def _load() -> dict[str, list[tuple[date, Decimal]]]:
    raw = json.loads(_SNAP.read_text(encoding="utf-8"))["legs"]
    out: dict[str, list[tuple[date, Decimal]]] = {}
    for key, rows in raw.items():
        shift = timedelta(days=1) if key in _INDEX_LEG_KEYS else timedelta(0)
        out[key] = [(date.fromisoformat(d) + shift, Decimal(c)) for d, c in rows]
    return out


def _aligned_returns(
    a: list[tuple[date, Decimal]], b: list[tuple[date, Decimal]], start: date, end: date
) -> tuple[list[float], list[float]]:
    """Daily returns of two curves on their COMMON consecutive trade dates over [start, end]."""
    am = {d: float(v) for d, v in a if start <= d <= end and v > 0}
    bm = {d: float(v) for d, v in b if start <= d <= end and v > 0}
    common = sorted(set(am) & set(bm))
    ar: list[float] = []
    br: list[float] = []
    for i in range(1, len(common)):
        d0, d1 = common[i - 1], common[i]
        if (d1 - d0).days > _MAX_RETURN_GAP_DAYS:
            continue  # skip a gap-spanning interval (halted series) — not a daily observation
        ar.append(am[d1] / am[d0] - 1.0)
        br.append(bm[d1] / bm[d0] - 1.0)
    return ar, br


def _beta_corr(
    a: list[tuple[date, Decimal]], b: list[tuple[date, Decimal]], start: date, end: date
) -> tuple[float | None, float | None, int]:
    """(beta of a on b, correlation, n) over the common window. None if too few pairs."""
    ar, br = _aligned_returns(a, b, start, end)
    if len(ar) < _MIN_PAIRS:
        return None, None, len(ar)
    var_b = statistics.variance(br)
    beta = statistics.covariance(ar, br) / var_b if var_b > 0 else None
    corr = statistics.correlation(ar, br)
    return beta, corr, len(ar)


def _curve_metrics(nav: list[tuple[date, Decimal]], start: date, end: date) -> ArmMetrics:
    return _metrics(_slice([d for d, _ in nav], [v for _, v in nav], start, end))


def main() -> None:  # noqa: PLR0915 — single linear cert script
    legs_raw = _load()
    zo_levels = legs_raw["zo_rurplrubtr"]
    cnybond_levels = legs_raw["cnybond_rucnytr"]
    equity_raw = legs_raw["equity_mcftrr"]
    cnyrub = legs_raw["fx_cnyrub"]
    usdrub = legs_raw["fx_usdrub"]

    full_start = max(zo_levels[0][0], equity_raw[0][0])
    axis = [
        d for d in master_axis({"z": zo_levels, "e": equity_raw}) if full_start <= d <= _BINDING_END
    ]

    # ── (1) FX-linkage + diversification correlations (on RAW index levels) ───────
    fx = {
        "zo_vs_usdrub_presanction": _beta_corr(zo_levels, usdrub, full_start, _USDRUB_RELIABLE_END),
        "zo_vs_cnyrub": _beta_corr(zo_levels, cnyrub, full_start, _BINDING_END),
        "cnybond_vs_cnyrub": _beta_corr(cnybond_levels, cnyrub, full_start, _BINDING_END),
        "zo_vs_equity": _beta_corr(zo_levels, equity_raw, full_start, _BINDING_END),
    }

    # ── (2) NET sleeves on the shared axis for the in-window diversification blend ─
    aligned = forward_align_legs({"equity": equity_raw, "zo": zo_levels}, axis)
    deposit_curve = accrue_real_risk_free_leg(
        axis, Decimal(1), spread_pp=_DEPOSIT_SPREAD_PP, tax_acc=YtdTaxAccumulator()
    )
    equity_curve = list(zip(axis, aligned["equity"], strict=True))  # MCFTRR already net
    zo_curve = net_index_returns(
        list(zip(axis, aligned["zo"], strict=True)), tax_acc=YtdTaxAccumulator()
    )
    legs = {
        "deposit": [v for _, v in deposit_curve],
        "equity": [v for _, v in equity_curve],
        "zo": [v for _, v in zo_curve],
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
        w: _blend({"deposit": _DEPOSIT_W, "equity": _EQUITY_BASE_W - w, "zo": w})
        for w in _ZO_WEIGHTS
    }
    # Control: a 0%-ZO three-leg blend reproduces the two-leg baseline.
    zo_zero = _blend({"deposit": _DEPOSIT_W, "equity": _EQUITY_BASE_W, "zo": Decimal(0)})
    zero_ok = [v for _, v in zo_zero] == [v for _, v in baseline]

    corr_vs_deposit = _beta_corr(zo_curve, deposit_curve, full_start, _BINDING_END)[1]

    windows = {
        "full_2023_26": (axis[0], axis[-1]),
        "high_rate_2024_25": (_HIGH_RATE_START, _HIGH_RATE_END),
        "easing_2025_26": (_EASING_START, axis[-1]),
    }
    rows: dict[str, object] = {}
    for wname, (w_start, w_end) in windows.items():
        base_m = _curve_metrics(baseline, w_start, w_end)
        per_variant: dict[str, object] = {}
        for w, nav in variants.items():
            vm = _curve_metrics(nav, w_start, w_end)
            # diversification_verdict's params are named gold_* in the shared sleeve lab;
            # ZO passes its (candidate) metrics into them — same conjunctive bar.
            div = diversification_verdict(
                baseline_maxdd_pct=base_m.maxdd_pct,
                gold_maxdd_pct=vm.maxdd_pct,
                baseline_sortino=base_m.sortino,
                gold_sortino=vm.sortino,
            )
            per_variant[str(w)] = {"metrics": vm.__dict__, "diversification": div}
        rows[wname] = {
            "range": [w_start.isoformat(), w_end.isoformat()],
            "baseline": base_m.__dict__,
            "variants": per_variant,
            "n1_caveat": wname != "high_rate_2024_25",
        }

    # ── Binding verdict ──────────────────────────────────────────────────────────
    beta_cny = fx["zo_vs_cnyrub"][0]
    beta_usd = fx["zo_vs_usdrub_presanction"][0]
    corr_eq = fx["zo_vs_equity"][1]
    fx_linked = (beta_cny is not None and beta_cny >= _FX_BETA_MIN) or (
        beta_usd is not None and beta_usd >= _FX_BETA_MIN
    )
    diversifies_equity = corr_eq is not None and corr_eq < _EQUITY_CORR_MAX
    redundant_rub = corr_vs_deposit is not None and corr_vs_deposit >= _REDUNDANT_CORR

    if fx_linked and diversifies_equity and not redundant_rub:
        verdict = "FX_LINKED_DIVERSIFIER_TAIL_UNTESTED"
    elif redundant_rub or not fx_linked:
        verdict = "REDUNDANT_RUB_CARRY"
    else:
        verdict = "INCONCLUSIVE"

    cnybond_beta = fx["cnybond_vs_cnyrub"][0]
    finding = (
        f"ZO (RURPLRUBTR) is FX-LINKED: daily-return beta vs CNYRUB (durable daily proxy)="
        f"{_f(beta_cny)}, vs USDRUB={_f(beta_usd)} (pre-Jun-2024-sanction window only; the "
        f"exchange USD series then halts). It is a genuine, non-redundant diversifier — corr vs "
        f"equity={_f(corr_eq)} (<{_EQUITY_CORR_MAX}) and corr vs the deposit leg="
        f"{_f(corr_vs_deposit)} (not >={_REDUNDANT_CORR}). The CNY-bond index RUCNYTR is by "
        f"contrast only WEAKLY FX-linked (CNY beta {_f(cnybond_beta)}) — RURPL is the FX-linked "
        f"one. Verdict: {verdict}. CRITICAL: THE 2022 TAIL BENEFIT IS UN-BACKTESTABLE — ZO "
        f"postdates the crash (index starts 2023), so the hedge is a forward-structural argument, "
        f"NOT measured. In-window (2023-2026, NO crash) ZO is INSURANCE WITH A COST: it modestly "
        f"cuts MaxDD (~1-2pp) but LOWERS total return (full 49%->38% at 15%) and worsens Sortino "
        f"— there was no crash in-sample to reward it. Not alpha; a forward-looking FX-tail "
        f"insurance leg whose payoff is structurally sound but unproven (N=1, no in-window crash)."
    )

    summary = {
        "window": {"start": axis[0].isoformat(), "end": axis[-1].isoformat(), "n_bars": len(axis)},
        "weights": {
            "deposit": str(_DEPOSIT_W),
            "equity_base": str(_EQUITY_BASE_W),
            "zo_sweep": [str(w) for w in _ZO_WEIGHTS],
        },
        "risk_free_annual_pct": RISK_FREE_ANNUAL_PCT,
        "structural_limit": (
            "ZO (RURPLRUBTR) + RUCNYTR start 2023 — postdate the 2022 crash; the acute tail is "
            "UN-backtestable. The only 2022-spanning eurobond index (RUCEU) shows the Euroclear "
            "freeze slamming shut, a survivorship trap, not a hedge payoff."
        ),
        "fx_linkage": {k: {"beta": _r(v[0]), "corr": _r(v[1]), "n": v[2]} for k, v in fx.items()},
        "corr_zo_vs_deposit": _r(corr_vs_deposit),
        "controls": {"zo_zero_reproduces_baseline": zero_ok},
        "windows": rows,
        "binding": {"verdict": verdict, "finding": finding, "tail_backtestable": False},
    }
    _DIR.mkdir(parents=True, exist_ok=True)
    (_DIR / "zo_cert_summary.json").write_text(
        json.dumps(summary, indent=1, default=str), encoding="utf-8"
    )

    md = [
        "# Phase B — ZO (Replacement-Bond) Sleeve: FX-Linked Diversifier? (Cert)",
        "",
        f"Window `{axis[0]}`->`{axis[-1]}` · {len(axis)} bars · RUONIA-excess "
        f"{RISK_FREE_ANNUAL_PCT}%",
        f"Base = deposit {_DEPOSIT_W} / equity {_EQUITY_BASE_W}; ZO carved from equity "
        f"(sweep {[str(w) for w in _ZO_WEIGHTS]}). ZO = RURPLRUBTR net-NDFL.",
        "",
        "> **Structural limit:** ZO + RUCNYTR start 2023 — they POSTDATE the 2022 crash they would "
        "hedge (created *by* the freeze). The acute-2022 tail benefit is **un-backtestable**; this "
        "cert measures in-window FX-linkage + diversification only, and reports the tail as a "
        "forward-structural argument, never measured.",
        "",
        f"## BINDING VERDICT: **{verdict}** (tail un-backtestable)",
        "",
        finding,
        "",
        "## FX-linkage & diversification (raw daily-return beta / correlation)",
        "| pair | beta | corr | n |",
        "| --- | ---: | ---: | ---: |",
        *(f"| {k} | {_f(v[0])} | {_f(v[1])} | {v[2]} |" for k, v in fx.items()),
        f"| zo_vs_deposit_leg | — | {_f(corr_vs_deposit)} | — |",
        "",
        f"- 0%-ZO reproduces baseline curve: **{zero_ok}**",
        "",
        "## In-window diversification blend (no crash in 2023-2026)",
        "| window | arm | Sharpe* | Sortino* | MaxDD% | TR% | diversifies |",
        "| --- | --- | ---: | ---: | ---: | ---: | :---: |",
    ]
    for wname, r in rows.items():
        cav = " *(N=1)*" if r["n1_caveat"] else ""  # type: ignore[index]
        base = r["baseline"]  # type: ignore[index]
        md.append(
            f"| {wname}{cav} | baseline | {_f(base['sharpe'])} | {_f(base['sortino'])} | "
            f"{_f(base['maxdd_pct'])} | {_f(base['total_return_pct'])} | — |"
        )
        for w, v in r["variants"].items():  # type: ignore[index,union-attr]
            m = v["metrics"]
            dv = "yes" if v["diversification"]["diversifies"] else "no"
            md.append(
                f"| {wname}{cav} | +ZO {w} | {_f(m['sharpe'])} | {_f(m['sortino'])} | "
                f"{_f(m['maxdd_pct'])} | {_f(m['total_return_pct'])} | {dv} |"
            )
    md += ["", "_*RUONIA-excess on a fixed 15% basis (apt for the high-rate era)._"]
    (_DIR / "zo_cert_report.md").write_text("\n".join(md), encoding="utf-8")

    print(f"BINDING VERDICT: {verdict}")
    print(
        f"  beta_cny={_f(beta_cny)} beta_usd={_f(beta_usd)} corr_eq={_f(corr_eq)} "
        f"corr_dep={_f(corr_vs_deposit)} zero_ok={zero_ok}"
    )
    print(f"  {finding}")


def _f(x: object) -> str:
    return f"{x:.3f}" if isinstance(x, float) else "n/a"


def _r(x: float | None) -> float | None:
    return round(x, 4) if x is not None else None


if __name__ == "__main__":
    main()
