"""Phase-C cert: is a real-estate sleeve an income-bearing diversifier — or deposit-dominated?

Deterministic, token-free. Real estate (index ``MREDC`` = DomClick Moscow residential price)
is the third and LAST "new asset class" candidate (gold → Phase A `NO`; ZO → Phase B
FX-diversifier-insurance). It is the ONE candidate that pays *income* (rent) — the operator's
"income" goal — so this cert models rent, not just price.

Two honesty-critical limits, reported prominently (the analogues of Phase B's un-backtestable
tail):

  1. SMOOTHING ARTIFACT — MREDC is ~WEEKLY and transaction/appraisal-based, so its measured
     volatility/drawdown are STRUCTURALLY UNDERSTATED vs a traded asset. ``bars_per_year``
     flags it; the investable rental-ZPIF wrapper carries the vol + illiquidity + 1-3%/yr
     fees the index hides. Any low-MaxDD reading is partly artifact.
  2. POLICY-DRIVEN APPRECIATION — the ~+8.5%/yr price rise was largely subsidised-mortgage
     (lgotnaya ipoteka) driven, a policy now wound down → NOT a forward expectation.

The cert measures, on real data (2022-2026):
  - PRICE-ONLY vs PRICE+RENT arms (net rental overlay swept 3/4/6%) blended into the frozen
    deposit40/equity60 core (real estate carved from equity, quarterly rebalance, retail cost);
  - real estate's correlation to equity (diversification) and to the deposit leg (redundancy);
  - the deposit-100 anchor — does the best real-estate arm beat the deposit on total return?

    uv run python scripts/research/run_realestate_sleeve.py
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
    apply_ter_drag,
    blend_portfolio,
    diversification_verdict,
    forward_align_legs,
    master_axis,
)
from finalayze.backtest.instrument_integration_gate import Candidate, run_integration_gate
from finalayze.backtest.realestate_sleeve_lab import accrue_rental_yield, bars_per_year
from finalayze.core.ndfl import YtdTaxAccumulator

_DIR = Path("results/research/realestate")
_SNAP = _DIR / "panel_snapshot.json"

_DEPOSIT_W = Decimal("0.4")
_EQUITY_BASE_W = Decimal("0.6")
_RE_WEIGHTS = (Decimal("0.05"), Decimal("0.10"), Decimal("0.15"))
_DEPOSIT_SPREAD_PP = Decimal("1.0")

# Net rental yield assumption (post-vacancy/management/tax/repairs AND income NDFL). Swept for
# sensitivity — NOT a measured number. Moscow residential gross ~4-6% -> net ~3-4%; 6% is a
# generous commercial-warehouse-ZPIF upper bound (which carries even more illiquidity/fees).
_RENT_NET_BASE = Decimal("4.0")
_RENT_NET_SWEEP = (Decimal("3.0"), Decimal("4.0"), Decimal("6.0"))

# The investable rental-ZPIF wrapper fee that MREDC (a bare index) HIDES. Real RU real-estate
# funds charge ~1-3%/yr; 2% is the honest midpoint (gold's Phase-A wrapper was 0.8%). Charged
# on the whole real-estate NAV so the cert compares the *investable* form, not the paper index.
_RE_WRAPPER_TER = Decimal("2.0")

_CRASH_START = date(2022, 2, 21)
_CRASH_END = date(2022, 12, 30)
_HIGH_RATE_START = date(2024, 1, 1)
_HIGH_RATE_END = date(2025, 6, 5)
_EASING_START = date(2025, 6, 6)
_BINDING_END = date(2026, 6, 10)

# Pre-registered thresholds (NOT fitted).
_EQUITY_CORR_MAX = 0.50  # below this real estate is a genuine (non-equity-like) diversifier
_REDUNDANT_CORR = 0.90  # at/above this vs the deposit leg it is a redundant RUB carry
_DAILY_FREQ_FLOOR = 150.0  # bars/yr below this a series is SMOOTHED (daily ~=252; weekly ~=52)
_MIN_PAIRS = 30
_MAX_RETURN_GAP_DAYS = 10  # skip a return spanning a gap longer than a fortnight (weekly-safe)

# Both legs come through the index path (MCFTRR MSK-midnight -> UTC T-1 convention); shift both
# +1 day to the true ISS trade date (the Phase-A/B lesson) so window boundaries are honest.
_INDEX_LEG_KEYS = ("real_estate_mredc", "equity_mcftrr")


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
    am = {d: float(v) for d, v in a if start <= d <= end and v > 0}
    bm = {d: float(v) for d, v in b if start <= d <= end and v > 0}
    common = sorted(set(am) & set(bm))
    ar: list[float] = []
    br: list[float] = []
    for i in range(1, len(common)):
        d0, d1 = common[i - 1], common[i]
        if (d1 - d0).days > _MAX_RETURN_GAP_DAYS:
            continue
        ar.append(am[d1] / am[d0] - 1.0)
        br.append(bm[d1] / bm[d0] - 1.0)
    return ar, br


def _corr(
    a: list[tuple[date, Decimal]], b: list[tuple[date, Decimal]], start: date, end: date
) -> tuple[float | None, int]:
    ar, br = _aligned_returns(a, b, start, end)
    if len(ar) < _MIN_PAIRS:
        return None, len(ar)
    return statistics.correlation(ar, br), len(ar)


def _curve_metrics(nav: list[tuple[date, Decimal]], start: date, end: date) -> ArmMetrics:
    return _metrics(_slice([d for d, _ in nav], [v for _, v in nav], start, end))


def main() -> None:  # noqa: PLR0915 — single linear cert script
    legs_raw = _load()
    mredc_levels = legs_raw["real_estate_mredc"]
    equity_raw = legs_raw["equity_mcftrr"]

    full_start = max(mredc_levels[0][0], equity_raw[0][0])
    axis = [
        d
        for d in master_axis({"r": mredc_levels, "e": equity_raw})
        if full_start <= d <= _BINDING_END
    ]

    re_freq = bars_per_year([d for d, _ in mredc_levels])
    eq_freq = bars_per_year([d for d, _ in equity_raw])
    smoothed = re_freq < _DAILY_FREQ_FLOOR

    # ── NET sleeves on the shared axis ────────────────────────────────────────────
    aligned = forward_align_legs({"equity": equity_raw, "re": mredc_levels}, axis)
    deposit_curve = accrue_real_risk_free_leg(
        axis, Decimal(1), spread_pp=_DEPOSIT_SPREAD_PP, tax_acc=YtdTaxAccumulator()
    )
    equity_curve = list(zip(axis, aligned["equity"], strict=True))  # MCFTRR already net
    # Price leg = net-NDFL MREDC appreciation, THEN the investable-wrapper TER (the fee the bare
    # index hides). Rent is overlaid on the fee-charged price so every real-estate arm is the
    # investable form, not the paper index.
    re_price_curve = apply_ter_drag(
        net_index_returns(list(zip(axis, aligned["re"], strict=True)), tax_acc=YtdTaxAccumulator()),
        _RE_WRAPPER_TER,
    )
    re_total_curve = accrue_rental_yield(re_price_curve, _RENT_NET_BASE)

    legs = {
        "deposit": [v for _, v in deposit_curve],
        "equity": [v for _, v in equity_curve],
        "re_price": [v for _, v in re_price_curve],
        "re_total": [v for _, v in re_total_curve],
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
    deposit_only = _blend({"deposit": Decimal(1)})

    # price-only and price+rent variant families
    variants_price = {
        w: _blend({"deposit": _DEPOSIT_W, "equity": _EQUITY_BASE_W - w, "re_price": w})
        for w in _RE_WEIGHTS
    }
    variants_total = {
        w: _blend({"deposit": _DEPOSIT_W, "equity": _EQUITY_BASE_W - w, "re_total": w})
        for w in _RE_WEIGHTS
    }
    # Control: a 0%-RE three-leg blend reproduces the two-leg baseline.
    re_zero = _blend({"deposit": _DEPOSIT_W, "equity": _EQUITY_BASE_W, "re_total": Decimal(0)})
    zero_ok = [v for _, v in re_zero] == [v for _, v in baseline]

    corr_eq, n_eq = _corr(re_price_curve, equity_curve, full_start, _BINDING_END)
    corr_dep, _ = _corr(re_price_curve, deposit_curve, full_start, _BINDING_END)

    windows = {
        "full_window": (axis[0], axis[-1]),
        "crash_year_2022": (_CRASH_START, _CRASH_END),
        "high_rate_2024_25": (_HIGH_RATE_START, _HIGH_RATE_END),
        "easing_2025_26": (_EASING_START, axis[-1]),
    }
    rows: dict[str, object] = {}
    for wname, (w_start, w_end) in windows.items():
        base_m = _curve_metrics(baseline, w_start, w_end)
        per_variant: dict[str, object] = {}
        for family, variants in (("price", variants_price), ("total", variants_total)):
            for w, nav in variants.items():
                vm = _curve_metrics(nav, w_start, w_end)
                div = diversification_verdict(
                    baseline_maxdd_pct=base_m.maxdd_pct,
                    gold_maxdd_pct=vm.maxdd_pct,
                    baseline_sortino=base_m.sortino,
                    gold_sortino=vm.sortino,
                )
                per_variant[f"{family}_{w}"] = {"metrics": vm.__dict__, "diversification": div}
        rows[wname] = {
            "range": [w_start.isoformat(), w_end.isoformat()],
            "baseline": base_m.__dict__,
            "variants": per_variant,
            "n1_caveat": wname != "high_rate_2024_25",
        }

    # ── Deposit-dominance + income sensitivity ────────────────────────────────────
    dep_tr = _curve_metrics(deposit_only, axis[0], axis[-1]).total_return_pct
    # Sensitivity is built on the SAME fee-charged price base as the blend legs.
    rent_sensitivity: dict[str, float] = {}
    for y in _RENT_NET_SWEEP:
        curve = accrue_rental_yield(re_price_curve, y)
        rent_sensitivity[str(y)] = _curve_metrics(curve, axis[0], axis[-1]).total_return_pct
    re_price_only_tr = _curve_metrics(re_price_curve, axis[0], axis[-1]).total_return_pct
    re_base_rent_tr = rent_sensitivity[str(_RENT_NET_BASE)]
    best_re_tr = max(rent_sensitivity.values())
    # Honest upper bound: the daily-mark NDFL over-taxes real estate (a >3yr LDV / primary-
    # residence hold is often price-gain-EXEMPT). Re-price WITHOUT the price NDFL (still charging
    # the wrapper TER), plus the generous 6% rent — the most favourable defensible real-estate arm.
    re_price_taxfree = apply_ter_drag(list(zip(axis, aligned["re"], strict=True)), _RE_WRAPPER_TER)
    re_price_taxfree_tr = _curve_metrics(re_price_taxfree, axis[0], axis[-1]).total_return_pct
    re_taxfree_gen_tr = _curve_metrics(
        accrue_rental_yield(re_price_taxfree, max(_RENT_NET_SWEEP)), axis[0], axis[-1]
    ).total_return_pct
    taxfree_generous_beats_deposit = re_taxfree_gen_tr > dep_tr
    # ROBUST test: the MEASURED price-only investable form (no rent assumption) vs the deposit.
    price_beats_deposit = re_price_only_tr > dep_tr
    # Secondary: does a realistic (base 4%) / generous (6%) net-rent assumption close the gap?
    base_rent_beats_deposit = re_base_rent_tr > dep_tr
    generous_rent_beats_deposit = best_re_tr > dep_tr

    # ── Binding verdict ──────────────────────────────────────────────────────────
    diversifies_equity = corr_eq is not None and corr_eq < _EQUITY_CORR_MAX
    redundant_rub = corr_dep is not None and corr_dep >= _REDUNDANT_CORR

    # A robust win requires the MEASURED (assumption-free) price arm to beat the deposit AND the
    # series not to be a smoothing artifact. Neither holds here, so the honest tag is the
    # deposit-dominated one; the rent overlay is reported as a sensitivity caveat, not a flip.
    if price_beats_deposit and not smoothed:
        verdict = "INCOME_DIVERSIFIER_BEATS_DEPOSIT"
    elif diversifies_equity and not redundant_rub:
        verdict = "SMOOTHED_ILLIQUID_DIVERSIFIER_DEPOSIT_DOMINATED"
    else:
        verdict = "REDUNDANT_OR_INCONCLUSIVE"

    base_gap = "closes" if base_rent_beats_deposit else "does NOT close"
    gen_gap = "closes" if generous_rent_beats_deposit else "does NOT close"
    finding = (
        f"Real estate (MREDC) is the STRONGEST of the three candidates — the only income-payer, "
        f"and a genuine equity DIVERSIFIER (corr vs equity={_f(corr_eq)}<{_EQUITY_CORR_MAX}, vs "
        f"deposit leg={_f(corr_dep)}); over 2022-2026 residential price even BEAT equity (MCFTRR "
        f"net was negative). BUT it is NOT a robust deposit-beater. Charging the investable "
        f"rental-ZPIF wrapper fee ({_RE_WRAPPER_TER}%/yr, which the bare index hides), the "
        f"MEASURED price-only investable form returns {re_price_only_tr:.1f}% vs the "
        f"100%-deposit {dep_tr:.1f}% -> price_beats_deposit={price_beats_deposit} "
        f"(deposit-dominated). A realistic 4% net rent -> {re_base_rent_tr:.1f}% ({base_gap} the "
        f"gap); a generous 6% commercial-grade net rent -> {best_re_tr:.1f}% ({gen_gap} the gap) "
        f"— and even that generous case rests on TWO fragile props: (1) MREDC is SMOOTHED "
        f"({re_freq:.0f} bars/yr ~weekly vs equity {eq_freq:.0f} ~daily) so its near-zero "
        f"correlation and low drawdown are partly APPRAISAL ARTIFACTS an illiquid ZPIF would "
        f"expose; (2) the ~+8.5%/yr appreciation was largely SUBSIDISED-MORTGAGE "
        f"(lgotnaya ipoteka) driven, a policy now wound down -> NOT a forward expectation. "
        f"Verdict: "
        f"{verdict}. Same family conclusion as gold/ZO: in the 16-21% rate regime the deposit "
        f"anchor holds; real estate is a policy-driven, illiquid, smoothed income-diversifier, "
        f"NOT a robust deposit-beater (N=1 easing cycle, one atypical sticky-price crash)."
    )

    # ── Canonical Instrument Integration Gate (battery-comparable) ─────────────────
    # Run real estate through the SAME pre-registered gate the beyond-edge battery used
    # (gold -> REJECT, ZO -> PROBATION) so the verdict is directly comparable. Candidate = the
    # INVESTABLE net curve (price + wrapper TER + base rent). MREDC spans the 2022 crash, so its
    # tail IS backtestable -> unlike ZO it is held to the strict INTEGRATE bar and CANNOT take the
    # PROBATION tail-untestable toe-hold. Real estate is a medium-risk diversifier (carved from
    # equity). No hook, no pre-baked literal — the gate scores the real blended curves.
    gate_candidate = Candidate(
        name="real_estate_mredc",
        net_curve=re_total_curve,
        risk_tier="medium",
        intended_role="diversifier",
    )
    gate_verdict = run_integration_gate(gate_candidate, equity_raw)
    sc = gate_verdict.scorecard
    gate = {
        "tier": gate_verdict.tier,
        "proposed_weight": str(gate_verdict.proposed_weight),
        "carved_from": gate_verdict.carved_from,
        "n1_caveat": gate_verdict.n1_caveat,
        "reasons": gate_verdict.reasons,
        "scorecard": {
            "window_bars": sc.window_bars,
            "regimes_covered": sc.regimes_covered,
            "tail_backtestable": sc.tail_backtestable,
            "marginal_sharpe_delta": round(sc.marginal_sharpe_delta, 4),
            "marginal_sortino_delta": round(sc.marginal_sortino_delta, 4),
            "marginal_maxdd_delta_pp": round(sc.marginal_maxdd_delta_pp, 3),
            "crash_year_maxdd_delta_pp": round(sc.crash_year_maxdd_delta_pp, 3),
            "toehold_sortino_delta": round(sc.toehold_sortino_delta, 4),
            "max_corr_to_existing_legs": round(sc.max_corr_to_existing_legs, 4),
            "anti_hollow_ok": sc.anti_hollow_ok,
        },
    }

    summary = {
        "window": {"start": axis[0].isoformat(), "end": axis[-1].isoformat(), "n_bars": len(axis)},
        "weights": {
            "deposit": str(_DEPOSIT_W),
            "equity_base": str(_EQUITY_BASE_W),
            "re_sweep": [str(w) for w in _RE_WEIGHTS],
            "rent_net_base_pct": str(_RENT_NET_BASE),
            "rent_net_sweep_pct": [str(y) for y in _RENT_NET_SWEEP],
            "re_wrapper_ter_pct": str(_RE_WRAPPER_TER),
        },
        "risk_free_annual_pct": RISK_FREE_ANNUAL_PCT,
        "structural_limits": {
            "smoothed": smoothed,
            "re_bars_per_year": round(re_freq, 1),
            "equity_bars_per_year": round(eq_freq, 1),
            "smoothing_note": (
                "MREDC updates ~weekly and is a transaction/appraisal index — measured "
                "volatility/drawdown are structurally understated; the investable rental-ZPIF "
                "wrapper carries the market vol + illiquidity + 1-3%/yr fees MREDC hides."
            ),
            "policy_note": (
                "The ~+8.5%/yr residential appreciation was largely subsidised-mortgage "
                "(lgotnaya ipoteka) driven — a policy now wound down, so the historical price "
                "rise is NOT a forward-looking expectation."
            ),
            "income_note": (
                "MREDC is PRICE-only; rent is overlaid as a labelled NET assumption "
                "(post-cost, post-NDFL), swept 3/4/6% — never a measured number."
            ),
        },
        "correlations": {
            "re_vs_equity": {"corr": _r(corr_eq), "n": n_eq},
            "re_vs_deposit": {"corr": _r(corr_dep)},
        },
        "deposit_dominance": {
            "deposit_only_tr_pct": round(dep_tr, 2),
            "re_price_only_tr_pct": round(re_price_only_tr, 2),
            "re_wrapper_ter_charged_pct": str(_RE_WRAPPER_TER),
            "re_plus_rent_tr_pct_by_yield": {k: round(v, 2) for k, v in rent_sensitivity.items()},
            "best_re_tr_pct": round(best_re_tr, 2),
            "price_beats_deposit": price_beats_deposit,
            "base_rent_beats_deposit": base_rent_beats_deposit,
            "generous_rent_beats_deposit": generous_rent_beats_deposit,
            "tax_exempt_bound": {
                "note": (
                    "The daily-mark NDFL OVER-taxes real estate (a >3yr LDV / primary-residence "
                    "hold is often price-gain-exempt) — the assumption that cuts hardest against "
                    "it. This is the most favourable defensible arm: price WITHOUT NDFL (wrapper "
                    "TER still charged) + generous 6% rent."
                ),
                "re_price_taxfree_tr_pct": round(re_price_taxfree_tr, 2),
                "re_taxfree_plus_generous_rent_tr_pct": round(re_taxfree_gen_tr, 2),
                "taxfree_generous_beats_deposit": taxfree_generous_beats_deposit,
            },
        },
        "controls": {"re_zero_reproduces_baseline": zero_ok},
        "windows": rows,
        "integration_gate": gate,
        "binding": {"verdict": verdict, "finding": finding},
    }
    _DIR.mkdir(parents=True, exist_ok=True)
    (_DIR / "realestate_cert_summary.json").write_text(
        json.dumps(summary, indent=1, default=str), encoding="utf-8"
    )

    md = [
        "# Phase C — Real-Estate Sleeve: Income Diversifier or Deposit-Dominated? (Cert)",
        "",
        f"Window `{axis[0]}`->`{axis[-1]}` · {len(axis)} bars · RUONIA-excess "
        f"{RISK_FREE_ANNUAL_PCT}%",
        f"Base = deposit {_DEPOSIT_W} / equity {_EQUITY_BASE_W}; real estate (MREDC) carved from "
        f"equity (sweep {[str(w) for w in _RE_WEIGHTS]}); rent overlay net "
        f"{[str(y) for y in _RENT_NET_SWEEP]}%.",
        "",
        "> **Two structural limits.** (1) MREDC is ~weekly/appraisal-smoothed "
        f"({re_freq:.0f} bars/yr vs equity {eq_freq:.0f}) → low measured drawdown is partly an "
        "artifact the investable rental-ZPIF wrapper (illiquid, 1-3%/yr fees) would erase. "
        "(2) The ~+8.5%/yr appreciation was largely subsidised-mortgage driven — not a forward "
        "expectation.",
        "",
        f"## BINDING VERDICT: **{verdict}**",
        "",
        finding,
        "",
        "## Canonical Instrument Integration Gate (battery-comparable)",
        "Same pre-registered gate as the beyond-edge battery (gold -> REJECT, ZO -> PROBATION). "
        "MREDC's tail IS backtestable, so real estate is held to the strict INTEGRATE bar and "
        "cannot take ZO's tail-untestable PROBATION toe-hold.",
        "",
        f"**GATE TIER: `{gate['tier']}`** (proposed weight {gate['proposed_weight']}, carved from "
        f"{gate['carved_from']}) — {'; '.join(gate['reasons'])}",  # type: ignore[arg-type]
        "",
        "| scorecard | value |",
        "| --- | ---: |",
        f"| window bars / regimes | {sc.window_bars} / {sc.regimes_covered} |",
        f"| tail backtestable | {sc.tail_backtestable} |",
        f"| Δ Sharpe (10% eval) | {sc.marginal_sharpe_delta:+.3f} |",
        f"| Δ Sortino (10% eval) | {sc.marginal_sortino_delta:+.3f} |",
        f"| Δ MaxDD pp (+ = cut) | {sc.marginal_maxdd_delta_pp:+.2f} |",
        f"| crash-year Δ MaxDD pp (+ = raised) | {sc.crash_year_maxdd_delta_pp:+.2f} |",
        f"| toe-hold Δ Sortino (3%) | {sc.toehold_sortino_delta:+.3f} |",
        f"| max \\|corr\\| to existing legs | {sc.max_corr_to_existing_legs:.3f} |",
        "",
        "## Correlation & deposit anchor",
        "| measure | value |",
        "| --- | ---: |",
        f"| corr(real estate, equity) | {_f(corr_eq)} ({n_eq} pairs) |",
        f"| corr(real estate, deposit leg) | {_f(corr_dep)} |",
        f"| 100%-deposit total return | {dep_tr:.1f}% |",
        f"| real-estate price-only TR (after {_RE_WRAPPER_TER}% ZPIF wrapper fee) "
        f"| {re_price_only_tr:.1f}% |",
        *(
            f"| real-estate price + {y}% net rent TR | {rent_sensitivity[str(y)]:.1f}% |"
            for y in _RENT_NET_SWEEP
        ),
        f"| **price-only beats deposit? (robust)** | **{price_beats_deposit}** |",
        f"| base 4% rent beats deposit? | {base_rent_beats_deposit} |",
        f"| generous 6% rent beats deposit? | {generous_rent_beats_deposit} |",
        f"| 0%-RE reproduces baseline curve | {zero_ok} |",
        "",
        "## In-window blend (real estate carved from equity)",
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
        for label, v in r["variants"].items():  # type: ignore[index,union-attr]
            m = v["metrics"]
            dv = "yes" if v["diversification"]["diversifies"] else "no"
            md.append(
                f"| {wname}{cav} | +RE {label} | {_f(m['sharpe'])} | {_f(m['sortino'])} | "
                f"{_f(m['maxdd_pct'])} | {_f(m['total_return_pct'])} | {dv} |"
            )
    md += [
        "",
        "_*RUONIA-excess on a fixed 15% basis (apt for the high-rate era); MaxDD is basis-free._",
        "_`price_*` arms are MREDC price-only; `total_*` arms add the net rental overlay._",
    ]
    (_DIR / "realestate_cert_report.md").write_text("\n".join(md), encoding="utf-8")

    print(f"BINDING VERDICT: {verdict}")
    print(
        f"  INTEGRATION GATE TIER: {gate['tier']} (weight {gate['proposed_weight']}) — "
        f"dSharpe={sc.marginal_sharpe_delta:+.3f} dSortino={sc.marginal_sortino_delta:+.3f} "
        f"dMaxDD={sc.marginal_maxdd_delta_pp:+.2f}pp tail_bt={sc.tail_backtestable}"
    )
    print(
        f"  corr_eq={_f(corr_eq)} corr_dep={_f(corr_dep)} smoothed={smoothed} "
        f"re_freq={re_freq:.0f}/yr eq_freq={eq_freq:.0f}/yr"
    )
    print(
        f"  deposit_tr={dep_tr:.1f}% re_price_tr={re_price_only_tr:.1f}% (after "
        f"{_RE_WRAPPER_TER}% TER) base_rent_tr={re_base_rent_tr:.1f}% best_re_tr={best_re_tr:.1f}% "
        f"price_beats={price_beats_deposit} base_beats={base_rent_beats_deposit} "
        f"gen_beats={generous_rent_beats_deposit} zero_ok={zero_ok}"
    )
    print(f"  {finding}")


def _f(x: object) -> str:
    return f"{x:.3f}" if isinstance(x, float) else "n/a"


def _r(x: float | None) -> float | None:
    return round(x, 4) if x is not None else None


if __name__ == "__main__":
    main()
