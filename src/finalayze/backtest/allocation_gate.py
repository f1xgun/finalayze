"""Allocator-gate measurement layer (Phase 73, GATE-01/02/03).

This module is the L5 *measurement* layer for the FROZEN W2 allocator: it builds
no new allocation logic. It ORCHESTRATES the existing
``AllocationOrchestrator`` (via the ``profiles=`` benchmark-injection seam) and
the canonical RUONIA-excess daily-rate machinery to judge whether the allocator
beats its naive benchmarks on a strict conjunctive rule (D-01).

Two honesty code-traps are reconciled here with pinned tests:

- **TRAP A (V-1):** ``AllocationResult.max_drawdown_pct`` is a PERCENT (e.g.
  ``8.0``) but the profile caps are FRACTIONS (``Decimal("0.08")``).
  :func:`realized_dd_fraction` divides by 100 so the gate never compares
  ``8.0 <= 0.08``.
- **TRAP B (V-2):** the snapshot-based ``performance.sortino_ratio`` clamps
  ``mean_excess <= 0`` to ``Decimal(0)``, which would corrupt the strict
  ``>= best_naive`` comparison. :func:`excess_sortino_from_equity` returns the
  TRUE (possibly negative) curve-based Sortino and never clamps a losing leg.

The conjunctive verdict (:func:`verdict_for_profile`, D-01) PASSes IFF the
allocator's Sharpe AND Sortino both clear the best-of-three naive bar AND the
realized MaxDD is within the profile cap — never an "either/or".

The auto-tighten (V-5) and OOS walk-forward (V-8) primitives, the regime split
(V-9) and the Markdown/JSON report renderer (D-11) live here. The synthetic
framing cut-path (V-7) was RETIRED in Phase 74 (D-07): the real binding window now
contains the real easing (high-rate plateau → the verified 2025 CBR cuts from
2025-06-06), so the evidence-based "cut scenario" is the real easing sub-window —
the post-:data:`REGIME_SPLIT_BOUNDARY` segment of :func:`regime_split` — reported
with real metrics, not a synthetic glide.
"""

from __future__ import annotations

import json
import math
import statistics
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from finalayze.backtest.bond_walk_forward import (
    _compute_excess_sharpe_from_equity,
    generate_wf_windows,
)
from finalayze.core.allocation import tighten
from finalayze.core.exceptions import ConfigurationError
from finalayze.core.schemas import AllocationProfile, AssetClass, RiskProfile
from finalayze.data.fetchers.cbr import (
    MacroContextProvider,
    deposit_rate_as_of,
)
from finalayze.strategies.bond_duration_rotation import CBRRegime, classify_regime

if TYPE_CHECKING:
    from finalayze.core.ndfl import YtdTaxAccumulator
    from finalayze.orchestration.allocation import AllocationOrchestrator, AllocationResult

# ── Constants (named — no PLR2004 magic numbers) ─────────────────────────────
_PERCENT = Decimal(100)
_TRADING_DAYS_PER_YEAR = 252
# The RUONIA-excess annual risk-free rate (mirrors bond_walk_forward._DEFAULT_RUONIA_ANNUAL_PCT).
_DEFAULT_RUONIA_ANNUAL_PCT = 15.0
# Below this many daily returns the Sortino is undefined → 0.0.
_MIN_RETURNS = 2
# Zero-downside (monotone-up) sentinel. Mirrors performance._LARGE_RATIO_SENTINEL's
# CONVENTION (a large-but-finite stand-in for an undefined ratio) as a float, since
# the gate's Sortino is a float to sit on the same footing as AllocationResult.sharpe.
_LARGE_SORTINO_SENTINEL = 1e9
# OOS walk-forward geometry (D-02). Passed EXPLICITLY to generate_wf_windows — its
# own defaults are 24/12/6, so these named constants are REQUIRED to honor the 12/6/3
# cadence GATE-01 mandates (V-8).
_WF_TRAIN_MONTHS = 12
_WF_TEST_MONTHS = 6
_WF_STEP_MONTHS = 3


def excess_sortino_from_equity(
    equity: list[float],
    risk_free_annual_pct: float = _DEFAULT_RUONIA_ANNUAL_PCT,
    trading_days_per_year: int = _TRADING_DAYS_PER_YEAR,
) -> float:
    """Annualised RUONIA-excess Sortino from an equity curve — TRUE negatives kept.

    Mirrors ``bond_walk_forward._compute_excess_sharpe_from_equity`` EXACTLY for the
    returns + daily risk-free computation (same RUONIA-excess basis), then swaps the
    full-volatility denominator for the downside-only deviation.

    Convention (TRAP B, V-2): the target is the daily RUONIA risk-free rate — the
    SAME footing as the Sharpe excess basis — so ``excess = daily_return - daily_rf``.
    Unlike ``performance.sortino_ratio`` (which clamps ``mean_excess <= 0`` to 0),
    this returns the genuine ``mean_excess / dd_std * sqrt(252)`` even when negative,
    so the strict ``>= best_naive`` comparison stays honest.

    Edge cases:
    - fewer than ``_MIN_RETURNS`` daily returns → ``0.0`` (undefined);
    - zero downside (monotone-up, all excess > 0) → ``_LARGE_SORTINO_SENTINEL`` when
      ``mean_excess > 0`` else ``0.0``.
    """
    if len(equity) < _MIN_RETURNS + 1:
        return 0.0

    daily_returns = [
        equity[i] / equity[i - 1] - 1.0 for i in range(1, len(equity)) if equity[i - 1] > 0
    ]
    if len(daily_returns) < _MIN_RETURNS:
        return 0.0

    # Daily risk-free rate (continuous compounding approximation) — byte-identical to
    # _compute_excess_sharpe_from_equity so Sharpe and Sortino share one excess basis.
    daily_rf = (1 + risk_free_annual_pct / 100) ** (1 / trading_days_per_year) - 1.0
    excess = [r - daily_rf for r in daily_returns]

    mean_excess = statistics.mean(excess)
    downside = [min(0.0, e) for e in excess]
    dd_std = (sum(d * d for d in downside) / len(downside)) ** 0.5

    if dd_std <= 0:
        # No downside at all: undefined ratio → large finite sentinel if winning.
        return _LARGE_SORTINO_SENTINEL if mean_excess > 0 else 0.0

    # CRITICAL (TRAP B): do NOT clamp mean_excess <= 0 — return the real negative ratio.
    return float(mean_excess / dd_std * math.sqrt(trading_days_per_year))


def realized_dd_fraction(max_drawdown_pct: float) -> Decimal:
    """Reconcile a realized MaxDD PERCENT to a FRACTION before the cap compare (TRAP A).

    ``AllocationResult.max_drawdown_pct`` is a PERCENT (e.g. ``8.0`` for an 8% drop)
    but the YAML profile caps are FRACTIONS (``Decimal("0.08")``). This divides by 100
    so the gate compares ``0.08 <= 0.08`` — never the off-by-100 ``8.0 <= 0.08`` (V-1).
    """
    return Decimal(str(max_drawdown_pct)) / _PERCENT


# ── Naive-leg weight constants (degenerate benchmark vectors) ────────────────
_ONE = Decimal(1)
_ZERO = Decimal(0)
_STATIC_DEP = Decimal("0.1")
_STATIC_OFZ = Decimal("0.3")
_STATIC_EQ = Decimal("0.6")
# A benchmark leg has no binding cap (it is the bar, not the candidate) → 1.0.
_NAIVE_CAP = Decimal("1.0")


def _zero_curve(reference: list[tuple[date, Decimal]]) -> list[tuple[date, Decimal]]:
    """A collapsed (all-``Decimal(0)``) TR curve on ``reference``'s exact date index.

    Used to take a leg OUT of a single-asset benchmark book. The orchestrator seeds
    every leg at ``scale=1`` regardless of target weight, so passing a leg's REAL
    rising curve into a 0%-target book makes the first boundary LIQUIDATE that leg
    (a real sell → cost). A zeroed curve carries no value (``price <= 0`` is skipped
    by ``_charge_rebalance``), so the held leg never trades against it — this is the
    orchestrator's own ``reproduce_legacy_60_40`` collapse pattern (allocation.py:449)
    applied to the eq/OFZ legs instead of the deposit leg. The held-leg basis is
    UNCHANGED (same curve, same dates), so the three legs still share one basis (R-3).
    """
    return [(d, _ZERO) for d, _ in reference]


def _naive_orchestrator(weights: dict[AssetClass, Decimal], cap: Decimal) -> AllocationOrchestrator:
    """Build an AllocationOrchestrator pinned to a single degenerate benchmark vector.

    Uses the ``profiles=`` injection seam (allocation.py:347) so the FROZEN W2
    allocator runs the naive leg on the SAME cost/NDFL path as the real candidate —
    guaranteeing the same basis (R-3). The profile KEY is arbitrary (only the
    weights/cap matter for a benchmark leg). Deferred inline import for the L5→L5
    hop (ARCHITECTURE Pattern 4 / bond_engine.py:180).
    """
    from finalayze.orchestration.allocation import AllocationOrchestrator  # noqa: PLC0415

    p = RiskProfile.BALANCED  # arbitrary key — only weights/cap matter for a benchmark leg
    profile = AllocationProfile(profile=p, weights=weights, max_drawdown_pct=cap)
    return AllocationOrchestrator(risk_profile=p, profiles={p: profile})


def build_naive_legs(
    deposit_curve: list[tuple[date, Decimal]],
    ofz_pk_curve: list[tuple[date, Decimal]],
    equity_curve: list[tuple[date, Decimal]],
) -> dict[str, AllocationResult]:
    """Build the three naive benchmark legs on ONE basis via degenerate injection (V-6 / R-3).

    Each leg is the FROZEN allocator run with a degenerate fixed-weight profile on the
    SAME basis (D-04/D-05). For the two single-asset legs the NON-held curves are
    collapsed to zero (:func:`_zero_curve`) so the 100%-target book never liquidates a
    phantom leg the orchestrator seeded at ``scale=1`` (the held leg's curve is
    UNCHANGED → same basis, R-3). The 60/30/10 leg keeps all three curves and trades:

    - ``deposit_100`` — 100% deposit, eq/OFZ collapsed: only the cost-free deposit leg
      is ever held → ``rebalance_cost == 0`` (D-09 deposit is free).
    - ``equity_100`` — 100% equity, deposit/OFZ collapsed: a single-leg book always at
      its 100% target → ~zero turnover.
    - ``static_60_30_10`` — 60% equity / 30% OFZ-PK / 10% deposit on the FULL curves:
      quarterly-rebalances the traded eq+OFZ legs → ``rebalance_cost > 0``
      (MOEX_RETAIL_COSTS round-trip).
    """
    deposit_weights = {AssetClass.DEPOSIT: _ONE, AssetClass.OFZ_PK: _ZERO, AssetClass.EQUITY: _ZERO}
    equity_weights = {AssetClass.DEPOSIT: _ZERO, AssetClass.OFZ_PK: _ZERO, AssetClass.EQUITY: _ONE}
    static_weights = {
        AssetClass.DEPOSIT: _STATIC_DEP,
        AssetClass.OFZ_PK: _STATIC_OFZ,
        AssetClass.EQUITY: _STATIC_EQ,
    }
    zero_dep = _zero_curve(deposit_curve)
    zero_ofz = _zero_curve(ofz_pk_curve)
    zero_eq = _zero_curve(equity_curve)
    return {
        # 100% deposit: collapse eq + OFZ so nothing trades against them → cost 0.
        "deposit_100": _naive_orchestrator(deposit_weights, _NAIVE_CAP).run(
            deposit_curve, zero_ofz, zero_eq
        ),
        # 100% equity: collapse deposit + OFZ; the lone equity leg holds its 100% target.
        "equity_100": _naive_orchestrator(equity_weights, _NAIVE_CAP).run(
            zero_dep, zero_ofz, equity_curve
        ),
        # 60/30/10: all three legs live → eq + OFZ rebalance and incur real cost.
        "static_60_30_10": _naive_orchestrator(static_weights, _NAIVE_CAP).run(
            deposit_curve, ofz_pk_curve, equity_curve
        ),
    }


def verdict_for_profile(
    *,
    alloc_sharpe: float,
    alloc_sortino: float,
    alloc_max_drawdown_pct: float,
    naive_sharpes: list[float],
    naive_sortinos: list[float],
    cap_fraction: Decimal,
) -> dict[str, object]:
    """The strict conjunctive §7 PASS verdict (V-3/V-4 / D-01/D-04).

    PASS IFF **all three** hold (NOT either/or):

    1. ``alloc_sharpe  >= max(naive_sharpes)``  — beats the best-of-three naive bar;
    2. ``alloc_sortino >= max(naive_sortinos)`` — same, on downside-risk footing;
    3. ``realized_dd_fraction(alloc_max_drawdown_pct) <= cap_fraction`` — within the
       profile MaxDD cap (TRAP A reconciled).

    ``>=`` is inclusive: a metric EXACTLY at its bar PASSes. Flipping any single
    condition flips the verdict to fail.

    Edge guard (WR-01): the best-of-three bar is ``max(naive_sharpes)`` /
    ``max(naive_sortinos)`` — both lists MUST be non-empty or ``max([])`` raises an opaque
    ``ValueError``. The production caller always passes three legs from
    :func:`build_naive_legs`, so this never fires on the cert path; it defends the exported
    surface against any future caller that filters degenerate legs.
    """
    if not naive_sharpes or not naive_sortinos:
        msg = "naive_sharpes and naive_sortinos must be non-empty (the best-of-three bar)"
        raise ValueError(msg)
    best_naive_sharpe = max(naive_sharpes)
    best_naive_sortino = max(naive_sortinos)
    realized_frac = realized_dd_fraction(alloc_max_drawdown_pct)
    # WR-02: ``excess_sortino_from_equity`` returns the fixed ``_LARGE_SORTINO_SENTINEL`` for a
    # zero-downside (monotone-up) leg. A candidate that is ITSELF zero-downside would otherwise
    # satisfy ``alloc_sortino >= best_naive_sortino`` by sentinel EQUALITY (1e9 >= 1e9), passing
    # the Sortino leg without a real risk-adjusted comparison. Treat sentinel-vs-sentinel as
    # UNDEFINED -> the Sortino condition fails (never an automatic pass). This never moves this
    # cert's verdict: an equity-holding allocation always has downside, so ``alloc_sortino`` is
    # never the sentinel here -- the real Sortino value still compares normally.
    sortino_sentinel_tie = (
        alloc_sortino >= _LARGE_SORTINO_SENTINEL and best_naive_sortino >= _LARGE_SORTINO_SENTINEL
    )
    sortino_passes = (not sortino_sentinel_tie) and alloc_sortino >= best_naive_sortino
    passes = alloc_sharpe >= best_naive_sharpe and sortino_passes and realized_frac <= cap_fraction
    return {
        "pass": passes,
        "sharpe": alloc_sharpe,
        "best_naive_sharpe": best_naive_sharpe,
        "sortino": alloc_sortino,
        "best_naive_sortino": best_naive_sortino,
        "realized_maxdd_frac": float(realized_frac),
        "cap_frac": float(cap_fraction),
    }


# ── Regime framing surface (D-09/R-6) ───────────────────────────────────────
# The synthetic framing cut-path (V-7 / D-07/D-08) was RETIRED in Phase 74 (D-07):
# the real binding window now CONTAINS the real easing (high-rate plateau → the real
# 2025 cuts), so the "cut scenario" is the real easing sub-window (the post-boundary
# segment of regime_split), reported with real metrics — not a synthetic glide.

# Trading days per year for the daily-compounding accrual (accrue_real_risk_free_leg).
_TRADING_DAYS = 252

# REGIME_SPLIT_BOUNDARY: the early-cut regime boundary (V-9 / D-09 / R-6). Phase 74 (R-C)
# shifts it to the VERIFIED first real 2025 CBR cut (2025-06-06 → 20.00) — the calendar
# previously listed a spurious 2025-07-25 first cut; Plan 01 corrected CBR_MEETINGS to the
# cbr.ru archive (first cut 2025-06-06). This is the documented high-rate/plateau vs
# early-cut split point. NAMED (no magic date in logic).
REGIME_SPLIT_BOUNDARY: date = date(2025, 6, 6)

# The binding HARD_FAIL verdict string (Phase 75 / REGIME-02). NAMED so the per-regime
# driver and Plan 02's derive_escalation never inline the raw literal. It mirrors the
# terminal verdict gate_with_autotighten emits ("PASS" / "PASS_AFTER_TIGHTEN" / "HARD_FAIL").
_HARD_FAIL = "HARD_FAIL"

# The single-cycle (N=1) caveat for the easing verdict (Phase 75 / REGIME-05 / D-04). Pinned
# VERBATIM by the report (the verifier greps for this literal), mirroring _HIGH_RATE_CAVEAT.
# It is SEPARATE metadata (an always-on n1_caveat flag) -- NEVER fused into a verdict-status
# string (no "HARD_FAIL (N=1)"); the strict conjunctive easing test is UNCHANGED (no softening).
_N1_CAVEAT = (
    "The easing verdict is based on a SINGLE observed easing cycle (N=1) — it is "
    "suggestive, not statistically robust; a future milestone accumulating additional "
    "easing cycles could upgrade it."
)

# The machine-readable escalation flag value recorded when BOTH regimes HARD_FAIL (REGIME-05 /
# D-03): anchor on the near-vol-free deposit for now, document-defer the redesign branch. NAMED
# so derive_escalation never inlines the literal (anti-hollow: the flag is DERIVED, not pre-baked).
_ESCALATION_DEPOSIT_ANCHOR = "deposit_anchor_vs_redesign"

# regime_split / regime_verdicts emit the pre-cut plateau under the key "high_rate" and the
# post-cut binding unit under the key "early_cut"; the report renders the post-cut unit under the
# human-facing label "easing" (REGIME-02). NAMED so neither regime_split, render_report, nor the
# CLI (scripts/run_allocation_gate.py, IN-03) ever inlines a copy of these unit-key strings — a
# single source of truth so a future relabel propagates everywhere.
_HIGH_RATE_UNIT_KEY = "high_rate"
_EASING_UNIT_KEY = "early_cut"
_EASING_UNIT_LABEL = "easing"

# The mandatory honesty caveat (D-08 / Pitfall 6): a deposit-wins-raw-return outcome in a
# 16-21% high-rate regime is correctly framed as NOT a failure. Pinned VERBATIM by the
# report; the verifier greps for this literal string.
_HIGH_RATE_CAVEAT = "100% deposit winning raw return in a 16-21% high-rate regime is NOT a failure"

# Methodology note (operator follow-up, framing-only — NOT a metric/verdict/logic change):
# in a 16-21% high-rate regime the 100%-deposit leg is a near-vol-free ~18% return (≈0
# downside, MaxDD 0), so its risk-adjusted Sharpe/Sortino is enormous (e.g. Sortino ~4.8e13
# is the genuine value of a zero-downside curve, NOT a rendering bug). Because that
# near-risk-free leg sets the best-naive bar, the conjunctive Sharpe ∧ Sortino test is
# STRUCTURALLY unwinnable for any equity-holding allocation while the high rate holds — so a
# HARD_FAIL here reflects the RATE REGIME, not an allocator defect. This note re-frames the
# numbers only; it changes no metric, verdict, cap or the binding gate logic.
_RISK_FREE_BAR_NOTE = (
    "Methodology note (framing-only): in a 16-21% high-rate regime the 100%-deposit leg is a "
    "near-vol-free ~18% return (near-zero downside, MaxDD 0), so its Sharpe/Sortino bar is "
    "enormous (a Sortino ~4.8e13 is the TRUE value of a zero-downside curve, NOT a rendering "
    "bug). That near-risk-free leg sets the best-naive bar, which makes the conjunctive "
    "Sharpe ∧ Sortino test structurally unwinnable for any equity-holding allocation while "
    "the high rate holds -- so a HARD_FAIL here reflects the RATE REGIME, not an allocator "
    "defect."
)


def _run_and_score(
    _profile_key: RiskProfile,
    weights: dict[AssetClass, Decimal],
    cap_fraction: Decimal,
    deposit_curve: list[tuple[date, Decimal]],
    ofz_pk_curve: list[tuple[date, Decimal]],
    equity_curve: list[tuple[date, Decimal]],
    naive_sharpes: list[float],
    naive_sortinos: list[float],
) -> dict[str, object]:
    """Run the FROZEN allocator on a single weight vector and score it against the naive bar.

    Constructs the degenerate-profile orchestrator via :func:`_naive_orchestrator` (the same
    ``profiles=`` injection seam the naive legs use — the candidate runs on the IDENTICAL
    cost/NDFL basis, R-3), runs it on the three curves, derives the curve-based Sortino
    (TRAP B), and returns the conjunctive :func:`verdict_for_profile` dict merged with the
    ``result`` carrier and the reported :func:`mean_wf_sharpe` (R-1). The ``_profile_key`` is
    accepted for caller symmetry only — the verdict depends solely on weights/cap/curves
    (the degenerate orchestrator pins its own arbitrary internal profile key).
    """
    result = _naive_orchestrator(weights, cap_fraction).run(
        deposit_curve, ofz_pk_curve, equity_curve
    )
    alloc_sortino = excess_sortino_from_equity([float(v) for v in result.merged_equity_curve])
    verdict = verdict_for_profile(
        alloc_sharpe=result.sharpe,
        alloc_sortino=alloc_sortino,
        alloc_max_drawdown_pct=result.max_drawdown_pct,
        naive_sharpes=naive_sharpes,
        naive_sortinos=naive_sortinos,
        cap_fraction=cap_fraction,
    )
    return {**verdict, "result": result, "mean_wf_sharpe": mean_wf_sharpe(result)}


def gate_with_autotighten(
    *,
    profile_key: RiskProfile,
    base_weights: dict[AssetClass, Decimal],
    cap_fraction: Decimal,
    deposit_curve: list[tuple[date, Decimal]],
    ofz_pk_curve: list[tuple[date, Decimal]],
    equity_curve: list[tuple[date, Decimal]],
    naive_sharpes: list[float],
    naive_sortinos: list[float],
) -> dict[str, object]:
    """Execute the W2-deferred D-05 auto-tighten loop: freeze + OOS re-gate (V-5 / D-03).

    EXECUTES the dormant L0 :func:`finalayze.core.allocation.tighten` rule that W2 shipped
    tested-but-unwired. Sequence:

    1. Score the untightened ``base_weights``. If it PASSes → ``"PASS"`` (no tighten needed).
    2. On a cap breach: feed the realized DD (reconciled to a FRACTION via
       :func:`realized_dd_fraction`) and the cap into ``tighten`` — a parameter-free, monotone
       5pp equity→deposit shift that clamps equity at 0. FREEZE that vector.
    3. Re-run the allocator on the FROZEN vector and re-gate OOS. If it now PASSes →
       ``"PASS_AFTER_TIGHTEN"``; otherwise → ``"HARD_FAIL"``.

    This is the HONESTY GUARD (T-73-06): a still-failing frozen vector is a binding
    HARD_FAIL — there is NO further widening, no search, no optimizer after the single
    freeze (Pitfall 8). The returned dict carries ``frozen_weights`` for traceability.
    """
    first = _run_and_score(
        profile_key,
        base_weights,
        cap_fraction,
        deposit_curve,
        ofz_pk_curve,
        equity_curve,
        naive_sharpes,
        naive_sortinos,
    )
    if first["pass"]:
        return {"verdict": "PASS", **first}

    first_result = cast("AllocationResult", first["result"])
    realized_dd_frac = realized_dd_fraction(first_result.max_drawdown_pct)
    # Single parameter-free 5pp equity->deposit freeze — NEVER a search/widening loop (Pitfall 8).
    frozen = tighten(base_weights, realized_dd_frac, cap_fraction)
    regated = _run_and_score(
        profile_key,
        frozen,
        cap_fraction,
        deposit_curve,
        ofz_pk_curve,
        equity_curve,
        naive_sharpes,
        naive_sortinos,
    )
    if regated["pass"]:
        return {"verdict": "PASS_AFTER_TIGHTEN", "frozen_weights": frozen, **regated}
    # No further widening after the freeze — a persistent breach is a binding FAIL (D-03).
    return {"verdict": "HARD_FAIL", "frozen_weights": frozen, **regated}


def oos_wf_sharpes(
    result: AllocationResult,
    risk_free_annual_pct: float = _DEFAULT_RUONIA_ANNUAL_PCT,
) -> list[float]:
    """OOS walk-forward Sharpes sliced from the merged curve — NO engine re-run (V-8 / D-02).

    Slices the ALREADY-merged ``AllocationResult.merged_equity_curve`` per
    ``generate_wf_windows(result.dates[0], result.dates[-1], 12, 6, 3)`` and computes one
    RUONIA-excess Sharpe per OOS test slice via
    :func:`bond_walk_forward._compute_excess_sharpe_from_equity` — the SAME daily-rf footing
    as the gate Sortino. This is "walk-forward ANALYSIS" (time-based slicing), never
    walk-forward OPTIMIZATION: no ``AllocationOrchestrator`` is constructed, no backtest
    engine is run (D-02). The 12/6/3 cadence is passed EXPLICITLY (generate_wf_windows
    defaults to 24/12/6).

    A test slice with too few daily returns (``< _MIN_RETURNS + 1`` bars) is skipped, so the
    returned count is the number of windows that yielded a defined fold Sharpe.
    """
    windows = generate_wf_windows(
        result.dates[0], result.dates[-1], _WF_TRAIN_MONTHS, _WF_TEST_MONTHS, _WF_STEP_MONTHS
    )
    idx = {d: i for i, d in enumerate(result.dates)}
    curve = [float(v) for v in result.merged_equity_curve]
    out: list[float] = []
    for _ts, _te, test_start, test_end in windows:
        slc = [curve[idx[d]] for d in result.dates if test_start <= d <= test_end]
        if len(slc) >= _MIN_RETURNS + 1:
            out.append(_compute_excess_sharpe_from_equity(slc, risk_free_annual_pct))
    return out


def mean_wf_sharpe(
    result: AllocationResult,
    risk_free_annual_pct: float = _DEFAULT_RUONIA_ANNUAL_PCT,
) -> float:
    """The REPORTED mean OOS walk-forward Sharpe (D-02 / R-1) — NOT binding.

    The mean of :func:`oos_wf_sharpes` (``0.0`` if no folds). This is the robustness number
    GATE-01 reports AGAINST the naive bar so "OOS via walk-forward" is visibly honored
    (R-1 mitigation); the BINDING metric is the full-window Sharpe/Sortino in
    :func:`verdict_for_profile`, not this average. Deliberately does NOT route through
    ``compute_walk_forward_sharpe`` (it wants a ``PortfolioBacktestResult``, has 0 callers,
    and would re-run an engine — the anti-pattern this avoids).
    """
    vals = oos_wf_sharpes(result, risk_free_annual_pct)
    return sum(vals) / len(vals) if vals else 0.0


def accrue_real_risk_free_leg(
    dates: list[date],
    base: Decimal,
    *,
    spread_pp: Decimal,
    tax_acc: YtdTaxAccumulator | None = None,
) -> list[tuple[date, Decimal]]:
    """Accrue a risk-free leg from the REAL CBR key-rate path (the live-cert builder).

    Daily-compounding convention (``(1 + annual) ** (1/252)``) reading the REAL
    look-ahead-safe :func:`finalayze.data.fetchers.cbr.deposit_rate_as_of` (which resolves
    the most-recent ``CBR_MEETINGS`` decision on/before each bar) — the real realized
    calendar, not a synthetic framing path (the synthetic cut-path was retired, D-07).
    With ``spread_pp = 1.0`` this is the deposit leg (key - 1pp, mirroring W1's deposit
    accrual); with ``spread_pp = 0`` it is the OFZ-PK floater leg (tracks the full key
    rate ~1:1). The leg opens at ``base`` on ``dates[0]`` and compounds one trading day per
    bar at the as-of annual rate — strictly as-of, NO look-ahead.

    This is the genuine real-data deposit/OFZ-PK total-return series the operator's
    ``--live`` cert requires (an explicit, operator-authorized override of D-10): both
    risk-free legs derive from the REAL CBR calendar, the equity leg from the REAL MCFTR
    series. The returned ``(date, Decimal)`` series shares the supplied common date axis
    (R-3), so the three legs forward-align identically for ``build_naive_legs``.

    Net-of-NDFL step (REGIME-04 / D-01 / D-04 / R-E): when ``tax_acc`` is supplied, the
    per-bar income INCREMENT (``value * daily_factor - value`` — the day's accrued interest,
    NOT the level) is netted through the shared progressive 13/15% band
    (:class:`finalayze.core.ndfl.YtdTaxAccumulator`); only the after-tax income compounds.
    Pass ONE shared accumulator per run so the deposit + OFZ legs share one cross-leg YTD
    (the W1 cross-sleeve design). With ``tax_acc=None`` the leg is GROSS — byte-identical
    to the pre-Phase-74 behaviour (no NDFL on the daily income delta). Never net the curve
    LEVEL — that would tax principal (Pitfall 1). MCFTRR is already net and must NOT be
    routed through an accumulator.
    """
    if not dates:
        return []
    out: list[tuple[date, Decimal]] = [(dates[0], base)]
    value = base
    for d in dates[1:]:
        # deposit_rate_as_of returns a FRACTION already net of the spread (key-spread)/100.
        annual = deposit_rate_as_of(d, spread_pp=spread_pp)
        daily_factor = Decimal(str((1.0 + float(annual)) ** (1.0 / _TRADING_DAYS)))
        # Net the day's income INCREMENT (not the level) through the shared NDFL band (R-E).
        gross_income_delta = value * daily_factor - value
        tax = tax_acc.tax(gross_income_delta, year=d.year) if tax_acc else Decimal(0)
        value = value + (gross_income_delta - tax)
        out.append((d, value))
    return out


def net_index_returns(
    level_series: list[tuple[date, Decimal]],
    *,
    tax_acc: YtdTaxAccumulator | None = None,
) -> list[tuple[date, Decimal]]:
    """Re-base a fetched GROSS index (RUFLBITR) to a net-of-NDFL TR curve (REGIME-04 / R-E).

    The deposit/OFZ accrual analogue for a FETCHED index level series: the real RUFLBITR
    floating-coupon-bond TR index is published GROSS of investor NDFL (D-04 derived
    implication), so to honour D-01 (net both sides) its daily return must be netted of the
    same progressive 13/15% band the deposit leg uses.

    The curve opens at the same base (``level_series[0]``). For each subsequent bar the day's
    income is the RETURN INCREMENT applied to the prior NET value
    (``prior_value * (level[i] / level[i - 1] - 1)`` — NOT the index level, Pitfall 1). A
    POSITIVE daily income is netted through ``tax_acc.tax(income, year=d.year)``; a NEGATIVE
    daily return is passed through untaxed (a loss is not a refund). When ``tax_acc`` is
    ``None`` the increment compounds gross. Principal is never taxed: a flat (zero-return)
    index returns the level unchanged.

    Pass the SAME shared ``YtdTaxAccumulator`` used by :func:`accrue_real_risk_free_leg` so
    the index leg shares one cross-leg YTD with the deposit leg (the W1 cross-sleeve design).
    MCFTRR is ALREADY net — do NOT route it through this helper (Pitfall 1: double-taxing
    equity).
    """
    if not level_series:
        return []
    out: list[tuple[date, Decimal]] = [level_series[0]]
    value = level_series[0][1]
    prev_level = level_series[0][1]
    for d, level in level_series[1:]:
        if prev_level > 0:
            daily_return = level / prev_level - _ONE
            income = value * daily_return
            if income > _ZERO:
                income -= tax_acc.tax(income, year=d.year) if tax_acc else _ZERO
            value = value + income
        prev_level = level
        out.append((d, value))
    return out


def net_fixed_income_legs_interleaved(
    ofz_level_series: list[tuple[date, Decimal]],
    deposit_dates: list[date],
    deposit_base: Decimal,
    *,
    deposit_spread_pp: Decimal,
    tax_acc: YtdTaxAccumulator,
) -> tuple[list[tuple[date, Decimal]], list[tuple[date, Decimal]]]:
    """Net BOTH fixed-income legs in ONE date-ordered pass through a shared YTD (CR-01).

    The W1 cross-sleeve contract is that the deposit + OFZ-PK legs share ONE cross-leg
    progressive-band YTD per run (one :class:`YtdTaxAccumulator`). Netting them
    LEG-BY-LEG (the full OFZ pass across the whole multi-year axis, THEN the full deposit
    pass) silently BREAKS that contract on a multi-tax-year window: after the OFZ pass the
    accumulator's ``_current_year`` is the LAST year, so the deposit leg's FIRST (earliest)
    bar triggers a Jan-1 reset (:meth:`YtdTaxAccumulator.tax` resets on a ``year`` change),
    discarding the OFZ leg's accumulated YTD — the two legs are then taxed as if each had
    its own per-year YTD (CR-01).

    This helper instead nets both legs interleaved BY DATE: for each bar (in date order)
    the OFZ income increment AND the deposit income increment are taxed through the SAME
    accumulator at ``year=d.year``, so BOTH increments hit the same running YTD BEFORE any
    year-boundary reset. The two legs MUST share one common date axis (R-3) — the OFZ level
    series is forward-aligned onto the deposit ``dates`` by the caller, so
    ``[d for d, _ in ofz_level_series] == deposit_dates``.

    Per-leg arithmetic is IDENTICAL to :func:`net_index_returns` (OFZ) and
    :func:`accrue_real_risk_free_leg` (deposit): a positive income increment is netted
    through the band, a loss (OFZ) passes through untaxed, principal is never taxed
    (Pitfall 1). On a window where the cross-leg YTD never crosses the 2.4M threshold (every
    increment is taxed at the 13% base rate) the netted curves are BYTE-IDENTICAL to the
    leg-by-leg result — only a band crossover (the exact thing the shared YTD exists to
    detect) makes the interleaved result differ. Returns ``(deposit_curve, ofz_pk_curve)``.
    """
    if not deposit_dates:
        return [], []
    if [d for d, _ in ofz_level_series] != deposit_dates:
        msg = "net_fixed_income_legs_interleaved requires both legs on one shared date axis (R-3)"
        raise ConfigurationError(msg)

    deposit_out: list[tuple[date, Decimal]] = [(deposit_dates[0], deposit_base)]
    ofz_out: list[tuple[date, Decimal]] = [ofz_level_series[0]]
    deposit_value = deposit_base
    ofz_value = ofz_level_series[0][1]
    ofz_prev_level = ofz_level_series[0][1]

    for i in range(1, len(deposit_dates)):
        d = deposit_dates[i]
        # OFZ leg increment (net_index_returns arithmetic) — taxed FIRST against the shared
        # YTD so both legs' increments hit the SAME running YTD before any year reset.
        ofz_level = ofz_level_series[i][1]
        if ofz_prev_level > 0:
            ofz_income = ofz_value * (ofz_level / ofz_prev_level - _ONE)
            if ofz_income > _ZERO:
                ofz_income -= tax_acc.tax(ofz_income, year=d.year)
            ofz_value = ofz_value + ofz_income
        ofz_prev_level = ofz_level

        # Deposit leg increment (accrue_real_risk_free_leg arithmetic) — taxed against the
        # SAME shared YTD at the SAME year, before the next year's reset.
        annual = deposit_rate_as_of(d, spread_pp=deposit_spread_pp)
        daily_factor = Decimal(str((1.0 + float(annual)) ** (1.0 / _TRADING_DAYS)))
        gross_income_delta = deposit_value * daily_factor - deposit_value
        deposit_tax = tax_acc.tax(max(gross_income_delta, _ZERO), year=d.year)
        deposit_value = deposit_value + (gross_income_delta - deposit_tax)

        ofz_out.append((d, ofz_value))
        deposit_out.append((d, deposit_value))

    return deposit_out, ofz_out


# ── Committed real-data snapshot (REGIME-01 / D-05, Phase-65 fail-closed pattern) ──
# The binding cert reads a committed JSON snapshot of the fetched real series (MCFTRR
# net equity + RUFLBITR-derived net OFZ + net deposit) so CI reproduces the gate
# deterministically with NO network. A missing/corrupt/future-dated file fails closed
# (ConfigurationError) — there is NO silent fallback to synthetic data (T-74-03 / V5).
# Plan 03 writes the committed file (and creates the data/ dir); this loader reads it.
_GATE_SNAPSHOT = Path(__file__).parent / "data" / "allocation_gate_snapshot.json"
# The binding window endpoint (the look-ahead clamp, Pitfall 3 / T-74-04): NO bar may
# post-date this. Named — no magic date in the guard.
_BINDING_END = date(2026, 6, 10)
# The three leg keys the snapshot must carry (R-F shape). Validated fail-closed.
_SNAPSHOT_LEG_KEYS = ("equity_mcftrr_net", "ofz_rgbitr_net", "deposit_net")


def _rehydrate_leg(rows: Any, *, window_end: date) -> list[tuple[date, Decimal]]:
    """Re-hydrate one snapshot leg ``[[iso_date, decimal_str], ...]`` fail-closed (Pitfall 3 / V5).

    Coerces each row to ``(date.fromisoformat(d), Decimal(str(v)))`` (the Phase-65
    ``_row_to_instrument`` convention) and REJECTS (raises ``ConfigurationError``) any bar
    dated after ``window_end`` OR after :data:`_BINDING_END` (the look-ahead clamp). A
    malformed row shape raises via the caller's ``except`` tuple.
    """
    out: list[tuple[date, Decimal]] = []
    for d_str, v_str in rows:
        d = date.fromisoformat(str(d_str))
        if d > window_end or d > _BINDING_END:
            msg = (
                f"allocation-gate snapshot bar {d.isoformat()} post-dates the binding window "
                f"end ({window_end.isoformat()} / clamp {_BINDING_END.isoformat()}) — "
                "look-ahead leak (Pitfall 3 / T-74-04)"
            )
            raise ConfigurationError(msg)
        out.append((d, Decimal(str(v_str))))
    return out


def _load_gate_snapshot(
    path: Path = _GATE_SNAPSHOT,
) -> tuple[
    list[tuple[date, Decimal]],
    list[tuple[date, Decimal]],
    list[tuple[date, Decimal]],
]:
    """Read the committed real-data gate snapshot, fail-closed (REGIME-01 / D-05 / V5).

    Copies the Phase-65 ``instruments.py:239-251`` committed-snapshot pattern EXACTLY: read
    the JSON, pull the three required legs (``equity_mcftrr_net`` / ``ofz_ruflbitr_net`` /
    ``deposit_net``), and re-hydrate each ``[iso_date, decimal_str]`` row to a
    ``(date, Decimal)`` pair. On a missing/corrupt file or a missing required key the loader
    raises :class:`finalayze.core.exceptions.ConfigurationError` — there is NO silent
    fallback to synthetic data (the committed file is the CI trust boundary, T-74-03).

    Look-ahead clamp (Pitfall 3 / T-74-04): every bar must be ``<= window.end`` AND
    ``<= _BINDING_END`` (2026-06-10) — a snapshot refreshed on a later date cannot leak a
    future bar. Returns ``(equity, ofz, deposit)`` curves on their committed date axes.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        window_end = date.fromisoformat(str(raw["window"]["end"]))
        legs = raw["legs"]
        equity = _rehydrate_leg(legs[_SNAPSHOT_LEG_KEYS[0]], window_end=window_end)
        ofz = _rehydrate_leg(legs[_SNAPSHOT_LEG_KEYS[1]], window_end=window_end)
        deposit = _rehydrate_leg(legs[_SNAPSHOT_LEG_KEYS[2]], window_end=window_end)
    except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        msg = f"allocation-gate snapshot missing/corrupt at {path}: {exc}"
        raise ConfigurationError(msg) from exc  # NO fallback to synthetic data (D-05)
    # IN-02 (defense-in-depth): reject a window.end that post-dates the binding clamp. The
    # per-bar clamp already rejects any bar > _BINDING_END, but a mis-stamped window.end field
    # itself was never checked — surface it explicitly rather than relying on the bar clamp.
    if window_end > _BINDING_END:
        msg = (
            f"allocation-gate snapshot window.end {window_end.isoformat()} post-dates the "
            f"binding clamp {_BINDING_END.isoformat()} at {path} (mis-stamped window)"
        )
        raise ConfigurationError(msg)
    # WR-01 (R-3): the whole gate runs on ONE basis and regime_split keys off only the deposit
    # leg's dates, so the three legs MUST share one identical, non-empty date axis. Enforce the
    # documented one-basis invariant fail-closed (the committed fixture always shares it).
    axes = ([d for d, _ in equity], [d for d, _ in ofz], [d for d, _ in deposit])
    if not all(axis == axes[0] and axis for axis in axes):
        msg = f"allocation-gate snapshot legs do not share one date axis (R-3) at {path}"
        raise ConfigurationError(msg)
    return equity, ofz, deposit


def regime_split(dates: list[date]) -> dict[str, tuple[date, date]]:
    """Partition a date window at :data:`REGIME_SPLIT_BOUNDARY` (V-9 / D-09 / R-6).

    The HEADLINE regime report (D-09): a documented high-rate/plateau vs early-cut date
    split at 2025-06-06 — the VERIFIED first real 2025 CBR cut (R-C). Returns:

    - ``{"high_rate": (start, 2025-06-05), "early_cut": (2025-06-06, end)}`` for a window
      spanning the boundary;
    - ``{"high_rate": (start, end)}`` (single regime) when the whole window ends before the
      boundary — a single high-rate plateau (Pitfall 6).

    This date split (NOT the ``classify_regime`` cross-check) is the binding-readable
    headline because it matches Pitfall 6's wording literally and is trivial to verify.

    Edge guard (WR-02): ``dates[0]``/``dates[-1]`` index the window — an empty list would
    raise an opaque ``IndexError``. The real cert path always passes a >= 300-bar window
    (offline ``_N_BARS`` / ``--live`` ``_N_LIVE_MIN_BARS``), so this never fires there; it
    defends the exported surface against a degenerate empty window.
    """
    if not dates:
        msg = "regime_split requires a non-empty date window"
        raise ValueError(msg)
    start, end = dates[0], dates[-1]
    if end < REGIME_SPLIT_BOUNDARY:
        # single high-rate plateau (no easing sub-window)
        return {_HIGH_RATE_UNIT_KEY: (start, end)}
    day_before = REGIME_SPLIT_BOUNDARY - timedelta(days=1)
    return {
        _HIGH_RATE_UNIT_KEY: (start, day_before),
        _EASING_UNIT_KEY: (REGIME_SPLIT_BOUNDARY, end),
    }


def _slice_leg(
    leg: list[tuple[date, Decimal]], start: date, end: date
) -> list[tuple[date, Decimal]]:
    """Inclusive date slice of an ALREADY-NETTED ``(date, Decimal)`` leg. No re-net (T-75-02).

    A PURE date filter — it never constructs or invokes a :class:`YtdTaxAccumulator` and never
    calls :func:`net_fixed_income_legs_interleaved`. The snapshot legs were netted ONCE at
    creation time (Phase 74 CR-01 single-pass), so slicing the already-net levels keeps the
    CR-01 year-boundary bug structurally unreachable: ``_slice_leg`` returns the date-matched
    full-window values byte-for-byte, no NDFL delta.
    """
    return [(d, v) for d, v in leg if start <= d <= end]


def regime_verdicts(
    deposit_net: list[tuple[date, Decimal]],
    ofz_net: list[tuple[date, Decimal]],
    equity_net: list[tuple[date, Decimal]],
    profiles: dict[RiskProfile, AllocationProfile],
    profile_order: tuple[RiskProfile, ...],
) -> dict[str, dict[str, object]]:
    """Per-regime BINDING verdicts on the ALREADY-NETTED legs (REGIME-02 / D-01).

    Slices each leg at :data:`REGIME_SPLIT_BOUNDARY` (via :func:`regime_split`) and runs the
    EXISTING :func:`build_naive_legs` -> :func:`gate_with_autotighten` per profile on each
    sub-window — the REAL frozen path (anti-hollow), never a pre-baked literal. NO re-net (the
    slice path never touches :class:`YtdTaxAccumulator`), so the CR-01 year-boundary bug is
    structurally unreachable. Each sub-window recomputes its OWN best-of-three naive bar
    (apples-to-apples within regime, D-01 derived).

    :func:`regime_split` emits ``high_rate`` / ``early_cut``; ``early_cut`` IS the easing
    binding unit (the post-cut segment). On a window ending before the boundary only
    ``high_rate`` is present (single-regime edge) — the easing unit is then absent.

    ``profiles`` and ``profile_order`` are PARAMETERS (the CLI injects them) so this module
    gains no config dependency and the orchestrator stays the only profile owner. Returns a
    dict keyed by regime unit; each value is a per-profile verdict dict with the
    non-serializable ``result`` / ``frozen_weights`` carriers stripped (mirroring the CLI loop).
    """
    regime = regime_split([d for d, _ in deposit_net])
    out: dict[str, dict[str, object]] = {}
    for unit, (w_start, w_end) in regime.items():
        sub_dep = _slice_leg(deposit_net, w_start, w_end)
        sub_ofz = _slice_leg(ofz_net, w_start, w_end)
        sub_eq = _slice_leg(equity_net, w_start, w_end)
        # Recompute the slice's OWN best-of-three naive bar (apples-to-apples within regime).
        naives = build_naive_legs(sub_dep, sub_ofz, sub_eq)
        naive_sharpes = [n.sharpe for n in naives.values()]
        naive_sortinos = [
            excess_sortino_from_equity([float(v) for v in n.merged_equity_curve])
            for n in naives.values()
        ]
        per_profile: dict[str, object] = {}
        for pk in profile_order:
            p = profiles[pk]
            v = gate_with_autotighten(
                profile_key=pk,
                base_weights=p.weights,
                cap_fraction=p.max_drawdown_pct,
                deposit_curve=sub_dep,
                ofz_pk_curve=sub_ofz,
                equity_curve=sub_eq,
                naive_sharpes=naive_sharpes,
                naive_sortinos=naive_sortinos,
            )
            v.pop("result", None)  # non-serializable AllocationResult carrier
            v.pop("frozen_weights", None)  # weight dict not JSON-key-safe; verdict suffices
            per_profile[pk.value] = v
        out[unit] = per_profile
    return out


def derive_escalation(high_rate_verdict: str, easing_verdict: str) -> dict[str, object]:
    """Derive the escalation flag + N=1 caveat from the REAL per-regime verdicts (D-03a).

    Anti-hollow: ``escalation == _ESCALATION_DEPOSIT_ANCHOR`` ONLY when BOTH the ``high_rate``
    AND the ``easing`` unit verdicts are :data:`_HARD_FAIL`; otherwise ``None`` (if easing
    somehow PASSed, NO deposit-anchor escalation is recorded — the recorded decision tracks the
    cert, never a pre-baked literal). ``n1_caveat`` is ALWAYS-ON separate metadata (D-04) for the
    single-cycle easing read — NEVER fused into a verdict-status string and NOT a threshold change.

    The verdict strings are the terminal outputs of :func:`gate_with_autotighten`
    (``"PASS"`` / ``"PASS_AFTER_TIGHTEN"`` / ``"HARD_FAIL"``); callers pass the per-regime unit
    verdicts (see :func:`regime_verdicts`).
    """
    both_hard_fail = high_rate_verdict == _HARD_FAIL and easing_verdict == _HARD_FAIL
    return {
        "escalation": _ESCALATION_DEPOSIT_ANCHOR if both_hard_fail else None,
        "n1_caveat": True,
    }


# Defensive defaults for the classify_regime cross-check before the calendar's first
# meeting (snapshot fields can be None). NEUTRAL-leaning so a missing snapshot never
# fabricates a HAWKISH/DOVISH headline — the date split is the headline, this is secondary.
_DEFAULT_KEY_RATE_PP = Decimal("21.0")
_DEFAULT_CPI_YOY = Decimal("0.0")
_DEFAULT_LAST_DECISION = "hold"


def classify_regime_for_date(d: date) -> CBRRegime:
    """Thin per-date wrapper reusing :func:`classify_regime` VERBATIM (D-09 cross-check).

    Assembles the ``(key_rate, ruonia_7d_avg, cpi_yoy, last_cbr_decision)`` tuple from
    :meth:`MacroContextProvider.get_snapshot` (look-ahead-safe, as-of *d*) and feeds it to
    the FROZEN :func:`finalayze.strategies.bond_duration_rotation.classify_regime`. The
    wrapper is the small new code; ``classify_regime`` is reused, not re-implemented — this
    is the SECONDARY cross-check (the headline is the date split in :func:`regime_split`).

    Missing snapshot fields (before the calendar's first meeting) fall back to NEUTRAL-safe
    defaults so a gap never fabricates a regime label.
    """
    snap = MacroContextProvider().get_snapshot(d)
    key_rate = snap.key_rate if snap.key_rate is not None else _DEFAULT_KEY_RATE_PP
    ruonia = snap.ruonia_7d_avg if snap.ruonia_7d_avg is not None else key_rate
    cpi = snap.cpi_yoy if snap.cpi_yoy is not None else _DEFAULT_CPI_YOY
    last = snap.last_cbr_decision if snap.last_cbr_decision is not None else _DEFAULT_LAST_DECISION
    return classify_regime(key_rate, ruonia, cpi, last)


def render_json(
    per_profile_verdicts: dict[str, object],
    naive_metrics: dict[str, object],
    regime: dict[str, tuple[date, date]],
    *,
    git_sha: str,
    per_regime: dict[str, dict[str, object]] | None = None,
    escalation: str | None = None,
    n1_caveat: bool = False,
) -> dict[str, object]:
    """Assemble all gate metrics into a JSON-serializable sidecar (D-11).

    Mirrors the ``scripts/phase72_allocation_ab.py`` ``summary.json`` shape: a flat dict of
    per-profile / per-naive metric blocks plus the ``git_sha`` and the regime split
    (rendered as ISO date pairs). The synthetic cut-path block was retired in Phase 74
    (D-07) — the real easing sub-window is now the post-boundary segment of
    ``regime_split``. Decimal values are stringified so the dict is ``json.dumps``-able
    without a custom encoder. The machine-readable feed a future dashboard (deferred)
    consumes.

    Phase 75 (REGIME-02/05) ADDS three keys ADDITIVELY (every existing key preserved): the
    ``per_regime`` binding-verdict block (:func:`regime_verdicts` output), the derived
    ``escalation`` flag (:func:`derive_escalation`), and the always-on ``n1_caveat`` metadata
    flag (D-04). The keyword-only params default to empty/None/False so existing callers keep
    working unchanged.
    """
    return {
        "git_sha": git_sha,
        "per_profile": per_profile_verdicts,
        "naive": naive_metrics,
        "regime_split": {k: [v[0].isoformat(), v[1].isoformat()] for k, v in regime.items()},
        "high_rate_caveat": _HIGH_RATE_CAVEAT,
        # Phase 75 additive decision block (REGIME-02/05).
        "per_regime": per_regime if per_regime is not None else {},
        "escalation": escalation,
        "n1_caveat": n1_caveat,
    }


def _fmt(value: object) -> str:
    """Render a metric cell — floats to 4dp, everything else via ``str`` (report-only)."""
    return f"{value:.4f}" if isinstance(value, float) else str(value)


def render_report(payload: dict[str, object]) -> str:
    """Render the human-readable Markdown gate report (D-11).

    Sections: a header, the per-profile verdict table (profile | Sharpe vs best-naive |
    Sortino vs best-naive | realized MaxDD vs cap | mean WF-fold Sharpe | verdict), the
    naive-comparison block (prefixed with the framing-only :data:`_RISK_FREE_BAR_NOTE`
    explaining why a near-vol-free risk-free leg inflates the naive Sharpe/Sortino bar in a
    high-rate regime), the regime split block, the REAL easing sub-window block (the
    post-`REGIME_SPLIT_BOUNDARY` segment — the evidence-based cut scenario that replaced the
    retired synthetic cut-path, D-07), the Phase-75 Per-Regime Verdict section (the binding
    ``high_rate`` / ``easing`` units + the derived escalation line + the verbatim N=1 caveat,
    REGIME-02/05 / D-01/D-04), and the mandatory honesty caveat line
    (:data:`_HIGH_RATE_CAVEAT`, verbatim).

    The BINDING number is the full-window metric (``verdict_for_profile``); the mean
    WF-fold Sharpe (R-1 / D-02) is REPORTED alongside so GATE-01's "OOS via walk-forward"
    is visibly honored without being binding. ``payload`` is the :func:`render_json` dict
    (or a compatible shape).
    """
    per_profile = cast("dict[str, object]", payload.get("per_profile", {}))
    naive = cast("dict[str, object]", payload.get("naive", {}))
    regime = cast("dict[str, object]", payload.get("regime_split", {}))
    per_regime = cast("dict[str, object]", payload.get("per_regime", {}))

    # Phase 75 (WR-01 / WR-02): the verbatim N=1 easing caveat is rendered ONLY when the easing
    # unit is actually present in the per-regime block AND the machine-readable n1_caveat flag is
    # set. Gating on the flag keeps the human report and the JSON sidecar a single source of truth
    # (WR-01); gating on easing-presence stops the report from asserting an "easing read" on a
    # single-regime (no-easing) window where it just printed "easing sub-window: none" (WR-02).
    # D-04 still holds: the caveat is SEPARATE metadata, never fused into a verdict-status string,
    # and on the normal both-regime cert (easing present + flag True) it still renders EXACTLY once.
    easing_present = _EASING_UNIT_KEY in per_regime
    n1_caveat_on = bool(payload.get("n1_caveat", False))
    render_n1_caveat = easing_present and n1_caveat_on

    lines: list[str] = [
        "# Allocation Gate Report (GATE-01/02/03)",
        "",
        f"git_sha: `{payload.get('git_sha', 'unknown')}`",
        "",
        "## Per-Profile Verdict (binding = full-window; WF mean reported-only)",
        "",
        "| Profile | Sharpe | Best-naive Sharpe | Sortino | Best-naive Sortino "
        "| Realized MaxDD | Cap | Mean WF Sharpe | Verdict |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, raw in per_profile.items():
        v = cast("dict[str, object]", raw)
        lines.append(
            f"| {name} | {_fmt(v.get('sharpe'))} | {_fmt(v.get('best_naive_sharpe'))} "
            f"| {_fmt(v.get('sortino'))} | {_fmt(v.get('best_naive_sortino'))} "
            f"| {_fmt(v.get('realized_maxdd_frac'))} | {_fmt(v.get('cap_frac'))} "
            f"| {_fmt(v.get('mean_wf_sharpe'))} | {_fmt(v.get('verdict', v.get('pass')))} |"
        )

    # Phase 75: per-regime binding-verdict rows (REGIME-02 / D-01). regime_verdicts emits
    # "high_rate" / "early_cut"; "early_cut" IS the easing binding unit — render it labeled
    # "easing". Each row reuses the per-profile _fmt(...) cell pattern (no recompute here).
    per_regime_rows: list[str] = []
    for unit_key, unit_raw in per_regime.items():
        unit_label = _EASING_UNIT_LABEL if unit_key == _EASING_UNIT_KEY else unit_key
        unit_profiles = cast("dict[str, object]", unit_raw)
        for prof_name, prof_raw in unit_profiles.items():
            pv = cast("dict[str, object]", prof_raw)
            per_regime_rows.append(
                f"| {unit_label} | {prof_name} | {_fmt(pv.get('sharpe'))} "
                f"| {_fmt(pv.get('best_naive_sharpe'))} | {_fmt(pv.get('sortino'))} "
                f"| {_fmt(pv.get('best_naive_sortino'))} | {_fmt(pv.get('realized_maxdd_frac'))} "
                f"| {_fmt(pv.get('cap_frac'))} | {_fmt(pv.get('verdict', pv.get('pass')))} |"
            )

    lines += [
        "",
        "## Naive Benchmark Comparison (best-of-three is the bar, D-04)",
        "",
        f"> {_RISK_FREE_BAR_NOTE}",
        "",
        *(f"- `{k}`: {_fmt(val)}" for k, val in naive.items()),
        "",
        "## Regime Split (headline = documented date split, D-09 / R-6)",
        "",
        *(f"- `{k}`: {val}" for k, val in regime.items()),
        "",
        "## Real Easing Sub-Window (post-REGIME_SPLIT_BOUNDARY, evidence-based, D-07)",
        "",
        "_The synthetic framing cut-path is RETIRED (D-07). The real binding window now "
        "CONTAINS the real easing (high-rate plateau → the verified 2025 CBR cuts from "
        "2025-06-06), so the cut scenario is the REAL easing sub-window below — sourced from "
        "the regime split, not a synthetic glide._",
        "",
        *(
            [f"- easing sub-window: `{regime['early_cut']}`"]
            if "early_cut" in regime
            else ["- easing sub-window: none (window ends before the first real cut)"]
        ),
        "",
        "## Per-Regime Verdict (binding: high_rate AND easing, D-01)",
        "",
        "| Regime | Profile | Sharpe | Best-naive Sharpe | Sortino | Best-naive Sortino "
        "| Realized MaxDD | Cap | Verdict |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        *per_regime_rows,
        "",
        f"- escalation: `{payload.get('escalation')}`",
        "",
        *(
            ["## N=1 Caveat (easing single-cycle, D-04)", "", f"> {_N1_CAVEAT}", ""]
            if render_n1_caveat
            else []
        ),
        "## Honesty Caveat (Pitfall 6 / D-08)",
        "",
        f"> {_HIGH_RATE_CAVEAT}",
        "",
    ]
    return "\n".join(lines)
