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

The auto-tighten (V-5), OOS walk-forward (V-8), cut-path (V-7) and regime-split
(V-9) primitives are implemented by Plans 03/04; their public names are declared
here so the module imports cleanly, but they raise ``NotImplementedError`` until
those plans land.
"""

from __future__ import annotations

import math
import statistics
from decimal import Decimal
from typing import TYPE_CHECKING, cast

from finalayze.backtest.bond_walk_forward import (
    _compute_excess_sharpe_from_equity,
    generate_wf_windows,
)
from finalayze.core.allocation import tighten
from finalayze.core.schemas import AllocationProfile, AssetClass, RiskProfile

if TYPE_CHECKING:
    from datetime import date

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
    """
    best_naive_sharpe = max(naive_sharpes)
    best_naive_sortino = max(naive_sortinos)
    realized_frac = realized_dd_fraction(alloc_max_drawdown_pct)
    passes = (
        alloc_sharpe >= best_naive_sharpe
        and alloc_sortino >= best_naive_sortino
        and realized_frac <= cap_fraction
    )
    return {
        "pass": passes,
        "sharpe": alloc_sharpe,
        "best_naive_sharpe": best_naive_sharpe,
        "sortino": alloc_sortino,
        "best_naive_sortino": best_naive_sortino,
        "realized_maxdd_frac": float(realized_frac),
        "cap_frac": float(cap_fraction),
    }


# ── Deferred public surface (Plans 03/04) ────────────────────────────────────
# These names are part of the V-1..V-9 import contract pinned by Plan 01, so they
# MUST exist for the test module to import. Their behaviour lands in Plans 03
# (auto-tighten V-5, OOS walk-forward V-8) and 04 (cut-path V-7, regime-split V-9);
# until then they raise so the four deferred tests FAIL (not error at collection).

# CUT_GLIDE: the synthetic declining-rate meeting calendar (V-7 / D-07, Plan 04).
CUT_GLIDE: object | None = None

# REGIME_SPLIT_BOUNDARY: the early-cut regime boundary (V-9 / D-09 / R-6, Plan 04).
# Pinned to date(2025, 7, 25); finalised in Plan 04.
REGIME_SPLIT_BOUNDARY: date | None = None


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


def run_cut_path(*args: object, **kwargs: object) -> AllocationResult:
    """Synthetic rate-cut framing path (V-7 / D-07) — implemented in Plan 04."""
    raise NotImplementedError("run_cut_path lands in Plan 73-04")


def regime_split(*args: object, **kwargs: object) -> dict[str, tuple[date, date]]:
    """Partition a date window at the regime boundary (V-9 / D-09) — Plan 04."""
    raise NotImplementedError("regime_split lands in Plan 73-04")
