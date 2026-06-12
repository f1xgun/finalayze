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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import date

    from finalayze.orchestration.allocation import AllocationResult

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


def build_naive_legs(*args: object, **kwargs: object) -> dict[str, AllocationResult]:
    """Three degenerate naive legs on one basis (V-6 / R-3) — Task 2 of this plan."""
    raise NotImplementedError("build_naive_legs lands in Task 2 of Plan 73-02")


def verdict_for_profile(*args: object, **kwargs: object) -> dict[str, object]:
    """Conjunctive per-profile PASS verdict (V-3/V-4 / D-01/D-04) — Task 2 of this plan."""
    raise NotImplementedError("verdict_for_profile lands in Task 2 of Plan 73-02")


def gate_with_autotighten(*args: object, **kwargs: object) -> dict[str, object]:
    """Auto-tighten execute path (V-5 / D-03) — implemented in Plan 03."""
    raise NotImplementedError("gate_with_autotighten lands in Plan 73-03")


def oos_wf_sharpes(*args: object, **kwargs: object) -> list[float]:
    """OOS walk-forward Sharpes sliced from the merged curve (V-8 / D-02) — Plan 03."""
    raise NotImplementedError("oos_wf_sharpes lands in Plan 73-03")


def run_cut_path(*args: object, **kwargs: object) -> AllocationResult:
    """Synthetic rate-cut framing path (V-7 / D-07) — implemented in Plan 04."""
    raise NotImplementedError("run_cut_path lands in Plan 73-04")


def regime_split(*args: object, **kwargs: object) -> dict[str, tuple[date, date]]:
    """Partition a date window at the regime boundary (V-9 / D-09) — Plan 04."""
    raise NotImplementedError("regime_split lands in Plan 73-04")
