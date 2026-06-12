"""RED scaffold: GATE-01/02/03 allocation-gate harness (Phase 73 Wave-0).

This is the CLAUDE.md #2 TDD invariant -- the failing-first binding contract for
the entire ``finalayze.backtest.allocation_gate`` module. It pins V-1..V-9 from
``73-VALIDATION.md`` BEFORE any implementation, so the implementer (Plans 02/03/04)
cannot drift from the locked decisions. Every test below MUST FAIL at collection
time because ``src/finalayze/backtest/allocation_gate.py`` does not exist yet --
that absence IS the RED state.

The contract pinned here:
- V-1 (TRAP A): ``realized_dd_fraction`` reconciles the percent-vs-fraction unit
  trap -- ``AllocationResult.max_drawdown_pct`` is a PERCENT (e.g. ``8.0``) but caps
  are FRACTIONS (``Decimal("0.08")``); the gate must never compare ``8.0 <= 0.08``.
- V-2 (TRAP B): ``excess_sortino_from_equity`` returns the TRUE (possibly negative)
  Sortino on a losing curve -- it does NOT clamp ``mean_excess <= 0`` to 0 the way
  the snapshot-based ``performance.sortino_ratio`` does (which would corrupt the
  strict ``>= best_naive`` comparison).
- V-3 (D-01): the conjunctive PASS rule -- pass IFF Sharpe >= best-naive Sharpe AND
  Sortino >= best-naive Sortino AND realized MaxDD <= profile cap (>= inclusive).
- V-4 (D-04): the naive bar is ``max()`` over all THREE naive legs.
- V-5 (D-03): the auto-tighten execute path freezes after a 5pp eq->deposit step
  and returns HARD_FAIL when still breaching (no further widening).
- V-6 (R-3): ``build_naive_legs`` builds the three legs on the SAME basis via
  degenerate-profile injection -- the deposit leg is cost-free, the rebalanced
  60/30/10 leg charges MOEX_RETAIL_COSTS.
- V-7 (D-07): the cut-path lowers the risk-free legs while holding the MCFTR equity
  curve BYTE-IDENTICAL (framing-only, not fed into the binding verdict).
- V-8 (D-02): OOS walk-forward Sharpes are sliced from the merged curve via
  ``generate_wf_windows`` -- NO engine re-run.
- V-9 (D-09/R-6): the regime split boundary is 2025-07-25.

RED now: ``finalayze.backtest.allocation_gate`` (Plans 02/03/04) does not exist yet.
"""

from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal

from finalayze.backtest.allocation_gate import (  # noqa: E402 -- RED: module absent until Plans 02-04
    build_naive_legs,
    excess_sortino_from_equity,
    realized_dd_fraction,
    verdict_for_profile,
)
from finalayze.core.schemas import AssetClass, RiskProfile

# -- Constants (named -- no magic numbers, ruff PLR2004) ----------------------

# Profile MaxDD caps are FRACTIONS (mirror config/allocation_profiles.yaml).
_CAP_CONSERVATIVE = Decimal("0.08")
_CAP_BALANCED = Decimal("0.15")

# TRAP A: a known realized MaxDD expressed as a PERCENT and its fraction form.
_KNOWN_MAXDD_PCT = 12.0
_KNOWN_MAXDD_FRAC = Decimal("0.12")
_MAXDD_PASS_CONSERVATIVE_PCT = 6.0  # 0.06 -> under the 8% cap
_MAXDD_BALANCED_BOUNDARY_PCT = 15.0  # 0.15 -> exactly at the 15% cap (>= inclusive)

# Curve fixtures span ~2 years of daily bars.
_N_BARS = 504
_FIRST_BAR = date(2023, 1, 1)
_RUONIA_PCT = 15.0  # the RUONIA-excess annual risk-free rate

# Geometric curve parameters (deterministic, no network).
_EQUITY_BASE = Decimal(100)
_EQUITY_DAILY = Decimal("1.0008")  # rising equity leg
_DEPOSIT_BASE = Decimal(100)
_DEPOSIT_DAILY = Decimal("1.00055")  # flat-ish deposit accrual
_OFZ_BASE = Decimal(100)
_OFZ_DAILY = Decimal("1.0004")  # slowly-rising OFZ-PK carry leg
_LOSING_BASE = Decimal(100)
_LOSING_DAILY = Decimal("0.998")  # monotonically declining -> negative Sortino

# Sharpe/Sortino are FLOATS (per AllocationResult.sharpe); caps are Decimal.
# Best-naive bar = max over the three naive legs (V-4 / D-04).
_NAIVE_SHARPES = [0.10, 0.90, 0.40]
_NAIVE_SORTINOS = [0.20, 0.80, 0.30]
_BEST_NAIVE_SHARPE = 0.90  # max(_NAIVE_SHARPES)
_BEST_NAIVE_SORTINO = 0.80  # max(_NAIVE_SORTINOS)

# Alloc metrics that strictly clear the bar.
_ALLOC_SHARPE_PASS = 1.20
_ALLOC_SORTINO_PASS = 1.00
_ALLOC_MAXDD_PASS_PCT = 10.0  # 0.10 -> under the 0.15 balanced cap

# Alloc metrics that individually fail each of the three conditions.
_ALLOC_SHARPE_FAIL = 0.50  # below best-naive Sharpe
_ALLOC_SORTINO_FAIL = 0.40  # below best-naive Sortino
_ALLOC_MAXDD_FAIL_PCT = 20.0  # 0.20 -> over the 0.15 balanced cap

_ZERO = Decimal(0)


def _daily_index(first: date, n: int) -> list[date]:
    return [first + timedelta(days=i) for i in range(n)]


def _curve(base: Decimal, daily: Decimal, dates: list[date]) -> list[tuple[date, Decimal]]:
    """Deterministic geometric (date, Decimal) total-return curve."""
    return [(d, base * daily**i) for i, d in enumerate(dates)]


def test_verdict_conjunctive() -> None:
    """The PASS verdict is the conjunction of all three conditions (V-3 / D-01).

    pass IFF alloc Sharpe >= best-naive Sharpe AND alloc Sortino >= best-naive
    Sortino AND realized MaxDD fraction <= cap. Flipping ANY one condition flips
    the verdict to fail; the >= boundary (each metric exactly at its bar) passes.
    """
    base = verdict_for_profile(
        alloc_sharpe=_ALLOC_SHARPE_PASS,
        alloc_sortino=_ALLOC_SORTINO_PASS,
        alloc_max_drawdown_pct=_ALLOC_MAXDD_PASS_PCT,
        naive_sharpes=_NAIVE_SHARPES,
        naive_sortinos=_NAIVE_SORTINOS,
        cap_fraction=_CAP_BALANCED,
    )
    assert base["pass"] is True

    # Flip ONLY the Sharpe condition below the best naive -> fail.
    fail_sharpe = verdict_for_profile(
        alloc_sharpe=_ALLOC_SHARPE_FAIL,
        alloc_sortino=_ALLOC_SORTINO_PASS,
        alloc_max_drawdown_pct=_ALLOC_MAXDD_PASS_PCT,
        naive_sharpes=_NAIVE_SHARPES,
        naive_sortinos=_NAIVE_SORTINOS,
        cap_fraction=_CAP_BALANCED,
    )
    assert fail_sharpe["pass"] is False

    # Flip ONLY the Sortino condition below the best naive -> fail.
    fail_sortino = verdict_for_profile(
        alloc_sharpe=_ALLOC_SHARPE_PASS,
        alloc_sortino=_ALLOC_SORTINO_FAIL,
        alloc_max_drawdown_pct=_ALLOC_MAXDD_PASS_PCT,
        naive_sharpes=_NAIVE_SHARPES,
        naive_sortinos=_NAIVE_SORTINOS,
        cap_fraction=_CAP_BALANCED,
    )
    assert fail_sortino["pass"] is False

    # Flip ONLY the MaxDD condition above the cap -> fail.
    fail_maxdd = verdict_for_profile(
        alloc_sharpe=_ALLOC_SHARPE_PASS,
        alloc_sortino=_ALLOC_SORTINO_PASS,
        alloc_max_drawdown_pct=_ALLOC_MAXDD_FAIL_PCT,
        naive_sharpes=_NAIVE_SHARPES,
        naive_sortinos=_NAIVE_SORTINOS,
        cap_fraction=_CAP_BALANCED,
    )
    assert fail_maxdd["pass"] is False

    # Boundary: each metric EXACTLY at its bar (>= is inclusive) -> pass.
    boundary = verdict_for_profile(
        alloc_sharpe=_BEST_NAIVE_SHARPE,
        alloc_sortino=_BEST_NAIVE_SORTINO,
        alloc_max_drawdown_pct=_MAXDD_BALANCED_BOUNDARY_PCT,
        naive_sharpes=_NAIVE_SHARPES,
        naive_sortinos=_NAIVE_SORTINOS,
        cap_fraction=_CAP_BALANCED,
    )
    assert boundary["pass"] is True


def test_maxdd_unit_reconcile() -> None:
    """MaxDD percent reconciles to a fraction before the cap compare (V-1 / TRAP A).

    ``AllocationResult.max_drawdown_pct`` is a PERCENT (12.0), caps are FRACTIONS
    (0.08 / 0.15). The gate must divide by 100 -- never compare ``12.0 <= 0.08``.
    A 12% MaxDD FAILs an 8% cap and PASSes a 15% cap.
    """
    assert realized_dd_fraction(_KNOWN_MAXDD_PCT) == _KNOWN_MAXDD_FRAC

    fails_8pct = verdict_for_profile(
        alloc_sharpe=_ALLOC_SHARPE_PASS,
        alloc_sortino=_ALLOC_SORTINO_PASS,
        alloc_max_drawdown_pct=_KNOWN_MAXDD_PCT,
        naive_sharpes=_NAIVE_SHARPES,
        naive_sortinos=_NAIVE_SORTINOS,
        cap_fraction=_CAP_CONSERVATIVE,
    )
    assert fails_8pct["pass"] is False  # 0.12 > 0.08

    passes_15pct = verdict_for_profile(
        alloc_sharpe=_ALLOC_SHARPE_PASS,
        alloc_sortino=_ALLOC_SORTINO_PASS,
        alloc_max_drawdown_pct=_KNOWN_MAXDD_PCT,
        naive_sharpes=_NAIVE_SHARPES,
        naive_sortinos=_NAIVE_SORTINOS,
        cap_fraction=_CAP_BALANCED,
    )
    assert passes_15pct["pass"] is True  # 0.12 <= 0.15


def test_sortino_negative_not_clamped() -> None:
    """A losing curve yields a TRUE negative Sortino -- never clamped to 0 (V-2 / TRAP B).

    The old ``performance.sortino_ratio`` clamps ``mean_excess <= 0`` to ``Decimal(0)``;
    the gate's curve-based helper must return the genuine negative value so the strict
    ``>= best_naive`` comparison stays honest.
    """
    dates = _daily_index(_FIRST_BAR, _N_BARS)
    losing = [float(v) for _, v in _curve(_LOSING_BASE, _LOSING_DAILY, dates)]
    sortino = excess_sortino_from_equity(losing, risk_free_annual_pct=_RUONIA_PCT)
    assert sortino < 0.0
    assert sortino != 0.0


def test_best_naive_max_over_three() -> None:
    """The naive bar is max() over all three naive legs (V-4 / D-04).

    With naive Sharpes [0.10, 0.90, 0.40] the bar an alloc must clear is 0.90; an
    alloc Sharpe between the lowest two naives (e.g. 0.50) does NOT pass even though
    it beats two of the three.
    """
    middling = verdict_for_profile(
        alloc_sharpe=_ALLOC_SHARPE_FAIL,  # 0.50 -- beats 0.10 and 0.40 but not 0.90
        alloc_sortino=_ALLOC_SORTINO_PASS,
        alloc_max_drawdown_pct=_ALLOC_MAXDD_PASS_PCT,
        naive_sharpes=_NAIVE_SHARPES,
        naive_sortinos=_NAIVE_SORTINOS,
        cap_fraction=_CAP_BALANCED,
    )
    assert middling["pass"] is False

    clears_bar = verdict_for_profile(
        alloc_sharpe=_BEST_NAIVE_SHARPE,  # exactly the max naive -> clears (>=)
        alloc_sortino=_ALLOC_SORTINO_PASS,
        alloc_max_drawdown_pct=_ALLOC_MAXDD_PASS_PCT,
        naive_sharpes=_NAIVE_SHARPES,
        naive_sortinos=_NAIVE_SORTINOS,
        cap_fraction=_CAP_BALANCED,
    )
    assert clears_bar["pass"] is True


def test_naive_legs_same_basis() -> None:
    """The three naive legs share one basis; deposit is cost-free, 60/30/10 is not (V-6 / R-3).

    ``build_naive_legs`` builds {deposit_100, equity_100, static_60_30_10} via
    degenerate-profile injection on the SAME curves. The 100% deposit leg never
    trades -> ``rebalance_cost == 0``; the quarterly-rebalanced 60/30/10 leg trades
    eq+OFZ -> ``rebalance_cost > 0`` (MOEX_RETAIL_COSTS round-trip).
    """
    dates = _daily_index(_FIRST_BAR, _N_BARS)
    deposit_curve = _curve(_DEPOSIT_BASE, _DEPOSIT_DAILY, dates)
    ofz_pk_curve = _curve(_OFZ_BASE, _OFZ_DAILY, dates)
    equity_curve = _curve(_EQUITY_BASE, _EQUITY_DAILY, dates)

    legs = build_naive_legs(deposit_curve, ofz_pk_curve, equity_curve)

    assert set(legs) == {"deposit_100", "equity_100", "static_60_30_10"}
    assert legs["deposit_100"].rebalance_cost == _ZERO
    assert legs["static_60_30_10"].rebalance_cost > _ZERO
