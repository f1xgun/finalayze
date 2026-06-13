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
- V-7 (D-07): RETIRED in Phase 74 -- the synthetic framing cut-path is deleted; the
  real easing sub-window (post-boundary) is now the evidence-based cut scenario.
- V-8 (D-02): OOS walk-forward Sharpes are sliced from the merged curve via
  ``generate_wf_windows`` -- NO engine re-run.
- V-9 (D-09/R-6): the regime split boundary is 2025-06-06 (Phase 74 R-C shifted it
  from 2025-07-25 to the verified first real 2025 CBR cut).

Phase 74 (Plan 02) reproduces this contract minus the retired V-7, plus the new
net-of-NDFL / snapshot / no-drift tests.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from finalayze.backtest.allocation_gate import (
    _LARGE_SORTINO_SENTINEL,
    REGIME_SPLIT_BOUNDARY,
    _load_gate_snapshot,
    accrue_real_risk_free_leg,
    build_naive_legs,
    excess_sortino_from_equity,
    gate_with_autotighten,
    net_fixed_income_legs_interleaved,
    net_index_returns,
    oos_wf_sharpes,
    realized_dd_fraction,
    regime_split,
    render_json,
    render_report,
    verdict_for_profile,
)
from finalayze.backtest.bond_walk_forward import generate_wf_windows
from finalayze.core.allocation import tighten  # used only to derive the EXPECTED frozen vector
from finalayze.core.exceptions import ConfigurationError
from finalayze.core.ndfl import YtdTaxAccumulator
from finalayze.core.schemas import AssetClass, RiskProfile
from finalayze.orchestration.allocation import AllocationOrchestrator, AllocationResult

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

# -- Task 2 constants (V-5, V-7, V-8, V-9) ------------------------------------

# V-9 / D-09 / R-6: the early-cut regime boundary. Phase 74 (R-C) shifts it to the
# VERIFIED first real 2025 CBR cut (2025-06-06 -> 20.00, was wrongly 2025-07-25).
_BOUNDARY = date(2025, 6, 6)
_HIGH_RATE_LAST = date(2025, 6, 5)  # last day of the high-rate window

# V-8 / D-02: the WF window cadence the gate MUST pass explicitly to
# generate_wf_windows (train/test/step months).
_WF_TRAIN_M = 12
_WF_TEST_M = 6
_WF_STEP_M = 3
# A >= 4-year daily window so generate_wf_windows yields >= 3 folds.
_WF_FIRST_BAR = date(2021, 1, 1)
_WF_N_BARS = 4 * 365 + 1  # ~4 years of daily bars (inclusive of a leap day)
# A SPARSE axis (IN-04b skip path): ~70-day spacing over a long span so the 6-month test
# slices carry only 1-2 bars and SOME WF windows are skipped (folds < windows, but > 0).
_WF_SPARSE_FIRST = date(2021, 1, 1)
_WF_SPARSE_SPACING_DAYS = 70  # > a 6-month/12-step cadence -> some test slices < _MIN_RETURNS+1
_WF_SPARSE_BARS = 40  # ~7.7y at 70-day spacing -> many windows, a fraction skipped

# V-5 / D-03: a growth-like cap-breaching base vector. A static breach drains
# equity 5pp/step into deposit until equity clamps at 0 (the tighten terminal
# state): deposit -> base_deposit + base_equity, ofz_pk fixed, equity -> 0.
_GROWTHISH_DEPOSIT_W = Decimal("0.20")
_GROWTHISH_OFZ_W = Decimal("0.25")
_GROWTHISH_EQUITY_W = Decimal("0.55")
_GROWTH_BASE_WEIGHTS = {
    AssetClass.DEPOSIT: _GROWTHISH_DEPOSIT_W,
    AssetClass.OFZ_PK: _GROWTHISH_OFZ_W,
    AssetClass.EQUITY: _GROWTHISH_EQUITY_W,
}
# A realized MaxDD (percent) that breaches even after equity drains to 0.
_BREACH_MAXDD_PCT = 30.0  # 0.30 -> over any sane cap -> HARD_FAIL
_HARD_FAIL = "HARD_FAIL"


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


def test_sortino_sentinel_vs_sentinel_does_not_auto_pass() -> None:
    """A zero-downside candidate does NOT pass the Sortino leg by sentinel equality (WR-02).

    ``excess_sortino_from_equity`` returns the fixed ``_LARGE_SORTINO_SENTINEL`` for a
    zero-downside (monotone-up) curve. If a candidate leg is ALSO zero-downside, the naive
    comparison ``alloc_sortino >= best_naive_sortino`` would be ``sentinel >= sentinel`` ->
    ``True``, satisfying the Sortino condition by sentinel EQUALITY rather than a real
    risk-adjusted measurement. The contract: sentinel-vs-sentinel is UNDEFINED -> it must NOT
    auto-pass the Sortino leg (treat it as a fail, not an automatic pass).

    This guards a future caller whose candidate is itself near-zero-downside. It does NOT move
    this cert's HARD_FAIL: any equity-holding allocation has downside, so ``alloc_sortino``
    never hits the sentinel and a real Sortino value still compares normally (pinned by the
    sibling conjunctive tests, whose verdicts are unchanged).
    """
    sentinel_vs_sentinel = verdict_for_profile(
        alloc_sharpe=_ALLOC_SHARPE_PASS,
        alloc_sortino=_LARGE_SORTINO_SENTINEL,
        alloc_max_drawdown_pct=_ALLOC_MAXDD_PASS_PCT,
        naive_sharpes=_NAIVE_SHARPES,
        naive_sortinos=[*_NAIVE_SORTINOS, _LARGE_SORTINO_SENTINEL],  # a zero-downside naive leg
        cap_fraction=_CAP_BALANCED,
    )
    # The sentinel-vs-sentinel Sortino case must NOT auto-pass (undefined, not a real win).
    assert sentinel_vs_sentinel["pass"] is False

    # A REAL (finite) candidate Sortino still clears a finite best-naive bar normally -- the
    # sentinel guard only fires when BOTH sides are the sentinel.
    finite = verdict_for_profile(
        alloc_sharpe=_ALLOC_SHARPE_PASS,
        alloc_sortino=_ALLOC_SORTINO_PASS,
        alloc_max_drawdown_pct=_ALLOC_MAXDD_PASS_PCT,
        naive_sharpes=_NAIVE_SHARPES,
        naive_sortinos=_NAIVE_SORTINOS,
        cap_fraction=_CAP_BALANCED,
    )
    assert finite["pass"] is True


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


def test_equity_leg_basis_identical() -> None:
    """The ``equity_100`` naive leg IS the MCFTR series fed in -- Pitfall 1 guard (GATE-01).

    The headline dead trap (Pitfall 1 / Pitfall C) is measuring the equity benchmark on
    price-IMOEX while the allocator's equity sleeve accrues dividends. This pins that the
    degenerate ``equity_100`` leg's equity-contribution curve is the SAME MCFTR series the
    allocator's equity sleeve uses (within forward-fill tolerance) -- no basis mismatch.
    """
    dates = _daily_index(_FIRST_BAR, _N_BARS)
    deposit_curve = _curve(_DEPOSIT_BASE, _DEPOSIT_DAILY, dates)
    ofz_pk_curve = _curve(_OFZ_BASE, _OFZ_DAILY, dates)
    equity_curve = _curve(_EQUITY_BASE, _EQUITY_DAILY, dates)

    legs = build_naive_legs(deposit_curve, ofz_pk_curve, equity_curve)

    # The equity_100 leg's equity contribution curve == the MCFTR series fed to the
    # allocator (same series -> Pitfall 1 solved). Dates are pre-aligned, so the
    # forward-fill is the identity and the comparison is exact Decimal equality.
    assert list(legs["equity_100"].equity_curve) == [v for _, v in equity_curve]


def test_autotighten_hard_fail() -> None:
    """A still-breaching profile freezes after the 5pp step and HARD_FAILs (V-5 / D-03).

    A cap-breaching base vector drains equity 5pp/step into deposit until equity
    clamps at 0 (the tighten terminal state). If the cap STILL breaches at that
    frozen vector the gate returns ``HARD_FAIL`` -- it does NOT widen further. The
    frozen vector matches the real ``tighten`` terminal: equity 0, deposit =
    base_deposit + base_equity, OFZ-PK unchanged.
    """
    dates = _daily_index(_FIRST_BAR, _N_BARS)
    deposit_curve = _curve(_DEPOSIT_BASE, _DEPOSIT_DAILY, dates)
    ofz_pk_curve = _curve(_OFZ_BASE, _OFZ_DAILY, dates)
    # A losing equity leg so the breach persists even after equity drains to 0.
    equity_curve = _curve(_LOSING_BASE, _LOSING_DAILY, dates)

    result = gate_with_autotighten(
        profile_key=RiskProfile.GROWTH,
        base_weights=_GROWTH_BASE_WEIGHTS,
        cap_fraction=_CAP_CONSERVATIVE,
        deposit_curve=deposit_curve,
        ofz_pk_curve=ofz_pk_curve,
        equity_curve=equity_curve,
        naive_sharpes=_NAIVE_SHARPES,
        naive_sortinos=_NAIVE_SORTINOS,
    )

    # Derive the EXPECTED frozen vector with the real tighten (the gate must call
    # it internally); pin equity -> 0 and deposit -> base_deposit + base_equity.
    expected_frozen = tighten(
        _GROWTH_BASE_WEIGHTS, realized_dd=_KNOWN_MAXDD_FRAC, cap=_CAP_CONSERVATIVE
    )
    assert result["verdict"] == _HARD_FAIL
    assert result["frozen_weights"][AssetClass.EQUITY] == _ZERO
    assert (
        result["frozen_weights"][AssetClass.DEPOSIT] == _GROWTHISH_DEPOSIT_W + _GROWTHISH_EQUITY_W
    )
    assert result["frozen_weights"][AssetClass.OFZ_PK] == _GROWTHISH_OFZ_W
    assert result["frozen_weights"] == expected_frozen


def test_wf_folds_no_rerun() -> None:
    """OOS WF Sharpes slice the merged curve with 12/6/3 -- no engine re-run (V-8 / D-02).

    ``oos_wf_sharpes`` slices the already-merged AllocationResult curve per
    ``generate_wf_windows(start, end, 12, 6, 3)`` and computes one excess Sharpe per
    fold. It NEVER constructs or runs a backtest engine.

    The GENERAL invariant (IN-04b) is ``len(folds) <= len(expected_windows)``: a window
    whose test slice has too few daily returns (< ``_MIN_RETURNS + 1`` bars) is SKIPPED. For
    DENSE daily data no window is ever skipped, so the equality ``==`` holds -- this dense
    case keeps the strict equality. The skip path is exercised separately by
    ``test_wf_folds_skip_sparse_windows`` so the inequality is documented, not masked.
    """
    dates = _daily_index(_WF_FIRST_BAR, _WF_N_BARS)
    orch = AllocationOrchestrator(risk_profile=RiskProfile.BALANCED)
    result: AllocationResult = orch.run(
        deposit_curve=_curve(_DEPOSIT_BASE, _DEPOSIT_DAILY, dates),
        ofz_pk_curve=_curve(_OFZ_BASE, _OFZ_DAILY, dates),
        equity_curve=_curve(_EQUITY_BASE, _EQUITY_DAILY, dates),
    )
    expected_windows = generate_wf_windows(
        result.dates[0], result.dates[-1], _WF_TRAIN_M, _WF_TEST_M, _WF_STEP_M
    )
    folds = oos_wf_sharpes(result)
    # General invariant: never MORE folds than windows (a sparse window is skipped).
    assert len(folds) <= len(expected_windows)
    # Dense daily data -> no window is skipped -> the strict equality still holds.
    assert len(folds) == len(expected_windows)
    assert all(isinstance(s, float) for s in folds)


def test_wf_folds_skip_sparse_windows() -> None:
    """A sparse window SKIPS folds whose test slice has too few bars (IN-04b skip path).

    ``oos_wf_sharpes`` skips any 6-month test slice with ``< _MIN_RETURNS + 1`` bars. On a
    SPARSE axis (bars ~70 days apart) some 6-month test slices carry only 1-2 bars and are
    skipped, so ``len(folds) < len(expected_windows)`` while still ``> 0``. This exercises
    the skip path the dense ``test_wf_folds_no_rerun`` cannot reach, so the general
    ``<=`` invariant is documented rather than masked by the dense fixture's shape.
    """
    dates = [
        _WF_SPARSE_FIRST + timedelta(days=i * _WF_SPARSE_SPACING_DAYS)
        for i in range(_WF_SPARSE_BARS)
    ]
    orch = AllocationOrchestrator(risk_profile=RiskProfile.BALANCED)
    result: AllocationResult = orch.run(
        deposit_curve=_curve(_DEPOSIT_BASE, _DEPOSIT_DAILY, dates),
        ofz_pk_curve=_curve(_OFZ_BASE, _OFZ_DAILY, dates),
        equity_curve=_curve(_EQUITY_BASE, _EQUITY_DAILY, dates),
    )
    expected_windows = generate_wf_windows(
        result.dates[0], result.dates[-1], _WF_TRAIN_M, _WF_TEST_M, _WF_STEP_M
    )
    folds = oos_wf_sharpes(result)
    # Some windows are SKIPPED (sparse test slices) -> strictly fewer folds than windows,
    # but the kept folds are still real -> > 0. This is the skip path the dense case misses.
    assert len(expected_windows) > 0
    assert 0 < len(folds) < len(expected_windows)
    assert all(isinstance(s, float) for s in folds)


def test_regime_split() -> None:
    """The early-cut regime boundary is 2025-06-06 (V-9 / D-09 / R-6, R-C-corrected).

    ``regime_split`` partitions a date window at 2025-06-06 -- the VERIFIED first real
    2025 CBR cut (21 -> 20). The high-rate window ends 2025-06-05, the early-cut window
    starts 2025-06-06. A window entirely before the boundary is a single high-rate regime.
    """
    assert REGIME_SPLIT_BOUNDARY == _BOUNDARY

    spanning = [
        _BOUNDARY - timedelta(days=30),
        _HIGH_RATE_LAST,
        _BOUNDARY,
        _BOUNDARY + timedelta(days=30),
    ]
    split = regime_split(spanning)
    assert split["high_rate"][1] == _HIGH_RATE_LAST
    assert split["early_cut"][0] == _BOUNDARY

    # A window entirely before the boundary -> the whole span is high_rate.
    pre_only = [_BOUNDARY - timedelta(days=60), _HIGH_RATE_LAST]
    pre_split = regime_split(pre_only)
    assert pre_split["high_rate"][0] == pre_only[0]
    assert pre_split["high_rate"][1] == _HIGH_RATE_LAST


# -- Live-path real-rate accrual (73-05 --live, operator-directed) -------------

# The real CBR key-rate window the operator's live cert spans: it BRACKETS the
# 2025-07-25 first-cut boundary so both regimes are exercised on REAL data.
_LIVE_BASE = Decimal(100_000)
_LIVE_FIRST = date(2024, 1, 2)
_LIVE_LAST = date(2025, 11, 27)
# The real CBR key rate (percentage points) on/before _LIVE_FIRST is 16.00%; the
# deposit leg accrues at key-1pp = 15% -> the first daily step is strictly > base.
_DEPOSIT_SPREAD_PP = Decimal("1.0")
_OFZ_SPREAD_PP = Decimal(0)
_DAYS_BETWEEN = 30


def test_accrue_real_risk_free_leg_grows_from_real_key_rate() -> None:
    """The live deposit/OFZ leg accrues from the REAL CBR key-rate path (operator D-10 override).

    ``accrue_real_risk_free_leg`` reads the REAL ``deposit_rate_as_of`` (the look-ahead-safe
    ``CBR_MEETINGS`` calendar) — the real realized calendar, not the retired synthetic
    framing path (D-07). On a real window opening in the
    16-21% high-rate regime the leg compounds upward monotonically (the rate is strictly
    positive across the whole window), opens at ``base``, and the no-spread (OFZ-PK
    floater) leg out-accrues the 1pp-spread (deposit) leg because it tracks the full key
    rate. The dates are preserved EXACTLY (one common axis, R-3).
    """
    dates = [_LIVE_FIRST + timedelta(days=i * _DAYS_BETWEEN) for i in range(24)]
    dates = [d for d in dates if d <= _LIVE_LAST]

    deposit = accrue_real_risk_free_leg(dates, _LIVE_BASE, spread_pp=_DEPOSIT_SPREAD_PP)
    ofz = accrue_real_risk_free_leg(dates, _LIVE_BASE, spread_pp=_OFZ_SPREAD_PP)

    # Same axis, opens at base.
    assert [d for d, _ in deposit] == dates
    assert deposit[0][1] == _LIVE_BASE
    assert ofz[0][1] == _LIVE_BASE
    # Real high-rate regime -> strictly rising, and the final value exceeds the opening.
    assert deposit[-1][1] > _LIVE_BASE
    assert ofz[-1][1] > _LIVE_BASE
    # OFZ-PK floater (no spread, full key rate) out-accrues the deposit (key-1pp).
    assert ofz[-1][1] > deposit[-1][1]


# -- NDFL net step (REGIME-04, D-01/D-04, R-E) --------------------------------

# A rising fetched-index level series whose daily return is the per-bar income the
# net step taxes. Strictly increasing -> every daily return is positive -> taxed.
_RISING_INDEX_BASE = Decimal(100)
_RISING_INDEX_DAILY = Decimal("1.0005")
# A flat (zero-return) index: every daily return is 0 -> nothing is taxed, so net == gross
# == the unchanged level (principal is never taxed -- Pitfall 1).
_FLAT_INDEX_LEVEL = Decimal(100)
# Enough bars for the cross-leg YTD to accumulate meaningfully (still well under the
# 2.4M threshold on a 100k base, so the cross-leg test reasons about ordering, not the band).
_NET_N_STEPS = 24
# A SINGLE-tax-year window (all bars in 2025) so the shared accumulator's YTD is monotonic
# across both legs -- no Jan-1 reset confounds the cross-leg assertion. 2025 opens in the
# high-rate regime (20-21%), so the deposit leg accrues real taxable income.
_NET_SINGLE_YEAR_FIRST = date(2025, 1, 9)
_NET_SINGLE_YEAR_STEPS = 11  # ~monthly bars through 2025, all in one tax year

# A TWO-tax-year daily window (CR-01 / IN-04a): the interleaved netter must NOT reset the
# cross-leg YTD between legs at a year boundary. On a LARGE base the cross-leg YTD crosses
# the 2.4M progressive threshold WITHIN the first tax year, so the band crossover (13% ->
# 15%) is the exact thing the shared YTD must detect. The OLD leg-by-leg structure (full OFZ
# pass, then full deposit pass) would reset the accumulator before the deposit leg's first
# (earliest) bar, undertaxing the deposit leg; the interleaved netter taxes both legs' daily
# increments against the SAME running YTD before any year reset.
_TWO_YEAR_FIRST = date(2024, 1, 9)  # window opens in 2024, runs into 2025 (two tax years)
_TWO_YEAR_STEPS = 24  # ~monthly bars spanning 2024 + 2025
# A LARGE base so the COMBINED cross-leg income crosses the 2.4M progressive band WITHIN the
# first tax year (200M base -> the deposit + OFZ legs each accrue ~1.6M+ income in 2024, so
# their combined cross-leg YTD crosses 2.4M and the band-crossover the shared YTD must detect
# fires). On a small base both legs stay in the 13% band and the netted curves are identical
# either way -- that is exactly why the orchestrator's real-window cert is byte-unchanged.
_LARGE_BASE = Decimal(200_000_000)
_LARGE_INDEX_DAILY = Decimal("1.0006")  # a high-carry index so the OFZ leg income is material


def _rebased_index(base: Decimal, daily: Decimal, dates: list[date]) -> list[tuple[date, Decimal]]:
    """A deterministic rising (date, Decimal) index *level* series (e.g. a RUFLBITR proxy)."""
    return [(d, base * daily**i) for i, d in enumerate(dates)]


def test_accrue_real_risk_free_leg_nets_via_accumulator() -> None:
    """The net deposit/OFZ leg loses income to NDFL: net final value < gross (REGIME-04 / R-E).

    Threading a ``YtdTaxAccumulator`` into ``accrue_real_risk_free_leg`` nets the per-bar
    income INCREMENT (not the level -- Pitfall 1) through the progressive 13/15% band, so
    the net leg's final value is strictly BELOW the same call with ``tax_acc=None`` (gross).
    Both still open at ``base`` on ``dates[0]`` and rise across the 16-21% high-rate regime.
    The gross path (``tax_acc=None``) stays byte-identical to today (no regression).
    """
    dates = [_LIVE_FIRST + timedelta(days=i * _DAYS_BETWEEN) for i in range(_NET_N_STEPS)]
    dates = [d for d in dates if d <= _LIVE_LAST]

    gross = accrue_real_risk_free_leg(dates, _LIVE_BASE, spread_pp=_DEPOSIT_SPREAD_PP)
    net = accrue_real_risk_free_leg(
        dates, _LIVE_BASE, spread_pp=_DEPOSIT_SPREAD_PP, tax_acc=YtdTaxAccumulator()
    )

    # Same axis, both open at base (R-3).
    assert [d for d, _ in net] == dates
    assert net[0][1] == _LIVE_BASE
    assert gross[0][1] == _LIVE_BASE
    # Both rise (high-rate regime), but the net leg loses income to NDFL -> strictly below gross.
    assert net[-1][1] > _LIVE_BASE
    assert net[-1][1] < gross[-1][1]


def test_net_index_returns_taxes_increment_not_principal() -> None:
    """``net_index_returns`` taxes the daily return increment, not principal (REGIME-04 / P-1).

    A rising fetched index re-based to a net total-return curve has a total gain strictly
    BELOW the gross index's total gain (income lost to NDFL), opening at the same base. A
    FLAT (zero daily return) index returns net == gross == the level unchanged -- principal
    is never taxed (Pitfall 1).
    """
    dates = [_LIVE_FIRST + timedelta(days=i * _DAYS_BETWEEN) for i in range(_NET_N_STEPS)]
    dates = [d for d in dates if d <= _LIVE_LAST]

    rising = _rebased_index(_RISING_INDEX_BASE, _RISING_INDEX_DAILY, dates)
    net = net_index_returns(rising, tax_acc=YtdTaxAccumulator())

    # Opens at the same base; total net gain is strictly below the gross index gain.
    assert net[0][1] == rising[0][1]
    gross_gain = rising[-1][1] - rising[0][1]
    net_gain = net[-1][1] - net[0][1]
    assert net_gain > _ZERO
    assert net_gain < gross_gain

    # A flat index: zero daily return -> nothing taxed -> net == gross == level unchanged.
    flat = [(d, _FLAT_INDEX_LEVEL) for d in dates]
    flat_net = net_index_returns(flat, tax_acc=YtdTaxAccumulator())
    assert [v for _, v in flat_net] == [v for _, v in flat]


def test_shared_accumulator_cross_leg_ytd() -> None:
    """One shared accumulator accrues a single cross-leg YTD over deposit + index legs (R-E / A2).

    Passing the SAME ``YtdTaxAccumulator`` through a deposit-leg call THEN a
    ``net_index_returns`` call accumulates one cross-leg YTD (the W1 cross-sleeve design):
    the second leg's tax is computed on top of the first leg's YTD, not from zero. Using a
    single-tax-year window (no Jan-1 reset) the accumulator's running YTD strictly grows on
    the second (index) leg, so it saw the deposit leg's prior YTD. Modeling the
    cross-leg-shared behavior (not per-leg) is the deliverable.
    """
    dates = [
        _NET_SINGLE_YEAR_FIRST + timedelta(days=i * _DAYS_BETWEEN)
        for i in range(_NET_SINGLE_YEAR_STEPS)
    ]
    rising = _rebased_index(_RISING_INDEX_BASE, _RISING_INDEX_DAILY, dates)

    shared = YtdTaxAccumulator()
    deposit_net = accrue_real_risk_free_leg(
        dates, _LIVE_BASE, spread_pp=_DEPOSIT_SPREAD_PP, tax_acc=shared
    )
    ytd_after_deposit = shared.ytd_taxable
    index_net = net_index_returns(rising, tax_acc=shared)
    ytd_after_both = shared.ytd_taxable

    # The shared accumulator advanced on BOTH legs (one tax year, no reset): the YTD strictly
    # grew on the index leg, so the second leg saw a non-zero starting YTD (cross-leg, not
    # per-leg).
    assert ytd_after_deposit > _ZERO
    assert ytd_after_both > ytd_after_deposit
    # Cross-leg, not per-leg: had the index leg used a FRESH accumulator, its standalone YTD
    # would be far below the shared running total (which already carries the deposit YTD).
    index_only_ytd = YtdTaxAccumulator()
    net_index_returns(rising, tax_acc=index_only_ytd)
    assert ytd_after_both > index_only_ytd.ytd_taxable
    # Both nets are real curves (open at base, taxed below their gross).
    assert deposit_net[0][1] == _LIVE_BASE
    assert index_net[0][1] == rising[0][1]


def test_shared_accumulator_cross_leg_ytd_two_tax_years() -> None:
    """Interleaved netting honors the cross-leg YTD across a TWO-tax-year window (CR-01 / IN-04a).

    The W1 contract is ONE cross-leg progressive-band YTD per run. Netting the two
    fixed-income legs LEG-BY-LEG (full OFZ pass, then full deposit pass through the same
    accumulator) silently breaks that contract on a multi-tax-year window: after the OFZ pass
    ``_current_year`` is the LAST year, so the deposit leg's earliest bar triggers a Jan-1
    reset that wipes the OFZ leg's accumulated YTD (CR-01). On a LARGE base where the combined
    cross-leg income crosses the 2.4M band within the first year, that reset UNDERTAXES the
    deposit leg (its first-year income is taxed from zero instead of on top of the OFZ YTD).

    :func:`net_fixed_income_legs_interleaved` taxes both legs' daily increments against the
    SAME running YTD before any year-boundary reset, so the deposit leg crosses into the 15%
    band sooner -> strictly MORE tax -> a strictly LOWER deposit final value than the broken
    leg-by-leg structure. This assertion is RED on the old leg-by-leg driver and GREEN after
    the interleaved fix.
    """
    dates = [_TWO_YEAR_FIRST + timedelta(days=i * _DAYS_BETWEEN) for i in range(_TWO_YEAR_STEPS)]
    # Two distinct tax years are actually spanned (guards the fixture against drift).
    assert len({d.year for d in dates}) >= 2  # noqa: PLR2004
    ofz_levels = _rebased_index(_LARGE_BASE, _LARGE_INDEX_DAILY, dates)

    # Interleaved (the fix): both legs net against ONE shared, date-ordered YTD.
    shared = YtdTaxAccumulator()
    deposit_interleaved, ofz_interleaved = net_fixed_income_legs_interleaved(
        ofz_levels, dates, _LARGE_BASE, deposit_spread_pp=_DEPOSIT_SPREAD_PP, tax_acc=shared
    )

    # Leg-by-leg (the OLD broken structure): full OFZ pass, THEN full deposit pass, one acc.
    leg_by_leg = YtdTaxAccumulator()
    net_index_returns(ofz_levels, tax_acc=leg_by_leg)  # OFZ pass advances _current_year to 2025
    deposit_leg_by_leg = accrue_real_risk_free_leg(
        dates, _LARGE_BASE, spread_pp=_DEPOSIT_SPREAD_PP, tax_acc=leg_by_leg
    )

    # The interleaved deposit leg crosses into the 15% band sooner (it sees the OFZ YTD before
    # the year boundary), so it pays strictly MORE tax than the reset-confounded leg-by-leg.
    assert deposit_interleaved[-1][1] < deposit_leg_by_leg[-1][1]
    # Both still open at base (principal never taxed) and share the one axis (R-3).
    assert deposit_interleaved[0][1] == _LARGE_BASE
    assert ofz_interleaved[0][1] == _LARGE_BASE
    assert [d for d, _ in deposit_interleaved] == dates
    assert [d for d, _ in ofz_interleaved] == dates


# -- Risk-free-bar methodology note (operator follow-up, framing-only) ---------

# The verbatim substrings the operator's methodology note MUST render, explaining why a
# near-vol-free risk-free leg inflates the naive Sharpe/Sortino bar in a high-rate regime
# (so a HARD_FAIL reflects the RATE REGIME, not an allocator defect) -- and that a huge
# Sortino is the TRUE value of a zero-downside curve, NOT a rendering bug.
_NOTE_FRAMING = "Methodology note (framing-only)"
_NOTE_NEAR_VOL_FREE = "near-vol-free"
_NOTE_NOT_A_BUG = "NOT a rendering"
_NOTE_STRUCTURALLY = "structurally unwinnable"
_NOTE_REGIME = "reflects the RATE REGIME, not an allocator defect"

# A minimal render payload: one HARD_FAIL profile + the zero-downside deposit bar that
# motivates the note (a near-infinite Sortino on a flat 16-21% deposit leg).
_NOTE_GIT_SHA = "deadbeef"
_NOTE_DEPOSIT_SORTINO = 48082830638875.85
_NOTE_DEPOSIT_SHARPE = 21.6448
_NOTE_PROFILE_SHARPE = -0.0429
_NOTE_PROFILE_SORTINO = -0.0628


def test_report_renders_risk_free_bar_note() -> None:
    """render_report carries the operator's framing-only risk-free-bar methodology note.

    The note explains the economically-real, NOT-a-bug provenance of the enormous
    near-vol-free deposit Sharpe/Sortino bar in a 16-21% high-rate regime, and that it makes
    the conjunctive test structurally unwinnable for any equity-holding allocation while the
    high rate holds -- so a HARD_FAIL reflects the REGIME, not a defect. It is FRAMING-ONLY:
    it must not alter any rendered metric or verdict (the HARD_FAIL stays HARD_FAIL).
    """
    per_profile = {
        "conservative": {
            "verdict": _HARD_FAIL,
            "sharpe": _NOTE_PROFILE_SHARPE,
            "best_naive_sharpe": _NOTE_DEPOSIT_SHARPE,
            "sortino": _NOTE_PROFILE_SORTINO,
            "best_naive_sortino": _NOTE_DEPOSIT_SORTINO,
            "realized_maxdd_frac": 0.021,
            "cap_frac": 0.08,
            "mean_wf_sharpe": 0.457,
        }
    }
    naive_metrics = {
        "deposit_100_sharpe": _NOTE_DEPOSIT_SHARPE,
        "deposit_100_sortino": _NOTE_DEPOSIT_SORTINO,
        "deposit_100_maxdd_pct": 0.0,
    }
    # The synthetic cut-path is retired (D-07): render_json no longer takes a cut_path arg.
    regime = regime_split([_BOUNDARY - timedelta(days=30), _BOUNDARY + timedelta(days=30)])

    payload = render_json(per_profile, naive_metrics, regime, git_sha=_NOTE_GIT_SHA)
    report = render_report(payload)

    # The methodology note renders verbatim (all framing pieces present).
    assert _NOTE_FRAMING in report
    assert _NOTE_NEAR_VOL_FREE in report
    assert _NOTE_NOT_A_BUG in report
    assert _NOTE_STRUCTURALLY in report
    assert _NOTE_REGIME in report

    # FRAMING-ONLY: the note changes no verdict/metric. The HARD_FAIL stays HARD_FAIL and the
    # mandatory honesty caveat is still rendered exactly once.
    assert _HARD_FAIL in report
    assert report.count("100% deposit winning raw return in a 16-21% high-rate regime") == 1


# -- Edge-input guards on the exported surface (WR-01 / WR-02) -----------------


def test_verdict_for_profile_rejects_empty_naive_lists() -> None:
    """verdict_for_profile defends the best-of-three bar against an empty leg list (WR-01).

    The conjunctive bar is ``max(naive_sharpes)``/``max(naive_sortinos)`` -- an empty list
    would raise an opaque ``ValueError: max() arg is an empty sequence``. The exported,
    test-pinned contract must reject the degenerate input with a CLEAR message instead. The
    happy path (three legs) is unchanged and still PASSes -- pinned by
    ``test_best_naive_max_over_three`` -- so no real-cert verdict moves.
    """
    with pytest.raises(ValueError, match="non-empty"):
        verdict_for_profile(
            alloc_sharpe=_ALLOC_SHARPE_PASS,
            alloc_sortino=_ALLOC_SORTINO_PASS,
            alloc_max_drawdown_pct=_ALLOC_MAXDD_PASS_PCT,
            naive_sharpes=[],
            naive_sortinos=_NAIVE_SORTINOS,
            cap_fraction=_CAP_BALANCED,
        )

    with pytest.raises(ValueError, match="non-empty"):
        verdict_for_profile(
            alloc_sharpe=_ALLOC_SHARPE_PASS,
            alloc_sortino=_ALLOC_SORTINO_PASS,
            alloc_max_drawdown_pct=_ALLOC_MAXDD_PASS_PCT,
            naive_sharpes=_NAIVE_SHARPES,
            naive_sortinos=[],
            cap_fraction=_CAP_BALANCED,
        )


def test_regime_split_rejects_empty_window() -> None:
    """regime_split defends the exported surface against an empty date window (WR-02).

    ``start, end = dates[0], dates[-1]`` indexes an empty list -> an opaque ``IndexError``.
    The real cert path always passes a >= 300-bar window, so the non-empty behaviour
    (pinned by ``test_regime_split``) is untouched; only the degenerate empty input now
    raises a CLEAR ``ValueError``.
    """
    with pytest.raises(ValueError, match="non-empty"):
        regime_split([])


# -- Fail-closed committed snapshot loader (REGIME-01 / D-05 / V5) -------------

# The three leg keys the snapshot loader expects (R-F shape). The fixture serializes
# Decimal -> str and date -> ISO string (the Phase-65 _row_to_instrument convention).
_SNAP_EQUITY_KEY = "equity_mcftrr_net"
_SNAP_OFZ_KEY = "ofz_ruflbitr_net"
_SNAP_DEPOSIT_KEY = "deposit_net"
# The binding window end (the look-ahead clamp, Pitfall 3): no bar may post-date this.
_SNAP_WINDOW_START = "2024-01-01"
_SNAP_WINDOW_END = "2026-06-10"
# A bar dated AFTER the window end -> must be rejected (look-ahead guard).
_SNAP_FUTURE_BAR = "2026-06-11"
_SNAP_BASE_EQUITY = "6423.95"
_SNAP_BASE_FIXED = "100000.00"


def _well_formed_snapshot() -> dict[str, object]:
    """A minimal valid snapshot dict (3 legs, 2 bars each, all <= the window end)."""
    return {
        "generated_at": "2026-06-12T00:00:00Z",
        "window": {"start": _SNAP_WINDOW_START, "end": _SNAP_WINDOW_END},
        "git_sha": "deadbeef",
        "legs": {
            _SNAP_EQUITY_KEY: [["2024-01-03", _SNAP_BASE_EQUITY], [_SNAP_WINDOW_END, "6110.63"]],
            _SNAP_OFZ_KEY: [["2024-01-03", _SNAP_BASE_FIXED], [_SNAP_WINDOW_END, "112000.00"]],
            _SNAP_DEPOSIT_KEY: [["2024-01-03", _SNAP_BASE_FIXED], [_SNAP_WINDOW_END, "118000.00"]],
        },
    }


def test_gate_snapshot_missing_raises(tmp_path: Path) -> None:
    """A MISSING snapshot file fails closed -> ConfigurationError (REGIME-01 / V5).

    The committed snapshot is the CI trust boundary -- a missing file must raise, never
    silently fall back to synthetic data (the Phase-65 fail-closed pattern).
    """
    missing = tmp_path / "does_not_exist.json"
    with pytest.raises(ConfigurationError):
        _load_gate_snapshot(missing)


def test_gate_snapshot_corrupt_raises(tmp_path: Path) -> None:
    """A CORRUPT / malformed snapshot fails closed -> ConfigurationError (REGIME-01 / V5).

    Both invalid JSON and a well-formed JSON missing a required leg key must raise -- no
    silent synthetic fallback (T-74-03).
    """
    bad_json = tmp_path / "corrupt.json"
    bad_json.write_text("{ this is not json", encoding="utf-8")
    with pytest.raises(ConfigurationError):
        _load_gate_snapshot(bad_json)

    # Valid JSON but a required leg key is absent -> still fails closed.
    snap = _well_formed_snapshot()
    del snap["legs"][_SNAP_OFZ_KEY]  # type: ignore[index]
    missing_leg = tmp_path / "missing_leg.json"
    missing_leg.write_text(json.dumps(snap), encoding="utf-8")
    with pytest.raises(ConfigurationError):
        _load_gate_snapshot(missing_leg)


def test_gate_snapshot_round_trip_clamped(tmp_path: Path) -> None:
    """A well-formed snapshot re-hydrates Decimal-exact + rejects future bars (REGIME-01 / P-3).

    Round-trip: a valid snapshot loads into three ``(date, Decimal)`` curves with exact
    Decimal re-hydration. A bar dated AFTER ``window.end`` (the look-ahead clamp) makes the
    loader fail closed (T-74-04).
    """
    good = tmp_path / "good.json"
    good.write_text(json.dumps(_well_formed_snapshot()), encoding="utf-8")
    equity, ofz, deposit = _load_gate_snapshot(good)

    # Three (date, Decimal) curves, Decimal-exact after re-hydration.
    assert equity[0] == (date(2024, 1, 3), Decimal(_SNAP_BASE_EQUITY))
    assert ofz[0] == (date(2024, 1, 3), Decimal(_SNAP_BASE_FIXED))
    assert deposit[0] == (date(2024, 1, 3), Decimal(_SNAP_BASE_FIXED))
    assert all(isinstance(d, date) and isinstance(v, Decimal) for d, v in equity)
    assert equity[-1][0] == date.fromisoformat(_SNAP_WINDOW_END)

    # A bar dated AFTER window.end -> rejected (look-ahead clamp, Pitfall 3).
    leaky = _well_formed_snapshot()
    leaky["legs"][_SNAP_EQUITY_KEY].append([_SNAP_FUTURE_BAR, "9999.99"])  # type: ignore[index,union-attr]
    leak_file = tmp_path / "leaky.json"
    leak_file.write_text(json.dumps(leaky), encoding="utf-8")
    with pytest.raises(ConfigurationError):
        _load_gate_snapshot(leak_file)


def test_gate_snapshot_rejects_misaligned_axes(tmp_path: Path) -> None:
    """The loader rejects legs whose date axes are not identical (WR-01 / R-3).

    The whole gate runs on ONE basis (R-3) and ``regime_split`` keys off only the deposit
    leg's dates, so a snapshot whose three legs have misaligned date axes would silently
    produce a WRONG basis without raising. The committed fixture always shares the MCFTRR
    axis (so the assertion accepts it -- pinned by ``test_gate_snapshot_round_trip_clamped``),
    but a fail-closed loader that documents R-3 must also ENFORCE it: a hand-crafted snapshot
    with three different per-leg axes must raise ``ConfigurationError``.
    """
    snap = _well_formed_snapshot()
    # Shift the OFZ leg onto a DIFFERENT date axis (same length, different dates) -> misaligned.
    snap["legs"][_SNAP_OFZ_KEY] = [  # type: ignore[index]
        ["2024-01-04", _SNAP_BASE_FIXED],
        [_SNAP_WINDOW_END, "112000.00"],
    ]
    misaligned = tmp_path / "misaligned.json"
    misaligned.write_text(json.dumps(snap), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="one date axis"):
        _load_gate_snapshot(misaligned)


def test_gate_snapshot_rejects_window_end_past_binding_end(tmp_path: Path) -> None:
    """The loader rejects a window.end that post-dates _BINDING_END (IN-02 defense-in-depth).

    The per-bar clamp already rejects any bar > _BINDING_END, so no future bar can leak. But
    the ``window.end`` field itself was never checked against _BINDING_END: a snapshot
    declaring ``window.end = 2027-01-01`` loaded fine (its bars were still clamped). Surface a
    mis-stamped window EXPLICITLY rather than relying only on the bar-level clamp -- a
    window.end past _BINDING_END raises ``ConfigurationError``.
    """
    snap = _well_formed_snapshot()
    # window.end declared well past the binding clamp (bars stay <= clamp; only the field lies).
    snap["window"] = {"start": _SNAP_WINDOW_START, "end": "2027-01-01"}  # type: ignore[index]
    bad_window = tmp_path / "bad_window.json"
    bad_window.write_text(json.dumps(snap), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="binding"):
        _load_gate_snapshot(bad_window)


def test_no_allocation_logic_drift() -> None:
    """The frozen merge path is byte-identical given identical input curves (frozen-allocator).

    ``build_naive_legs`` + the frozen ``AllocationOrchestrator`` merge produce IDENTICAL
    results across two calls on the SAME input curves. This pins that the Phase-74
    measurement changes (net basis / snapshot / boundary) did NOT perturb the allocation
    logic -- the candidate and the naive legs both flow through the unchanged frozen path.
    """
    dates = _daily_index(_FIRST_BAR, _N_BARS)
    deposit_curve = _curve(_DEPOSIT_BASE, _DEPOSIT_DAILY, dates)
    ofz_pk_curve = _curve(_OFZ_BASE, _OFZ_DAILY, dates)
    equity_curve = _curve(_EQUITY_BASE, _EQUITY_DAILY, dates)

    legs_a = build_naive_legs(deposit_curve, ofz_pk_curve, equity_curve)
    legs_b = build_naive_legs(deposit_curve, ofz_pk_curve, equity_curve)

    assert set(legs_a) == set(legs_b)
    for name in legs_a:
        # Byte-identical merged equity curve + headline metrics across the two runs.
        assert list(legs_a[name].merged_equity_curve) == list(legs_b[name].merged_equity_curve)
        assert legs_a[name].sharpe == legs_b[name].sharpe
        assert legs_a[name].max_drawdown_pct == legs_b[name].max_drawdown_pct
        assert legs_a[name].rebalance_cost == legs_b[name].rebalance_cost
