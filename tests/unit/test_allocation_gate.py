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
    CUT_GLIDE,
    REGIME_SPLIT_BOUNDARY,
    build_naive_legs,
    excess_sortino_from_equity,
    gate_with_autotighten,
    oos_wf_sharpes,
    realized_dd_fraction,
    regime_split,
    run_cut_path,
    verdict_for_profile,
)
from finalayze.backtest.bond_walk_forward import generate_wf_windows
from finalayze.core.allocation import tighten  # used only to derive the EXPECTED frozen vector
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

# V-9 / D-09 / R-6: the early-cut regime boundary.
_BOUNDARY = date(2025, 7, 25)
_HIGH_RATE_LAST = date(2025, 7, 24)  # last day of the high-rate window

# V-8 / D-02: the WF window cadence the gate MUST pass explicitly to
# generate_wf_windows (train/test/step months).
_WF_TRAIN_M = 12
_WF_TEST_M = 6
_WF_STEP_M = 3
# A >= 4-year daily window so generate_wf_windows yields >= 3 folds.
_WF_FIRST_BAR = date(2021, 1, 1)
_WF_N_BARS = 4 * 365 + 1  # ~4 years of daily bars (inclusive of a leap day)

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
    fold. It NEVER constructs or runs a backtest engine; pinned by asserting the
    fold count equals the window count for the SAME date span.
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
    assert len(folds) == len(expected_windows)
    assert all(isinstance(s, float) for s in folds)


def test_cutpath_equity_fixed() -> None:
    """The cut-path holds the MCFTR equity curve byte-identical (V-7 / D-07).

    ``run_cut_path`` rebuilds ONLY the deposit + OFZ legs under the synthetic
    declining ``CUT_GLIDE`` meeting calendar; the equity (MCFTR) leg is passed
    through UNCHANGED (no fabricated uplift). The deposit leg under CUT_GLIDE
    differs from the high-rate baseline deposit leg.
    """
    dates = _daily_index(_FIRST_BAR, _N_BARS)
    deposit_curve = _curve(_DEPOSIT_BASE, _DEPOSIT_DAILY, dates)
    ofz_pk_curve = _curve(_OFZ_BASE, _OFZ_DAILY, dates)
    equity_curve = _curve(_EQUITY_BASE, _EQUITY_DAILY, dates)

    cut = run_cut_path(deposit_curve, ofz_pk_curve, equity_curve)

    # Equity leg is held FIXED to exact Decimal equality (no uplift).
    assert [v for _, v in equity_curve] == list(cut.equity_curve)
    # The deposit leg under CUT_GLIDE diverges from the high-rate baseline.
    assert list(cut.deposit_curve) != [v for _, v in deposit_curve]
    assert CUT_GLIDE is not None  # the synthetic declining meeting calendar exists


def test_regime_split() -> None:
    """The early-cut regime boundary is 2025-07-25 (V-9 / D-09 / R-6).

    ``regime_split`` partitions a date window at 2025-07-25 -- the high-rate window
    ends 2025-07-24, the early-cut window starts 2025-07-25. A window entirely
    before the boundary is a single high-rate regime.
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
