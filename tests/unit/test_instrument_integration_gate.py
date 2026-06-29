"""Unit tests for the Instrument Integration Gate pure logic (autonomous diversification program).

Deterministic, no-network tests of the threshold classifier, the weight proposer, and the small
analytic helpers. The end-to-end gold→REJECT / ZO→PROBATION validation runs from committed panels
via scripts/research/run_instrument_gate.py (an integration cert, not a unit test).
"""

from __future__ import annotations

import dataclasses
from datetime import date
from decimal import Decimal

from finalayze.backtest.instrument_integration_gate import (
    _PROBATION_NOMINAL_CAP,
    _TIER_NOMINAL_CAPS,
    Scorecard,
    classify,
    daily_returns,
    leg_correlations,
    propose_weight,
)

_PLACES = Decimal("0.0001")


def _sound_integrate_scorecard() -> Scorecard:
    """A scorecard meeting EVERY INTEGRATE condition (the theoretical free-improvement case)."""
    return Scorecard(
        window_bars=900,
        regimes_covered=2,
        tail_backtestable=True,
        marginal_sharpe_delta=0.20,
        marginal_sortino_delta=0.10,
        marginal_maxdd_delta_pp=4.0,
        crash_year_maxdd_delta_pp=-1.0,
        toehold_sortino_delta=0.02,
        corr_to_legs={"deposit": 0.0, "equity": 0.2},
        max_corr_to_existing_legs=0.20,
        anti_hollow_ok=True,
    )


def _with(sc: Scorecard, **kw: object) -> Scorecard:
    return dataclasses.replace(sc, **kw)


def test_insufficient_data_when_too_few_bars() -> None:
    sc = _with(_sound_integrate_scorecard(), window_bars=100)
    tier, _ = classify(sc)
    assert tier == "INSUFFICIENT_DATA"


def test_insufficient_data_when_anti_hollow_fails() -> None:
    sc = _with(_sound_integrate_scorecard(), anti_hollow_ok=False)
    tier, _ = classify(sc)
    assert tier == "INSUFFICIENT_DATA"


def test_integrate_when_all_conditions_met() -> None:
    tier, _ = classify(_sound_integrate_scorecard())
    assert tier == "INTEGRATE"


def test_reject_when_redundant_high_correlation() -> None:
    sc = _with(
        _sound_integrate_scorecard(),
        max_corr_to_existing_legs=0.75,
        corr_to_legs={"deposit": 0.1, "equity": 0.75},
    )
    tier, reasons = classify(sc)
    assert tier == "REJECT"
    assert any("corr" in r.lower() for r in reasons)


def test_reject_when_tested_tail_raises_crash_drawdown() -> None:
    # gold's signature: tail in-window, but it RAISED the crash-year drawdown.
    sc = _with(
        _sound_integrate_scorecard(),
        marginal_sharpe_delta=-0.05,
        marginal_sortino_delta=-0.16,
        marginal_maxdd_delta_pp=2.0,
        crash_year_maxdd_delta_pp=2.4,  # raised the crash drawdown
        toehold_sortino_delta=-0.05,
        corr_to_legs={"deposit": 0.0, "equity": 0.03},
        max_corr_to_existing_legs=0.03,
    )
    tier, reasons = classify(sc)
    assert tier == "REJECT"
    assert any("crash" in r.lower() for r in reasons)


def test_probation_when_uncorrelated_hedge_with_untestable_tail() -> None:
    # ZO's signature: uncorrelated + small drawdown relief + un-backtestable tail + mild toe-hold.
    sc = _with(
        _sound_integrate_scorecard(),
        tail_backtestable=False,
        marginal_sharpe_delta=-0.10,
        marginal_sortino_delta=-0.28,  # hurts at a real 10% weight...
        marginal_maxdd_delta_pp=1.5,
        crash_year_maxdd_delta_pp=0.0,
        toehold_sortino_delta=-0.06,  # ...but tolerable at the 3% toe-hold
        corr_to_legs={"deposit": -0.06, "equity": 0.05},
        max_corr_to_existing_legs=0.06,
    )
    tier, _ = classify(sc)
    assert tier == "PROBATION"


def test_probation_blocked_when_toehold_still_hurts_sortino() -> None:
    # Same shape but the drag is bad even at the toe-hold -> REJECT, not PROBATION.
    sc = _with(
        _sound_integrate_scorecard(),
        tail_backtestable=False,
        marginal_sortino_delta=-0.40,
        marginal_maxdd_delta_pp=1.5,
        toehold_sortino_delta=-0.20,  # hurts even at 3%
        corr_to_legs={"deposit": -0.06, "equity": 0.05},
        max_corr_to_existing_legs=0.06,
    )
    tier, _ = classify(sc)
    assert tier == "REJECT"


def test_propose_weight_probation_is_the_toehold() -> None:
    sc = _with(_sound_integrate_scorecard(), tail_backtestable=False)
    w = propose_weight("PROBATION", "medium", sc)
    assert w == _PROBATION_NOMINAL_CAP


def test_propose_weight_integrate_capped_by_tier() -> None:
    sc = _sound_integrate_scorecard()
    assert propose_weight("INTEGRATE", "high", sc) == _TIER_NOMINAL_CAPS["high"]
    assert propose_weight("INTEGRATE", "medium", sc) == _TIER_NOMINAL_CAPS["medium"]


def test_propose_weight_reject_is_zero() -> None:
    sc = _sound_integrate_scorecard()
    assert propose_weight("REJECT", "high", sc) == Decimal(0)
    assert propose_weight("INSUFFICIENT_DATA", "low", sc) == Decimal(0)


def test_daily_returns_basic() -> None:
    curve = [(date(2022, 1, 3), Decimal(100)), (date(2022, 1, 4), Decimal(110))]
    rets = daily_returns(curve)
    assert len(rets) == 1
    assert abs(rets[0] - 0.10) < float(_PLACES)


def test_leg_correlations_perfectly_correlated_leg() -> None:
    # A candidate equal to the equity leg (scaled) -> corr 1.0 with equity.
    base = [(date(2022, 1, d), Decimal(100 + d)) for d in range(3, 13)]
    cand = [(d, v * Decimal(2)) for d, v in base]
    corrs = leg_correlations(cand, {"equity": base})
    assert abs(corrs["equity"] - 1.0) < float(_PLACES)
