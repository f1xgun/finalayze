"""Unit tests for the gold-sleeve allocation blender (beyond-MOEX-edge R&D, Phase A).

Deterministic, no-network tests of the pure simulator primitives: TER drag, the
shared-axis forward-fill, the fixed-weight quarterly-rebalanced blend (single-leg
identity, weighted blend, cost charging, free-leg exemption), and the
pre-registered diversification verdict.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

from finalayze.backtest.gold_sleeve_lab import (
    apply_ter_drag,
    blend_portfolio,
    diversification_verdict,
    forward_align_legs,
    master_axis,
)

# ── test constants (no magic numbers — ruff PLR2004) ─────────────────────────
_D0 = date(2022, 1, 3)
_D1 = date(2022, 1, 4)
_D2 = date(2022, 1, 5)
_HUNDRED = Decimal(100)
_UP_10 = Decimal(110)
_UP_21 = Decimal(121)
_DOWN_10 = Decimal(90)
_TER_080 = Decimal("0.8")
_ZERO_TER = Decimal(0)
_PER_SIDE = Decimal("0.0055")  # 0.55%/side retail
_HALF = Decimal("0.5")
_ONE = Decimal(1)
_NAV1 = Decimal(1)
_MAXDD_CUT_MIN_PP = Decimal("3.0")
_PLACES = Decimal("0.000001")


def _q(x: Decimal) -> Decimal:
    return x.quantize(_PLACES)


def test_apply_ter_drag_zero_is_identity() -> None:
    curve = [(_D0, _HUNDRED), (_D1, _UP_10), (_D2, _UP_21)]
    assert apply_ter_drag(curve, _ZERO_TER) == curve


def test_apply_ter_drag_reduces_a_flat_curve_monotonically() -> None:
    flat = [(_D0, _HUNDRED), (_D1, _HUNDRED), (_D2, _HUNDRED)]
    out = apply_ter_drag(flat, _TER_080)
    vals = [v for _, v in out]
    assert vals[0] == _HUNDRED  # opens at base
    assert vals[1] < vals[0]
    assert vals[2] < vals[1]  # strictly decreasing under a positive holding cost


def test_master_axis_is_sorted_union() -> None:
    legs = {"a": [(_D0, _HUNDRED), (_D2, _HUNDRED)], "b": [(_D0, _HUNDRED), (_D1, _HUNDRED)]}
    assert master_axis(legs) == [_D0, _D1, _D2]


def test_forward_align_holds_last_known_value_through_a_gap() -> None:
    # leg "a" has no bar on _D1 (e.g. equity halt) — it must forward-fill _D0's value.
    legs = {"a": [(_D0, _HUNDRED), (_D2, _UP_10)]}
    axis = [_D0, _D1, _D2]
    aligned = forward_align_legs(legs, axis)
    assert aligned["a"] == [_HUNDRED, _HUNDRED, _UP_10]


def test_blend_single_leg_reproduces_its_net_curve() -> None:
    # 100% one leg, rebalanced every day, any cost -> reproduces the leg's return path.
    legs = {"a": [_HUNDRED, _UP_10, _UP_21]}
    axis = [_D0, _D1, _D2]
    out = blend_portfolio(
        legs=legs,
        dates=axis,
        target_weights={"a": _ONE},
        rebalance_dates=axis,
        per_side_cost=_PER_SIDE,
        free_legs=set(),
        initial_nav=_NAV1,
    )
    vals = [_q(v) for _, v in out]
    assert vals == [_q(_NAV1), _q(Decimal("1.1")), _q(Decimal("1.21"))]


def test_blend_weighted_two_legs_no_rebalance_drift() -> None:
    # +10% and -10% at 50/50 with no interim rebalance -> net ~flat on day 1.
    legs = {"a": [_HUNDRED, _UP_10], "b": [_HUNDRED, _DOWN_10]}
    axis = [_D0, _D1]
    out = blend_portfolio(
        legs=legs,
        dates=axis,
        target_weights={"a": _HALF, "b": _HALF},
        rebalance_dates=[_D0],  # initial only -> day 1 just drifts
        per_side_cost=_PER_SIDE,
        free_legs=set(),
        initial_nav=_NAV1,
    )
    assert _q(out[-1][1]) == _q(_NAV1)  # 0.5*1.1 + 0.5*0.9 = 1.0


def test_blend_charges_cost_on_traded_legs_but_not_free_legs() -> None:
    # Two legs drift apart, then rebalance. Charging cost on the traded leg must
    # leave a LOWER nav than exempting both as free legs.
    legs = {"dep": [_HUNDRED, _HUNDRED, _HUNDRED], "eq": [_HUNDRED, _UP_10, _UP_21]}
    axis = [_D0, _D1, _D2]
    weights = {"dep": _HALF, "eq": _HALF}
    rebal = [_D0, _D2]
    charged = blend_portfolio(
        legs=legs,
        dates=axis,
        target_weights=weights,
        rebalance_dates=rebal,
        per_side_cost=_PER_SIDE,
        free_legs=set(),
        initial_nav=_NAV1,
    )
    free = blend_portfolio(
        legs=legs,
        dates=axis,
        target_weights=weights,
        rebalance_dates=rebal,
        per_side_cost=_PER_SIDE,
        free_legs={"dep", "eq"},
        initial_nav=_NAV1,
    )
    assert charged[-1][1] < free[-1][1]


def test_diversification_verdict_requires_maxdd_cut_and_sortino_improvement() -> None:
    # gold cuts MaxDD by 5pp AND improves Sortino -> diversifies.
    good = diversification_verdict(
        baseline_maxdd_pct=50.0,
        gold_maxdd_pct=45.0,
        baseline_sortino=-1.0,
        gold_sortino=-0.8,
        maxdd_cut_min_pp=_MAXDD_CUT_MIN_PP,
    )
    assert good["diversifies"] is True
    # MaxDD cut too small -> fails even with better Sortino.
    small = diversification_verdict(
        baseline_maxdd_pct=50.0,
        gold_maxdd_pct=49.0,
        baseline_sortino=-1.0,
        gold_sortino=-0.5,
        maxdd_cut_min_pp=_MAXDD_CUT_MIN_PP,
    )
    assert small["diversifies"] is False
    # Big MaxDD cut but WORSE Sortino -> fails (must satisfy both).
    worse = diversification_verdict(
        baseline_maxdd_pct=50.0,
        gold_maxdd_pct=40.0,
        baseline_sortino=-0.5,
        gold_sortino=-1.0,
        maxdd_cut_min_pp=_MAXDD_CUT_MIN_PP,
    )
    assert worse["diversifies"] is False
