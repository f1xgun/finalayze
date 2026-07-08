"""Pure L0 SAA auto-tighten rule (SAA-05 / D-05 / R-4).

The deterministic, monotone, parameter-free equity->deposit shift that W3 (Phase 73)
will wire into the freeze + out-of-sample re-gate. This module is a pure data transform:
zero project imports above L0 (only ``finalayze.core.schemas`` + stdlib ``Decimal``),
no I/O, no randomness, no solver.

See docs/architecture/DEPENDENCY_LAYERS.md for layering rules.
"""

from __future__ import annotations

from decimal import Decimal

from finalayze.core.schemas import AssetClass

_DEFAULT_TIGHTEN_STEP = Decimal("0.05")  # 5pp equity->deposit per step (D-05, parameter-free)
_ZERO = Decimal(0)
_ONE = Decimal(1)


def graded_regime_weights(
    high_rate: dict[AssetClass, Decimal],
    easing: dict[AssetClass, Decimal],
    current_rate: Decimal,
    peak_rate: Decimal,
    neutral_rate: Decimal,
) -> dict[AssetClass, Decimal]:
    """Interpolate high_rate<->easing weights by how far the key rate has fallen (graded regime).

    The shipped Phase-76 tilt is a BINARY switch: full ``high_rate`` until the first CBR
    cut, then a full jump to ``easing`` regardless of how far rates actually fall. This
    grades that response to the DEPTH of easing:

        t = clamp((peak_rate - current_rate) / (peak_rate - neutral_rate), 0, 1)
        weights = high_rate * (1 - t) + easing * t

    At/above the cycle ``peak_rate`` -> full ``high_rate`` (deposit-anchored). At/below the
    ``neutral_rate`` anchor -> full ``easing``. In between the shift is proportional to how
    far the rate has travelled toward neutral, so a shallow early cut moves the book only a
    little and conviction builds as rates approach neutral.

    Pure, monotone, parameter-light: ``peak_rate`` and ``neutral_rate`` are ECONOMIC anchors
    (cycle peak, long-run neutral), NOT free parameters fitted to a verdict (Pitfall 8). The
    result is a convex combination of two vectors that each sum to 1.0 and are non-negative,
    so it also sums to 1.0 and is non-negative -- no renormalization, no solver. The binary
    Phase-76 switch is the ``t in {0, 1}`` special case of this rule.

    Raises:
        ValueError: if ``peak_rate <= neutral_rate`` (a non-positive easing span) or the two
            weight vectors do not carry the same asset classes.
    """
    if peak_rate <= neutral_rate:
        msg = f"peak_rate {peak_rate} must exceed neutral_rate {neutral_rate}"
        raise ValueError(msg)
    if set(high_rate) != set(easing):
        msg = "high_rate and easing must carry the same asset classes"
        raise ValueError(msg)

    t = (peak_rate - current_rate) / (peak_rate - neutral_rate)
    t = max(_ZERO, min(_ONE, t))
    one_minus = _ONE - t
    return {cls: high_rate[cls] * one_minus + easing[cls] * t for cls in high_rate}


def tighten(
    weights: dict[AssetClass, Decimal],
    realized_dd: Decimal,
    cap: Decimal,
    step: Decimal = _DEFAULT_TIGHTEN_STEP,
) -> dict[AssetClass, Decimal]:
    """Deterministic, monotone, parameter-free equity->deposit shift until the MaxDD cap holds.

    SAA-05 / D-05 / R-4. Pure data transform: consumes an already-computed in-sample
    ``realized_dd`` (no data dependency, no look-ahead). Shifts ``step`` from EQUITY to
    DEPOSIT each step while ``realized_dd > cap``, clamping equity at 0; OFZ-PK is the flat
    leg (D-01) and is never touched. NOT a solver (Pitfall 8): no covariance, no
    expected-return, no search, no free parameter to fit.

    Contract (pinned by tests/unit/test_auto_tighten.py): the supplied ``realized_dd`` is a
    CONSTANT scalar, so the monotone rule deterministically drains EQUITY into DEPOSIT in
    ``step`` increments until EQUITY clamps at 0 -- the terminal vector under any breach is
    ``{deposit: deposit + equity, ofz_pk: ofz_pk, equity: 0}``. The output always re-sums to
    the input total and every weight is >= 0.

    DORMANT in W2: ships tested-but-unwired. W3 (Phase 73) supplies the realized DD, calls
    this, FREEZES the result, and runs the OOS re-gate (the honesty guard).
    """
    if realized_dd <= cap:
        return dict(weights)

    deposit = weights[AssetClass.DEPOSIT]
    ofz = weights[AssetClass.OFZ_PK]
    equity = weights[AssetClass.EQUITY]

    # Monotone shift equity -> deposit while the (static input) DD breaches the cap.
    # With a constant realized_dd the deterministic terminal state is equity clamped to 0;
    # the loop is bounded by equity / step (no infinite loop).
    while realized_dd > cap and equity > _ZERO:
        shift = min(step, equity)
        equity -= shift
        deposit += shift

    return {AssetClass.DEPOSIT: deposit, AssetClass.OFZ_PK: ofz, AssetClass.EQUITY: equity}
