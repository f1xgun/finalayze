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
