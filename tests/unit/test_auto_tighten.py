"""RED scaffold: SAA-05 / R-4 parameter-free monotone tighten() (Phase 72 Wave-0).

Pins the L0 ``finalayze.core.allocation.tighten`` contract before it exists:
- no MaxDD breach (realized_dd <= cap) is the identity (no shift);
- a static breach drives a fixed 5pp equity->deposit shift each step until the
  cap holds OR equity clamps at 0 -- deterministic, monotone, NO solver (R-4);
- OFZ-PK is never touched (flat 25% leg, D-01);
- equity is clamped non-negative and the vector always re-sums to 1.0;
- the function is pure (two identical calls return equal dicts -- no state, no
  randomness).

W2 ships this rule tested-but-DORMANT (it must NOT be wired into the live
allocation path; W3 wires the freeze + OOS re-gate, R-4).

RED now: ``finalayze.core.allocation`` (tighten, Plan 02) +
``finalayze.core.schemas.AssetClass`` (Plan 02) do not exist yet.
"""

from __future__ import annotations

from decimal import Decimal

from finalayze.core.allocation import tighten
from finalayze.core.schemas import AssetClass

# -- Constants (named -- no magic numbers, ruff PLR2004) ----------------------

_BALANCED = {
    AssetClass.DEPOSIT: Decimal("0.45"),
    AssetClass.OFZ_PK: Decimal("0.25"),
    AssetClass.EQUITY: Decimal("0.30"),
}
_CAP = Decimal("0.15")
_FLAT_OFZ_PK = Decimal("0.25")  # D-01 flat leg
_VECTOR_SUM = Decimal("1.0")
_ZERO = Decimal(0)

# realized-drawdown scenarios.
_DD_UNDER_CAP = Decimal("0.10")  # <= cap -> identity
_DD_JUST_OVER = Decimal("0.16")  # just over cap -> tighten fires
_DD_HIGHER = Decimal("0.30")  # a higher breach than _DD_JUST_OVER (monotone check)
_DD_EXTREME = Decimal("0.99")  # drives equity to the clamp

# Terminal vector under a STATIC breach: equity is shifted to deposit in 5pp
# steps until equity clamps at 0 (0.30 -> 0.00, +0.30 to deposit), OFZ-PK fixed.
_TERMINAL_DEPOSIT = Decimal("0.75")
_TERMINAL_EQUITY = Decimal("0.00")


def test_no_breach_is_identity() -> None:
    """realized_dd <= cap -> no shift (identity)."""
    assert tighten(_BALANCED, realized_dd=_DD_UNDER_CAP, cap=_CAP) == _BALANCED


def test_static_breach_clamps_equity_to_zero() -> None:
    """A static breach shifts 5pp equity->deposit until equity clamps at 0 (R-4).

    The supplied realized_dd is CONSTANT, so the parameter-free monotone rule
    keeps moving equity into deposit (5pp/step) until equity hits 0 -- the
    terminal deterministic vector is {deposit 0.75, ofz_pk 0.25, equity 0.00}.

    NOTE to executor: this pins the parameter-free monotone+clamp contract R-4
    specifies. If Plan 02 instead applies a single step per call, adjust BOTH
    this assertion AND the impl to one consistent deterministic contract and
    document it -- the binding requirement is deterministic + monotone +
    parameter-free + equity >= 0.
    """
    result = tighten(_BALANCED, realized_dd=_DD_JUST_OVER, cap=_CAP)
    assert result == {
        AssetClass.DEPOSIT: _TERMINAL_DEPOSIT,
        AssetClass.OFZ_PK: _FLAT_OFZ_PK,
        AssetClass.EQUITY: _TERMINAL_EQUITY,
    }


def test_monotone_more_dd_not_less_deposit() -> None:
    """A higher realized_dd never yields LESS deposit weight (monotone, R-4)."""
    higher = tighten(_BALANCED, _DD_HIGHER, _CAP)
    lower = tighten(_BALANCED, _DD_JUST_OVER, _CAP)
    assert higher[AssetClass.DEPOSIT] >= lower[AssetClass.DEPOSIT]


def test_equity_clamped_nonnegative() -> None:
    """An extreme breach clamps equity at 0 (never negative) and re-sums to 1.0."""
    result = tighten(_BALANCED, _DD_EXTREME, _CAP)
    assert result[AssetClass.EQUITY] >= _ZERO
    assert sum(result.values()) == _VECTOR_SUM


def test_ofz_pk_never_touched() -> None:
    """For any realized_dd the OFZ-PK leg stays fixed at 25% (flat leg, D-01)."""
    for dd in (_DD_UNDER_CAP, _DD_JUST_OVER, _DD_HIGHER, _DD_EXTREME):
        assert tighten(_BALANCED, dd, _CAP)[AssetClass.OFZ_PK] == _FLAT_OFZ_PK


def test_deterministic_pure_fn() -> None:
    """Two calls with identical inputs return equal dicts (no state, no randomness)."""
    first = tighten(_BALANCED, _DD_JUST_OVER, _CAP)
    second = tighten(_BALANCED, _DD_JUST_OVER, _CAP)
    assert first == second
