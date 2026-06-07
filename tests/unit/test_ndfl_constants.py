"""RED scaffold: single-source NDFL constant module (ACCT-03 / D-12).

Pins the L0 ``finalayze.core.constants`` contract before it exists:
- the canonical NDFL band rates, progressive threshold, ASV cap, deposit
  demand rate and the deposit non-taxable-floor base are all single-sourced;
- neither broker hard-codes a second ``Decimal("0.13")`` literal (the dedup
  guard that turns GREEN once Plan 02 re-points both brokers at the L0 source).

RED now: ``finalayze.core.constants`` does not exist yet and both brokers
still carry their own ``Decimal("0.13")`` copy.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

from finalayze.core.constants import (
    ASV_CAP_PER_BANK,
    DEPOSIT_DEMAND_RATE,
    DEPOSIT_FLOOR_BASE,
    NDFL_PROGRESSIVE_THRESHOLD,
    NDFL_RATE,
    NDFL_RATE_HIGH,
)

# ── Constants ──────────────────────────────────────────────────────────────

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BROKERS = [
    _REPO_ROOT / "src" / "finalayze" / "execution" / "bond_simulated_broker.py",
    _REPO_ROOT / "src" / "finalayze" / "execution" / "sandbox_tracker.py",
]
_NDFL_LITERAL = 'Decimal("0.13")'

# Expected single-source values (D-12 / D-10 / D-07 / D-03).
_EXPECTED_NDFL_RATE = Decimal("0.13")
_EXPECTED_NDFL_RATE_HIGH = Decimal("0.15")
_EXPECTED_PROGRESSIVE_THRESHOLD = Decimal(2_400_000)
_EXPECTED_ASV_CAP = Decimal(1_400_000)
_EXPECTED_DEPOSIT_FLOOR_BASE = Decimal(1_000_000)
_EXPECTED_DEPOSIT_DEMAND_RATE = Decimal("0.0001")


def test_ndfl_rate_is_single_source() -> None:
    """The L0 constants module exposes the canonical financial values exactly."""
    assert NDFL_RATE == _EXPECTED_NDFL_RATE
    assert NDFL_RATE_HIGH == _EXPECTED_NDFL_RATE_HIGH
    assert NDFL_PROGRESSIVE_THRESHOLD == _EXPECTED_PROGRESSIVE_THRESHOLD
    assert ASV_CAP_PER_BANK == _EXPECTED_ASV_CAP
    assert DEPOSIT_FLOOR_BASE == _EXPECTED_DEPOSIT_FLOOR_BASE
    assert DEPOSIT_DEMAND_RATE == _EXPECTED_DEPOSIT_DEMAND_RATE


def test_no_duplicate_ndfl_literal_in_brokers() -> None:
    """Neither broker hard-codes a second ``Decimal("0.13")`` literal (D-12).

    Comment-only lines are stripped so header prose does not self-invalidate
    the gate; only executable code is scanned.
    """
    for path in _BROKERS:
        code = "\n".join(
            line
            for line in path.read_text(encoding="utf-8").splitlines()
            if not line.lstrip().startswith("#")
        )
        assert _NDFL_LITERAL not in code, (
            f"{path} still hard-codes the NDFL literal "
            "-- re-point at core.constants.NDFL_RATE (D-12)"
        )
