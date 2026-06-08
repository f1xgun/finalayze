"""RED scaffold: SAA-01 profile config + SAA-05 MaxDD caps (Phase 72 Wave-0).

Pins the L0/L1 allocation-profile contract before it exists:
- the three deposit-anchored risk profiles (conservative / balanced / growth)
  load from a fixed config (D-01 / D-02), each weight vector is byte-exact and
  sums to 1.0 (V5), the OFZ-PK leg is flat 25% across all three (D-01);
- each profile carries a tight absolute MaxDD cap (8/15/25%, SAA-05 / D-04);
- the loader is fail-closed: bad weights (not summing to 1.0) or a missing file
  raise ``ConfigurationError`` (V5 / Pitfall-8 guard);
- the vectors are STATIC config, never solver output (D-03) -- two successive
  loads are byte-identical.

RED now: ``finalayze.core.schemas`` (RiskProfile / AssetClass / AllocationProfile,
Plan 02) + ``finalayze.config.allocation_profiles`` (the loader, Plan 03) do not
exist yet. (Loader module path pinned here = ``finalayze.config.allocation_profiles``;
Plan 03 wires this exact path.)
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

import pytest

from finalayze.config.allocation_profiles import load_allocation_profiles
from finalayze.core.exceptions import ConfigurationError
from finalayze.core.schemas import AllocationProfile, AssetClass, RiskProfile

if TYPE_CHECKING:
    from pathlib import Path

# -- Constants (named -- no magic numbers, ruff PLR2004) ----------------------

# D-01 deposit-anchored weight vectors (each sums to 1.0).
_CONSERVATIVE_DEPOSIT = Decimal("0.60")
_CONSERVATIVE_OFZ_PK = Decimal("0.25")
_CONSERVATIVE_EQUITY = Decimal("0.15")

_BALANCED_DEPOSIT = Decimal("0.45")
_BALANCED_OFZ_PK = Decimal("0.25")
_BALANCED_EQUITY = Decimal("0.30")

_GROWTH_DEPOSIT = Decimal("0.25")
_GROWTH_OFZ_PK = Decimal("0.25")
_GROWTH_EQUITY = Decimal("0.50")

_FLAT_OFZ_PK = Decimal("0.25")  # D-01 flat leg, identical across all three profiles
_VECTOR_SUM = Decimal("1.0")

# D-04 tight absolute MaxDD caps.
_CONSERVATIVE_CAP = Decimal("0.08")
_BALANCED_CAP = Decimal("0.15")
_GROWTH_CAP = Decimal("0.25")

# A deliberately invalid vector: deposit/ofz/equity each 0.5 -> sums to 1.5 != 1.0.
_BAD_WEIGHT = Decimal("0.5")
_MISSING_YAML_NAME = "does_not_exist.yaml"


def test_three_profiles_load() -> None:
    """All three deposit-anchored profiles ship (D-02)."""
    profiles = load_allocation_profiles()
    assert set(profiles) == {
        RiskProfile.CONSERVATIVE,
        RiskProfile.BALANCED,
        RiskProfile.GROWTH,
    }


def test_conservative_weights_exact() -> None:
    """Conservative = {deposit 0.60, ofz_pk 0.25, equity 0.15} (D-01, Decimal-exact)."""
    profiles = load_allocation_profiles()
    assert profiles[RiskProfile.CONSERVATIVE].weights == {
        AssetClass.DEPOSIT: _CONSERVATIVE_DEPOSIT,
        AssetClass.OFZ_PK: _CONSERVATIVE_OFZ_PK,
        AssetClass.EQUITY: _CONSERVATIVE_EQUITY,
    }


def test_balanced_weights_exact() -> None:
    """Balanced = {deposit 0.45, ofz_pk 0.25, equity 0.30} (D-01 / D-02 default)."""
    profiles = load_allocation_profiles()
    assert profiles[RiskProfile.BALANCED].weights == {
        AssetClass.DEPOSIT: _BALANCED_DEPOSIT,
        AssetClass.OFZ_PK: _BALANCED_OFZ_PK,
        AssetClass.EQUITY: _BALANCED_EQUITY,
    }


def test_growth_weights_exact() -> None:
    """Growth = {deposit 0.25, ofz_pk 0.25, equity 0.50} (D-01)."""
    profiles = load_allocation_profiles()
    assert profiles[RiskProfile.GROWTH].weights == {
        AssetClass.DEPOSIT: _GROWTH_DEPOSIT,
        AssetClass.OFZ_PK: _GROWTH_OFZ_PK,
        AssetClass.EQUITY: _GROWTH_EQUITY,
    }


def test_each_vector_sums_to_one() -> None:
    """Every profile's weight vector sums to exactly 1.0 (V5, Decimal-exact)."""
    profiles = load_allocation_profiles()
    for profile in profiles.values():
        assert sum(profile.weights.values()) == _VECTOR_SUM


def test_caps_are_tight_absolute() -> None:
    """Each profile carries its tight absolute MaxDD cap (SAA-05 / D-04)."""
    profiles = load_allocation_profiles()
    assert profiles[RiskProfile.CONSERVATIVE].max_drawdown_pct == _CONSERVATIVE_CAP
    assert profiles[RiskProfile.BALANCED].max_drawdown_pct == _BALANCED_CAP
    assert profiles[RiskProfile.GROWTH].max_drawdown_pct == _GROWTH_CAP


def test_ofz_pk_is_flat_25() -> None:
    """The OFZ-PK leg is flat 25% across all three profiles (D-01 flat leg)."""
    profiles = load_allocation_profiles()
    for profile in profiles.values():
        assert profile.weights[AssetClass.OFZ_PK] == _FLAT_OFZ_PK


def test_fail_closed_on_bad_yaml(tmp_path: Path) -> None:
    """A weight vector not summing to 1.0 fails closed (V5 / Pitfall-8 guard)."""
    bad = tmp_path / "bad_profiles.yaml"
    # deposit 0.5 + ofz_pk 0.5 + equity 0.5 = 1.5 != 1.0 -> must be rejected.
    bad.write_text(
        "conservative:\n"
        f"  max_drawdown_pct: '{_CONSERVATIVE_CAP}'\n"
        "  weights:\n"
        f"    deposit: '{_BAD_WEIGHT}'\n"
        f"    ofz_pk: '{_BAD_WEIGHT}'\n"
        f"    equity: '{_BAD_WEIGHT}'\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigurationError):
        load_allocation_profiles(bad)


def test_fail_closed_on_missing_file(tmp_path: Path) -> None:
    """A missing config path fails closed with ConfigurationError (no stale fallback)."""
    with pytest.raises(ConfigurationError):
        load_allocation_profiles(tmp_path / _MISSING_YAML_NAME)


def test_no_solver_static_vectors() -> None:
    """Loaded vectors are STATIC config -- byte-identical across two loads (D-03)."""
    first = load_allocation_profiles()
    second = load_allocation_profiles()
    assert first == second
    # Belt-and-braces: the dataclass instances compare equal field-for-field.
    for profile in RiskProfile:
        a: AllocationProfile = first[profile]
        b: AllocationProfile = second[profile]
        assert a.weights == b.weights
        assert a.max_drawdown_pct == b.max_drawdown_pct
