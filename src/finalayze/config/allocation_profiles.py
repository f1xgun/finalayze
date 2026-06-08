"""Fail-closed loader for the SAA risk-profile weight vectors (SAA-01 / SAA-05).

Reads the committed ``config/allocation_profiles.yaml`` snapshot (the FIXED config
weight vectors, D-03 -- never solver output, Pitfall 8) and parses it into
``dict[RiskProfile, AllocationProfile]`` (the Plan-02 L0 types). Validation is
mandatory (V5): each weight vector MUST sum to 1.0 and be non-negative, every profile
MUST carry a MaxDD cap, and all three deposit-anchored profiles MUST be present.

Layout: each top-level YAML key is a ``RiskProfile`` name mapping to
``{weights: {deposit, ofz_pk, equity}, max_drawdown_pct}``. The reserved
``rebalance_cadence`` scalar (D-08) is informational metadata and is skipped -- the
orchestrator drives the quarterly trigger off ``_quarter_key``.

Honesty rules (fail-closed, mirroring ``backtest/dividend_schedule.py:46``):
- A missing/unparseable/non-mapping file raises ``ConfigurationError`` (no stale
  fallback, Pattern 4 / T-72-09).
- A weight vector not summing to 1.0, a negative weight, an unknown profile name, a
  missing weight, a missing cap, or a missing required profile raises
  ``ConfigurationError`` (V5 / Pitfall-8 guard / T-72-08).
- The loader has NO covariance/expected-return/solver path -- two successive loads are
  byte-identical (D-03 / T-72-10).
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import structlog
import yaml

from finalayze.core.exceptions import ConfigurationError
from finalayze.core.schemas import AllocationProfile, AssetClass, RiskProfile

_LOGGER = structlog.get_logger(__name__)

# ``allocation_profiles.py`` lives at ``src/finalayze/config/``; the committed YAML at
# the repo-root ``config/``. ``.parent`` x4: config -> finalayze -> src -> repo-root.
_PROFILES_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent / "config" / "allocation_profiles.yaml"
)

# The three SAA asset classes every profile vector must carry (D-01).
_REQUIRED_CLASSES = (AssetClass.DEPOSIT, AssetClass.OFZ_PK, AssetClass.EQUITY)

# Each weight vector must sum to exactly this (V5, Decimal-exact).
_WEIGHT_SUM_TARGET = Decimal("1.0")
_ZERO = Decimal(0)

# Reserved informational top-level key (D-08); not a profile entry.
_CADENCE_KEY = "rebalance_cadence"


def load_allocation_profiles(path: Path | None = None) -> dict[RiskProfile, AllocationProfile]:
    """Load + validate the SAA risk-profile weight vectors (SAA-01 / SAA-05).

    Fail-closed (V5): raises ``ConfigurationError`` on a missing/corrupt YAML file, an
    unknown or missing profile, a weight vector that does not sum to 1.0, a negative or
    missing weight, or a missing MaxDD cap. Returns FIXED config vectors (D-03 -- never
    solver output); two successive loads are byte-identical.

    Args:
        path: Optional override for the snapshot location (defaults to the committed
            ``config/allocation_profiles.yaml``).

    Returns:
        Mapping of each ``RiskProfile`` to its validated ``AllocationProfile``
        (Decimal-exact weights summing to 1.0 + its MaxDD cap).

    Raises:
        ConfigurationError: on any defect -- missing/unparseable/non-mapping file,
            unknown profile, missing/negative weight, weights not summing to 1.0,
            missing cap, or a required profile absent.
    """
    target = path or _PROFILES_PATH
    try:
        raw: Any = yaml.safe_load(target.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        msg = f"allocation profiles missing/corrupt at {target}: {exc}"
        raise ConfigurationError(msg) from exc

    if not isinstance(raw, dict):
        msg = f"allocation profiles malformed (expected a mapping) at {target}"
        raise ConfigurationError(msg)

    out: dict[RiskProfile, AllocationProfile] = {}
    for name, body in raw.items():
        if name == _CADENCE_KEY:
            # Informational cadence scalar (D-08), not a profile entry.
            continue
        out[_parse_profile(name, target)] = _build_profile(name, body, target)

    missing = set(RiskProfile) - set(out)
    if missing:
        absent = sorted(p.value for p in missing)
        msg = f"allocation profiles missing required profiles {absent} in {target}"
        raise ConfigurationError(msg)

    _LOGGER.debug("allocation_profiles_loaded", path=str(target), profiles=len(out))
    return out


def _parse_profile(name: Any, target: Path) -> RiskProfile:
    """Resolve a top-level key to a ``RiskProfile``, fail-closed on an unknown name."""
    # ``name`` is an untrusted YAML key (any scalar); a non-matching value raises
    # ValueError from the StrEnum lookup, which we re-raise fail-closed.
    try:
        return RiskProfile(name)
    except ValueError as exc:
        msg = f"unknown risk profile {name!r} in {target}"
        raise ConfigurationError(msg) from exc


def _build_profile(name: Any, body: object, target: Path) -> AllocationProfile:
    """Validate one profile body into an ``AllocationProfile`` (fail-closed, V5)."""
    profile = _parse_profile(name, target)
    if not isinstance(body, dict) or not isinstance(body.get("weights"), dict):
        msg = f"profile {name!r} missing a weights mapping in {target}"
        raise ConfigurationError(msg)

    raw_weights: dict[Any, Any] = body["weights"]
    weights: dict[AssetClass, Decimal] = {}
    for cls in _REQUIRED_CLASSES:
        if cls.value not in raw_weights:
            msg = f"profile {name!r} missing weight for {cls.value!r} in {target}"
            raise ConfigurationError(msg)
        weights[cls] = _to_decimal(raw_weights[cls.value], name, cls.value, target)
        if weights[cls] < _ZERO:
            msg = f"profile {name!r} has a negative weight for {cls.value!r} in {target}"
            raise ConfigurationError(msg)

    if sum(weights.values()) != _WEIGHT_SUM_TARGET:
        total = sum(weights.values())
        msg = f"profile {name!r} weights sum to {total}, expected 1.0, in {target}"
        raise ConfigurationError(msg)

    if "max_drawdown_pct" not in body:
        msg = f"profile {name!r} missing max_drawdown_pct in {target}"
        raise ConfigurationError(msg)
    cap = _to_decimal(body["max_drawdown_pct"], name, "max_drawdown_pct", target)
    if cap < _ZERO:
        msg = f"profile {name!r} has a negative max_drawdown_pct in {target}"
        raise ConfigurationError(msg)

    return AllocationProfile(profile=profile, weights=weights, max_drawdown_pct=cap)


def _to_decimal(value: object, name: object, field: str, target: Path) -> Decimal:
    """Convert a YAML scalar to ``Decimal`` via ``str`` (float-exact), fail-closed."""
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:
        msg = f"profile {name!r} has a non-numeric {field} {value!r} in {target}"
        raise ConfigurationError(msg) from exc
