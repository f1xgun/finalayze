"""Lightweight bootstrap-time validator for strategy preset YAML files (Layer 4).

The combiner already handles malformed presets gracefully at signal time
(``StrategyCombiner._load_config`` catches all parse errors and downstream
callers use ``.get()`` with defaults). This module surfaces *silent* schema
drift before any trades execute:

* Typo'd top-level keys (e.g. ``min_combined_confidance``) that would be
  silently swallowed by ``.get(known_key, default)``.
* Wrong types on known keys (e.g. ``strategies`` set to a list instead of a
  dict, ``regime_routing`` set to a string).
* Unparseable Decimal-like fields that would have raised ``InvalidOperation``
  on the relevant trade path.
* ``normalize_mode`` values outside the known set ``{firing, total, active}``
  (which would silently fall through to ``firing``).

Issues are returned as a list; the bootstrap caller decides whether to log,
escalate, or fail. Validation never raises — its purpose is observability,
not gatekeeping. The combiner's per-segment fail-soft behaviour is preserved
on purpose so a single bad preset cannot take down the entire trading loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    import structlog


# Top-level keys recognised by ``StrategyCombiner`` for equity-style presets.
# Anything else triggers a typo warning.
_EQUITY_TOP_KEYS: frozenset[str] = frozenset(
    {
        "segment_id",
        "normalize_mode",
        "min_combined_confidence",
        "min_exit_confidence",
        "regime_routing",
        "strategies",
    }
)

# Bond-segment presets (ru_ofz_*) have a different shape: they carry their own
# risk + costs blocks and are consumed by bond_carry / bond strategies rather
# than the equity combiner. Validate them against their own key set.
_BOND_TOP_KEYS: frozenset[str] = frozenset(
    {
        "segment_id",
        "market",
        "instrument_type",
        "strategies",
        "risk",
        "costs",
    }
)

_VALID_NORMALIZE_MODES: frozenset[str] = frozenset({"firing", "total", "active"})

_VALID_REGIME_ROUTING_KEYS: frozenset[str] = frozenset(
    {"enabled", "adx_period", "trend_threshold", "mr_threshold"},
)


class PresetSeverity(StrEnum):
    """Severity tier for a preset issue."""

    WARNING = "warning"  # silent drift, current default kicks in
    ERROR = "error"  # would crash mid-trade or produce wrong behaviour


@dataclass(frozen=True, slots=True)
class PresetIssue:
    """A single schema observation about a preset file."""

    file: str
    severity: PresetSeverity
    key: str
    message: str


def _is_dividend_mapping(data: dict[str, object]) -> bool:
    """Return True when the YAML looks like the dividend-data file.

    ``moex_dividends.yaml`` is not a combiner preset — its top-level keys are
    uppercase MOEX tickers mapping to lists of historical dividend records.
    Identify it by the absence of ``segment_id`` together with majority-list
    values keyed by uppercase identifiers.
    """
    if "segment_id" in data:
        return False
    if not data:
        return False
    list_valued_upper = sum(
        1 for k, v in data.items() if isinstance(k, str) and k.isupper() and isinstance(v, list)
    )
    return list_valued_upper >= max(1, len(data) // 2)


def _classify_schema(data: dict[str, object]) -> frozenset[str] | None:
    """Pick the expected key set for this preset, or None if not a preset."""
    if _is_dividend_mapping(data):
        return None
    if "instrument_type" in data and data.get("instrument_type") == "bond":
        return _BOND_TOP_KEYS
    return _EQUITY_TOP_KEYS


def _validate_decimal_field(
    file: str,
    data: dict[str, object],
    key: str,
) -> PresetIssue | None:
    """Verify a Decimal-like field parses cleanly, if present."""
    if key not in data:
        return None
    raw = data[key]
    try:
        Decimal(str(raw))
    except InvalidOperation:
        return PresetIssue(
            file=file,
            severity=PresetSeverity.ERROR,
            key=key,
            message=f"{key}={raw!r} is not parseable as Decimal",
        )
    return None


def _validate_equity_preset(
    file: str,
    data: dict[str, object],
) -> list[PresetIssue]:
    """Run equity-specific checks (strategies dict, normalize_mode, weights)."""
    issues: list[PresetIssue] = []

    strategies_raw = data.get("strategies")
    if strategies_raw is not None and not isinstance(strategies_raw, dict):
        issues.append(
            PresetIssue(
                file=file,
                severity=PresetSeverity.ERROR,
                key="strategies",
                message=f"strategies must be a dict, got {type(strategies_raw).__name__}",
            )
        )
        strategies_raw = {}

    normalize = data.get("normalize_mode")
    if normalize is not None and str(normalize) not in _VALID_NORMALIZE_MODES:
        issues.append(
            PresetIssue(
                file=file,
                severity=PresetSeverity.WARNING,
                key="normalize_mode",
                message=(
                    f"normalize_mode={normalize!r} is unknown; "
                    f"combiner will fall back to 'firing'. "
                    f"Valid values: {sorted(_VALID_NORMALIZE_MODES)}"
                ),
            )
        )

    routing = data.get("regime_routing")
    if routing is not None:
        if not isinstance(routing, dict):
            issues.append(
                PresetIssue(
                    file=file,
                    severity=PresetSeverity.ERROR,
                    key="regime_routing",
                    message=f"regime_routing must be a dict, got {type(routing).__name__}",
                )
            )
        else:
            issues.extend(
                PresetIssue(
                    file=file,
                    severity=PresetSeverity.WARNING,
                    key=f"regime_routing.{k}",
                    message=(
                        f"unknown regime_routing key {k!r}; "
                        f"valid keys: {sorted(_VALID_REGIME_ROUTING_KEYS)}"
                    ),
                )
                for k in routing
                if k not in _VALID_REGIME_ROUTING_KEYS
            )

    for decimal_key in ("min_combined_confidence", "min_exit_confidence"):
        issue = _validate_decimal_field(file, data, decimal_key)
        if issue is not None:
            issues.append(issue)

    if isinstance(strategies_raw, dict):
        for sname, scfg in strategies_raw.items():
            if not isinstance(scfg, dict):
                issues.append(
                    PresetIssue(
                        file=file,
                        severity=PresetSeverity.ERROR,
                        key=f"strategies.{sname}",
                        message=(f"strategy block must be a dict, got {type(scfg).__name__}"),
                    )
                )
                continue
            if "weight" in scfg:
                try:
                    Decimal(str(scfg["weight"]))
                except InvalidOperation:
                    issues.append(
                        PresetIssue(
                            file=file,
                            severity=PresetSeverity.ERROR,
                            key=f"strategies.{sname}.weight",
                            message=(f"weight={scfg['weight']!r} is not parseable as Decimal"),
                        )
                    )

    return issues


def _validate_one_file(path: Path) -> list[PresetIssue]:
    """Validate a single YAML preset file."""
    file = path.name
    try:
        raw = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as exc:
        return [
            PresetIssue(
                file=file,
                severity=PresetSeverity.ERROR,
                key="<file>",
                message=f"could not parse YAML: {exc}",
            )
        ]

    if raw is None:
        return [
            PresetIssue(
                file=file,
                severity=PresetSeverity.WARNING,
                key="<file>",
                message="preset file is empty",
            )
        ]

    if not isinstance(raw, dict):
        return [
            PresetIssue(
                file=file,
                severity=PresetSeverity.ERROR,
                key="<file>",
                message=f"top-level value must be a mapping, got {type(raw).__name__}",
            )
        ]

    expected_keys = _classify_schema(raw)
    if expected_keys is None:
        # Non-preset data file (e.g. moex_dividends.yaml) — skip schema checks.
        return []

    issues: list[PresetIssue] = [
        PresetIssue(
            file=file,
            severity=PresetSeverity.WARNING,
            key=k,
            message=(
                f"unknown top-level key {k!r}; "
                f"valid keys for this preset shape: {sorted(expected_keys)}"
            ),
        )
        for k in raw
        if k not in expected_keys
    ]

    if expected_keys is _EQUITY_TOP_KEYS:
        issues.extend(_validate_equity_preset(file, raw))

    return issues


def validate_presets(presets_dir: Path) -> list[PresetIssue]:
    """Validate every ``*.yaml`` file under ``presets_dir``.

    Returns a flat list of issues sorted by file then severity. Never raises;
    a missing directory yields an empty list so the caller can decide whether
    the absence is itself an error.
    """
    if not presets_dir.is_dir():
        return []

    issues: list[PresetIssue] = []
    for path in sorted(presets_dir.glob("*.yaml")):
        issues.extend(_validate_one_file(path))
    return issues


def log_preset_issues(
    issues: Iterable[PresetIssue],
    logger: structlog.stdlib.BoundLogger,
) -> None:
    """Emit a structured log entry per issue, grouped by severity."""
    for issue in issues:
        log_fn = logger.error if issue.severity is PresetSeverity.ERROR else logger.warning
        log_fn(
            "preset_schema_issue",
            file=issue.file,
            severity=issue.severity.value,
            key=issue.key,
            message=issue.message,
        )
