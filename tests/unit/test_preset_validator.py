"""Unit tests for ``finalayze.strategies.preset_validator``.

The validator's job is observability, not gatekeeping — it must never raise,
must classify the three known YAML shapes (equity preset, bond preset,
dividend data file), and must surface real silent-drift risks:
typo'd top-level keys, wrong types, unparseable Decimals, unknown
``normalize_mode`` values, and unknown ``regime_routing`` sub-keys.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from finalayze.strategies.preset_validator import (
    PresetIssue,
    PresetSeverity,
    log_preset_issues,
    validate_presets,
)


def _write(path: Path, text: str) -> Path:
    path.write_text(text)
    return path


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


def test_validate_presets_empty_dir_yields_no_issues(tmp_path: Path) -> None:
    """Directory with no YAML files returns an empty list, never raises."""
    assert validate_presets(tmp_path) == []


def test_validate_presets_missing_dir_yields_no_issues(tmp_path: Path) -> None:
    """A nonexistent directory yields an empty list, never raises."""
    assert validate_presets(tmp_path / "does_not_exist") == []


def test_valid_equity_preset_has_no_issues(tmp_path: Path) -> None:
    """A minimal valid equity preset produces zero issues."""
    _write(
        tmp_path / "us_test.yaml",
        """
segment_id: us_test
normalize_mode: firing
min_combined_confidence: 0.50
min_exit_confidence: 0.25
regime_routing:
  enabled: true
  adx_period: 14
  trend_threshold: 30
  mr_threshold: 20
strategies:
  momentum:
    enabled: true
    weight: 0.5
""",
    )
    assert validate_presets(tmp_path) == []


def test_valid_bond_preset_has_no_issues(tmp_path: Path) -> None:
    """A bond-shape preset is validated against the bond key set."""
    _write(
        tmp_path / "ru_ofz_test.yaml",
        """
segment_id: ru_ofz_test
market: moex
instrument_type: bond
strategies:
  bond_duration_rotation:
    enabled: true
    weight: 0.7
risk:
  max_positions: 5
costs:
  normal: moex_bond
""",
    )
    assert validate_presets(tmp_path) == []


def test_dividend_data_file_is_skipped(tmp_path: Path) -> None:
    """``moex_dividends.yaml`` shape (ticker→list mapping) is not a preset."""
    _write(
        tmp_path / "moex_dividends.yaml",
        """
SBER:
  - {ex_date: "2025-01-01", amount: 10.0, status: paid}
GAZP:
  - {ex_date: "2025-02-01", amount: 5.0, status: paid}
""",
    )
    assert validate_presets(tmp_path) == []


# ---------------------------------------------------------------------------
# Drift detection — the actual point of this module
# ---------------------------------------------------------------------------


def test_typo_in_top_level_key_is_flagged(tmp_path: Path) -> None:
    """A typo'd key like 'min_combined_confidance' must surface as a WARNING."""
    _write(
        tmp_path / "us_typo.yaml",
        """
segment_id: us_typo
min_combined_confidance: 0.50
strategies: {}
""",
    )
    issues = validate_presets(tmp_path)
    matches = [i for i in issues if i.key == "min_combined_confidance"]
    assert len(matches) == 1
    assert matches[0].severity is PresetSeverity.WARNING
    assert "unknown top-level key" in matches[0].message


def test_unknown_normalize_mode_is_flagged(tmp_path: Path) -> None:
    """A normalize_mode outside {firing,total,active} surfaces as WARNING."""
    _write(
        tmp_path / "us_norm.yaml",
        """
segment_id: us_norm
normalize_mode: weighted_geometric_mean
strategies: {}
""",
    )
    issues = validate_presets(tmp_path)
    matches = [i for i in issues if i.key == "normalize_mode"]
    assert len(matches) == 1
    assert matches[0].severity is PresetSeverity.WARNING


def test_unparseable_min_confidence_decimal_is_error(tmp_path: Path) -> None:
    """A non-numeric min_combined_confidence value is an ERROR."""
    _write(
        tmp_path / "us_bad_dec.yaml",
        """
segment_id: us_bad_dec
min_combined_confidence: not_a_number
strategies: {}
""",
    )
    issues = validate_presets(tmp_path)
    matches = [i for i in issues if i.key == "min_combined_confidence"]
    assert len(matches) == 1
    assert matches[0].severity is PresetSeverity.ERROR
    assert "not parseable as Decimal" in matches[0].message


def test_strategies_must_be_dict(tmp_path: Path) -> None:
    """``strategies: []`` triggers a type ERROR."""
    _write(
        tmp_path / "us_bad_strat.yaml",
        """
segment_id: us_bad_strat
strategies:
  - foo
  - bar
""",
    )
    issues = validate_presets(tmp_path)
    matches = [i for i in issues if i.key == "strategies"]
    assert len(matches) == 1
    assert matches[0].severity is PresetSeverity.ERROR


def test_regime_routing_must_be_dict(tmp_path: Path) -> None:
    """``regime_routing: 'enabled'`` triggers a type ERROR."""
    _write(
        tmp_path / "us_bad_routing.yaml",
        """
segment_id: us_bad_routing
regime_routing: enabled
strategies: {}
""",
    )
    issues = validate_presets(tmp_path)
    matches = [i for i in issues if i.key == "regime_routing"]
    assert len(matches) == 1
    assert matches[0].severity is PresetSeverity.ERROR


def test_unknown_regime_routing_subkey_is_warning(tmp_path: Path) -> None:
    """Typo inside ``regime_routing`` is surfaced as a sub-key WARNING."""
    _write(
        tmp_path / "us_routing_typo.yaml",
        """
segment_id: us_routing_typo
regime_routing:
  enabled: true
  trend_treshold: 30
strategies: {}
""",
    )
    issues = validate_presets(tmp_path)
    matches = [i for i in issues if i.key == "regime_routing.trend_treshold"]
    assert len(matches) == 1
    assert matches[0].severity is PresetSeverity.WARNING


def test_bad_strategy_weight_is_error(tmp_path: Path) -> None:
    """A non-numeric ``weight`` inside a strategy block is an ERROR."""
    _write(
        tmp_path / "us_bad_weight.yaml",
        """
segment_id: us_bad_weight
strategies:
  momentum:
    enabled: true
    weight: abc
""",
    )
    issues = validate_presets(tmp_path)
    matches = [i for i in issues if i.key == "strategies.momentum.weight"]
    assert len(matches) == 1
    assert matches[0].severity is PresetSeverity.ERROR


def test_strategy_block_must_be_dict(tmp_path: Path) -> None:
    """``strategies.momentum: "enabled"`` (string) triggers an ERROR."""
    _write(
        tmp_path / "us_bad_block.yaml",
        """
segment_id: us_bad_block
strategies:
  momentum: enabled
""",
    )
    issues = validate_presets(tmp_path)
    matches = [i for i in issues if i.key == "strategies.momentum"]
    assert len(matches) == 1
    assert matches[0].severity is PresetSeverity.ERROR


# ---------------------------------------------------------------------------
# File-level edge cases
# ---------------------------------------------------------------------------


def test_malformed_yaml_is_error(tmp_path: Path) -> None:
    """A YAML parse error becomes an ERROR file-level issue."""
    _write(tmp_path / "broken.yaml", ": bad: yaml: ][")
    issues = validate_presets(tmp_path)
    assert len(issues) == 1
    assert issues[0].severity is PresetSeverity.ERROR
    assert issues[0].key == "<file>"


def test_empty_yaml_file_is_warning(tmp_path: Path) -> None:
    """An empty file is suspicious but not catastrophic; surface as WARNING."""
    _write(tmp_path / "empty.yaml", "")
    issues = validate_presets(tmp_path)
    assert len(issues) == 1
    assert issues[0].severity is PresetSeverity.WARNING
    assert issues[0].key == "<file>"


def test_top_level_list_is_error(tmp_path: Path) -> None:
    """A YAML file with a top-level list is an ERROR (not a mapping)."""
    _write(
        tmp_path / "list.yaml",
        """
- one
- two
""",
    )
    issues = validate_presets(tmp_path)
    assert len(issues) == 1
    assert issues[0].severity is PresetSeverity.ERROR


def test_log_preset_issues_routes_by_severity() -> None:
    """ERROR severities call logger.error; WARNING calls logger.warning."""
    issues = [
        PresetIssue(
            file="a.yaml",
            severity=PresetSeverity.ERROR,
            key="x",
            message="boom",
        ),
        PresetIssue(
            file="b.yaml",
            severity=PresetSeverity.WARNING,
            key="y",
            message="meh",
        ),
    ]
    seen: list[tuple[str, str, str]] = []

    class _StubLog:
        def error(self, event: str, **kw: object) -> None:
            seen.append(("error", event, str(kw.get("file"))))

        def warning(self, event: str, **kw: object) -> None:
            seen.append(("warning", event, str(kw.get("file"))))

    log_preset_issues(issues, _StubLog())  # type: ignore[arg-type]

    assert seen == [
        ("error", "preset_schema_issue", "a.yaml"),
        ("warning", "preset_schema_issue", "b.yaml"),
    ]


# ---------------------------------------------------------------------------
# Real-repo regression — the validator must not flag the shipping presets.
# ---------------------------------------------------------------------------


def test_real_presets_directory_passes() -> None:
    """The shipping presets under src/.../presets/ must not produce ERRORs.

    WARNINGs are tolerated (e.g. ru_tech missing optional keys), but a fresh
    ERROR here means an existing preset would have been silently broken
    before this validator landed.
    """
    presets_dir = (
        Path(__file__).resolve().parents[2] / "src" / "finalayze" / "strategies" / "presets"
    )
    issues = validate_presets(presets_dir)
    errors = [i for i in issues if i.severity is PresetSeverity.ERROR]
    assert errors == [], f"Unexpected ERRORs in shipping presets: {errors}"


_PRESETS_DIR = Path(__file__).resolve().parents[2] / "src" / "finalayze" / "strategies" / "presets"

# Phase 68 activation presets (Wave 3 liquid + Wave 4 thin). Each must validate
# with zero ERROR-severity issues (UNIV-03).
_LIQUID_ACTIVATED_PRESETS = ("ru_metals", "ru_consumer", "ru_construction")
_THIN_ACTIVATED_PRESETS = ("ru_telecom", "ru_transport", "ru_chemicals")
_ACTIVATED_PRESETS = _LIQUID_ACTIVATED_PRESETS + _THIN_ACTIVATED_PRESETS


@pytest.mark.parametrize("segment_id", _ACTIVATED_PRESETS)
def test_liquid_activation_preset_has_no_errors(segment_id: str) -> None:
    """UNIV-03: each activation preset (liquid + thin) yields zero ERROR issues."""
    preset_path = _PRESETS_DIR / f"{segment_id}.yaml"
    assert preset_path.is_file(), f"missing activation preset {preset_path}"
    issues = validate_presets(_PRESETS_DIR)
    errors = [
        i for i in issues if i.file == f"{segment_id}.yaml" and i.severity is PresetSeverity.ERROR
    ]
    assert errors == [], f"{segment_id}.yaml ERRORs: {errors}"


def test_combiner_exposes_presets_dir() -> None:
    """``StrategyCombiner.presets_dir`` is the public seam this hook depends on."""
    pytest.importorskip("finalayze.strategies.combiner")
    from finalayze.strategies.combiner import StrategyCombiner

    combiner = StrategyCombiner(strategies=[])
    assert combiner.presets_dir.name == "presets"
    assert combiner.presets_dir.is_dir()
