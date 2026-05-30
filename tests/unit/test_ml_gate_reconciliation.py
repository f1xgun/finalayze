"""Phase 61 Stage 0 — reconciliation invariant: ml_ensemble.enabled vs gate status.

Guards the truth between each MOEX ``ru_*`` preset's ``ml_ensemble.enabled`` flag and
the model's recorded walk-forward gate status. The core invariant is fail-closed: a
preset may only enable ``ml_ensemble`` if its model *legitimately* passed the gate
(model dir present, ``wf_gate_results.json`` present, a truthy ``gate_passed`` or an
exactly-``True`` ``overall_passed``, and NOT ``force_saved``). Absent dir / absent file
/ absent-or-null gate status / force-saved => treated as "not passed" => the preset MUST
disable ml_ensemble.

The models dir is resolved via the ``FINALAYZE_MODELS_DIR`` env override (default
project-root ``models/``) so the test works both in this worktree (where ``models/`` is
absent — absent-dir => not-passed => disabled is the conservative correct outcome) and
against the primary checkout's real artefacts.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import yaml

_PRESETS_DIR = (
    Path(__file__).resolve().parent.parent.parent / "src" / "finalayze" / "strategies" / "presets"
)

_MODELS_DIR = Path(
    os.environ.get(
        "FINALAYZE_MODELS_DIR",
        str(Path(__file__).resolve().parent.parent.parent / "models"),
    )
)

# All ru_* equity segments whose presets carry an ml_ensemble block (or could).
_RU_ML_SEGMENTS = ["ru_blue_chips", "ru_tech", "ru_energy", "ru_finance"]

# Segments expected to be force-disabled by Phase 61 (gate not legitimately passed).
_DISABLED_SEGMENTS = ["ru_blue_chips", "ru_tech"]

# Segments that must remain disabled per D-02 (disable-only): this phase never flips on.
_REMAIN_DISABLED_SEGMENTS = ["ru_energy", "ru_finance"]

# Bond presets that must never carry an ml_ensemble block.
_RU_BOND_SEGMENTS = ["ru_ofz_pd", "ru_ofz_pk"]

_GATE_FILE = "wf_gate_results.json"
_EXPECTED_DISABLED_WEIGHT = 0.0


def _load_preset(name: str) -> dict:
    """Load a YAML preset by name (mirrors test_moex_preset_validation pattern)."""
    path = _PRESETS_DIR / f"{name}.yaml"
    assert path.exists(), f"Preset file {path} does not exist"
    with path.open() as f:
        return yaml.safe_load(f)


def _model_legitimately_passed(seg: str) -> bool:
    """Fail-closed check that a segment's model legitimately passed the gate.

    Returns False on any of: absent model dir, absent gate file, force-saved artefact,
    or a gate result lacking a truthy ``gate_passed`` AND an exactly-``True``
    ``overall_passed``. Only an honestly-recorded pass returns True.
    """
    seg_dir = _MODELS_DIR / seg
    if not seg_dir.exists():
        return False

    gate_path = seg_dir / _GATE_FILE
    if not gate_path.exists():
        return False

    with gate_path.open() as f:
        gate = json.load(f)

    if gate.get("force_saved"):
        return False

    return bool(gate.get("gate_passed")) or gate.get("overall_passed") is True


def test_no_ru_preset_enables_ml_on_unpassed_model() -> None:
    """Core invariant: no ru_* preset enables ml_ensemble on an unpassed model."""
    for seg in _RU_ML_SEGMENTS:
        preset = _load_preset(seg)
        ml_enabled = preset["strategies"].get("ml_ensemble", {}).get("enabled", False)
        if ml_enabled:
            assert _model_legitimately_passed(seg), (
                f"{seg}: ml_ensemble enabled but model did not legitimately pass the gate"
            )


@pytest.mark.parametrize("seg", _DISABLED_SEGMENTS)
def test_disabled_ml_ensemble_has_zero_weight(seg: str) -> None:
    """ru_blue_chips and ru_tech: ml_ensemble disabled with weight zeroed."""
    preset = _load_preset(seg)
    ml_block = preset["strategies"].get("ml_ensemble", {})
    assert ml_block.get("enabled") is False, f"{seg}: ml_ensemble must be disabled"
    assert ml_block.get("weight") == pytest.approx(_EXPECTED_DISABLED_WEIGHT), (
        f"{seg}: disabled ml_ensemble must have weight 0.00"
    )


@pytest.mark.parametrize("seg", _RU_BOND_SEGMENTS)
def test_ru_ofz_presets_have_no_ml_ensemble(seg: str) -> None:
    """Guard: bond presets carry no ml_ensemble block (defensive against future adds)."""
    preset = _load_preset(seg)
    assert "ml_ensemble" not in preset["strategies"], (
        f"{seg}: bond preset must not declare an ml_ensemble block"
    )


@pytest.mark.parametrize("seg", _REMAIN_DISABLED_SEGMENTS)
def test_ru_energy_and_ru_finance_remain_disabled(seg: str) -> None:
    """Regression guard for D-02 disable-only: these must NOT be flipped on by Phase 61."""
    preset = _load_preset(seg)
    ml_enabled = preset["strategies"].get("ml_ensemble", {}).get("enabled", False)
    assert ml_enabled is False, f"{seg}: must remain disabled (disable-only, D-02)"
