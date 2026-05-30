"""Reconciliation invariant: ``ml_ensemble.enabled`` vs recorded gate status.

Guards the truth between each MOEX ``ru_*`` preset's ``ml_ensemble.enabled`` flag and
the model's recorded walk-forward gate status. The core invariant is **fail-closed**: a
preset may only enable ``ml_ensemble`` if its model *legitimately* passed the gate
(model dir present, ``wf_gate_results.json`` present, a truthy ``gate_passed`` or an
exactly-``True`` ``overall_passed``, and NOT ``force_saved``). Absent dir / absent file
/ absent-or-null gate status / force-saved => treated as "not passed" => the preset MUST
disable ml_ensemble.

Phase 62 (Stage 1) supersedes the Phase-61 disable-only rule (D-04): segments MAY be
legitimately re-enabled once their model honestly passes the gate. This module therefore
NO LONGER hard-asserts that ru_energy / ru_finance stay disabled (Pitfall 1 — that
assertion would break CI on a legitimate enable). Instead:

  - the zero-weight guard is **preset-driven** (iterate segments whose preset disables
    ml_ensemble) so a later legitimate enable auto-excludes that segment, and
  - an enabled segment must satisfy the fail-closed ``enabled => legitimately passed``
    invariant (``test_enabled_segment_requires_legitimate_pass``).

The models dir is resolved via the ``FINALAYZE_MODELS_DIR`` env override (default
project-root ``models/``). Tests that exercise the legitimacy check against a controlled
artefact build a tmp models dir (``monkeypatch`` on the module ``_MODELS_DIR``) so they
never depend on the gitignored primary ``models/`` tree.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
import yaml

# Reference to this module so the fixture can monkeypatch the module-level
# ``_MODELS_DIR`` (avoids a self-import; ruff PLW0406).
_self_module = sys.modules[__name__]

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


def _ml_block(seg: str) -> dict:
    """Return the ml_ensemble block for a segment preset (empty dict if absent)."""
    return _load_preset(seg)["strategies"].get("ml_ensemble", {})


def _is_ml_enabled(seg: str) -> bool:
    """True iff the segment preset enables ml_ensemble."""
    return _ml_block(seg).get("enabled", False) is True


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


def _disabled_ru_segments() -> list[str]:
    """ru_* segments whose preset currently DISABLES ml_ensemble (preset-driven)."""
    return [seg for seg in _RU_ML_SEGMENTS if not _is_ml_enabled(seg)]


def _enabled_ru_segments() -> list[str]:
    """ru_* segments whose preset currently ENABLES ml_ensemble (preset-driven)."""
    return [seg for seg in _RU_ML_SEGMENTS if _is_ml_enabled(seg)]


def test_no_ru_preset_enables_ml_on_unpassed_model() -> None:
    """Core invariant: no ru_* preset enables ml_ensemble on an unpassed model."""
    for seg in _RU_ML_SEGMENTS:
        preset = _load_preset(seg)
        ml_enabled = preset["strategies"].get("ml_ensemble", {}).get("enabled", False)
        if ml_enabled:
            assert _model_legitimately_passed(seg), (
                f"{seg}: ml_ensemble enabled but model did not legitimately pass the gate"
            )


def test_disabled_ml_ensemble_has_zero_weight() -> None:
    """Preset-driven: every ru_* segment that DISABLES ml_ensemble must have weight 0.00.

    Not static-list driven (Pitfall 1): a segment that a later wave legitimately ENABLES
    is automatically excluded here (its ``enabled`` flips to True), so the per-segment
    enable waves never need to touch this test and a genuine pass cannot break CI.
    """
    disabled = _disabled_ru_segments()
    # Sanity: there must be at least one disabled ru_ segment for this guard to be meaningful
    # (all four are disabled today; a future wave may enable some, never all four at once
    # in a single Stage-1 plan).
    assert disabled, "expected at least one disabled ru_ ml_ensemble segment"
    for seg in disabled:
        ml_block = _ml_block(seg)
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


@pytest.fixture
def _tmp_models_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the module ``_MODELS_DIR`` at a controlled tmp tree.

    Avoids depending on the gitignored primary ``models/`` tree (threat T-62-02).
    """
    models = tmp_path / "models"
    models.mkdir()
    monkeypatch.setattr(_self_module, "_MODELS_DIR", models)
    return models


def _seed_gate(models: Path, seg: str, *, overall_passed: bool, force_saved: bool) -> None:
    """Write a ``wf_gate_results.json`` for a segment under the tmp models dir."""
    seg_dir = models / seg
    seg_dir.mkdir(parents=True, exist_ok=True)
    (seg_dir / _GATE_FILE).write_text(
        json.dumps({"overall_passed": overall_passed, "force_saved": force_saved})
    )


def test_enabled_segment_requires_legitimate_pass(_tmp_models_dir: Path) -> None:
    """Fail-closed: any preset-enabled ru_ segment must have an honest passing artefact.

    For every currently-enabled segment we seed a genuine pass and assert the invariant
    holds. We also assert directly that a NOT-passed (overall_passed:false) artefact would
    FAIL the legitimacy check — so a segment flipped on without an honest pass is rejected.
    """
    enabled = _enabled_ru_segments()

    # Positive case: each enabled segment with a genuine pass satisfies the invariant.
    for seg in enabled:
        _seed_gate(_tmp_models_dir, seg, overall_passed=True, force_saved=False)
        assert _model_legitimately_passed(seg), (
            f"{seg}: ml_ensemble enabled but the seeded genuine pass was not accepted"
        )

    # Negative case (always exercised, independent of today's enabled set): an
    # overall_passed:false artefact must NOT count as a legitimate pass, so an enabled
    # segment backed by it would fail the fail-closed invariant.
    probe = "ru_blue_chips"
    _seed_gate(_tmp_models_dir, probe, overall_passed=False, force_saved=False)
    assert not _model_legitimately_passed(probe), (
        f"{probe}: overall_passed:false must NOT be treated as a legitimate gate pass"
    )

    # Negative case: a force_saved artefact is never legitimate, even if overall_passed.
    _seed_gate(_tmp_models_dir, probe, overall_passed=True, force_saved=True)
    assert not _model_legitimately_passed(probe), (
        f"{probe}: force_saved artefact must NOT be treated as a legitimate gate pass"
    )
