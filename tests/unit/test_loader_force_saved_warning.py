"""RETRAIN-03 — loader ``ml_force_saved_artifact_loaded`` warning, both polarities.

Pins the legitimacy signal emitted by ``finalayze.ml.loader._load_segment``: the loader
must warn loudly when a segment's ``wf_gate_results.json`` shows an illegitimate artefact
(``overall_passed:false`` OR ``force_saved:true``) and stay silent on a genuine pass
(``overall_passed:true`` and not ``force_saved``).

The warning is logged BEFORE any model loading (loader.py: "Logged BEFORE any model
loading so the warning fires even when downstream loads raise"), so the absence of real
``.pkl`` files is expected — the subsequent load failure is suppressed and we assert only
on the captured warning. Everything is driven against a ``tmp_path`` segment dir; the
gitignored primary ``models/`` tree is never read or written (threat T-62-02).
"""

from __future__ import annotations

import contextlib
import json
from typing import TYPE_CHECKING
from unittest.mock import patch

if TYPE_CHECKING:
    from pathlib import Path

_WARNING_EVENT = "ml_force_saved_artifact_loaded"
_SEGMENT = "ru_blue_chips"


def _write_gate(seg_dir: Path, *, overall_passed: bool, force_saved: bool) -> None:
    """Write a minimal ``wf_gate_results.json`` for the segment dir."""
    (seg_dir / "wf_gate_results.json").write_text(
        json.dumps(
            {
                "overall_passed": overall_passed,
                "force_saved": force_saved,
                "best_accuracy": 0.50,
                "bh_passed": overall_passed,
            }
        )
    )


def _load_capturing_warnings(seg_dir: Path) -> list:
    """Drive ``_load_segment`` against *seg_dir*, returning the captured warning calls.

    Patches the module-level structlog ``_log`` (caplog only captures stdlib logging, not
    structlog's direct writes). The load is expected to raise once it reaches the missing
    ``.pkl`` files — that is suppressed; the warning fires first.
    """
    from finalayze.ml import loader

    with patch.object(loader, "_log") as mock_log, contextlib.suppress(Exception):
        loader._load_segment(_SEGMENT, seg_dir)

    return [
        call
        for call in mock_log.warning.call_args_list
        if call.args and call.args[0] == _WARNING_EVENT
    ]


def test_warning_fires_on_overall_passed_false(tmp_path: Path) -> None:
    """overall_passed:false (force_saved:false) → warning fires."""
    seg_dir = tmp_path / _SEGMENT
    seg_dir.mkdir()
    _write_gate(seg_dir, overall_passed=False, force_saved=False)

    warnings = _load_capturing_warnings(seg_dir)

    assert warnings, f"loader must emit {_WARNING_EVENT} when overall_passed is false"


def test_warning_fires_on_force_saved_true(tmp_path: Path) -> None:
    """force_saved:true (overall_passed:true) → warning still fires."""
    seg_dir = tmp_path / _SEGMENT
    seg_dir.mkdir()
    _write_gate(seg_dir, overall_passed=True, force_saved=True)

    warnings = _load_capturing_warnings(seg_dir)

    assert warnings, f"loader must emit {_WARNING_EVENT} when force_saved is true"


def test_warning_silent_on_genuine_pass(tmp_path: Path) -> None:
    """Genuine pass (overall_passed:true, not force_saved) → NO warning."""
    seg_dir = tmp_path / _SEGMENT
    seg_dir.mkdir()
    _write_gate(seg_dir, overall_passed=True, force_saved=False)

    warnings = _load_capturing_warnings(seg_dir)

    assert not warnings, f"loader must stay silent on a genuine pass, got: {warnings}"
