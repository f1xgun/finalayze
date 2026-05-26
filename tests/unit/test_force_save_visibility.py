"""S2.3 — Force-saved artefacts must be detectable at load time.

Pre-S2.3 the only signal that a segment had bypassed the quality gate via
`--force-save` was the operator's memory. The audit found that ALL seven
shipped artefacts had `overall_passed=false` yet were silently loaded.

Contract:
  - wf_gate_results.json carries a `force_saved` flag written by the trainer.
  - The loader emits an ml_force_saved_artifact_loaded warning whenever a
    segment with overall_passed=false OR force_saved=true is loaded.
"""

from __future__ import annotations

import contextlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import numpy as np

if TYPE_CHECKING:
    import pytest


def _write_gate_results(seg_dir: Path, *, overall_passed: bool, force_saved: bool) -> None:
    (seg_dir / "wf_gate_results.json").write_text(
        json.dumps(
            {
                "overall_passed": overall_passed,
                "force_saved": force_saved,
                "best_accuracy": 0.50,
                "n_folds": 3,
                "gate_pass_rates": {},
            }
        )
    )


def test_force_saved_artefact_emits_loader_warning(tmp_path: Path) -> None:
    """A force_saved artefact must trigger ml_force_saved_artifact_loaded."""
    from finalayze.ml import loader

    seg_dir = tmp_path / "ru_blue_chips"
    seg_dir.mkdir()
    _write_gate_results(seg_dir, overall_passed=False, force_saved=True)

    # Patch the structlog logger inside the module (caplog only captures
    # stdlib logging, not structlog's direct writes). Loading will fail
    # (no real model files) — we only care that the warning fires before
    # the failure. _load_segment reads wf_gate_results near the top.
    with patch.object(loader, "_log") as mock_log, contextlib.suppress(Exception):
        loader._load_segment("ru_blue_chips", seg_dir)

    warning_events = [
        call_args
        for call_args in mock_log.warning.call_args_list
        if call_args.args and call_args.args[0] == "ml_force_saved_artifact_loaded"
    ]
    assert warning_events, (
        "loader must emit a ml_force_saved_artifact_loaded warning "
        f"(all warning calls: {mock_log.warning.call_args_list})"
    )


def test_gate_passed_artefact_no_warning(tmp_path: Path) -> None:
    """Clean (overall_passed=true, force_saved=false) artefact: no warning."""
    from finalayze.ml import loader

    seg_dir = tmp_path / "ru_blue_chips"
    seg_dir.mkdir()
    _write_gate_results(seg_dir, overall_passed=True, force_saved=False)

    with patch.object(loader, "_log") as mock_log, contextlib.suppress(Exception):
        loader._load_segment("ru_blue_chips", seg_dir)

    warning_events = [
        call_args
        for call_args in mock_log.warning.call_args_list
        if call_args.args and call_args.args[0] == "ml_force_saved_artifact_loaded"
    ]
    assert not warning_events, (
        f"clean artefact must not trigger force_saved warning, got: {warning_events}"
    )


def _make_synthetic(
    n: int = 700,
) -> tuple[list[dict[str, float]], list[int], list[int], list[datetime]]:
    rng = np.random.default_rng(42)
    features = [{f"feat_{j}": float(rng.random()) for j in range(8)} for _ in range(n)]
    labels = rng.integers(0, 2, n).tolist()
    hold_bars = [5] * n
    timestamps = [datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(n)]
    return features, labels, hold_bars, timestamps


def _stub_models(monkeypatch: pytest.MonkeyPatch) -> None:
    import scripts.training.walk_forward as wf

    mock_model = MagicMock()
    mock_model.fit.return_value = None
    mock_model.predict_proba.return_value = 0.6
    mock_model._trained = True
    mock_model._model = object()
    monkeypatch.setattr(wf, "XGBoostModel", lambda **kw: mock_model)
    monkeypatch.setattr(wf, "LightGBMModel", lambda **kw: mock_model)
    monkeypatch.setattr(wf, "CatBoostModel", lambda **kw: mock_model)
    monkeypatch.setattr(wf, "fit_and_save_meta_learner", lambda *a, **kw: None)


def test_walk_forward_writes_force_saved_true_when_gates_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """force_save=True + failing gates → wf_gate_results carries force_saved=true."""
    import scripts.training.walk_forward as wf

    features, labels, hold_bars, timestamps = _make_synthetic()
    _stub_models(monkeypatch)
    select_spy = MagicMock(return_value=["feat_0", "feat_1"])

    with (
        patch.object(
            wf,
            "build_dataset_with_timestamps",
            return_value=(features, labels, None, hold_bars, timestamps),
        ),
        patch.object(wf, "select_features", new=select_spy),
        patch(
            "finalayze.ml.training.quality_gates.evaluate_walk_forward", return_value=(False, {})
        ),  # noqa: E501
    ):
        wf.train_walk_forward(
            segment_id="ru_blue_chips",
            symbols=["SBER"],
            output_dir=tmp_path,
            force_save=True,
            seq_bootstrap=False,
        )

    gate_path = tmp_path / "ru_blue_chips" / "wf_gate_results.json"
    assert gate_path.exists()
    data = json.loads(gate_path.read_text())
    assert data["force_saved"] is True
    assert data["overall_passed"] is False


def test_walk_forward_writes_force_saved_false_when_gates_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Gate passes → force_saved=false even with force_save=True at the CLI."""
    import scripts.training.walk_forward as wf

    features, labels, hold_bars, timestamps = _make_synthetic()
    _stub_models(monkeypatch)
    select_spy = MagicMock(return_value=["feat_0", "feat_1"])

    with (
        patch.object(
            wf,
            "build_dataset_with_timestamps",
            return_value=(features, labels, None, hold_bars, timestamps),
        ),
        patch.object(wf, "select_features", new=select_spy),
        patch("finalayze.ml.training.quality_gates.evaluate_walk_forward", return_value=(True, {})),  # noqa: E501
    ):
        wf.train_walk_forward(
            segment_id="ru_blue_chips",
            symbols=["SBER"],
            output_dir=tmp_path,
            force_save=True,  # CLI flag is set but gates pass → no force-save needed
            seq_bootstrap=False,
        )

    gate_path = tmp_path / "ru_blue_chips" / "wf_gate_results.json"
    data = json.loads(gate_path.read_text())
    assert data["force_saved"] is False
    assert data["overall_passed"] is True
