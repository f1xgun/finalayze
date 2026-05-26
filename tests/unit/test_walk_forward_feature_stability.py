"""S2.1 — Feature-selection stability in the production walk-forward trainer.

The Phase-46 stability fix landed in scripts/auto_ml_research.py (covered by
test_auto_ml_research_feature_selection_stability.py) but was never ported
to the prod path in scripts/training/walk_forward.py. Tests below assert
the same contract for the prod trainer.

Contract:
  FSEL-PROD-01: select_features is called exactly once per train_walk_forward run.
  FSEL-PROD-02: The DataFrame passed to select_features contains the union of
                ALL fold training indices (max history, no test leakage).
  FSEL-PROD-03: Every fold uses the same selected feature list.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# 700 daily samples ≈ 23 months — enough for US WF (12 train + 2 cal + 4 test
# + step + purge gaps). MOEX would need less but US is the harder constraint.
_N_SAMPLES = 700
_N_FEATURES = 8
_FIXED_SELECTED = ["feat_0", "feat_2", "feat_5"]


@pytest.fixture
def synthetic_dataset() -> tuple[list[dict[str, float]], list[int], list[int], list[datetime]]:
    """Synthetic features / labels / hold_bars / timestamps spanning ~16 months daily."""
    rng = np.random.default_rng(42)
    raw = rng.random((_N_SAMPLES, _N_FEATURES))
    features = [
        {f"feat_{j}": float(raw[i, j]) for j in range(_N_FEATURES)} for i in range(_N_SAMPLES)
    ]
    labels = rng.integers(0, 2, _N_SAMPLES).tolist()
    hold_bars = [5] * _N_SAMPLES
    start = datetime(2024, 1, 1, tzinfo=UTC)
    timestamps = [start + timedelta(days=i) for i in range(_N_SAMPLES)]
    return features, labels, hold_bars, timestamps


def _patch_training_internals(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub out heavy trainers + data builder so the test runs in <1s."""
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

    # The fold-level calibrator is heavy; substitute with a no-op.
    from finalayze.ml import calibration as _cal

    class _NoOpCalib:
        is_fitted = False

        def fit(self, *a, **kw):
            self.is_fitted = True

        def predict_proba(self, x):
            return np.full_like(np.asarray(x, dtype=float), 0.6)

    monkeypatch.setattr(_cal, "EnsembleCalibrator", _NoOpCalib)


def _run_trainer(
    features: list[dict[str, float]],
    labels: list[int],
    hold_bars: list[int],
    timestamps: list[datetime],
    tmp_path: Path,
    select_features_spy: MagicMock,
):
    """Invoke train_walk_forward via patches that bypass real data loading."""
    import scripts.training.walk_forward as wf

    with (
        patch.object(
            wf,
            "build_dataset_with_timestamps",
            return_value=(features, labels, None, hold_bars, timestamps),
        ),
        patch.object(wf, "select_features", new=select_features_spy),
    ):
        return wf.train_walk_forward(
            segment_id="ru_blue_chips",
            symbols=["AAPL"],
            output_dir=tmp_path,
            force_save=True,  # skip the gate-pass requirement; we only care about FSEL calls
            seq_bootstrap=False,
        )


def test_select_features_called_once(
    synthetic_dataset: tuple[list[dict[str, float]], list[int], list[int], list[datetime]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FSEL-PROD-01: select_features is invoked exactly once across all folds."""
    features, labels, hold_bars, timestamps = synthetic_dataset
    _patch_training_internals(monkeypatch)

    select_spy = MagicMock(return_value=_FIXED_SELECTED)
    _run_trainer(features, labels, hold_bars, timestamps, tmp_path, select_spy)

    assert select_spy.call_count == 1, (
        f"select_features must be called exactly ONCE per train_walk_forward run, "
        f"got {select_spy.call_count} calls (one-per-fold is the bug this test catches)"
    )


def test_select_features_receives_union_of_train_indices(
    synthetic_dataset: tuple[list[dict[str, float]], list[int], list[int], list[datetime]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FSEL-PROD-02: The DataFrame passed has the UNION of all fold train indices."""
    features, labels, hold_bars, timestamps = synthetic_dataset
    _patch_training_internals(monkeypatch)

    select_spy = MagicMock(return_value=_FIXED_SELECTED)
    _run_trainer(features, labels, hold_bars, timestamps, tmp_path, select_spy)

    # Recompute expected union size from fold generator
    import scripts.training.walk_forward as wf

    folds = wf.generate_walk_forward_folds(timestamps, segment_id="ru_blue_chips")
    union: set[int] = set()
    for train_idx, _cal, _test in folds:
        union.update(train_idx)

    assert select_spy.call_args_list, "select_features was never called"
    first_call_args = select_spy.call_args_list[0]
    train_df_arg = first_call_args.args[0]
    assert len(train_df_arg) == len(union), (
        f"select_features received {len(train_df_arg)} samples but expected {len(union)} "
        f"(union of all train indices across {len(folds)} folds)"
    )
    # Test data must NOT be in the selection set: union size < total samples
    assert len(train_df_arg) < _N_SAMPLES, (
        "Union of training indices spans the whole dataset — test rows would leak"
    )


def test_all_folds_use_same_feature_set(
    synthetic_dataset: tuple[list[dict[str, float]], list[int], list[int], list[datetime]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FSEL-PROD-03: best_selected_features matches the single selection result."""
    features, labels, hold_bars, timestamps = synthetic_dataset
    _patch_training_internals(monkeypatch)

    select_spy = MagicMock(return_value=_FIXED_SELECTED)
    _run_trainer(features, labels, hold_bars, timestamps, tmp_path, select_spy)

    # After a run that force-saves, selected_features.json must contain the fixed set
    selected_path = tmp_path / "ru_blue_chips" / "selected_features.json"
    assert selected_path.exists(), "selected_features.json must be persisted"
    import json

    written = json.loads(selected_path.read_text())
    assert written == _FIXED_SELECTED, (
        f"selected_features.json should be the single stable selection {_FIXED_SELECTED}, "
        f"got {written}"
    )
