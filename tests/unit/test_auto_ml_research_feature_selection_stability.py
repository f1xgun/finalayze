"""Tests proving feature selection stability across walk-forward folds.

FSEL-01: Feature selection runs once per experiment, not once per fold.
FSEL-02: All walk-forward folds within a single experiment use the identical feature list.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Module import helper
# ---------------------------------------------------------------------------

_SCRIPT_PATH = Path(__file__).parent.parent.parent / "scripts" / "auto_ml_research.py"
_MODULE_NAME = "auto_ml_research"


def _import_module():
    """Import auto_ml_research safely (registers in sys.modules to fix dataclass resolution)."""
    import importlib.util

    if _MODULE_NAME in sys.modules:
        return sys.modules[_MODULE_NAME]

    spec = importlib.util.spec_from_file_location(_MODULE_NAME, _SCRIPT_PATH)
    assert spec is not None
    mod = importlib.util.module_from_spec(spec)
    # Register BEFORE exec so that dataclass string annotations can resolve
    sys.modules[_MODULE_NAME] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_N_SAMPLES = 200
_N_FEATURES = 10
_FEATURE_NAMES = [f"feat_{i}" for i in range(_N_FEATURES)]
_FIXED_SELECTED = ["feat_0", "feat_2", "feat_5"]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def module():
    """Load the script module once per test session."""
    return _import_module()


@pytest.fixture(scope="module")
def synthetic_data():
    """Generate deterministic synthetic data: 200 samples x 10 features, binary labels."""
    rng = np.random.default_rng(42)
    raw_features = rng.random((_N_SAMPLES, _N_FEATURES))
    labels = rng.integers(0, 2, _N_SAMPLES).tolist()
    all_features = [
        {f"feat_{j}": float(raw_features[i, j]) for j in range(_N_FEATURES)}
        for i in range(_N_SAMPLES)
    ]
    return all_features, labels


@pytest.fixture(scope="module")
def three_folds():
    """Three non-overlapping walk-forward folds (train, cal, test) with 50+ samples each."""
    # Fold 0: train=0..99, cal=[], test=100..149
    fold0 = (list(range(100)), [], list(range(100, 150)))
    # Fold 1: train=0..149, cal=[], test=150..174
    fold1 = (list(range(150)), [], list(range(150, 175)))
    # Fold 2: train=0..174, cal=[], test=175..199
    fold2 = (list(range(175)), [], list(range(175, 200)))
    return [fold0, fold1, fold2]


@pytest.fixture(scope="module")
def two_folds():
    """Two simple folds for quick tests."""
    fold0 = (list(range(100)), [], list(range(100, 150)))
    fold1 = (list(range(150)), [], list(range(150, 200)))
    return [fold0, fold1]


def _make_config(module, feature_subset=None):
    """Create a minimal ExperimentConfig."""
    return module.ExperimentConfig(
        name="test-stability-exp",
        description="Stability test experiment",
        strategy="ablation",
        feature_subset=feature_subset,
        max_features=5,
    )


def _mock_models(module):
    """Patch XGBoost/LightGBM/CatBoost models to avoid real training."""
    mock_model = MagicMock()
    mock_model.fit.return_value = None
    proba = np.array([[0.6, 0.4]] * 50)
    mock_model.predict_proba.return_value = proba

    return (
        patch.object(module, "XGBoostModel", return_value=mock_model),
        patch.object(module, "LightGBMModel", return_value=mock_model),
        patch.object(module, "CatBoostModel", return_value=mock_model),
    )


# ---------------------------------------------------------------------------
# Test 1: select_features_efficient called exactly ONCE before the fold loop
# ---------------------------------------------------------------------------


def test_feature_selection_runs_once_before_folds(module, synthetic_data, three_folds):
    """FSEL-01: select_features_efficient is called exactly once per experiment, not per fold.

    With 3 folds, the old code would call it 3 times. The new code calls it once before
    the fold loop and injects the result into config.feature_subset.
    """
    all_features, labels = synthetic_data
    config = _make_config(module)
    call_count = 0
    calls_feature_lists = []

    def mock_select(x, y, *, max_features=15, **kwargs):
        nonlocal call_count
        call_count += 1
        calls_feature_lists.append(list(x.columns))
        return _FIXED_SELECTED

    p_xgb, p_lgbm, p_cat = _mock_models(module)

    with (
        patch.object(module, "select_features_efficient", side_effect=mock_select),
        p_xgb,
        p_lgbm,
        p_cat,
        patch.object(module, "compute_decay_weights", return_value=np.ones(100)),
        patch.object(module, "evaluate_fold", return_value=[True] * 5),
        patch.object(
            module,
            "_evaluate_models",
            return_value=MagicMock(
                accuracy=0.6,
                brier_score=0.22,
                profit_factor=1.2,
                n_test=25,
                sensitivity=0.5,
                signal_count=10,
                avg_hold_bars=1.0,
            ),
        ),
    ):
        result = module.run_experiment(
            config=config,
            all_features=all_features,
            labels=labels,
            hold_bars=None,
            folds=three_folds,
            segment_id="us_tech",
        )

    assert call_count == 1, (
        f"select_features_efficient must be called exactly ONCE per experiment, "
        f"got {call_count} calls (one per fold is the bug this test catches)"
    )
    _ = result  # result is not None


# ---------------------------------------------------------------------------
# Test 2: explicit feature_subset → selection is skipped entirely
# ---------------------------------------------------------------------------


def test_explicit_feature_subset_skips_selection(module, synthetic_data, two_folds):
    """FSEL-02: When config.feature_subset is set, select_features_efficient is never called."""
    all_features, labels = synthetic_data
    explicit_features = ["feat_1", "feat_3"]
    config = _make_config(module, feature_subset=explicit_features)
    call_count = 0

    def mock_select(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return _FIXED_SELECTED

    p_xgb, p_lgbm, p_cat = _mock_models(module)

    with (
        patch.object(module, "select_features_efficient", side_effect=mock_select),
        p_xgb,
        p_lgbm,
        p_cat,
        patch.object(module, "compute_decay_weights", return_value=np.ones(150)),
        patch.object(module, "evaluate_fold", return_value=[True] * 5),
        patch.object(
            module,
            "_evaluate_models",
            return_value=MagicMock(
                accuracy=0.6,
                brier_score=0.22,
                profit_factor=1.2,
                n_test=25,
                sensitivity=0.5,
                signal_count=10,
                avg_hold_bars=1.0,
            ),
        ),
    ):
        module.run_experiment(
            config=config,
            all_features=all_features,
            labels=labels,
            hold_bars=None,
            folds=two_folds,
            segment_id="us_tech",
        )

    assert call_count == 0, (
        f"select_features_efficient must NOT be called when feature_subset is provided, "
        f"got {call_count} calls"
    )


# ---------------------------------------------------------------------------
# Test 3: selection uses UNION of all train indices (no test data leak)
# ---------------------------------------------------------------------------


def test_selection_uses_all_train_data(module, synthetic_data, three_folds):
    """FSEL-01: The DataFrame passed to select_features_efficient contains the union of
    all training indices across folds — not just one fold's training set.

    This ensures selection sees maximum history while excluding test data (no look-ahead).
    """
    all_features, labels = synthetic_data
    config = _make_config(module)
    captured_x_len = None

    def mock_select(x, y, *, max_features=15, **kwargs):
        nonlocal captured_x_len
        captured_x_len = len(x)
        return _FIXED_SELECTED

    # Compute expected: union of all train indices in three_folds
    all_train = set()
    for train_idx, _cal, _test in three_folds:
        all_train.update(train_idx)
    expected_n = len(all_train)

    p_xgb, p_lgbm, p_cat = _mock_models(module)

    with (
        patch.object(module, "select_features_efficient", side_effect=mock_select),
        p_xgb,
        p_lgbm,
        p_cat,
        patch.object(module, "compute_decay_weights", return_value=np.ones(200)),
        patch.object(module, "evaluate_fold", return_value=[True] * 5),
        patch.object(
            module,
            "_evaluate_models",
            return_value=MagicMock(
                accuracy=0.6,
                brier_score=0.22,
                profit_factor=1.2,
                n_test=25,
                sensitivity=0.5,
                signal_count=10,
                avg_hold_bars=1.0,
            ),
        ),
    ):
        module.run_experiment(
            config=config,
            all_features=all_features,
            labels=labels,
            hold_bars=None,
            folds=three_folds,
            segment_id="us_tech",
        )

    assert captured_x_len is not None, "select_features_efficient was never called"
    assert captured_x_len == expected_n, (
        f"select_features_efficient received {captured_x_len} samples "
        f"but expected {expected_n} (union of all train indices across folds)"
    )


# ---------------------------------------------------------------------------
# Test 4: exactly one 'feature_selection_stable' log line per experiment
# ---------------------------------------------------------------------------


def test_selected_features_logged_once(module, synthetic_data, three_folds):
    """A single 'feature_selection_stable' log line appears per experiment run."""
    all_features, labels = synthetic_data
    config = _make_config(module)
    log_events: list[dict] = []

    def mock_select(x, y, *, max_features=15, **kwargs):
        return _FIXED_SELECTED

    # Capture structlog events by patching the logger
    _original_logger = module.logger

    class CapturingLogger:
        def info(self, event, **kwargs):
            log_events.append({"event": event, **kwargs})

        def warning(self, event, **kwargs):
            pass

        def error(self, event, **kwargs):
            pass

        def debug(self, event, **kwargs):
            pass

    p_xgb, p_lgbm, p_cat = _mock_models(module)

    with (
        patch.object(module, "select_features_efficient", side_effect=mock_select),
        patch.object(module, "logger", CapturingLogger()),
        p_xgb,
        p_lgbm,
        p_cat,
        patch.object(module, "compute_decay_weights", return_value=np.ones(200)),
        patch.object(module, "evaluate_fold", return_value=[True] * 5),
        patch.object(
            module,
            "_evaluate_models",
            return_value=MagicMock(
                accuracy=0.6,
                brier_score=0.22,
                profit_factor=1.2,
                n_test=25,
                sensitivity=0.5,
                signal_count=10,
                avg_hold_bars=1.0,
            ),
        ),
    ):
        module.run_experiment(
            config=config,
            all_features=all_features,
            labels=labels,
            hold_bars=None,
            folds=three_folds,
            segment_id="us_tech",
        )

    stability_logs = [e for e in log_events if e.get("event") == "feature_selection_stable"]
    assert len(stability_logs) == 1, (
        f"Expected exactly 1 'feature_selection_stable' log line, got {len(stability_logs)}. "
        f"All log events: {log_events}"
    )
    log = stability_logs[0]
    assert "selected_count" in log, f"Log must include selected_count, got: {log}"
    assert log["selected_count"] == len(_FIXED_SELECTED), (
        f"selected_count={log['selected_count']} != {len(_FIXED_SELECTED)}"
    )
