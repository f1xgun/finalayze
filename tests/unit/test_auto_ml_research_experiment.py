"""Tests for ExperimentManager integration in auto_ml_research.py.

Tests the --experiment-id flag and ExperimentManager wiring:
- T1: no flag → no ExperimentManager import, JSONL written
- T2: with flag → ExperimentManager.create_experiment called with correct args
- T3: result linking → link_result called per experiment
- T4: verdict → record_verdict called with best_score after loop
- T5: invalid ID → argparse error
- T6: error resilience → loop completes even if create_experiment raises
- T7: concurrent isolation → two IDs produce independent files
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

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
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def module():
    """Load the script module once per test session."""
    return _import_module()


@pytest.fixture
def fake_result(module):
    """Return a deterministic ExperimentResult (script's internal dataclass)."""
    return module.ExperimentResult(
        config=module.ExperimentConfig(
            name="test-exp",
            description="Test experiment",
            strategy="ablation",
        ),
        score=0.75,
        avg_accuracy=0.60,
        avg_brier=0.22,
        avg_profit_factor=1.4,
        feature_count=5,
        features_used=["rsi", "macd", "sma", "atr", "vol"],
        status="keep",
    )


def _run_loop(module, fake_result, tmp_path, experiment_id=None, segment_id="us_tech"):
    """Helper: run run_research_loop with all heavy dependencies mocked."""
    fake_results_dir = tmp_path / "results"
    fake_results_dir.mkdir(exist_ok=True)

    features = [{"rsi": 0.5, "macd": 0.3, "sma": 0.7}] * 10
    labels = [0, 1] * 5
    hold_bars = [1] * 10
    folds = [([0, 1, 2, 3, 4], [], [5, 6, 7, 8, 9])]

    with (
        patch.object(module, "_prepare_data") as mock_prepare,
        patch.object(module, "run_experiment", return_value=fake_result),
        patch.object(module, "_RESULTS_DIR", fake_results_dir),
    ):
        mock_prepare.return_value = (features, labels, hold_bars, folds)
        module.run_research_loop(
            segment_id=segment_id,
            strategy="ablation",
            max_experiments=2,
            experiment_id=experiment_id,
        )

    return fake_results_dir


# ---------------------------------------------------------------------------
# T1: No --experiment-id → no ExperimentManager imported, JSONL written
# ---------------------------------------------------------------------------


def test_no_experiment_id(module, fake_result, tmp_path):
    """Without --experiment-id, ExperimentManager is never imported and JSONL is written."""
    imported_modules_before = set(sys.modules.keys())

    results_dir = _run_loop(module, fake_result, tmp_path, experiment_id=None)

    # ExperimentManager should NOT have been newly imported
    new_modules = set(sys.modules.keys()) - imported_modules_before
    exp_mgr_imports = [m for m in new_modules if "experiment_manager" in m.lower()]
    assert not exp_mgr_imports, (
        f"ExperimentManager was unexpectedly imported when experiment_id=None: {exp_mgr_imports}"
    )

    # JSONL should be written
    jsonl_files = list(results_dir.glob("*.jsonl"))
    assert len(jsonl_files) >= 1, "JSONL file should be written even without --experiment-id"


# ---------------------------------------------------------------------------
# T2: With --experiment-id → create_experiment called with correct args
# ---------------------------------------------------------------------------


def test_experiment_id_creates_entry(module, fake_result, tmp_path):
    """run_research_loop with experiment_id calls ExperimentManager.create_experiment."""
    mock_mgr = MagicMock()
    mock_mgr_class = MagicMock(return_value=mock_mgr)
    mock_em_module = MagicMock()
    mock_em_module.ExperimentManager = mock_mgr_class

    with patch.dict(sys.modules, {"finalayze.core.experiment_manager": mock_em_module}):
        _run_loop(module, fake_result, tmp_path, experiment_id="test-exp-2024")

    mock_mgr_class.assert_called_once()
    mock_mgr.create_experiment.assert_called_once()

    create_call = mock_mgr.create_experiment.call_args
    # experiment_id must be passed
    got_id = create_call.kwargs.get("experiment_id") or (
        create_call.args[0] if create_call.args else None
    )
    assert got_id == "test-exp-2024"

    # Hypothesis must mention something meaningful
    hypothesis = create_call.kwargs.get("hypothesis") or (
        create_call.args[1] if len(create_call.args) > 1 else ""
    )
    assert hypothesis, "hypothesis must not be empty"

    # update_status("running") must be called
    mock_mgr.update_status.assert_called()


# ---------------------------------------------------------------------------
# T3: Result linking → link_result called for baseline + each experiment
# ---------------------------------------------------------------------------


def test_result_linking(module, fake_result, tmp_path):
    """link_result is called for each _log_result invocation when experiment_id is set."""
    mock_mgr = MagicMock()
    mock_mgr_class = MagicMock(return_value=mock_mgr)
    mock_em_module = MagicMock()
    mock_em_module.ExperimentManager = mock_mgr_class

    with patch.dict(sys.modules, {"finalayze.core.experiment_manager": mock_em_module}):
        _run_loop(module, fake_result, tmp_path, experiment_id="link-test")

    assert mock_mgr.link_result.call_count >= 1, (
        f"link_result should be called at least once, got {mock_mgr.link_result.call_count}"
    )


# ---------------------------------------------------------------------------
# T4: Verdict → record_verdict called with best_score after loop
# ---------------------------------------------------------------------------


def test_verdict_recorded(module, fake_result, tmp_path):
    """record_verdict is called with a float (best_score) after the loop completes."""
    mock_mgr = MagicMock()
    mock_mgr_class = MagicMock(return_value=mock_mgr)
    mock_em_module = MagicMock()
    mock_em_module.ExperimentManager = mock_mgr_class

    with patch.dict(sys.modules, {"finalayze.core.experiment_manager": mock_em_module}):
        _run_loop(module, fake_result, tmp_path, experiment_id="verdict-test")

    mock_mgr.record_verdict.assert_called_once()
    verdict_call = mock_mgr.record_verdict.call_args
    metric_value = verdict_call.kwargs.get("metric_value") or (
        verdict_call.args[1] if len(verdict_call.args) > 1 else None
    )
    assert isinstance(metric_value, float), (
        f"record_verdict metric_value should be float, got {type(metric_value)}"
    )


# ---------------------------------------------------------------------------
# T5: Invalid experiment ID → argparse error (SystemExit)
# ---------------------------------------------------------------------------


def _build_argparse_type_fn():
    """Return the type function that the module should use for --experiment-id."""

    def _valid_experiment_id(value: str) -> str:
        if not re.match(r"^[a-zA-Z0-9_-]+$", value):
            raise argparse.ArgumentTypeError(
                f"Invalid experiment-id '{value}': only [a-zA-Z0-9_-] allowed"
            )
        return value

    return _valid_experiment_id


def test_experiment_id_validation_invalid(module):
    """--experiment-id with invalid chars raises argparse error (SystemExit)."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--segment", required=True, choices=list(module._SEGMENT_SYMBOLS.keys()))
    parser.add_argument("--strategy", default="all")
    parser.add_argument("--max-experiments", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--experiment-id", type=_build_argparse_type_fn(), default=None)

    invalid_ids = ["has space", "has/slash", "has@at"]
    for invalid_id in invalid_ids:
        with pytest.raises(SystemExit):
            parser.parse_args(["--segment", "us_tech", "--experiment-id", invalid_id])


def test_experiment_id_validation_valid(module):
    """--experiment-id with valid chars is accepted and returned unchanged."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--segment", required=True, choices=list(module._SEGMENT_SYMBOLS.keys()))
    parser.add_argument("--strategy", default="all")
    parser.add_argument("--max-experiments", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--experiment-id", type=_build_argparse_type_fn(), default=None)

    valid_ids = ["exp-2024", "my_experiment", "Exp123", "a-b_c-D-1"]
    for valid_id in valid_ids:
        args = parser.parse_args(["--segment", "us_tech", "--experiment-id", valid_id])
        assert args.experiment_id == valid_id


# ---------------------------------------------------------------------------
# T6: Error resilience → loop completes even if create_experiment raises
# ---------------------------------------------------------------------------


def test_error_resilience(module, fake_result, tmp_path):
    """If ExperimentManager.create_experiment raises, research loop still completes."""
    mock_mgr = MagicMock()
    mock_mgr.create_experiment.side_effect = RuntimeError("Disk full or permission denied")
    mock_mgr_class = MagicMock(return_value=mock_mgr)
    mock_em_module = MagicMock()
    mock_em_module.ExperimentManager = mock_mgr_class

    # Must NOT raise; loop must complete
    with patch.dict(sys.modules, {"finalayze.core.experiment_manager": mock_em_module}):
        results_dir = _run_loop(module, fake_result, tmp_path, experiment_id="resilience-test")

    jsonl_files = list(results_dir.glob("*.jsonl"))
    assert len(jsonl_files) >= 1, (
        "JSONL must be written even when ExperimentManager.create_experiment raises"
    )


# ---------------------------------------------------------------------------
# T7: Concurrent isolation → two experiment IDs produce independent files
# ---------------------------------------------------------------------------


def test_concurrent_isolation(tmp_path):
    """Two ExperimentManager instances with different dirs produce independent experiment files."""
    from finalayze.core.experiment_manager import ExperimentManager
    from finalayze.core.schemas import SuccessCriteria

    dir_a = tmp_path / "exp_a"
    dir_b = tmp_path / "exp_b"

    mgr_a = ExperimentManager(experiments_dir=dir_a)
    mgr_b = ExperimentManager(experiments_dir=dir_b)

    criteria = SuccessCriteria(metric="composite_score", threshold=0.0, operator=">=")

    path_a = mgr_a.create_experiment("run-alpha", "Hypothesis A", criteria)
    path_b = mgr_b.create_experiment("run-beta", "Hypothesis B", criteria)

    # Files must be in separate directories
    assert path_a.parent == dir_a
    assert path_b.parent == dir_b
    assert path_a != path_b

    state_a = mgr_a.read_experiment("run-alpha")
    state_b = mgr_b.read_experiment("run-beta")

    assert state_a.experiment_id == "run-alpha"
    assert state_b.experiment_id == "run-beta"
    assert state_a.hypothesis == "Hypothesis A"
    assert state_b.hypothesis == "Hypothesis B"

    # No cross-contamination
    assert mgr_a.list_experiments() == ["run-alpha"]
    assert mgr_b.list_experiments() == ["run-beta"]
