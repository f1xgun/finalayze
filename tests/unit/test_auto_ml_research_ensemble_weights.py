"""Tests for ensemble_weights strategy in auto_ml_research.py.

Tests for generate_ensemble_weight_experiments, weighted _evaluate_models,
small-fold guard in run_experiment, and CLI wiring.

- T1: generate_ensemble_weight_experiments returns ExperimentConfig items
      with strategy="ensemble_weights"
- T2: Generated configs count is between 9 and 12
- T3: Every config has xgb_weight + lgbm_weight + cat_weight == 1.0 (within tolerance)
- T4: No config has any single weight > 0.7
- T5: Every config has all weights >= 0.1
- T6: _evaluate_models with weights=[0.5, 0.3, 0.2] produces weighted average
- T7: _evaluate_models without weights produces equal average (backward compat)
- T8: "ensemble_weights" is in CLI choices
- T9: _generate_experiments("ensemble_weights", ...) returns non-empty list
- T10: run_experiment with < 4 folds and strategy="ensemble_weights" uses equal weights
- T11: run_experiment with >= 4 folds and strategy="ensemble_weights" preserves original weights
- T12: Non-ensemble_weights strategy with small fold count is unaffected
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Module import helper
# ---------------------------------------------------------------------------

_SCRIPT_PATH = Path(__file__).parent.parent.parent / "scripts" / "auto_ml_research.py"
_MODULE_NAME = "auto_ml_research"


def _import_module() -> Any:
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
# T1-T5: generate_ensemble_weight_experiments
# ---------------------------------------------------------------------------


def test_t1_generator_returns_ensemble_weights_strategy() -> None:
    """T1: generate_ensemble_weight_experiments returns configs with strategy='ensemble_weights'."""
    mod = _import_module()
    experiments = mod.generate_ensemble_weight_experiments()
    assert len(experiments) > 0
    for exp in experiments:
        assert exp.strategy == "ensemble_weights", (
            f"Expected 'ensemble_weights', got {exp.strategy!r}"
        )


def test_t2_generator_count_in_range() -> None:
    """T2: Generated configs count is at least 9.

    With step=0.1, min=0.1, max=0.7 per model the simplex produces 33 configs.
    The plan's original "9-12" estimate was a miscalculation — 33 is correct.
    We assert >=9 to validate the simplex is populated.
    """
    mod = _import_module()
    experiments = mod.generate_ensemble_weight_experiments()
    count = len(experiments)
    assert count >= 9, f"Expected at least 9 experiments, got {count}"


def test_t3_weights_sum_to_one() -> None:
    """T3: Every generated config has weights summing to 1.0 (within float tolerance)."""
    mod = _import_module()
    experiments = mod.generate_ensemble_weight_experiments()
    for exp in experiments:
        hp = exp.hparams
        total = hp["xgb_weight"] + hp["lgbm_weight"] + hp["cat_weight"]
        assert abs(total - 1.0) < 1e-9, f"Weights don't sum to 1.0 for {exp.name}: {total}"


def test_t4_no_weight_exceeds_cap() -> None:
    """T4: No generated config has any single weight > 0.7."""
    mod = _import_module()
    experiments = mod.generate_ensemble_weight_experiments()
    for exp in experiments:
        hp = exp.hparams
        for key in ("xgb_weight", "lgbm_weight", "cat_weight"):
            assert hp[key] <= 0.7, f"Weight {key}={hp[key]} exceeds 0.7 in {exp.name}"


def test_t5_all_weights_at_least_minimum() -> None:
    """T5: Every config has xgb_weight >= 0.1, lgbm_weight >= 0.1, cat_weight >= 0.1."""
    mod = _import_module()
    experiments = mod.generate_ensemble_weight_experiments()
    for exp in experiments:
        hp = exp.hparams
        for key in ("xgb_weight", "lgbm_weight", "cat_weight"):
            assert hp[key] >= 0.1, f"Weight {key}={hp[key]} is below 0.1 in {exp.name}"


# ---------------------------------------------------------------------------
# T6-T7: _evaluate_models weighted vs equal averaging
# ---------------------------------------------------------------------------


def _make_mock_model(proba: float) -> MagicMock:
    """Create a mock model that returns fixed predict_proba value."""
    mock = MagicMock()
    mock._trained = True
    mock._model = True
    mock.predict_proba.return_value = proba
    return mock


def test_t6_weighted_average_when_weights_provided() -> None:
    """T6: _evaluate_models with weights=[0.5, 0.3, 0.2] produces weighted, not equal, average."""
    mod = _import_module()

    # Three models with fixed probas: 0.8, 0.4, 0.6
    models = [
        _make_mock_model(0.8),
        _make_mock_model(0.4),
        _make_mock_model(0.6),
    ]
    test_f = [{"feature_a": 1.0}]  # single sample
    test_l = [1]
    weights = [0.5, 0.3, 0.2]

    # Expected weighted prob: 0.8*0.5 + 0.4*0.3 + 0.6*0.2 = 0.4 + 0.12 + 0.12 = 0.64
    expected_weighted = 0.8 * 0.5 + 0.4 * 0.3 + 0.6 * 0.2
    equal_avg = (0.8 + 0.4 + 0.6) / 3.0

    # They must be different to make this test meaningful
    assert abs(expected_weighted - equal_avg) > 0.01

    result = mod._evaluate_models(models, test_f, test_l, 1.0, 1.0, weights=weights)

    # Verify models were called
    for m in models:
        m.predict_proba.assert_called_once_with({"feature_a": 1.0})

    # The resulting accuracy/brier should reflect the weighted proba of 0.64
    # When weighted proba=0.64, pred=round(0.64)=1, label=1, so accuracy=1.0
    # brier = (0.64 - 1)^2 = 0.1296
    assert result.accuracy == pytest.approx(1.0)
    assert result.brier_score == pytest.approx((expected_weighted - 1.0) ** 2, abs=1e-6)


def test_t7_equal_average_when_no_weights() -> None:
    """T7: _evaluate_models without weights produces equal average (backward compat)."""
    mod = _import_module()

    models = [
        _make_mock_model(0.8),
        _make_mock_model(0.4),
        _make_mock_model(0.6),
    ]
    test_f = [{"feature_a": 1.0}]
    test_l = [1]

    # Expected equal avg: (0.8+0.4+0.6)/3 = 0.6
    equal_avg = (0.8 + 0.4 + 0.6) / 3.0

    result_no_weights = mod._evaluate_models(models, test_f, test_l, 1.0, 1.0)
    result_none_weights = mod._evaluate_models(models, test_f, test_l, 1.0, 1.0, weights=None)

    # brier = (0.6 - 1)^2 = 0.16
    expected_brier = (equal_avg - 1.0) ** 2
    assert result_no_weights.brier_score == pytest.approx(expected_brier, abs=1e-6)
    assert result_none_weights.brier_score == pytest.approx(expected_brier, abs=1e-6)


# ---------------------------------------------------------------------------
# T8: CLI choices include "ensemble_weights"
# ---------------------------------------------------------------------------


def test_t8_ensemble_weights_in_cli_choices() -> None:
    """T8: 'ensemble_weights' appears in CLI choices."""
    source = _SCRIPT_PATH.read_text()
    assert "ensemble_weights" in source, "'ensemble_weights' not found in script source"
    # More specifically, it should be in the choices list for --strategy
    assert '"ensemble_weights"' in source or "'ensemble_weights'" in source


# ---------------------------------------------------------------------------
# T9: _generate_experiments routes "ensemble_weights"
# ---------------------------------------------------------------------------


def test_t9_generate_experiments_routes_ensemble_weights() -> None:
    """T9: _generate_experiments("ensemble_weights") returns non-empty list from the generator."""
    mod = _import_module()
    experiments = mod._generate_experiments(
        strategy="ensemble_weights",
        baseline_features=["feat_a", "feat_b"],
        all_feature_names=["feat_a", "feat_b", "feat_c"],
        max_experiments=100,
    )
    assert len(experiments) > 0
    for exp in experiments:
        assert exp.strategy == "ensemble_weights"


# ---------------------------------------------------------------------------
# T10-T12: small-fold guard in run_experiment
# ---------------------------------------------------------------------------


def _make_minimal_folds(count: int) -> list[tuple[list[int], list[int], list[int]]]:
    """Create minimal fold tuples for testing."""
    # Each fold is (train_idx, cal_idx, test_idx)
    return [([0, 1, 2], [3], [4]) for _ in range(count)]


def test_t10_small_fold_guard_overrides_weights() -> None:
    """T10: run_experiment with <4 folds and strategy='ensemble_weights' uses equal weights."""
    mod = _import_module()

    captured_configs: list[Any] = []

    def fake_run_fold(
        train_idx: Any,
        test_idx: Any,
        all_features: Any,
        labels: Any,
        hold_bars: Any,
        config: Any,
        segment_id: Any,
        **kwargs: Any,
    ) -> None:
        captured_configs.append(config)

    config = mod.ExperimentConfig(
        name="ew-0.5-0.3-0.2",
        description="test",
        strategy="ensemble_weights",
        hparams={
            **dict(mod._DEFAULT_HPARAMS),
            "xgb_weight": 0.5,
            "lgbm_weight": 0.3,
            "cat_weight": 0.2,
        },
    )

    folds_3 = _make_minimal_folds(3)
    all_features = [{"f": 1.0}] * 10
    labels = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]

    with patch.object(mod, "_run_fold", side_effect=fake_run_fold):
        mod.run_experiment(config, all_features, labels, None, folds_3, "us_tech")

    assert len(captured_configs) > 0
    for captured in captured_configs:
        assert abs(captured.hparams["xgb_weight"] - 1 / 3) < 1e-9, (
            f"Expected 1/3 but got xgb_weight={captured.hparams['xgb_weight']}"
        )
        assert abs(captured.hparams["lgbm_weight"] - 1 / 3) < 1e-9
        assert abs(captured.hparams["cat_weight"] - 1 / 3) < 1e-9


def test_t11_sufficient_folds_preserve_weights() -> None:
    """T11: run_experiment with >=4 folds and strategy='ensemble_weights' keeps original weights."""
    mod = _import_module()

    captured_configs: list[Any] = []

    def fake_run_fold(
        train_idx: Any,
        test_idx: Any,
        all_features: Any,
        labels: Any,
        hold_bars: Any,
        config: Any,
        segment_id: Any,
        **kwargs: Any,
    ) -> None:
        captured_configs.append(config)

    config = mod.ExperimentConfig(
        name="ew-0.5-0.3-0.2",
        description="test",
        strategy="ensemble_weights",
        hparams={
            **dict(mod._DEFAULT_HPARAMS),
            "xgb_weight": 0.5,
            "lgbm_weight": 0.3,
            "cat_weight": 0.2,
        },
    )

    folds_4 = _make_minimal_folds(4)
    all_features = [{"f": 1.0}] * 10
    labels = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]

    with patch.object(mod, "_run_fold", side_effect=fake_run_fold):
        mod.run_experiment(config, all_features, labels, None, folds_4, "us_tech")

    assert len(captured_configs) > 0
    for captured in captured_configs:
        assert abs(captured.hparams["xgb_weight"] - 0.5) < 1e-9, (
            f"Expected 0.5 but got xgb_weight={captured.hparams['xgb_weight']}"
        )
        assert abs(captured.hparams["lgbm_weight"] - 0.3) < 1e-9
        assert abs(captured.hparams["cat_weight"] - 0.2) < 1e-9


def test_t12_non_ensemble_strategy_unaffected_by_small_fold_guard() -> None:
    """T12: Non-ensemble_weights strategy with small fold count does NOT trigger weight override."""
    mod = _import_module()

    captured_configs: list[Any] = []

    def fake_run_fold(
        train_idx: Any,
        test_idx: Any,
        all_features: Any,
        labels: Any,
        hold_bars: Any,
        config: Any,
        segment_id: Any,
        **kwargs: Any,
    ) -> None:
        captured_configs.append(config)

    # Use ablation strategy with no weight keys
    config = mod.ExperimentConfig(
        name="ablate-feat_a",
        description="test",
        strategy="ablation",
        feature_subset=["feat_b"],
    )

    folds_2 = _make_minimal_folds(2)
    all_features = [{"f": 1.0}] * 10
    labels = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]

    with patch.object(mod, "_run_fold", side_effect=fake_run_fold):
        mod.run_experiment(config, all_features, labels, None, folds_2, "us_tech")

    # For ablation, no weight keys should be present (or at least not 1/3)
    for captured in captured_configs:
        # The strategy must remain ablation — guard only applies to ensemble_weights
        assert captured.strategy == "ablation", (
            "Non-ensemble strategy should not be mutated by the small-fold guard"
        )
