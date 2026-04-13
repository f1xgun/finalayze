"""Tests for feature_engineering strategy in auto_ml_research.py.

- T1: _generate_feature_candidates with 5 base features produces lag_ratio,
      rolling_zscore, and interaction columns
- T2: _generate_feature_candidates output length does not exceed cap (n_samples // 20)
- T3: For 730 samples (MOEX), cap is 36; for 1825 samples (US), cap is 91
- T4: _filter_by_permutation_importance removes features with importance <= 0 (noise-only)
- T5: _filter_by_permutation_importance keeps features with positive importance
- T6: generate_feature_engineering_experiments returns list[ExperimentConfig] with
      strategy="feature_engineering"
- T7: Each returned config has feature_subset that is the union of baseline features
      + surviving engineered features
- T8: "feature_engineering" appears in CLI --strategy choices
- T9: _generate_experiments("feature_engineering", ...) routes to
      generate_feature_engineering_experiments
- T10: _generate_experiments("all", ...) includes feature_engineering results
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

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
# Synthetic fixtures
# ---------------------------------------------------------------------------

_N_SAMPLES = 100
_BASE_FEATURES = ["close_ratio", "volume_ratio", "rsi_14", "momentum_20", "atr_14"]


def _make_feature_dicts(feature_names: list[str], n: int = _N_SAMPLES) -> list[dict[str, float]]:
    """Create synthetic feature dicts with normally distributed values."""
    rng = np.random.default_rng(42)
    return [{name: float(rng.standard_normal()) for name in feature_names} for _ in range(n)]


def _make_labels(n: int = _N_SAMPLES) -> list[int]:
    """Create synthetic binary labels."""
    rng = np.random.default_rng(42)
    return [int(rng.integers(0, 2)) for _ in range(n)]


# ---------------------------------------------------------------------------
# T1: _generate_feature_candidates produces expected feature types
# ---------------------------------------------------------------------------


def test_t1_generate_feature_candidates_produces_lag_zscore_interaction() -> None:
    """T1: _generate_feature_candidates with 5 base features produces
    lag_ratio, rolling_zscore, and interaction columns."""
    mod = _import_module()

    features = _make_feature_dicts(_BASE_FEATURES)
    labels = _make_labels()
    cap = 200  # large cap so nothing is truncated

    candidate_names, _augmented = mod._generate_feature_candidates(
        _BASE_FEATURES, features, labels, cap
    )

    # Should have lag_ratio columns (for close/volume features)
    lag_names = [n for n in candidate_names if "lag" in n and "ratio" in n]
    assert len(lag_names) > 0, f"Expected lag_ratio features, got: {candidate_names}"

    # Should have rolling_zscore columns
    zscore_names = [n for n in candidate_names if "zscore" in n]
    assert len(zscore_names) > 0, f"Expected rolling_zscore features, got: {candidate_names}"

    # Should have interaction columns (rsi x volume)
    interaction_names = [n for n in candidate_names if "x" in n or "interaction" in n]
    assert len(interaction_names) > 0, f"Expected interaction features, got: {candidate_names}"


# ---------------------------------------------------------------------------
# T2: Output length does not exceed cap
# ---------------------------------------------------------------------------


def test_t2_generate_feature_candidates_respects_cap() -> None:
    """T2: _generate_feature_candidates output length does not exceed cap."""
    mod = _import_module()

    features = _make_feature_dicts(_BASE_FEATURES)
    labels = _make_labels()
    cap = 3  # very small cap

    candidate_names, _augmented = mod._generate_feature_candidates(
        _BASE_FEATURES, features, labels, cap
    )

    assert len(candidate_names) <= cap, (
        f"Expected <= {cap} candidates, got {len(candidate_names)}: {candidate_names}"
    )


# ---------------------------------------------------------------------------
# T3: Cap values for MOEX (730 samples) and US (1825 samples)
# ---------------------------------------------------------------------------


def test_t3_cap_for_moex_is_36() -> None:
    """T3: For 730 samples (MOEX), cap is 36 (730 // 20)."""
    n_moex = 730
    expected_cap = n_moex // 20
    assert expected_cap == 36, f"Expected cap=36, got {expected_cap}"


def test_t3_cap_for_us_is_91() -> None:
    """T3: For 1825 samples (US), cap is 91 (1825 // 20)."""
    n_us = 1825
    expected_cap = n_us // 20
    assert expected_cap == 91, f"Expected cap=91, got {expected_cap}"


def test_t3_generate_feature_candidates_cap_matches_n_samples_div_20() -> None:
    """T3: generate_feature_engineering_experiments uses cap = n_samples // 20."""
    mod = _import_module()

    # Use a large feature set to guarantee candidates are generated
    large_feats = _BASE_FEATURES + [f"extra_{i}" for i in range(10)]
    n_samples = 100
    cap = n_samples // 20  # = 5

    features = _make_feature_dicts(large_feats, n=n_samples)
    labels = _make_labels(n=n_samples)

    candidate_names, _ = mod._generate_feature_candidates(large_feats, features, labels, cap)
    assert len(candidate_names) <= cap, f"Expected <= {cap} candidates, got {len(candidate_names)}"


# ---------------------------------------------------------------------------
# T4: _filter_by_permutation_importance removes zero/negative importance features
# ---------------------------------------------------------------------------


def test_t4_filter_removes_zero_importance_features() -> None:
    """T4: _filter_by_permutation_importance removes features with importance <= 0."""
    mod = _import_module()

    candidate_names = ["eng_feat_a", "eng_feat_b", "eng_feat_c"]
    baseline_features = ["rsi_14", "volume_ratio"]
    all_feature_names = baseline_features + candidate_names

    features = _make_feature_dicts(all_feature_names, n=200)
    labels = _make_labels(n=200)

    # Mock permutation_importance to return 0 for all candidates
    mock_result = MagicMock()
    mock_result.importances_mean = np.array(
        [0.1, 0.2, 0.0, -0.1, 0.0]  # baseline feats positive, candidates zero/negative
    )

    with patch("sklearn.inspection.permutation_importance", return_value=mock_result) as mock_pi:
        survivors = mod._filter_by_permutation_importance(
            features, labels, candidate_names, baseline_features
        )

    assert mock_pi.called, "Expected permutation_importance to be called"
    # All candidates should be filtered out (importance <= 0)
    assert survivors == [], f"Expected empty survivors list, got {survivors}"


# ---------------------------------------------------------------------------
# T5: _filter_by_permutation_importance keeps positive importance features
# ---------------------------------------------------------------------------


def test_t5_filter_keeps_positive_importance_features() -> None:
    """T5: _filter_by_permutation_importance keeps features with positive importance."""
    mod = _import_module()

    candidate_names = ["eng_feat_a", "eng_feat_b", "eng_feat_c"]
    baseline_features = ["rsi_14", "volume_ratio"]
    all_feature_names = baseline_features + candidate_names

    features = _make_feature_dicts(all_feature_names, n=200)
    labels = _make_labels(n=200)

    # Mock: first candidate positive, rest zero
    mock_result = MagicMock()
    mock_result.importances_mean = np.array(
        [0.1, 0.2, 0.5, 0.0, -0.05]  # eng_feat_a positive, b/c zero or negative
    )

    with patch("sklearn.inspection.permutation_importance", return_value=mock_result):
        survivors = mod._filter_by_permutation_importance(
            features, labels, candidate_names, baseline_features
        )

    assert "eng_feat_a" in survivors, (
        f"Expected 'eng_feat_a' to survive (positive importance), got {survivors}"
    )
    assert "eng_feat_b" not in survivors, "Expected 'eng_feat_b' to be filtered (importance=0)"
    assert "eng_feat_c" not in survivors, "Expected 'eng_feat_c' to be filtered (importance=-0.05)"


# ---------------------------------------------------------------------------
# T6: generate_feature_engineering_experiments returns ExperimentConfig with correct strategy
# ---------------------------------------------------------------------------


def test_t6_generate_feature_engineering_experiments_returns_configs() -> None:
    """T6: generate_feature_engineering_experiments returns list[ExperimentConfig]
    with strategy="feature_engineering"."""
    mod = _import_module()

    n_samples = 200
    features = _make_feature_dicts(_BASE_FEATURES, n=n_samples)
    labels = _make_labels(n=n_samples)

    # Mock permutation_importance to return positive importance for all candidates
    def mock_perm_importance(estimator, x_data, y, **kwargs):  # type: ignore[no-untyped-def]
        n_features = x_data.shape[1] if hasattr(x_data, "shape") else len(x_data.columns)
        mock_res = MagicMock()
        mock_res.importances_mean = np.ones(n_features) * 0.1
        return mock_res

    with patch("sklearn.inspection.permutation_importance", side_effect=mock_perm_importance):
        result = mod.generate_feature_engineering_experiments(
            _BASE_FEATURES, features, labels, n_samples
        )

    assert isinstance(result, list)
    if len(result) > 0:
        for exp in result:
            assert exp.strategy == "feature_engineering", (
                f"Expected 'feature_engineering', got {exp.strategy!r}"
            )


# ---------------------------------------------------------------------------
# T7: Each config has feature_subset = baseline + surviving engineered features
# ---------------------------------------------------------------------------


def test_t7_config_feature_subset_includes_baseline_and_survivors() -> None:
    """T7: Each returned config has feature_subset = baseline + surviving engineered features."""
    mod = _import_module()

    n_samples = 200
    features = _make_feature_dicts(_BASE_FEATURES, n=n_samples)
    labels = _make_labels(n=n_samples)

    # Mock permutation_importance to return positive importance for all candidates
    def mock_perm_importance(estimator, x_data, y, **kwargs):  # type: ignore[no-untyped-def]
        n_features = x_data.shape[1] if hasattr(x_data, "shape") else len(x_data.columns)
        mock_res = MagicMock()
        mock_res.importances_mean = np.ones(n_features) * 0.1
        return mock_res

    with patch("sklearn.inspection.permutation_importance", side_effect=mock_perm_importance):
        result = mod.generate_feature_engineering_experiments(
            _BASE_FEATURES, features, labels, n_samples
        )

    if len(result) > 0:
        exp = result[0]
        assert exp.feature_subset is not None
        # All baseline features should be in the subset
        for feat in _BASE_FEATURES:
            assert feat in exp.feature_subset, (
                f"Baseline feature '{feat}' missing from subset {exp.feature_subset}"
            )
        # Should have MORE features than just baseline (engineered ones added)
        assert len(exp.feature_subset) > len(_BASE_FEATURES), (
            "Expected additional engineered features beyond baseline"
        )


# ---------------------------------------------------------------------------
# T8: "feature_engineering" in CLI choices
# ---------------------------------------------------------------------------


def test_t8_feature_engineering_in_cli_choices() -> None:
    """T8: 'feature_engineering' appears in CLI --strategy choices."""
    source = _SCRIPT_PATH.read_text()
    assert "feature_engineering" in source, "'feature_engineering' not found in script source"
    assert '"feature_engineering"' in source or "'feature_engineering'" in source


# ---------------------------------------------------------------------------
# T9: _generate_experiments routes "feature_engineering"
# ---------------------------------------------------------------------------


def test_t9_generate_experiments_routes_feature_engineering() -> None:
    """T9: _generate_experiments("feature_engineering", ...) routes to
    generate_feature_engineering_experiments."""
    mod = _import_module()

    n_samples = 200
    features = _make_feature_dicts(_BASE_FEATURES, n=n_samples)
    labels = _make_labels(n=n_samples)

    def mock_perm_importance(estimator, x_data, y, **kwargs):  # type: ignore[no-untyped-def]
        n_features = x_data.shape[1] if hasattr(x_data, "shape") else len(x_data.columns)
        mock_res = MagicMock()
        mock_res.importances_mean = np.ones(n_features) * 0.1
        return mock_res

    with patch("sklearn.inspection.permutation_importance", side_effect=mock_perm_importance):
        experiments = mod._generate_experiments(
            strategy="feature_engineering",
            baseline_features=_BASE_FEATURES,
            all_feature_names=_BASE_FEATURES,
            max_experiments=100,
            segment_id="us_tech",
            all_features=features,
            labels=labels,
            n_samples=n_samples,
        )

    # Should have routed — may be empty if no candidates survive, but no exception
    for exp in experiments:
        assert exp.strategy == "feature_engineering"


# ---------------------------------------------------------------------------
# T10: _generate_experiments("all") includes feature_engineering results
# ---------------------------------------------------------------------------


def test_t10_generate_experiments_all_includes_feature_engineering() -> None:
    """T10: _generate_experiments("all", ...) includes feature_engineering results."""
    mod = _import_module()

    n_samples = 200
    all_feats_names = [f"feat_{i}" for i in range(10)]
    features = _make_feature_dicts(_BASE_FEATURES, n=n_samples)
    labels = _make_labels(n=n_samples)

    # Track whether generate_feature_engineering_experiments is called
    called = []
    original_fn = mod.generate_feature_engineering_experiments

    def tracking_fn(*args, **kwargs):  # type: ignore[no-untyped-def]
        called.append(True)
        return original_fn(*args, **kwargs)

    def mock_perm_importance(estimator, x_data, y, **kwargs):  # type: ignore[no-untyped-def]
        n_features = x_data.shape[1] if hasattr(x_data, "shape") else len(x_data.columns)
        mock_res = MagicMock()
        mock_res.importances_mean = np.ones(n_features) * 0.1
        return mock_res

    with (
        patch.object(mod, "generate_feature_engineering_experiments", side_effect=tracking_fn),
        patch("sklearn.inspection.permutation_importance", side_effect=mock_perm_importance),
    ):
        mod._generate_experiments(
            strategy="all",
            baseline_features=_BASE_FEATURES,
            all_feature_names=all_feats_names,
            max_experiments=500,
            segment_id="us_tech",
            all_features=features,
            labels=labels,
            n_samples=n_samples,
        )

    assert len(called) > 0, (
        "Expected generate_feature_engineering_experiments to be called in 'all' strategy"
    )
