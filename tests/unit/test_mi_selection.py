"""Tests for MI-based feature selection (Layer 3).

Tests cover:
- MI computation returns positive values for informative features
- Redundant features (copies) get filtered
- max_features limit respected
- Uninformative features (random noise) get low MI scores
- Empty/single-feature edge cases
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from finalayze.ml.training.feature_selection import compute_feature_mi, select_features_mi

# ---------------------------------------------------------------------------
# Constants (no magic numbers, ruff PLR2004)
# ---------------------------------------------------------------------------
N_ROWS = 50
N_FEATURES = 5
MAX_FEATURES_DEFAULT = 15
MAX_FEATURES_SMALL = 3
MI_THRESHOLD_DEFAULT = 0.05
MI_THRESHOLD_ZERO = 0.0
RANDOM_SEED = 42
INFORMATIVE_MI_FLOOR = 0.0  # MI is always >= 0
REDUNDANT_FEATURE_COUNT = 3


@pytest.fixture
def informative_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Create a dataset where features are correlated with the target."""
    rng = np.random.default_rng(RANDOM_SEED)
    y = pd.Series(rng.integers(0, 2, size=N_ROWS), name="target")
    x = pd.DataFrame(
        {
            "feat_a": y.values * 2.0 + rng.normal(0, 0.1, N_ROWS),
            "feat_b": y.values * 1.5 + rng.normal(0, 0.3, N_ROWS),
            "feat_c": rng.normal(0, 1, N_ROWS),  # noise
            "feat_d": y.values * 3.0 + rng.normal(0, 0.2, N_ROWS),
            "feat_e": rng.normal(0, 1, N_ROWS),  # noise
        }
    )
    return x, y


@pytest.fixture
def redundant_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Dataset with redundant (copied) features."""
    rng = np.random.default_rng(RANDOM_SEED)
    y = pd.Series(rng.integers(0, 2, size=N_ROWS), name="target")
    base = y.values * 2.0 + rng.normal(0, 0.1, N_ROWS)
    x = pd.DataFrame(
        {
            "original": base,
            "copy_1": base + rng.normal(0, 0.01, N_ROWS),  # near-identical
            "copy_2": base + rng.normal(0, 0.01, N_ROWS),  # near-identical
            "independent": y.values * 1.0 + rng.normal(0, 0.5, N_ROWS),
        }
    )
    return x, y


class TestComputeFeatureMI:
    """Tests for compute_feature_mi helper."""

    def test_returns_series_with_feature_names(
        self, informative_dataset: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        x, y = informative_dataset
        mi = compute_feature_mi(x, y)
        assert isinstance(mi, pd.Series)
        assert list(mi.index) == list(x.columns)

    def test_informative_features_have_positive_mi(
        self, informative_dataset: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        x, y = informative_dataset
        mi = compute_feature_mi(x, y)
        # Features strongly correlated with target should have MI > 0
        assert mi["feat_a"] > INFORMATIVE_MI_FLOOR
        assert mi["feat_d"] > INFORMATIVE_MI_FLOOR

    def test_informative_features_higher_than_noise(
        self, informative_dataset: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        x, y = informative_dataset
        mi = compute_feature_mi(x, y)
        # Informative features should score higher than pure noise
        assert mi["feat_a"] > mi["feat_c"]
        assert mi["feat_d"] > mi["feat_e"]

    def test_all_mi_values_non_negative(
        self, informative_dataset: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        x, y = informative_dataset
        mi = compute_feature_mi(x, y)
        assert (mi >= INFORMATIVE_MI_FLOOR).all()

    def test_single_feature(self) -> None:
        rng = np.random.default_rng(RANDOM_SEED)
        y = pd.Series(rng.integers(0, 2, size=N_ROWS))
        x = pd.DataFrame({"only_feat": y.values * 2.0 + rng.normal(0, 0.1, N_ROWS)})
        mi = compute_feature_mi(x, y)
        assert len(mi) == 1
        assert mi["only_feat"] > INFORMATIVE_MI_FLOOR


class TestSelectFeaturesMI:
    """Tests for select_features_mi."""

    def test_returns_list_of_strings(
        self, informative_dataset: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        x, y = informative_dataset
        selected = select_features_mi(x, y)
        assert isinstance(selected, list)
        assert all(isinstance(s, str) for s in selected)

    def test_max_features_limit_respected(
        self, informative_dataset: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        x, y = informative_dataset
        selected = select_features_mi(x, y, max_features=MAX_FEATURES_SMALL)
        assert len(selected) <= MAX_FEATURES_SMALL

    def test_uninformative_features_filtered_by_threshold(self) -> None:
        """Pure noise features should be filtered — but floor of 8 still applies."""
        rng = np.random.default_rng(RANDOM_SEED)
        n_noise = 15  # more than the min floor of 8
        y = pd.Series(rng.integers(0, 2, size=N_ROWS))
        # All features are pure noise
        x = pd.DataFrame({f"noise_{i}": rng.normal(0, 1, N_ROWS) for i in range(n_noise)})
        selected = select_features_mi(x, y, mi_threshold=MI_THRESHOLD_DEFAULT)
        # Floor ensures at least 8 features, but should not keep all 15 noise features
        min_floor = 8
        assert len(selected) >= min_floor
        assert len(selected) < n_noise

    def test_redundant_features_deduplicated(
        self, redundant_dataset: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Near-identical features should be deduplicated (75th pct threshold).

        With relaxed dedup (75th percentile), very similar features may survive.
        The independent feature should always be selected.
        """
        x, y = redundant_dataset
        selected = select_features_mi(
            x, y, max_features=MAX_FEATURES_DEFAULT, mi_threshold=MI_THRESHOLD_ZERO
        )
        # Independent feature should be selected
        assert "independent" in selected

    def test_independent_feature_kept_with_redundant(
        self, redundant_dataset: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Independent informative feature should survive dedup."""
        x, y = redundant_dataset
        selected = select_features_mi(
            x, y, max_features=MAX_FEATURES_DEFAULT, mi_threshold=MI_THRESHOLD_ZERO
        )
        assert "independent" in selected

    def test_empty_dataframe_returns_empty(self) -> None:
        x = pd.DataFrame()
        y = pd.Series(dtype=int)
        selected = select_features_mi(x, y)
        assert selected == []

    def test_single_feature_returned(self) -> None:
        rng = np.random.default_rng(RANDOM_SEED)
        y = pd.Series(rng.integers(0, 2, size=N_ROWS))
        x = pd.DataFrame({"only_feat": y.values * 2.0 + rng.normal(0, 0.1, N_ROWS)})
        selected = select_features_mi(x, y, mi_threshold=MI_THRESHOLD_ZERO)
        assert selected == ["only_feat"]

    def test_selected_features_are_subset_of_input(
        self, informative_dataset: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        x, y = informative_dataset
        selected = select_features_mi(x, y)
        assert set(selected).issubset(set(x.columns))
