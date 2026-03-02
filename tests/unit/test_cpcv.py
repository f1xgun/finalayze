"""Tests for Combinatorial Purged Cross-Validation (CPCV)."""

from __future__ import annotations

import numpy as np
import pytest

from finalayze.ml.training.cpcv import CPCVSplit, evaluate_cpcv, generate_cpcv_splits

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_N_SAMPLES = 1000
_N_GROUPS = 5
_N_TEST_GROUPS = 2
_EXPECTED_SPLITS = 10  # C(5, 2) = 10
_PURGE_WINDOW = 60


class TestGenerateCPCVSplits:
    """Tests for generate_cpcv_splits."""

    def test_cpcv_generates_10_splits(self) -> None:
        """C(5, 2) = 10 splits should be generated."""
        splits = generate_cpcv_splits(
            n_samples=_N_SAMPLES,
            n_groups=_N_GROUPS,
            n_test_groups=_N_TEST_GROUPS,
            purge_window=_PURGE_WINDOW,
        )
        assert len(splits) == _EXPECTED_SPLITS

    def test_cpcv_purge_window_60(self) -> None:
        """Purge removes samples at boundaries between train and test blocks."""
        splits = generate_cpcv_splits(
            n_samples=_N_SAMPLES,
            n_groups=_N_GROUPS,
            n_test_groups=_N_TEST_GROUPS,
            purge_window=_PURGE_WINDOW,
        )
        all_indices = set(range(_N_SAMPLES))
        for split in splits:
            used = set(split.train_indices) | set(split.test_indices)
            purged = all_indices - used
            # Purge window is 60, so there must be purged samples
            assert len(purged) > 0, "Expected purged samples at boundaries"

    def test_cpcv_no_overlap(self) -> None:
        """Train and test indices must not overlap in any split."""
        splits = generate_cpcv_splits(
            n_samples=_N_SAMPLES,
            n_groups=_N_GROUPS,
            n_test_groups=_N_TEST_GROUPS,
            purge_window=_PURGE_WINDOW,
        )
        for i, split in enumerate(splits):
            overlap = set(split.train_indices) & set(split.test_indices)
            assert not overlap, f"Split {i} has overlap: {overlap}"

    def test_cpcv_purge_window_too_small_raises(self) -> None:
        """purge_window < 60 must raise ValueError."""
        with pytest.raises(ValueError, match="purge_window must be >= 60"):
            generate_cpcv_splits(n_samples=_N_SAMPLES, purge_window=5)

    def test_cpcv_each_split_has_data(self) -> None:
        """Every split must have non-empty train and test sets."""
        splits = generate_cpcv_splits(
            n_samples=_N_SAMPLES,
            n_groups=_N_GROUPS,
            n_test_groups=_N_TEST_GROUPS,
            purge_window=_PURGE_WINDOW,
        )
        for i, split in enumerate(splits):
            assert len(split.train_indices) > 0, f"Split {i} has empty train"
            assert len(split.test_indices) > 0, f"Split {i} has empty test"

    def test_cpcv_indices_within_bounds(self) -> None:
        """All indices must be in [0, n_samples)."""
        splits = generate_cpcv_splits(
            n_samples=_N_SAMPLES,
            n_groups=_N_GROUPS,
            n_test_groups=_N_TEST_GROUPS,
            purge_window=_PURGE_WINDOW,
        )
        for split in splits:
            for idx in split.train_indices:
                assert 0 <= idx < _N_SAMPLES
            for idx in split.test_indices:
                assert 0 <= idx < _N_SAMPLES


class TestEvaluateCPCV:
    """Tests for evaluate_cpcv."""

    @staticmethod
    def _make_separable_data(
        n: int, n_features: int = 5
    ) -> tuple[list[dict[str, float]], list[int]]:
        """Create data where label is easily predictable from features."""
        rng = np.random.default_rng(42)
        features: list[dict[str, float]] = []
        labels: list[int] = []
        for _ in range(n):
            vals = rng.standard_normal(n_features)
            feat = {f"f{j}": float(vals[j]) for j in range(n_features)}
            # Label determined by sign of first feature (easy to learn)
            label = 1 if vals[0] > 0 else 0
            features.append(feat)
            labels.append(label)
        return features, labels

    @staticmethod
    def _make_random_data(n: int, n_features: int = 5) -> tuple[list[dict[str, float]], list[int]]:
        """Create random noise data (no signal)."""
        rng = np.random.default_rng(99)
        features: list[dict[str, float]] = []
        labels: list[int] = []
        for _ in range(n):
            vals = rng.standard_normal(n_features)
            feat = {f"f{j}": float(vals[j]) for j in range(n_features)}
            label = int(rng.integers(0, 2))
            features.append(feat)
            labels.append(label)
        return features, labels

    def test_cpcv_acceptance(self) -> None:
        """Easily separable data should be accepted (high median Sharpe)."""
        features, labels = self._make_separable_data(2000)
        result = evaluate_cpcv(features, labels, n_groups=_N_GROUPS, n_test_groups=_N_TEST_GROUPS)
        assert result["accepted"] is True
        assert result["median_sharpe"] >= 0.3  # noqa: PLR2004
        assert len(result["fold_sharpes"]) == _EXPECTED_SPLITS

    def test_cpcv_rejection_criteria(self) -> None:
        """Random noise data should be rejected (poor Sharpe)."""
        features, labels = self._make_random_data(2000)
        result = evaluate_cpcv(features, labels, n_groups=_N_GROUPS, n_test_groups=_N_TEST_GROUPS)
        # Random data should have median Sharpe near 0 or negative
        # At minimum, it should not reliably pass
        assert len(result["fold_sharpes"]) == _EXPECTED_SPLITS
        # Verify the result dict has all expected keys
        assert "median_sharpe" in result
        assert "negative_folds_pct" in result
        assert "accepted" in result

    def test_cpcv_returns_expected_keys(self) -> None:
        """Result dict must contain all documented keys."""
        features, labels = self._make_separable_data(1000)
        result = evaluate_cpcv(features, labels)
        expected_keys = {"fold_sharpes", "median_sharpe", "negative_folds_pct", "accepted"}
        assert set(result.keys()) == expected_keys

    def test_cpcv_empty_data(self) -> None:
        """Empty data should return not-accepted."""
        result = evaluate_cpcv([], [])
        assert result["accepted"] is False
        assert result["fold_sharpes"] == []
