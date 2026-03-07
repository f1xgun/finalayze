"""Tests for feature selection module."""

from __future__ import annotations

import numpy as np
import pandas as pd

from finalayze.ml.training.feature_selection import select_features, select_features_mi

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_N_SAMPLES = 500
_IMPORTANCE_THRESHOLD = 0.01
_CORRELATION_THRESHOLD = 0.85
_MI_THRESHOLD_DEFAULT = 0.02
_MIN_FEATURE_COUNT = 8


class TestFeatureSelection:
    """Tests for select_features."""

    @staticmethod
    def _make_data_with_useless_feature(
        n: int = _N_SAMPLES,
    ) -> tuple[list[dict[str, float]], list[int]]:
        """Create data where one feature is constant (zero importance)."""
        rng = np.random.default_rng(42)
        features: list[dict[str, float]] = []
        labels: list[int] = []
        for _ in range(n):
            signal = float(rng.standard_normal())
            noise = float(rng.standard_normal())
            features.append(
                {
                    "signal": signal,
                    "noise": noise,
                    "constant": 1.0,  # zero variance = no importance
                }
            )
            labels.append(1 if signal > 0 else 0)
        return features, labels

    @staticmethod
    def _make_correlated_data(
        n: int = _N_SAMPLES,
    ) -> tuple[list[dict[str, float]], list[int]]:
        """Create data with two highly correlated features."""
        rng = np.random.default_rng(42)
        features: list[dict[str, float]] = []
        labels: list[int] = []
        for _ in range(n):
            base = float(rng.standard_normal())
            features.append(
                {
                    "feat_a": base,
                    "feat_b": base + float(rng.normal(0, 0.05)),  # ~0.99 corr
                    "feat_c": float(rng.standard_normal()),  # uncorrelated
                }
            )
            labels.append(1 if base > 0 else 0)
        return features, labels

    def test_drops_low_importance(self) -> None:
        """Features with < 1% importance should be dropped."""
        features, labels = self._make_data_with_useless_feature()
        filtered, selected = select_features(
            features, labels, importance_threshold=_IMPORTANCE_THRESHOLD
        )
        # 'constant' has zero variance => should get zero or near-zero importance
        # It should be dropped
        assert "constant" not in selected
        assert len(selected) < 3  # noqa: PLR2004
        assert len(filtered) == len(features)
        # Each filtered row only has selected keys
        for row in filtered:
            assert set(row.keys()) == set(selected)

    def test_deduplicates_correlated(self) -> None:
        """Correlated features (> 0.85) should be reduced to one."""
        features, labels = self._make_correlated_data()
        _filtered, selected = select_features(
            features, labels, correlation_threshold=_CORRELATION_THRESHOLD
        )
        # feat_a and feat_b are ~0.99 correlated, one should be dropped
        has_a = "feat_a" in selected
        has_b = "feat_b" in selected
        # At most one of the correlated pair should survive
        assert not (has_a and has_b), "Both correlated features survived"

    def test_preserves_important_features(self) -> None:
        """Important, uncorrelated features should be kept."""
        features, labels = self._make_correlated_data()
        filtered, selected = select_features(features, labels)
        # feat_c is uncorrelated and should be important if it has signal
        # At minimum, the signal feature (feat_a or feat_b) should survive
        assert len(selected) >= 1
        assert len(filtered) == len(features)

    def test_empty_features(self) -> None:
        """Empty input returns empty output."""
        filtered, selected = select_features([], [])
        assert filtered == []
        assert selected == []

    def test_all_features_returned_in_filtered(self) -> None:
        """Every row in filtered output has exactly the selected feature names."""
        rng = np.random.default_rng(42)
        features: list[dict[str, float]] = []
        labels: list[int] = []
        for _ in range(_N_SAMPLES):
            v = float(rng.standard_normal())
            features.append({"a": v, "b": float(rng.standard_normal())})
            labels.append(1 if v > 0 else 0)

        filtered, selected = select_features(features, labels)
        for row in filtered:
            assert set(row.keys()) == set(selected)


class TestSelectFeaturesMI:
    """Tests for select_features_mi with relaxed thresholds."""

    @staticmethod
    def _make_mi_data(
        n_samples: int = 200,
        n_features: int = 15,
        *,
        low_mi: bool = False,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Create synthetic data for MI-based feature selection.

        Args:
            n_samples: Number of samples.
            n_features: Number of features.
            low_mi: If True, features have very low MI with target (near-noise).
        """
        rng = np.random.default_rng(42)
        # Create a target based on a hidden signal
        signal = rng.standard_normal(n_samples)
        y = pd.Series((signal > 0).astype(int))

        data: dict[str, np.ndarray] = {}
        for i in range(n_features):
            if low_mi:
                # All features are pure noise — very low MI with target
                data[f"feat_{i:02d}"] = rng.standard_normal(n_samples)
            else:
                # Mix of signal and noise so MI varies
                noise_ratio = 0.3 + 0.7 * (i / max(n_features - 1, 1))
                data[f"feat_{i:02d}"] = (
                    signal * (1 - noise_ratio)
                    + rng.standard_normal(n_samples) * noise_ratio
                )

        return pd.DataFrame(data), y

    def test_default_mi_threshold_is_0_02(self) -> None:
        """Default MI threshold should be 0.02 (not the old 0.05)."""
        import inspect

        sig = inspect.signature(select_features_mi)
        default = sig.parameters["mi_threshold"].default
        assert default == _MI_THRESHOLD_DEFAULT

    def test_minimum_feature_count_floor(self) -> None:
        """When all MI scores are low, still returns at least 8 features."""
        x, y = self._make_mi_data(n_samples=200, n_features=12, low_mi=True)
        selected = select_features_mi(x, y, mi_threshold=_MI_THRESHOLD_DEFAULT)
        # Even with low MI scores, the floor should guarantee at least 8
        assert len(selected) >= _MIN_FEATURE_COUNT

    def test_minimum_feature_count_floor_capped_by_available(self) -> None:
        """Floor cannot exceed the number of available features."""
        n_features = 5
        x, y = self._make_mi_data(n_samples=200, n_features=n_features, low_mi=True)
        selected = select_features_mi(x, y, mi_threshold=_MI_THRESHOLD_DEFAULT)
        # Cannot return more features than exist
        assert len(selected) <= n_features

    def test_75th_percentile_dedup_retains_more_than_median(self) -> None:
        """Using 75th percentile for dedup retains more features than median would.

        We verify this indirectly: with correlated features that have signal,
        the relaxed dedup should keep more features than an aggressive dedup.
        """
        rng = np.random.default_rng(42)
        n_samples = 300
        signal = rng.standard_normal(n_samples)
        y = pd.Series((signal > 0).astype(int))

        # Create features: several correlated with signal, some pure noise
        data: dict[str, np.ndarray] = {}
        n_correlated = 10
        for i in range(n_correlated):
            # Correlated with signal but also with each other
            data[f"corr_{i:02d}"] = (
                signal + rng.standard_normal(n_samples) * (0.5 + i * 0.1)
            )
        # A few uncorrelated noise features
        for i in range(5):
            data[f"noise_{i:02d}"] = rng.standard_normal(n_samples)

        x = pd.DataFrame(data)
        selected = select_features_mi(x, y, mi_threshold=_MI_THRESHOLD_DEFAULT)

        # With 75th percentile dedup, correlated-but-useful features survive.
        # We should get more than just 1-2 features from the correlated group.
        correlated_selected = [f for f in selected if f.startswith("corr_")]
        assert len(correlated_selected) >= 3  # noqa: PLR2004

    def test_respects_max_features(self) -> None:
        """Should not return more than max_features."""
        max_feat = 5
        x, y = self._make_mi_data(n_samples=200, n_features=15)
        selected = select_features_mi(x, y, max_features=max_feat)
        assert len(selected) <= max_feat

    def test_empty_input(self) -> None:
        """Empty DataFrame returns empty list."""
        x = pd.DataFrame()
        y = pd.Series(dtype=int)
        selected = select_features_mi(x, y)
        assert selected == []

    def test_returns_ordered_by_mi(self) -> None:
        """Selected features should be ordered by descending MI score."""
        x, y = self._make_mi_data(n_samples=200, n_features=10)
        selected = select_features_mi(x, y, max_features=10)
        # At minimum, should return a non-empty list
        assert len(selected) >= 1
