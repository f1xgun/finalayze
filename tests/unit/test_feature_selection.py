"""Tests for feature selection module."""

from __future__ import annotations

import numpy as np

from finalayze.ml.training.feature_selection import select_features

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_N_SAMPLES = 500
_IMPORTANCE_THRESHOLD = 0.01
_CORRELATION_THRESHOLD = 0.85


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
