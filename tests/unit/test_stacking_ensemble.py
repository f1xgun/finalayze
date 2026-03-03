"""Tests for StackingEnsemble meta-learner."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from finalayze.ml.models.stacking import StackingEnsemble


class TestStackingPredictBeforeFit:
    """StackingEnsemble.predict_proba falls back to mean when not fitted."""

    def test_stacking_predict_before_fit_returns_mean(self) -> None:
        stacking = StackingEnsemble()
        probs = [0.3, 0.7, 0.5]
        result = stacking.predict_proba(probs)
        assert result == pytest.approx(0.5, abs=1e-6)

    def test_is_fitted_false_before_fit(self) -> None:
        stacking = StackingEnsemble()
        assert stacking.is_fitted is False


class TestStackingFitAndPredict:
    """StackingEnsemble.fit trains a LogisticRegression meta-learner."""

    def test_stacking_fit_and_predict(self) -> None:
        stacking = StackingEnsemble()

        rng = np.random.default_rng(42)
        n_samples = 200
        # Model A: good predictor for label=1
        # Model B: noisy predictor
        labels = [1] * (n_samples // 2) + [0] * (n_samples // 2)
        holdout: list[list[float]] = []
        for label in labels:
            if label == 1:
                holdout.append([float(rng.uniform(0.6, 0.9)), float(rng.uniform(0.3, 0.7))])
            else:
                holdout.append([float(rng.uniform(0.1, 0.4)), float(rng.uniform(0.3, 0.7))])

        stacking.fit(holdout, labels)
        assert stacking.is_fitted is True

        # High-probability inputs should give high output
        high_prob = stacking.predict_proba([0.85, 0.65])
        low_prob = stacking.predict_proba([0.15, 0.35])
        assert high_prob > low_prob
        assert 0.0 <= high_prob <= 1.0
        assert 0.0 <= low_prob <= 1.0

    def test_stacking_predict_returns_float(self) -> None:
        stacking = StackingEnsemble()
        rng = np.random.default_rng(7)
        n = 100
        holdout = [[float(rng.uniform(0, 1)), float(rng.uniform(0, 1))] for _ in range(n)]
        labels = [int(rng.integers(0, 2)) for _ in range(n)]
        stacking.fit(holdout, labels)

        result = stacking.predict_proba([0.5, 0.5])
        assert isinstance(result, float)


class TestStackingMinimumSamples:
    """StackingEnsemble.fit requires a minimum number of samples."""

    _MIN_SAMPLES = 10

    def test_stacking_minimum_samples_raises(self) -> None:
        stacking = StackingEnsemble()
        holdout = [[0.5, 0.5]] * 5
        labels = [1, 0, 1, 0, 1]
        with pytest.raises(ValueError, match="minimum"):
            stacking.fit(holdout, labels)

    def test_stacking_minimum_samples_exact_boundary(self) -> None:
        stacking = StackingEnsemble()
        holdout = [[0.5, 0.5]] * self._MIN_SAMPLES
        labels = [1, 0] * (self._MIN_SAMPLES // 2)
        # Should not raise
        stacking.fit(holdout, labels)
        assert stacking.is_fitted is True


class TestStackingWithEnsembleModel:
    """Integration: EnsembleModel uses StackingEnsemble when provided."""

    def test_stacking_with_ensemble_model(self) -> None:
        from finalayze.ml.models.ensemble import EnsembleModel

        # Create two mock models
        m1 = MagicMock()
        m1._model = MagicMock()
        m1.predict_proba.return_value = 0.8

        m2 = MagicMock()
        m2._model = MagicMock()
        m2.predict_proba.return_value = 0.6

        # Without stacking: simple mean = 0.7
        ensemble_plain = EnsembleModel(models=[m1, m2])
        result_plain = ensemble_plain.predict_proba({"a": 1.0})
        assert result_plain == pytest.approx(0.7)

        # With fitted stacking: uses meta-learner
        stacking = StackingEnsemble()
        rng = np.random.default_rng(42)
        n = 100
        holdout = [[float(rng.uniform(0, 1)), float(rng.uniform(0, 1))] for _ in range(n)]
        labels = [int(rng.integers(0, 2)) for _ in range(n)]
        stacking.fit(holdout, labels)

        ensemble_stacked = EnsembleModel(models=[m1, m2], stacking=stacking)
        result_stacked = ensemble_stacked.predict_proba({"a": 1.0})
        # Result should be a valid probability but likely different from simple mean
        assert 0.0 <= result_stacked <= 1.0

    def test_unfitted_stacking_falls_back_to_mean(self) -> None:
        from finalayze.ml.models.ensemble import EnsembleModel

        m1 = MagicMock()
        m1._model = MagicMock()
        m1.predict_proba.return_value = 0.8

        m2 = MagicMock()
        m2._model = MagicMock()
        m2.predict_proba.return_value = 0.6

        stacking = StackingEnsemble()  # not fitted
        ensemble = EnsembleModel(models=[m1, m2], stacking=stacking)
        result = ensemble.predict_proba({"a": 1.0})
        # Falls back to mean
        assert result == pytest.approx(0.7)
