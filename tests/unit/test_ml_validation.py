"""Tests for validation gate metrics (6C.7)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from finalayze.ml.training import ValidationResult, validate_ensemble


def _make_ensemble(probas: list[float]) -> MagicMock:
    """Create a mock EnsembleModel returning given probabilities."""
    ensemble = MagicMock()
    ensemble.predict_proba.side_effect = probas
    return ensemble


class TestValidationGate:
    def test_validate_ensemble_passes_good_model(self) -> None:
        """Model with good accuracy, Brier, and log-loss passes."""
        # Perfect predictions: labels=[1,1,0,0], probas=[0.9, 0.8, 0.1, 0.2]
        features = [{"a": float(i)} for i in range(4)]
        labels = [1, 1, 0, 0]
        probas = [0.9, 0.8, 0.1, 0.2]
        ensemble = _make_ensemble(probas)

        result = validate_ensemble(ensemble, features, labels)
        assert result.passed
        assert result.accuracy == 1.0
        assert result.brier_score < 0.25
        assert result.n_samples == 4

    def test_validate_ensemble_fails_bad_accuracy(self) -> None:
        """Accuracy below 52% -> not passed."""
        features = [{"a": float(i)} for i in range(10)]
        labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        # All predict 0.4 -> all round to 0 -> 5/10 correct = 50%
        probas = [0.4] * 10
        ensemble = _make_ensemble(probas)

        result = validate_ensemble(ensemble, features, labels)
        assert not result.passed
        assert result.accuracy < 0.52

    def test_validate_ensemble_fails_bad_brier(self) -> None:
        """Brier > 0.25 -> not passed even with OK accuracy."""
        features = [{"a": float(i)} for i in range(4)]
        labels = [1, 1, 0, 0]
        # Overconfident wrong: predict 0.9 for class 0, 0.1 for class 1
        # Actually: labels are 1,1,0,0 and probas 0.1,0.1,0.9,0.9
        # Brier = mean((1-0.1)^2, (1-0.1)^2, (0-0.9)^2, (0-0.9)^2) = 0.81
        probas = [0.1, 0.1, 0.9, 0.9]
        ensemble = _make_ensemble(probas)

        result = validate_ensemble(ensemble, features, labels)
        assert not result.passed
        assert result.brier_score > 0.25

    def test_validate_ensemble_fails_bad_logloss(self) -> None:
        """Log-loss above threshold -> not passed."""
        features = [{"a": float(i)} for i in range(4)]
        labels = [1, 1, 0, 0]
        # Probas close to 0.5 with some wrong direction push -> high log-loss
        # 0.501 rounds to 1 so accuracy=1.0, but log-loss stays near ln(2)
        probas = [0.501, 0.501, 0.499, 0.499]
        ensemble = _make_ensemble(probas)

        result = validate_ensemble(ensemble, features, labels)
        # log-loss for near-0.5 predictions is >= ln(2) ~ 0.693
        assert result.log_loss_val > 0.69
        assert not result.passed

    def test_validation_result_dataclass_fields(self) -> None:
        """Verify all fields present in ValidationResult."""
        result = ValidationResult(
            accuracy=0.6,
            brier_score=0.2,
            log_loss_val=0.5,
            n_samples=100,
            passed=True,
        )
        assert result.accuracy == 0.6
        assert result.brier_score == 0.2
        assert result.log_loss_val == 0.5
        assert result.n_samples == 100
        assert result.passed is True
