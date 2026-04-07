"""Tests for ML quality gate fixes: profit_factor computation and calibrated Brier."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from finalayze.ml.training.quality_gates import (
    FoldMetrics,
    check_profit_factor_gate,
)


class TestCheckProfitFactorGate:
    """Tests for check_profit_factor_gate with computed values."""

    def test_passes_when_profit_factor_above_threshold(self) -> None:
        metrics = FoldMetrics(
            accuracy=0.6,
            brier_score=0.2,
            log_loss=0.5,
            n_test=100,
            profit_factor=1.25,
        )
        result = check_profit_factor_gate(metrics)
        assert result.passed is True
        assert result.gate_name == "profit_factor"

    def test_fails_when_profit_factor_is_default_1_0(self) -> None:
        metrics = FoldMetrics(
            accuracy=0.6,
            brier_score=0.2,
            log_loss=0.5,
            n_test=100,
            profit_factor=1.0,  # the old default -- should fail >= 1.10
        )
        result = check_profit_factor_gate(metrics)
        assert result.passed is False

    def test_passes_at_exact_threshold(self) -> None:
        metrics = FoldMetrics(
            accuracy=0.6,
            brier_score=0.2,
            log_loss=0.5,
            n_test=100,
            profit_factor=1.10,
        )
        result = check_profit_factor_gate(metrics)
        assert result.passed is True


class TestEvaluateFoldMetricsProfitFactor:
    """Tests for _evaluate_fold_metrics profit_factor computation."""

    def test_all_correct_predictions_high_pf(self) -> None:
        """When all BUY predictions are correct, profit_factor > 1.0."""
        # Import here to avoid import issues at module level
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
        from train_models import _evaluate_fold_metrics

        # Create mock models that return high prob for label=1 and low for label=0
        models = _make_mock_models([0.8, 0.8, 0.2, 0.2])

        test_features = [{"f1": 1.0}] * 4
        test_labels = [1, 1, 0, 0]

        result = _evaluate_fold_metrics(models, test_features, test_labels)
        # BUY signals (prob >= 0.55): items 0,1 (both label=1 -> profit)
        # No BUY for items 2,3 (prob 0.2 < 0.55)
        # gross_profit=2, gross_loss=0 -> PF=2.0
        assert result.profit_factor > 1.0
        assert result.profit_factor != 1.0  # not the default

    def test_all_wrong_predictions_low_pf(self) -> None:
        """When all BUY predictions are wrong, profit_factor < 1.0."""
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
        from train_models import _evaluate_fold_metrics

        # Models predict BUY (high prob) on label=0, SELL (low prob) on label=1
        models = _make_mock_models([0.2, 0.8, 0.2, 0.8])

        test_features = [{"f1": 1.0}] * 4
        test_labels = [1, 0, 1, 0]  # wrong: BUY on label=0

        result = _evaluate_fold_metrics(models, test_features, test_labels)
        # BUY signals (prob >= 0.55): items 1,3 (both label=0 -> loss)
        # gross_profit=0, gross_loss=2 -> PF = 2.0 if gp > 0 else 1.0... wait
        # Actually gp=0, gl=2 -> 0/2 = 0.0
        assert result.profit_factor < 1.0

    def test_brier_uses_calibrator_when_provided(self) -> None:
        """When calibrator is provided, brier_score uses calibrated probabilities."""
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
        from train_models import _evaluate_fold_metrics

        # Raw probas will be [0.7, 0.7, 0.3, 0.3]
        models = _make_mock_models([0.7, 0.7, 0.3, 0.3])

        test_features = [{"f1": 1.0}] * 4
        test_labels = [1, 1, 0, 0]

        # Calibrator that maps probabilities closer to extremes (better calibration)
        import numpy as np

        calibrator = MagicMock()
        calibrator.is_fitted = True
        calibrator.predict_proba = MagicMock(
            return_value=np.array([0.9, 0.9, 0.1, 0.1])
        )

        result_with_cal = _evaluate_fold_metrics(
            models, test_features, test_labels, calibrator=calibrator
        )

        # Without calibrator
        result_without_cal = _evaluate_fold_metrics(
            models, test_features, test_labels
        )

        # Calibrated should have better (lower) Brier score
        assert result_with_cal.brier_score < result_without_cal.brier_score

    def test_brier_without_calibrator_uses_raw(self) -> None:
        """Without calibrator, brier_score uses raw probabilities (backward compat)."""
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
        from train_models import _evaluate_fold_metrics

        models = _make_mock_models([0.7, 0.7, 0.3, 0.3])
        test_features = [{"f1": 1.0}] * 4
        test_labels = [1, 1, 0, 0]

        result = _evaluate_fold_metrics(models, test_features, test_labels)

        # Brier score from raw [0.7, 0.7, 0.3, 0.3] vs [1, 1, 0, 0]
        # = mean((1-0.7)^2, (1-0.7)^2, (0-0.3)^2, (0-0.3)^2) = mean(0.09, 0.09, 0.09, 0.09) = 0.09
        assert abs(result.brier_score - 0.09) < 0.01


def _make_mock_models(probas: list[float]) -> list:
    """Create mock models that return predetermined probabilities.

    Each model returns the same probability for each sample index.
    """
    models = []
    mock = MagicMock()
    mock._trained = True

    # predict_proba is called per-feature-dict, so we need to track call index
    call_counter = {"idx": 0}
    total = len(probas)

    def predict_proba_side_effect(feat: dict) -> float:
        idx = call_counter["idx"] % total
        call_counter["idx"] += 1
        return probas[idx]

    mock.predict_proba = MagicMock(side_effect=predict_proba_side_effect)
    models.append(mock)
    return models
