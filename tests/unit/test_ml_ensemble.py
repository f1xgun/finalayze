"""Tests for ensemble exception handling (6C.4)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import structlog

from finalayze.core.exceptions import PredictionError
from finalayze.ml.models.ensemble import EnsembleModel


def _make_model(*, trained: bool = True, proba: float = 0.7, raises: bool = False) -> MagicMock:
    """Create a mock BaseMLModel."""
    model = MagicMock()
    model._model = MagicMock() if trained else None
    if raises:
        model.predict_proba.side_effect = RuntimeError("model error")
    else:
        model.predict_proba.return_value = proba
    return model


def _make_lstm(*, trained: bool = True, proba: float = 0.6, raises: bool = False) -> MagicMock:
    """Create a mock LSTMModel."""
    lstm = MagicMock()
    lstm._trained = trained
    if raises:
        lstm.predict_proba.side_effect = RuntimeError("lstm error")
    else:
        lstm.predict_proba.return_value = proba
    return lstm


class TestEnsembleExceptionHandling:
    """6C.4: Graceful degradation on predict_proba failures."""

    def test_ensemble_skips_failing_model(self) -> None:
        """One model raises, others succeed; average from surviving models."""
        good = _make_model(proba=0.8)
        bad = _make_model(raises=True)
        ensemble = EnsembleModel(models=[good, bad], lstm_model=None)
        result = ensemble.predict_proba({"a": 1.0})
        assert result == pytest.approx(0.8)

    def test_ensemble_all_fail_raises_prediction_error(self) -> None:
        """All trained models raise; PredictionError is raised."""
        bad1 = _make_model(raises=True)
        bad2 = _make_model(raises=True)
        ensemble = EnsembleModel(models=[bad1, bad2], lstm_model=None)
        with pytest.raises(PredictionError):
            ensemble.predict_proba({"a": 1.0})

    def test_ensemble_untrained_returns_default(self) -> None:
        """No trained models; returns 0.5."""
        untrained = _make_model(trained=False)
        ensemble = EnsembleModel(models=[untrained], lstm_model=None)
        result = ensemble.predict_proba({"a": 1.0})
        assert result == pytest.approx(0.5)

    def test_ensemble_partial_failure_logged(self) -> None:
        """Failing model generates a warning log via structlog."""
        good = _make_model(proba=0.7)
        bad = _make_model(raises=True)
        ensemble = EnsembleModel(models=[good, bad], lstm_model=None)
        with structlog.testing.capture_logs() as captured:
            ensemble.predict_proba({"a": 1.0})
        assert any("ensemble_model_failed" in entry.get("event", "") for entry in captured)

    def test_ensemble_lstm_failure_skipped(self) -> None:
        """LSTM raises but tree models succeed; returns tree average."""
        good = _make_model(proba=0.8)
        bad_lstm = _make_lstm(raises=True)
        ensemble = EnsembleModel(models=[good], lstm_model=bad_lstm)
        result = ensemble.predict_proba({"a": 1.0})
        assert result == pytest.approx(0.8)

    def test_ensemble_all_fail_including_lstm(self) -> None:
        """All models including LSTM fail; PredictionError is raised."""
        bad = _make_model(raises=True)
        bad_lstm = _make_lstm(raises=True)
        ensemble = EnsembleModel(models=[bad], lstm_model=bad_lstm)
        with pytest.raises(PredictionError):
            ensemble.predict_proba({"a": 1.0})

    def test_ensemble_mixed_trained_untrained(self) -> None:
        """Mix of trained and untrained; only trained contribute."""
        trained = _make_model(proba=0.9)
        untrained = _make_model(trained=False)
        ensemble = EnsembleModel(models=[trained, untrained], lstm_model=None)
        result = ensemble.predict_proba({"a": 1.0})
        assert result == pytest.approx(0.9)


class TestWeightedAveraging:
    """C2: Performance-weighted averaging in EnsembleModel."""

    def test_weighted_averaging_excludes_coinflip(self) -> None:
        """Model at 50% accuracy gets weight 0, excluded from average."""
        good = _make_model(proba=0.9)
        good.__class__.__name__ = "GoodModel"
        coinflip = _make_model(proba=0.1)
        coinflip.__class__.__name__ = "CoinflipModel"
        # GoodModel weight = (0.7 - 0.5)^2 = 0.04
        # CoinflipModel weight = max(0, 0.5 - 0.5)^2 = 0.0
        ensemble = EnsembleModel(
            models=[good, coinflip],
            lstm_model=None,
            model_weights={"GoodModel": 0.04, "CoinflipModel": 0.0},
        )
        result = ensemble.predict_proba({"a": 1.0})
        # Only GoodModel contributes → result should be 0.9
        assert result == pytest.approx(0.9)

    def test_weighted_averaging_favors_better_model(self) -> None:
        """Higher accuracy model has more influence on the average."""
        strong = _make_model(proba=0.9)
        strong.__class__.__name__ = "StrongModel"
        weak = _make_model(proba=0.3)
        weak.__class__.__name__ = "WeakModel"
        # strong weight = (0.7 - 0.5)^2 = 0.04
        # weak weight = (0.55 - 0.5)^2 = 0.0025
        ensemble = EnsembleModel(
            models=[strong, weak],
            lstm_model=None,
            model_weights={"StrongModel": 0.04, "WeakModel": 0.0025},
        )
        result = ensemble.predict_proba({"a": 1.0})
        # weighted = (0.04*0.9 + 0.0025*0.3) / (0.04 + 0.0025) = 0.03675/0.0425 ≈ 0.8647
        expected = (0.04 * 0.9 + 0.0025 * 0.3) / (0.04 + 0.0025)
        assert result == pytest.approx(expected, abs=1e-6)

    def test_equal_averaging_without_weights(self) -> None:
        """Without model_weights, falls back to equal averaging."""
        m1 = _make_model(proba=0.8)
        m2 = _make_model(proba=0.6)
        ensemble = EnsembleModel(models=[m1, m2], lstm_model=None)
        result = ensemble.predict_proba({"a": 1.0})
        assert result == pytest.approx(0.7)

    def test_all_zero_weights_returns_default(self) -> None:
        """When all model weights are 0, returns 0.5 (default)."""
        m1 = _make_model(proba=0.9)
        m1.__class__.__name__ = "M1"
        m2 = _make_model(proba=0.1)
        m2.__class__.__name__ = "M2"
        ensemble = EnsembleModel(
            models=[m1, m2],
            lstm_model=None,
            model_weights={"M1": 0.0, "M2": 0.0},
        )
        result = ensemble.predict_proba({"a": 1.0})
        assert result == pytest.approx(0.5)


class TestPredictionUncertainty:
    """C5: Ensemble disagreement tracking."""

    def test_prediction_uncertainty_high_disagreement(self) -> None:
        """std > 0 when models disagree."""
        m1 = _make_model(proba=0.9)
        m2 = _make_model(proba=0.1)
        ensemble = EnsembleModel(models=[m1, m2], lstm_model=None)
        ensemble.predict_proba({"a": 1.0})
        # Manually set distinct keys since mocks share type name
        ensemble.last_model_probas = {"XGBoostModel": 0.9, "LightGBMModel": 0.1}
        assert ensemble.prediction_uncertainty > 0.0

    def test_prediction_uncertainty_agreement(self) -> None:
        """std ~ 0 when models agree."""
        m1 = _make_model(proba=0.7)
        m2 = _make_model(proba=0.7)
        ensemble = EnsembleModel(models=[m1, m2], lstm_model=None)
        ensemble.predict_proba({"a": 1.0})
        # Manually set distinct keys since mocks share type name
        ensemble.last_model_probas = {"XGBoostModel": 0.7, "LightGBMModel": 0.7}
        assert ensemble.prediction_uncertainty == pytest.approx(0.0, abs=1e-9)

    def test_prediction_uncertainty_single_model(self) -> None:
        """With a single model, uncertainty is 0."""
        m1 = _make_model(proba=0.7)
        ensemble = EnsembleModel(models=[m1], lstm_model=None)
        ensemble.predict_proba({"a": 1.0})
        assert ensemble.prediction_uncertainty == 0.0

    def test_prediction_uncertainty_before_predict(self) -> None:
        """Before any prediction, uncertainty is 0."""
        ensemble = EnsembleModel(models=[], lstm_model=None)
        assert ensemble.prediction_uncertainty == 0.0


class TestEnsembleSelectedFeatures:
    """Feature mismatch fix: selected_features attribute on EnsembleModel."""

    def test_selected_features_default_none(self) -> None:
        """EnsembleModel defaults selected_features to None."""
        ensemble = EnsembleModel(models=[], lstm_model=None)
        assert ensemble.selected_features is None

    def test_selected_features_set_via_constructor(self) -> None:
        """selected_features can be passed through the constructor."""
        features = ["rsi_14", "macd_hist_pct", "bb_pct_b"]
        ensemble = EnsembleModel(models=[], lstm_model=None, selected_features=features)
        assert ensemble.selected_features == features

    def test_selected_features_assignable(self) -> None:
        """selected_features can be assigned after construction."""
        ensemble = EnsembleModel(models=[], lstm_model=None)
        features = ["rsi_14", "atr_14_pct"]
        ensemble.selected_features = features
        assert ensemble.selected_features == features
