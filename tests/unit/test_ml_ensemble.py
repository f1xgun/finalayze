"""Tests for ensemble exception handling (6C.4)."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

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


def _make_lstm(
    *, trained: bool = True, proba: float = 0.6, raises: bool = False
) -> MagicMock:
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

    def test_ensemble_partial_failure_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        """Failing model generates a warning log."""
        good = _make_model(proba=0.7)
        bad = _make_model(raises=True)
        ensemble = EnsembleModel(models=[good, bad], lstm_model=None)
        with caplog.at_level(logging.WARNING):
            ensemble.predict_proba({"a": 1.0})
        assert any("failed, skipping" in record.message for record in caplog.records)

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
