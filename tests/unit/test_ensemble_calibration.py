"""Tests for ensemble-level calibration consolidation (B.5)."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from finalayze.ml.calibration import EnsembleCalibrator
from finalayze.ml.models.ensemble import EnsembleModel

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_N_CALIBRATION_SAMPLES = 100
_N_TRAIN_SAMPLES = 100
_N_FEATURES = 2


class TestEnsembleCalibration:
    def test_ensemble_uses_calibrator_when_fitted(self) -> None:
        """EnsembleModel applies calibrator to raw ensemble output."""
        mock_model = MagicMock()
        mock_model._model = True
        mock_model.predict_proba.return_value = 0.7

        calibrator = EnsembleCalibrator()
        raw = np.array([0.3, 0.4, 0.5, 0.6, 0.7] * 20)
        labels = np.array([0, 0, 0, 1, 1] * 20)
        calibrator.fit(raw, labels)
        assert calibrator.is_fitted

        ensemble = EnsembleModel(models=[mock_model], calibrator=calibrator)
        result = ensemble.predict_proba({"feat": 1.0})
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_ensemble_returns_raw_when_no_calibrator(self) -> None:
        """Without calibrator, ensemble returns raw average."""
        mock_model = MagicMock()
        mock_model._model = True
        mock_model.predict_proba.return_value = 0.8

        ensemble = EnsembleModel(models=[mock_model])
        result = ensemble.predict_proba({"feat": 1.0})
        assert result == pytest.approx(0.8, abs=0.01)

    def test_ensemble_returns_raw_when_calibrator_unfitted(self) -> None:
        """Unfitted calibrator passes through raw probability."""
        mock_model = MagicMock()
        mock_model._model = True
        mock_model.predict_proba.return_value = 0.65

        calibrator = EnsembleCalibrator()
        assert not calibrator.is_fitted

        ensemble = EnsembleModel(models=[mock_model], calibrator=calibrator)
        result = ensemble.predict_proba({"feat": 1.0})
        assert result == pytest.approx(0.65, abs=0.01)

    def test_xgboost_returns_raw_proba(self) -> None:
        """XGBoostModel no longer has per-model calibrator after consolidation."""
        from finalayze.ml.models.xgboost_model import XGBoostModel

        model = XGBoostModel(segment_id="test")
        X = [{"a": float(i), "b": float(i * 2)} for i in range(_N_TRAIN_SAMPLES)]
        y = [1 if i > 50 else 0 for i in range(_N_TRAIN_SAMPLES)]
        model.fit(X, y)
        assert not hasattr(model, "_calibrator")
        result = model.predict_proba({"a": 75.0, "b": 150.0})
        assert 0.0 <= result <= 1.0

    def test_lightgbm_returns_raw_proba(self) -> None:
        """LightGBMModel no longer has per-model calibrator after consolidation."""
        from finalayze.ml.models.lightgbm_model import LightGBMModel

        model = LightGBMModel(segment_id="test")
        X = [{"a": float(i), "b": float(i * 2)} for i in range(_N_TRAIN_SAMPLES)]
        y = [1 if i > 50 else 0 for i in range(_N_TRAIN_SAMPLES)]
        model.fit(X, y)
        assert not hasattr(model, "_calibrator")
        result = model.predict_proba({"a": 75.0, "b": 150.0})
        assert 0.0 <= result <= 1.0

    def test_lstm_returns_raw_proba(self) -> None:
        """LSTMModel no longer has per-model Platt scaler after consolidation."""
        from finalayze.ml.models.lstm_model import LSTMModel

        seq_len = 3
        model = LSTMModel(segment_id="test", sequence_length=seq_len, hidden_size=8, num_layers=1)
        X = [{"a": float(i), "b": float(i * 2)} for i in range(10)]
        y = [i % 2 for i in range(10)]
        model.fit(X, y)
        assert not hasattr(model, "_platt_scaler")
        result = model.predict_proba({"a": 5.0, "b": 10.0})
        assert 0.0 <= result <= 1.0

    def test_xgboost_trains_on_all_data(self) -> None:
        """XGBoost trains on full dataset, no holdout split."""
        from finalayze.ml.models.xgboost_model import XGBoostModel

        model = XGBoostModel(segment_id="test")
        X = [{"a": float(i), "b": float(i * 2)} for i in range(_N_TRAIN_SAMPLES)]
        y = [1 if i > 50 else 0 for i in range(_N_TRAIN_SAMPLES)]
        model.fit(X, y)
        # Model should be trained and produce predictions
        assert model._model is not None  # noqa: SLF001
        result = model.predict_proba({"a": 75.0, "b": 150.0})
        assert isinstance(result, float)

    def test_lightgbm_trains_on_all_data(self) -> None:
        """LightGBM trains on full dataset, no holdout split."""
        from finalayze.ml.models.lightgbm_model import LightGBMModel

        model = LightGBMModel(segment_id="test")
        X = [{"a": float(i), "b": float(i * 2)} for i in range(_N_TRAIN_SAMPLES)]
        y = [1 if i > 50 else 0 for i in range(_N_TRAIN_SAMPLES)]
        model.fit(X, y)
        assert model._model is not None  # noqa: SLF001
        result = model.predict_proba({"a": 75.0, "b": 150.0})
        assert isinstance(result, float)
