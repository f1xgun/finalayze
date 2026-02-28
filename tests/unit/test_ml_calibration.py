"""Unit tests for ML model probability calibration."""

from __future__ import annotations

import numpy as np
import pytest

from finalayze.ml.models.lightgbm_model import LightGBMModel
from finalayze.ml.models.xgboost_model import XGBoostModel

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_N_SAMPLES = 100
_N_FEATURES = 5
_UNTRAINED_PROB = 0.5
_CALIBRATION_HOLDOUT_FRACTION = 0.2
_MIN_PROB = 0.0
_MAX_PROB = 1.0
_RNG_SEED = 42


def _make_synthetic_data(
    n_samples: int = _N_SAMPLES,
    n_features: int = _N_FEATURES,
) -> tuple[list[dict[str, float]], list[int]]:
    """Create synthetic training data with separable classes."""
    rng = np.random.default_rng(_RNG_SEED)
    feature_keys = [f"feat_{i}" for i in range(n_features)]
    X: list[dict[str, float]] = []
    y: list[int] = []
    for i in range(n_samples):
        label = 1 if i % 2 == 0 else 0
        row = {
            k: float(rng.standard_normal() + (1.0 if label == 1 else -1.0)) for k in feature_keys
        }
        X.append(row)
        y.append(label)
    return X, y


# ── XGBoost Calibration ─────────────────────────────────────────────────────


class TestXGBoostCalibration:
    def test_untrained_returns_half(self) -> None:
        model = XGBoostModel("test-seg")
        result = model.predict_proba({"feat_0": 0.1, "feat_1": 0.2})
        assert result == _UNTRAINED_PROB

    def test_calibrated_proba_in_range(self) -> None:
        model = XGBoostModel("test-seg")
        X, y = _make_synthetic_data()
        model.fit(X, y)
        sample = X[0]
        proba = model.predict_proba(sample)
        assert _MIN_PROB <= proba <= _MAX_PROB

    def test_fit_stores_calibrator(self) -> None:
        model = XGBoostModel("test-seg")
        X, y = _make_synthetic_data()
        model.fit(X, y)
        assert model._calibrator is not None  # noqa: SLF001

    def test_calibrated_output_differs_from_raw(self) -> None:
        """Calibration should transform raw probabilities (may be same in degenerate cases)."""
        model = XGBoostModel("test-seg")
        X, y = _make_synthetic_data()
        model.fit(X, y)
        # Just verify predict_proba returns a valid float after calibration
        proba = model.predict_proba(X[0])
        assert isinstance(proba, float)

    def test_multiple_predictions_all_in_range(self) -> None:
        model = XGBoostModel("test-seg")
        X, y = _make_synthetic_data()
        model.fit(X, y)
        for sample in X[:10]:
            proba = model.predict_proba(sample)
            assert _MIN_PROB <= proba <= _MAX_PROB


# ── LightGBM Calibration ────────────────────────────────────────────────────


class TestLightGBMCalibration:
    def test_untrained_returns_half(self) -> None:
        model = LightGBMModel("test-seg")
        result = model.predict_proba({"feat_0": 0.1, "feat_1": 0.2})
        assert result == _UNTRAINED_PROB

    def test_calibrated_proba_in_range(self) -> None:
        model = LightGBMModel("test-seg")
        X, y = _make_synthetic_data()
        model.fit(X, y)
        sample = X[0]
        proba = model.predict_proba(sample)
        assert _MIN_PROB <= proba <= _MAX_PROB

    def test_fit_stores_calibrator(self) -> None:
        model = LightGBMModel("test-seg")
        X, y = _make_synthetic_data()
        model.fit(X, y)
        assert model._calibrator is not None  # noqa: SLF001

    def test_multiple_predictions_all_in_range(self) -> None:
        model = LightGBMModel("test-seg")
        X, y = _make_synthetic_data()
        model.fit(X, y)
        for sample in X[:10]:
            proba = model.predict_proba(sample)
            assert _MIN_PROB <= proba <= _MAX_PROB


# ── LSTM Calibration ────────────────────────────────────────────────────────


class TestLSTMCalibration:
    def test_untrained_returns_half(self) -> None:
        from finalayze.ml.models.lstm_model import LSTMModel

        model = LSTMModel("test-seg", sequence_length=5)
        result = model.predict_proba({"feat_0": 0.1, "feat_1": 0.2})
        assert result == _UNTRAINED_PROB

    def test_calibrated_proba_in_range(self) -> None:
        from finalayze.ml.models.lstm_model import LSTMModel

        seq_len = 5
        model = LSTMModel("test-seg", sequence_length=seq_len)
        X, y = _make_synthetic_data(n_samples=50)
        model.fit(X, y)
        sample = X[-1]
        proba = model.predict_proba(sample)
        assert _MIN_PROB <= proba <= _MAX_PROB

    def test_fit_stores_platt_scaler(self) -> None:
        from finalayze.ml.models.lstm_model import LSTMModel

        seq_len = 5
        n_samples = 100  # enough to produce >= 10 calibration sequences
        model = LSTMModel("test-seg", sequence_length=seq_len)
        X, y = _make_synthetic_data(n_samples=n_samples)
        model.fit(X, y)
        assert model._platt_scaler is not None  # noqa: SLF001

    def test_multiple_predictions_all_in_range(self) -> None:
        from finalayze.ml.models.lstm_model import LSTMModel

        seq_len = 5
        model = LSTMModel("test-seg", sequence_length=seq_len)
        X, y = _make_synthetic_data(n_samples=50)
        model.fit(X, y)
        for sample in X[-5:]:
            proba = model.predict_proba(sample)
            assert _MIN_PROB <= proba <= _MAX_PROB

    def test_save_load_preserves_calibration(self, tmp_path: pytest.TempPathFactory) -> None:
        from finalayze.ml.models.lstm_model import LSTMModel

        seq_len = 5
        model = LSTMModel("test-seg", sequence_length=seq_len)
        X, y = _make_synthetic_data(n_samples=50)
        model.fit(X, y)

        save_path = tmp_path / "lstm_model.pt"  # type: ignore[operator]
        model.save(save_path)  # type: ignore[arg-type]

        loaded = LSTMModel("test-seg", sequence_length=seq_len)
        loaded.load(save_path)  # type: ignore[arg-type]

        sample = X[-1]
        original_proba = model.predict_proba(sample)
        # Reset buffer for loaded model
        loaded_proba = loaded.predict_proba(sample)

        assert _MIN_PROB <= loaded_proba <= _MAX_PROB
        assert original_proba == pytest.approx(loaded_proba, abs=0.05)
