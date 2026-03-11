"""Tests for ML model loader (Layer 3)."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from unittest.mock import MagicMock, patch

import joblib
import numpy as np
import pytest

from finalayze.ml.calibration import EnsembleCalibrator
from finalayze.ml.loader import load_registry, save_ensemble
from finalayze.ml.models.ensemble import EnsembleModel
from finalayze.ml.models.xgboost_model import XGBoostModel


class TestLoadRegistry:
    def test_load_registry_missing_dir_returns_empty(self, tmp_path: Path) -> None:
        """Loading from a non-existent directory returns an empty registry."""
        registry = load_registry(tmp_path / "nonexistent", ["us_tech"])
        assert registry.get("us_tech") is None

    def test_load_registry_missing_segment_dir(self, tmp_path: Path) -> None:
        """When the segment subdirectory doesn't exist, skip it."""
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        registry = load_registry(model_dir, ["us_tech"])
        assert registry.get("us_tech") is None

    def test_load_registry_with_existing_models(self, tmp_path: Path) -> None:
        """Verify that models are loaded and registered when files exist."""
        import joblib

        from finalayze.ml.models.xgboost_model import XGBoostModel

        segment_dir = tmp_path / "us_tech"
        segment_dir.mkdir(parents=True)

        # Create a minimal XGBoost model and save it
        xgb = XGBoostModel(segment_id="us_tech")
        xgb.fit(
            [{"a": 1.0, "b": 2.0}] * 20,
            [1, 0] * 10,
        )
        joblib.dump(xgb, segment_dir / "xgb.pkl")

        registry = load_registry(tmp_path, ["us_tech"])
        ensemble = registry.get("us_tech")
        assert ensemble is not None

    def test_load_registry_corrupt_file_skips(self, tmp_path: Path) -> None:
        """Corrupt model file should be logged and skipped."""
        segment_dir = tmp_path / "us_tech"
        segment_dir.mkdir(parents=True)
        (segment_dir / "xgb.pkl").write_text("not a pickle")

        registry = load_registry(tmp_path, ["us_tech"])
        assert registry.get("us_tech") is None


class TestSaveEnsemble:
    def test_save_ensemble_creates_files(self, tmp_path: Path) -> None:
        """save_ensemble should create model files in segment directory."""
        ensemble = MagicMock()
        # Simulate XGBoost model
        xgb_model = MagicMock()
        type(xgb_model).__name__ = "XGBoostModel"
        xgb_model.save = MagicMock()
        ensemble._models = [xgb_model]
        ensemble._lstm_model = None
        ensemble.selected_features = None
        ensemble._calibrator = None
        ensemble._model_weights = None
        ensemble.base_rate = None

        with patch("finalayze.ml.loader._atomic_save") as mock_save:
            save_ensemble(tmp_path, "us_tech", ensemble)
            mock_save.assert_called_once()

        # Directory should be created
        assert (tmp_path / "us_tech").is_dir()

    def test_save_ensemble_atomic_write(self, tmp_path: Path) -> None:
        """Atomic save should use temp file + rename pattern."""
        import joblib

        from finalayze.ml.models.ensemble import EnsembleModel
        from finalayze.ml.models.xgboost_model import XGBoostModel

        xgb = XGBoostModel(segment_id="us_tech")
        xgb.fit([{"a": 1.0, "b": 2.0}] * 20, [1, 0] * 10)
        ensemble = EnsembleModel(models=[xgb], lstm_model=None)

        save_ensemble(tmp_path, "us_tech", ensemble)

        # File should exist after save
        assert (tmp_path / "us_tech" / "xgb.pkl").exists()


class TestLSTMAtomicSave:
    """6C.9: LSTM atomic save tests."""

    def test_lstm_save_creates_weight_and_scaler_files(self, tmp_path: Path) -> None:
        """After save, weights and scaler files exist (platt removed in B.5)."""
        from finalayze.ml.models.lstm_model import LSTMModel

        model = LSTMModel(segment_id="test", sequence_length=5)
        X = [{"a": float(i), "b": float(i * 2)} for i in range(30)]
        y = [i % 2 for i in range(30)]
        model.fit(X, y)

        save_path = tmp_path / "lstm.pkl"
        model.save(save_path)

        assert save_path.exists()
        assert (tmp_path / "lstm.pkl.scaler.pkl").exists()

    def test_lstm_save_atomic_no_corrupt_on_interrupt(self, tmp_path: Path) -> None:
        """If torch.save raises, no partial file at the target path."""
        from finalayze.ml.models.lstm_model import LSTMModel

        model = LSTMModel(segment_id="test", sequence_length=5)
        X = [{"a": float(i), "b": float(i * 2)} for i in range(30)]
        y = [i % 2 for i in range(30)]
        model.fit(X, y)

        save_path = tmp_path / "lstm.pkl"
        with (
            patch(
                "finalayze.ml.models.lstm_model.torch.save",
                side_effect=OSError("disk full"),
            ),
            pytest.raises(OSError, match="disk full"),
        ):
            model.save(save_path)

        assert not save_path.exists()

    def test_lstm_save_scaler_atomic(self, tmp_path: Path) -> None:
        """If pickle.dump raises for scaler, no partial scaler file."""
        from finalayze.ml.models.lstm_model import LSTMModel

        model = LSTMModel(segment_id="test", sequence_length=5)
        X = [{"a": float(i), "b": float(i * 2)} for i in range(30)]
        y = [i % 2 for i in range(30)]
        model.fit(X, y)

        save_path = tmp_path / "lstm.pkl"

        original_dump = pickle.dump
        call_count = 0

        def failing_dump(*args: object, **kwargs: object) -> None:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise OSError("disk full")
            return original_dump(*args, **kwargs)  # type: ignore[arg-type]

        with (
            patch(
                "finalayze.ml.models.lstm_model.pickle.dump",
                side_effect=failing_dump,
            ),
            pytest.raises(OSError, match="disk full"),
        ):
            model.save(save_path)

        # Weights file should have been written atomically before scaler failed
        assert save_path.exists()
        # Scaler file should NOT exist (atomic save cleaned up)
        scaler_path = tmp_path / "lstm.pkl.scaler.pkl"
        assert not scaler_path.exists()


class TestSelectedFeaturesLoading:
    """Feature mismatch fix: selected_features.json persistence in loader."""

    def test_load_selected_features_when_file_exists(self, tmp_path: Path) -> None:
        """Loader reads selected_features.json and sets it on EnsembleModel."""
        segment_dir = tmp_path / "us_tech"
        segment_dir.mkdir(parents=True)

        # Create a minimal XGBoost model
        xgb = XGBoostModel(segment_id="us_tech")
        xgb.fit([{"a": 1.0, "b": 2.0}] * 20, [1, 0] * 10)
        joblib.dump(xgb, segment_dir / "xgb.pkl")

        # Write selected_features.json
        selected = ["a", "b"]
        (segment_dir / "selected_features.json").write_text(json.dumps(selected))

        registry = load_registry(tmp_path, ["us_tech"])
        ensemble = registry.get("us_tech")
        assert ensemble is not None
        assert ensemble.selected_features == ["a", "b"]

    def test_load_without_selected_features_file(self, tmp_path: Path) -> None:
        """Legacy models without selected_features.json get None (graceful degradation)."""
        segment_dir = tmp_path / "us_tech"
        segment_dir.mkdir(parents=True)

        xgb = XGBoostModel(segment_id="us_tech")
        xgb.fit([{"a": 1.0, "b": 2.0}] * 20, [1, 0] * 10)
        joblib.dump(xgb, segment_dir / "xgb.pkl")

        registry = load_registry(tmp_path, ["us_tech"])
        ensemble = registry.get("us_tech")
        assert ensemble is not None
        assert ensemble.selected_features is None

    def test_save_ensemble_writes_selected_features(self, tmp_path: Path) -> None:
        """save_ensemble persists selected_features.json when set on ensemble."""
        xgb = XGBoostModel(segment_id="us_tech")
        xgb.fit([{"a": 1.0, "b": 2.0}] * 20, [1, 0] * 10)
        ensemble = EnsembleModel(models=[xgb], lstm_model=None, selected_features=["a", "b"])

        save_ensemble(tmp_path, "us_tech", ensemble)

        features_path = tmp_path / "us_tech" / "selected_features.json"
        assert features_path.exists()
        loaded = json.loads(features_path.read_text())
        assert loaded == ["a", "b"]

    def test_save_ensemble_no_selected_features_no_file(self, tmp_path: Path) -> None:
        """save_ensemble does not write selected_features.json when None."""
        xgb = XGBoostModel(segment_id="us_tech")
        xgb.fit([{"a": 1.0, "b": 2.0}] * 20, [1, 0] * 10)
        ensemble = EnsembleModel(models=[xgb], lstm_model=None)

        save_ensemble(tmp_path, "us_tech", ensemble)

        features_path = tmp_path / "us_tech" / "selected_features.json"
        assert not features_path.exists()


# ---------------------------------------------------------------------------
# Constants for calibrator round-trip tests
# ---------------------------------------------------------------------------
_N_TRAIN = 20
_N_CAL_SAMPLES = 200


def _make_fitted_calibrator() -> EnsembleCalibrator:
    """Create a fitted EnsembleCalibrator with well-separated data."""
    rng = np.random.default_rng(42)
    raw_probas = np.concatenate(
        [
            rng.uniform(0.1, 0.4, _N_CAL_SAMPLES // 2),
            rng.uniform(0.6, 0.9, _N_CAL_SAMPLES // 2),
        ]
    )
    labels = np.array([0] * (_N_CAL_SAMPLES // 2) + [1] * (_N_CAL_SAMPLES // 2))
    cal = EnsembleCalibrator()
    cal.fit(raw_probas, labels)
    assert cal.is_fitted
    return cal


class TestCalibratorRoundTrip:
    """Save ensemble with calibrator, load it, verify calibrator survives."""

    def test_save_load_preserves_calibrator(self, tmp_path: Path) -> None:
        """Round-trip: save ensemble with fitted calibrator, load, calibrator is present."""
        xgb = XGBoostModel(segment_id="us_tech")
        xgb.fit([{"a": 1.0, "b": 2.0}] * _N_TRAIN, [1, 0] * (_N_TRAIN // 2))

        calibrator = _make_fitted_calibrator()
        ensemble = EnsembleModel(models=[xgb], lstm_model=None, calibrator=calibrator)

        save_ensemble(tmp_path, "us_tech", ensemble)

        # calibrator.pkl must exist
        assert (tmp_path / "us_tech" / "calibrator.pkl").exists()

        # Load back via registry
        registry = load_registry(tmp_path, ["us_tech"])
        loaded = registry.get("us_tech")
        assert loaded is not None
        assert loaded._calibrator is not None
        assert loaded._calibrator.is_fitted

    def test_loaded_calibrator_produces_same_output(self, tmp_path: Path) -> None:
        """Loaded calibrator produces the same calibrated value as the original."""
        xgb = XGBoostModel(segment_id="us_tech")
        xgb.fit([{"a": 1.0, "b": 2.0}] * _N_TRAIN, [1, 0] * (_N_TRAIN // 2))

        calibrator = _make_fitted_calibrator()
        original_cal_low = calibrator.calibrate(0.2)
        original_cal_high = calibrator.calibrate(0.8)

        ensemble = EnsembleModel(models=[xgb], lstm_model=None, calibrator=calibrator)
        save_ensemble(tmp_path, "us_tech", ensemble)

        registry = load_registry(tmp_path, ["us_tech"])
        loaded = registry.get("us_tech")
        assert loaded is not None
        assert loaded._calibrator is not None

        loaded_cal_low = loaded._calibrator.calibrate(0.2)
        loaded_cal_high = loaded._calibrator.calibrate(0.8)

        assert loaded_cal_low == pytest.approx(original_cal_low, abs=1e-6)
        assert loaded_cal_high == pytest.approx(original_cal_high, abs=1e-6)

    def test_load_without_calibrator_still_works(self, tmp_path: Path) -> None:
        """Graceful degradation: if calibrator.pkl is missing, ensemble loads fine."""
        segment_dir = tmp_path / "us_tech"
        segment_dir.mkdir(parents=True)

        xgb = XGBoostModel(segment_id="us_tech")
        xgb.fit([{"a": 1.0, "b": 2.0}] * _N_TRAIN, [1, 0] * (_N_TRAIN // 2))
        joblib.dump(xgb, segment_dir / "xgb.pkl")

        # No calibrator.pkl file
        assert not (segment_dir / "calibrator.pkl").exists()

        registry = load_registry(tmp_path, ["us_tech"])
        loaded = registry.get("us_tech")
        assert loaded is not None
        # Should work without calibrator (returns raw proba)
        result = loaded.predict_proba({"a": 1.0, "b": 2.0})
        assert 0.0 <= result <= 1.0

    def test_save_ensemble_without_calibrator_no_file(self, tmp_path: Path) -> None:
        """If ensemble has no calibrator, no calibrator.pkl is written."""
        xgb = XGBoostModel(segment_id="us_tech")
        xgb.fit([{"a": 1.0, "b": 2.0}] * _N_TRAIN, [1, 0] * (_N_TRAIN // 2))
        ensemble = EnsembleModel(models=[xgb], lstm_model=None)

        save_ensemble(tmp_path, "us_tech", ensemble)

        assert not (tmp_path / "us_tech" / "calibrator.pkl").exists()

    def test_calibrated_probas_pulled_toward_center(self) -> None:
        """For a ~50% accuracy model, calibration should pull extreme probas toward 0.5."""
        rng = np.random.default_rng(42)
        # Simulate a poorly calibrated model: outputs extreme probas
        # but actual labels are close to 50/50
        n = 200
        raw_probas = np.concatenate(
            [
                rng.uniform(0.0, 0.2, n // 2),
                rng.uniform(0.8, 1.0, n // 2),
            ]
        )
        # Labels are only slightly correlated (model is not that good)
        labels = rng.integers(0, 2, n)

        cal = EnsembleCalibrator()
        cal.fit(raw_probas, labels)
        assert cal.is_fitted

        # For a model with ~50% accuracy, extreme raw probas should be
        # pulled toward 0.5 after calibration
        cal_low = cal.calibrate(0.1)
        cal_high = cal.calibrate(0.9)
        # Calibrated should be less extreme than raw
        assert cal_low > 0.1  # pulled up from 0.1
        assert cal_high < 0.9  # pulled down from 0.9


# ---------------------------------------------------------------------------
# Task 9: model_weights.json loading round-trip
# ---------------------------------------------------------------------------


class TestModelWeightsRoundTrip:
    """model_weights.json persistence in loader (Task 9)."""

    def test_load_model_weights_when_file_exists(self, tmp_path: Path) -> None:
        """Loader reads model_weights.json and passes it to EnsembleModel."""
        segment_dir = tmp_path / "us_tech"
        segment_dir.mkdir(parents=True)

        xgb = XGBoostModel(segment_id="us_tech")
        xgb.fit([{"a": 1.0, "b": 2.0}] * 20, [1, 0] * 10)
        joblib.dump(xgb, segment_dir / "xgb.pkl")

        weights = {"xgboostmodel": 0.04, "lightgbmmodel": 0.01}
        (segment_dir / "model_weights.json").write_text(json.dumps(weights))

        registry = load_registry(tmp_path, ["us_tech"])
        ensemble = registry.get("us_tech")
        assert ensemble is not None
        assert ensemble._model_weights == weights

    def test_load_without_model_weights_file(self, tmp_path: Path) -> None:
        """Legacy models without model_weights.json get None (equal averaging)."""
        segment_dir = tmp_path / "us_tech"
        segment_dir.mkdir(parents=True)

        xgb = XGBoostModel(segment_id="us_tech")
        xgb.fit([{"a": 1.0, "b": 2.0}] * 20, [1, 0] * 10)
        joblib.dump(xgb, segment_dir / "xgb.pkl")

        registry = load_registry(tmp_path, ["us_tech"])
        ensemble = registry.get("us_tech")
        assert ensemble is not None
        assert ensemble._model_weights is None

    def test_save_ensemble_writes_model_weights(self, tmp_path: Path) -> None:
        """save_ensemble persists model_weights.json when set on ensemble."""
        xgb = XGBoostModel(segment_id="us_tech")
        xgb.fit([{"a": 1.0, "b": 2.0}] * 20, [1, 0] * 10)
        weights = {"xgboostmodel": 0.04}
        ensemble = EnsembleModel(models=[xgb], lstm_model=None, model_weights=weights)

        save_ensemble(tmp_path, "us_tech", ensemble)

        weights_path = tmp_path / "us_tech" / "model_weights.json"
        assert weights_path.exists()
        loaded = json.loads(weights_path.read_text())
        assert loaded == weights

    def test_save_ensemble_no_model_weights_no_file(self, tmp_path: Path) -> None:
        """save_ensemble does not write model_weights.json when None."""
        xgb = XGBoostModel(segment_id="us_tech")
        xgb.fit([{"a": 1.0, "b": 2.0}] * 20, [1, 0] * 10)
        ensemble = EnsembleModel(models=[xgb], lstm_model=None)

        save_ensemble(tmp_path, "us_tech", ensemble)

        weights_path = tmp_path / "us_tech" / "model_weights.json"
        assert not weights_path.exists()


# ---------------------------------------------------------------------------
# Task 10: segment_meta.json loading (base_rate)
# ---------------------------------------------------------------------------


class TestSegmentMetaRoundTrip:
    """segment_meta.json persistence for base_rate (Task 10)."""

    def test_load_base_rate_when_meta_exists(self, tmp_path: Path) -> None:
        """Loader reads segment_meta.json and sets base_rate on EnsembleModel."""
        from finalayze.ml.loader import FEATURE_SCHEMA_VERSION

        segment_dir = tmp_path / "us_tech"
        segment_dir.mkdir(parents=True)

        xgb = XGBoostModel(segment_id="us_tech")
        xgb.fit([{"a": 1.0, "b": 2.0}] * 20, [1, 0] * 10)
        joblib.dump(xgb, segment_dir / "xgb.pkl")

        meta = {"base_rate": 0.5312, "feature_schema_version": FEATURE_SCHEMA_VERSION}
        (segment_dir / "segment_meta.json").write_text(json.dumps(meta))

        registry = load_registry(tmp_path, ["us_tech"])
        ensemble = registry.get("us_tech")
        assert ensemble is not None
        assert ensemble.base_rate == pytest.approx(0.5312)

    def test_load_without_meta_file_gives_none(self, tmp_path: Path) -> None:
        """Legacy models without segment_meta.json get base_rate=None."""
        segment_dir = tmp_path / "us_tech"
        segment_dir.mkdir(parents=True)

        xgb = XGBoostModel(segment_id="us_tech")
        xgb.fit([{"a": 1.0, "b": 2.0}] * 20, [1, 0] * 10)
        joblib.dump(xgb, segment_dir / "xgb.pkl")

        registry = load_registry(tmp_path, ["us_tech"])
        ensemble = registry.get("us_tech")
        assert ensemble is not None
        assert ensemble.base_rate is None

    def test_save_ensemble_writes_meta(self, tmp_path: Path) -> None:
        """save_ensemble persists segment_meta.json when base_rate is set."""
        xgb = XGBoostModel(segment_id="us_tech")
        xgb.fit([{"a": 1.0, "b": 2.0}] * 20, [1, 0] * 10)
        ensemble = EnsembleModel(models=[xgb], lstm_model=None)
        ensemble.base_rate = 0.5312

        save_ensemble(tmp_path, "us_tech", ensemble)

        meta_path = tmp_path / "us_tech" / "segment_meta.json"
        assert meta_path.exists()
        loaded = json.loads(meta_path.read_text())
        assert loaded["base_rate"] == pytest.approx(0.5312)

    def test_save_ensemble_always_writes_meta(self, tmp_path: Path) -> None:
        """save_ensemble always writes segment_meta.json (even when base_rate is None) for feature_schema_version."""
        xgb = XGBoostModel(segment_id="us_tech")
        xgb.fit([{"a": 1.0, "b": 2.0}] * 20, [1, 0] * 10)
        ensemble = EnsembleModel(models=[xgb], lstm_model=None)
        # base_rate is None (default)

        save_ensemble(tmp_path, "us_tech", ensemble)

        meta_path = tmp_path / "us_tech" / "segment_meta.json"
        assert meta_path.exists()


# ---------------------------------------------------------------------------
# Task 13: feature_schema_version field in segment_meta.json
# ---------------------------------------------------------------------------


class TestFeatureSchemaVersion:
    """Task 13: feature_schema_version field in segment_meta.json."""

    def test_schema_version_mismatch_returns_none(self, tmp_path: Path) -> None:
        """If saved feature_schema_version differs from current, load_registry skips the segment."""
        from finalayze.ml.loader import FEATURE_SCHEMA_VERSION

        segment_dir = tmp_path / "us_tech"
        segment_dir.mkdir(parents=True)

        xgb = XGBoostModel(segment_id="us_tech")
        xgb.fit([{"a": 1.0, "b": 2.0}] * 20, [1, 0] * 10)
        joblib.dump(xgb, segment_dir / "xgb.pkl")

        # Write segment_meta with an outdated schema version
        outdated_version = FEATURE_SCHEMA_VERSION - 1
        meta = {"feature_schema_version": outdated_version, "base_rate": 0.5}
        (segment_dir / "segment_meta.json").write_text(json.dumps(meta))

        registry = load_registry(tmp_path, ["us_tech"])
        assert registry.get("us_tech") is None

    def test_schema_version_match_loads_successfully(self, tmp_path: Path) -> None:
        """If feature_schema_version matches current, loading proceeds normally."""
        from finalayze.ml.loader import FEATURE_SCHEMA_VERSION

        segment_dir = tmp_path / "us_tech"
        segment_dir.mkdir(parents=True)

        xgb = XGBoostModel(segment_id="us_tech")
        xgb.fit([{"a": 1.0, "b": 2.0}] * 20, [1, 0] * 10)
        joblib.dump(xgb, segment_dir / "xgb.pkl")

        meta = {"feature_schema_version": FEATURE_SCHEMA_VERSION, "base_rate": 0.5}
        (segment_dir / "segment_meta.json").write_text(json.dumps(meta))

        registry = load_registry(tmp_path, ["us_tech"])
        ensemble = registry.get("us_tech")
        assert ensemble is not None

    def test_no_segment_meta_loads_without_version_check(self, tmp_path: Path) -> None:
        """Legacy models with no segment_meta.json at all still load (no version field to mismatch)."""
        segment_dir = tmp_path / "us_tech"
        segment_dir.mkdir(parents=True)

        xgb = XGBoostModel(segment_id="us_tech")
        xgb.fit([{"a": 1.0, "b": 2.0}] * 20, [1, 0] * 10)
        joblib.dump(xgb, segment_dir / "xgb.pkl")

        # No segment_meta.json at all
        assert not (segment_dir / "segment_meta.json").exists()

        registry = load_registry(tmp_path, ["us_tech"])
        ensemble = registry.get("us_tech")
        assert ensemble is not None

    def test_save_ensemble_writes_feature_schema_version(self, tmp_path: Path) -> None:
        """save_ensemble writes feature_schema_version into segment_meta.json."""
        from finalayze.ml.loader import FEATURE_SCHEMA_VERSION

        xgb = XGBoostModel(segment_id="us_tech")
        xgb.fit([{"a": 1.0, "b": 2.0}] * 20, [1, 0] * 10)
        ensemble = EnsembleModel(models=[xgb], lstm_model=None)
        ensemble.base_rate = 0.5  # ensure segment_meta.json is written

        save_ensemble(tmp_path, "us_tech", ensemble)

        meta_path = tmp_path / "us_tech" / "segment_meta.json"
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert meta["feature_schema_version"] == FEATURE_SCHEMA_VERSION

    def test_save_ensemble_writes_version_even_without_base_rate(self, tmp_path: Path) -> None:
        """save_ensemble always writes segment_meta.json with version, even when base_rate is None."""
        from finalayze.ml.loader import FEATURE_SCHEMA_VERSION

        xgb = XGBoostModel(segment_id="us_tech")
        xgb.fit([{"a": 1.0, "b": 2.0}] * 20, [1, 0] * 10)
        ensemble = EnsembleModel(models=[xgb], lstm_model=None)
        # base_rate is None (default)

        save_ensemble(tmp_path, "us_tech", ensemble)

        meta_path = tmp_path / "us_tech" / "segment_meta.json"
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert meta["feature_schema_version"] == FEATURE_SCHEMA_VERSION
