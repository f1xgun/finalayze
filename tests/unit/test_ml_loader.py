"""Tests for ML model loader (Layer 3)."""

from __future__ import annotations

import pickle
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from finalayze.ml.loader import load_registry, save_ensemble


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

    def test_lstm_save_creates_all_three_files(self, tmp_path: Path) -> None:
        """After save, all 3 files (weights, scaler, platt) exist."""
        from finalayze.ml.models.lstm_model import LSTMModel

        model = LSTMModel(segment_id="test", sequence_length=5)
        X = [{"a": float(i), "b": float(i * 2)} for i in range(30)]
        y = [i % 2 for i in range(30)]
        model.fit(X, y)

        save_path = tmp_path / "lstm.pkl"
        model.save(save_path)

        assert save_path.exists()
        assert (tmp_path / "lstm.pkl.scaler.pkl").exists()
        assert (tmp_path / "lstm.pkl.platt.pkl").exists()

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
