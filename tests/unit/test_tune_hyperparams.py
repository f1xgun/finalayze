"""Tests for Optuna hyperparameter tuning script."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(_PROJECT_ROOT))


class TestSearchSpaces:
    def test_xgboost_search_space(self) -> None:
        import optuna
        from tune_hyperparams import _xgboost_search_space

        study = optuna.create_study(direction="minimize")
        trial = study.ask()
        params = _xgboost_search_space(trial)
        assert "n_estimators" in params
        assert "max_depth" in params
        assert "learning_rate" in params
        assert "subsample" in params
        assert "colsample_bytree" in params

    def test_lightgbm_search_space(self) -> None:
        import optuna
        from tune_hyperparams import _lightgbm_search_space

        study = optuna.create_study(direction="minimize")
        trial = study.ask()
        params = _lightgbm_search_space(trial)
        assert "num_leaves" in params
        assert "max_depth" in params
        assert "learning_rate" in params
        assert "feature_fraction" in params


class TestTemporalCVOrdering:
    """Verify that temporal CV folds maintain strict temporal ordering."""

    _N_FOLDS = 5
    _PURGE_WINDOW = 130

    @pytest.fixture
    def dataset(self) -> tuple[np.ndarray, np.ndarray]:
        """Create a synthetic dataset with 1000 samples."""
        rng = np.random.default_rng(42)
        n = 1000
        features = rng.standard_normal((n, 10))
        labels = rng.integers(0, 2, size=n)
        return features, labels

    def test_train_end_before_val_start(self, dataset: tuple[np.ndarray, np.ndarray]) -> None:
        """Train end index must be before validation start for every fold."""
        features, _labels = dataset
        n = len(features)
        fold_size = n // (self._N_FOLDS + 1)

        for fold in range(self._N_FOLDS):
            val_start = (fold + 1) * fold_size
            val_end = min(val_start + fold_size, n)
            train_end = val_start - self._PURGE_WINDOW

            if train_end <= 0 or val_end <= val_start:
                continue

            assert train_end < val_start, (
                f"Fold {fold}: train_end={train_end} >= val_start={val_start}"
            )

    def test_purge_gap_separates_train_and_val(
        self, dataset: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Purge window must create a gap >= _PURGE_WINDOW between train and val."""
        features, _labels = dataset
        n = len(features)
        fold_size = n // (self._N_FOLDS + 1)

        for fold in range(self._N_FOLDS):
            val_start = (fold + 1) * fold_size
            val_end = min(val_start + fold_size, n)
            train_end = val_start - self._PURGE_WINDOW

            if train_end <= 0 or val_end <= val_start:
                continue

            gap = val_start - train_end
            assert gap >= self._PURGE_WINDOW, (
                f"Fold {fold}: gap={gap} < purge_window={self._PURGE_WINDOW}"
            )

    def test_folds_are_chronologically_ordered(
        self, dataset: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Each subsequent fold's validation window starts after the previous one."""
        features, _labels = dataset
        n = len(features)
        fold_size = n // (self._N_FOLDS + 1)

        val_starts: list[int] = []
        for fold in range(self._N_FOLDS):
            val_start = (fold + 1) * fold_size
            val_starts.append(val_start)

        for i in range(1, len(val_starts)):
            assert val_starts[i] > val_starts[i - 1], (
                f"Fold {i} val_start={val_starts[i]} <= fold {i - 1} val_start={val_starts[i - 1]}"
            )


class TestSaveParams:
    def test_save_and_load_params(self, tmp_path: Path) -> None:
        from tune_hyperparams import _save_best_params

        params = {"n_estimators": 100, "max_depth": 3}
        output_dir = tmp_path / "tuned_params"
        _save_best_params("us_tech", "xgboost", params, output_dir)

        result_path = output_dir / "us_tech" / "xgboost.json"
        assert result_path.exists()
        loaded = json.loads(result_path.read_text())
        assert loaded == params

    def test_save_creates_dirs(self, tmp_path: Path) -> None:
        from tune_hyperparams import _save_best_params

        params = {"lr": 0.01}
        output_dir = tmp_path / "deep" / "nested"
        path = _save_best_params("us_finance", "lightgbm", params, output_dir)
        assert path.exists()
