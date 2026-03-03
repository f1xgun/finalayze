"""Tests for Optuna hyperparameter tuning script."""

from __future__ import annotations

import json
import sys
from pathlib import Path

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
