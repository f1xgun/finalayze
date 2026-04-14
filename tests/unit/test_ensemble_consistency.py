"""Tests for ensemble model consistency in class rebalancing strategy.

All three models (XGBoost, LightGBM, CatBoost) must use the same pattern:
- When sample_weight is provided → disable internal pos_weight multipliers
- When sample_weight is NOT provided → use class-ratio weighting

This prevents double-rebalancing (counting class imbalance twice).
"""

from __future__ import annotations

import numpy as np
import pytest

from finalayze.ml.models.xgboost_model import XGBoostModel
from finalayze.ml.models.lightgbm_model import LightGBMModel
from finalayze.ml.models.catboost_model import CatBoostModel

# Minimal synthetic dataset: 10 samples, 3 features, imbalanced classes (7 neg, 3 pos)
_N_SAMPLES = 20
_N_FEATURES = 3
_RNG = np.random.default_rng(42)

_X_RAW: list[dict[str, float]] = [
    {"f0": float(_RNG.random()), "f1": float(_RNG.random()), "f2": float(_RNG.random())}
    for _ in range(_N_SAMPLES)
]
# Imbalanced labels: 14 zeros, 6 ones
_Y: list[int] = [0] * 14 + [1] * 6
_SW: np.ndarray = np.ones(_N_SAMPLES, dtype=float)


class TestXGBoostScalePosWeight:
    """XGBoost scale_pos_weight must be 1.0 when sample_weight provided."""

    def test_xgb_spw_with_sample_weight(self) -> None:
        """XGBoostModel.fit() with sample_weight → scale_pos_weight == 1.0."""
        model = XGBoostModel(segment_id="test")
        model.fit(_X_RAW, _Y, sample_weight=_SW)
        assert model._model is not None
        params = model._model.get_params()
        assert params["scale_pos_weight"] == 1.0, (
            f"Expected scale_pos_weight=1.0 when sample_weight provided, "
            f"got {params['scale_pos_weight']}"
        )

    def test_xgb_spw_without_sample_weight(self) -> None:
        """XGBoostModel.fit() without sample_weight → scale_pos_weight == n_neg/n_pos."""
        model = XGBoostModel(segment_id="test")
        model.fit(_X_RAW, _Y, sample_weight=None)
        assert model._model is not None
        params = model._model.get_params()
        y_arr = np.array(_Y)
        n_pos = int(np.sum(y_arr == 1))
        n_neg = int(np.sum(y_arr == 0))
        expected_spw = n_neg / n_pos
        assert params["scale_pos_weight"] == pytest.approx(expected_spw), (
            f"Expected scale_pos_weight={expected_spw} when no sample_weight, "
            f"got {params['scale_pos_weight']}"
        )


class TestCatBoostAutoClassWeights:
    """CatBoost auto_class_weights must be None when sample_weight provided."""

    def test_catboost_acw_with_sample_weight(self) -> None:
        """CatBoostModel.fit() with sample_weight → auto_class_weights is None."""
        model = CatBoostModel(segment_id="test")
        model.fit(_X_RAW, _Y, sample_weight=_SW)
        assert model._model is not None
        params = model._model.get_params()
        assert params.get("auto_class_weights") is None, (
            f"Expected auto_class_weights=None when sample_weight provided, "
            f"got {params.get('auto_class_weights')!r}"
        )

    def test_catboost_acw_without_sample_weight(self) -> None:
        """CatBoostModel.fit() without sample_weight → auto_class_weights == 'Balanced'."""
        model = CatBoostModel(segment_id="test")
        model.fit(_X_RAW, _Y, sample_weight=None)
        assert model._model is not None
        params = model._model.get_params()
        assert params.get("auto_class_weights") == "Balanced", (
            f"Expected auto_class_weights='Balanced' when no sample_weight, "
            f"got {params.get('auto_class_weights')!r}"
        )


class TestLightGBMScalePosWeight:
    """LightGBM scale_pos_weight must be 1.0 when sample_weight provided (already correct)."""

    def test_lgbm_spw_with_sample_weight(self) -> None:
        """LightGBMModel.fit() with sample_weight → scale_pos_weight == 1.0."""
        model = LightGBMModel(segment_id="test")
        model.fit(_X_RAW, _Y, sample_weight=_SW)
        assert model._model is not None
        params = model._model.get_params()
        assert params["scale_pos_weight"] == 1.0, (
            f"Expected scale_pos_weight=1.0 when sample_weight provided, "
            f"got {params['scale_pos_weight']}"
        )

    def test_lgbm_spw_without_sample_weight(self) -> None:
        """LightGBMModel.fit() without sample_weight → scale_pos_weight == n_neg/n_pos."""
        model = LightGBMModel(segment_id="test")
        model.fit(_X_RAW, _Y, sample_weight=None)
        assert model._model is not None
        params = model._model.get_params()
        y_arr = np.array(_Y)
        n_pos = int(np.sum(y_arr == 1))
        n_neg = int(np.sum(y_arr == 0))
        expected_spw = n_neg / n_pos
        assert params["scale_pos_weight"] == pytest.approx(expected_spw), (
            f"Expected scale_pos_weight={expected_spw} when no sample_weight, "
            f"got {params['scale_pos_weight']}"
        )
