"""Tests for tree model regularization params (6C.3)."""

from __future__ import annotations

from finalayze.ml.models.lightgbm_model import LightGBMModel
from finalayze.ml.models.xgboost_model import XGBoostModel

_TRAIN_X = [{"a": float(i), "b": float(i * 2)} for i in range(50)]
_TRAIN_Y = [i % 2 for i in range(50)]


class TestXGBoostRegularization:
    def test_xgboost_regularization_params(self) -> None:
        model = XGBoostModel(segment_id="test")
        model.fit(_TRAIN_X, _TRAIN_Y)
        assert model._model is not None
        params = model._model.get_params()
        assert params["reg_alpha"] == 0.1
        assert params["reg_lambda"] == 1.0
        assert params["subsample"] == 0.8
        assert params["colsample_bytree"] == 0.8


class TestLightGBMRegularization:
    def test_lightgbm_regularization_params(self) -> None:
        model = LightGBMModel(segment_id="test")
        model.fit(_TRAIN_X, _TRAIN_Y)
        assert model._model is not None
        params = model._model.get_params()
        assert params["reg_alpha"] == 0.1
        assert params["reg_lambda"] == 1.0
        assert params["subsample"] == 0.8
        assert params["colsample_bytree"] == 0.8
