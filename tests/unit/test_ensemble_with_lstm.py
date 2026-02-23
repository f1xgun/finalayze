"""Unit tests for EnsembleModel with optional LSTMModel integration."""

from __future__ import annotations

import numpy as np
import pytest

# Constants
N_SAMPLES = 60
SEQUENCE_LENGTH = 20
HALF_PROB = 0.5
TOLERANCE = 1e-5


def _make_features(seed: int = 0) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    return {f"feat_{i:02d}": float(rng.standard_normal()) for i in range(6)}


def _make_dataset(n: int = N_SAMPLES) -> tuple[list[dict[str, float]], list[int]]:
    rng = np.random.default_rng(42)
    X = [_make_features(i) for i in range(n)]
    y = [int(rng.integers(0, 2)) for _ in range(n)]
    return X, y


@pytest.mark.unit
class TestEnsembleWithLSTM:
    def test_ensemble_no_lstm_behaves_as_before(self) -> None:
        """Passing lstm_model=None keeps existing XGB+LGBM averaging behaviour."""
        from finalayze.ml.models.ensemble import EnsembleModel
        from finalayze.ml.models.lightgbm_model import LightGBMModel
        from finalayze.ml.models.xgboost_model import XGBoostModel

        xgb = XGBoostModel(segment_id="us_tech")
        lgbm = LightGBMModel(segment_id="us_tech")
        ensemble = EnsembleModel(models=[xgb, lgbm], lstm_model=None)
        result = ensemble.predict_proba(_make_features())
        # Both untrained → average of 0.5 and 0.5 = 0.5
        assert result == pytest.approx(HALF_PROB)

    def test_ensemble_all_three_trained_averages_correctly(self) -> None:
        from finalayze.ml.models.ensemble import EnsembleModel
        from finalayze.ml.models.lightgbm_model import LightGBMModel
        from finalayze.ml.models.lstm_model import LSTMModel
        from finalayze.ml.models.xgboost_model import XGBoostModel

        X, y = _make_dataset()
        xgb = XGBoostModel(segment_id="us_tech")
        lgbm = LightGBMModel(segment_id="us_tech")
        lstm = LSTMModel(segment_id="us_tech", sequence_length=SEQUENCE_LENGTH)

        xgb.fit(X, y)
        lgbm.fit(X, y)
        lstm.fit(X, y)

        ensemble = EnsembleModel(models=[xgb, lgbm], lstm_model=lstm)
        features = _make_features()
        result = ensemble.predict_proba(features)

        # Verify it is a proper average of the three
        p_xgb = xgb.predict_proba(features)
        p_lgbm = lgbm.predict_proba(features)
        lstm._feature_buffer.clear()  # noqa: SLF001
        p_lstm = lstm.predict_proba(features)
        expected = (p_xgb + p_lgbm + p_lstm) / 3.0
        assert result == pytest.approx(expected, abs=TOLERANCE)

    def test_ensemble_only_two_trained_uses_two(self) -> None:
        """When lstm is untrained, only XGB+LGBM contribute to average."""
        from finalayze.ml.models.ensemble import EnsembleModel
        from finalayze.ml.models.lightgbm_model import LightGBMModel
        from finalayze.ml.models.lstm_model import LSTMModel
        from finalayze.ml.models.xgboost_model import XGBoostModel

        X, y = _make_dataset()
        xgb = XGBoostModel(segment_id="us_tech")
        lgbm = LightGBMModel(segment_id="us_tech")
        lstm = LSTMModel(segment_id="us_tech", sequence_length=SEQUENCE_LENGTH)  # NOT trained

        xgb.fit(X, y)
        lgbm.fit(X, y)

        ensemble = EnsembleModel(models=[xgb, lgbm], lstm_model=lstm)
        features = _make_features()
        result = ensemble.predict_proba(features)

        # Should be average of XGB and LGBM only
        p_xgb = xgb.predict_proba(features)
        p_lgbm = lgbm.predict_proba(features)
        expected = (p_xgb + p_lgbm) / 2.0
        assert result == pytest.approx(expected, abs=TOLERANCE)

    def test_ensemble_none_trained_returns_half(self) -> None:
        from finalayze.ml.models.ensemble import EnsembleModel

        ensemble = EnsembleModel(models=[], lstm_model=None)
        assert ensemble.predict_proba(_make_features()) == pytest.approx(HALF_PROB)


@pytest.mark.unit
class TestMLModelRegistryFactory:
    def test_create_ensemble_returns_ensemble_with_three_models(self) -> None:
        from finalayze.ml.models.ensemble import EnsembleModel
        from finalayze.ml.registry import MLModelRegistry

        registry = MLModelRegistry()
        ensemble = registry.create_ensemble("us_tech")
        assert isinstance(ensemble, EnsembleModel)
        # The ensemble should have exactly 2 base models + 1 lstm
        assert len(ensemble._models) == 2  # noqa: SLF001
        assert ensemble._lstm_model is not None  # noqa: SLF001

    def test_create_ensemble_registers_under_segment(self) -> None:
        from finalayze.ml.registry import MLModelRegistry

        registry = MLModelRegistry()
        ensemble = registry.create_ensemble("us_tech")
        registry.register("us_tech", ensemble)
        assert registry.get("us_tech") is ensemble
