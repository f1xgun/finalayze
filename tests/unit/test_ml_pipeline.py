"""Unit tests for ML pipeline scaffold."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import numpy as np
import pytest

from finalayze.core.exceptions import InsufficientDataError
from finalayze.core.schemas import Candle
from finalayze.ml.features.technical import compute_features
from finalayze.ml.models.ensemble import EnsembleModel
from finalayze.ml.models.lightgbm_model import LightGBMModel
from finalayze.ml.models.xgboost_model import XGBoostModel
from finalayze.ml.registry import MLModelRegistry

# ── Feature computation ──────────────────────────────────────────────────────
_FEATURE_NAMES = {"rsi_14", "macd_hist", "bb_pct_b", "volume_ratio_20d", "atr_14", "sentiment"}
_N_FEATURES = len(_FEATURE_NAMES)


class TestComputeFeatures:
    def test_returns_correct_keys(self) -> None:
        # 40 synthetic candles needed for all indicators
        features = _make_features()
        assert set(features.keys()) == _FEATURE_NAMES

    def test_all_values_are_floats(self) -> None:
        features = _make_features()
        assert all(isinstance(v, float) for v in features.values())

    def test_sentiment_passed_through(self) -> None:
        features = _make_features(sentiment=0.75)
        assert features["sentiment"] == pytest.approx(0.75)

    def test_insufficient_candles_raises(self) -> None:
        base_date = datetime(2024, 1, 1, tzinfo=UTC)
        candles = [
            Candle(
                symbol="AAPL",
                market_id="us",
                timeframe="1d",
                timestamp=base_date + timedelta(days=i),
                open=Decimal(100),
                high=Decimal(105),
                low=Decimal(95),
                close=Decimal(102),
                volume=1000,
            )
            for i in range(5)  # only 5, need at least 30
        ]
        with pytest.raises(InsufficientDataError):
            compute_features(candles)


class TestXGBoostModel:
    def test_predict_proba_before_fit_returns_half(self) -> None:
        model = XGBoostModel(segment_id="us_tech")
        features = _make_features()
        result = model.predict_proba(features)
        assert result == pytest.approx(0.5)

    def test_fit_and_predict(self) -> None:
        model = XGBoostModel(segment_id="us_tech")
        x_data = [_make_features()] * 50
        y = [1] * 25 + [0] * 25
        model.fit(x_data, y)
        result = model.predict_proba(_make_features())
        assert 0.0 <= result <= 1.0


class TestEnsembleModel:
    def test_predict_averages_two_models(self) -> None:
        xgb = XGBoostModel(segment_id="us_tech")
        lgb = LightGBMModel(segment_id="us_tech")
        ensemble = EnsembleModel(models=[xgb, lgb])
        features = _make_features()
        result = ensemble.predict_proba(features)
        assert 0.0 <= result <= 1.0

    def test_empty_models_returns_half(self) -> None:
        ensemble = EnsembleModel(models=[])
        assert ensemble.predict_proba(_make_features()) == pytest.approx(0.5)


class TestMLModelRegistry:
    def test_get_unregistered_returns_none(self) -> None:
        registry = MLModelRegistry()
        assert registry.get("us_tech") is None

    def test_register_and_get(self) -> None:
        registry = MLModelRegistry()
        xgb = XGBoostModel(segment_id="us_tech")
        lgb = LightGBMModel(segment_id="us_tech")
        model = EnsembleModel(models=[xgb, lgb])
        registry.register("us_tech", model)
        assert registry.get("us_tech") is model


# ── Helper ───────────────────────────────────────────────────────────────────


def _make_features(sentiment: float = 0.0) -> dict[str, float]:
    """Create a 40-candle set and return computed features."""
    rng = np.random.default_rng(42)
    prices = 100.0 + rng.standard_normal(40).cumsum()
    base_date = datetime(2024, 1, 1, tzinfo=UTC)
    candles = [
        Candle(
            symbol="AAPL",
            market_id="us",
            timeframe="1d",
            timestamp=base_date + timedelta(days=i),
            open=Decimal(str(round(float(prices[i]) * 0.999, 2))),
            high=Decimal(str(round(float(prices[i]) * 1.005, 2))),
            low=Decimal(str(round(float(prices[i]) * 0.995, 2))),
            close=Decimal(str(round(float(prices[i]), 2))),
            volume=int(1000 + rng.integers(0, 500)),
        )
        for i in range(40)
    ]
    return compute_features(candles, sentiment_score=sentiment)
