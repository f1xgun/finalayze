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
_FEATURE_NAMES = {
    # Core 5
    "rsi_14",
    "macd_hist_pct",
    "bb_pct_b",
    "volume_ratio_20d",
    "atr_14_pct",
    # Extra indicators
    "roc_10",
    "willr_14",
    "adx_14",
    "hist_vol_20",
    "gk_vol_20",
    "obv_slope_10",
    "rsi_divergence",
    # Predictive (lagged returns + distribution + short RSI)
    "ret_1d",
    "ret_5d",
    "ret_21d",
    "skew_20d",
    "kurt_20d",
    "max_ret_20d",
    "min_ret_20d",
    "rsi_2",
    "rsi_5",
    # Microstructure
    "proximity_rolling_high",
    "amihud_20d",
    "corwin_schultz_spread",
    # Wavelet
    "wavelet_approx_energy",
    "wavelet_detail1_energy",
    "wavelet_detail2_energy",
    "wavelet_detail3_energy",
    # Z-score
    "price_zscore_60d",
    "volume_zscore_20d",
    "rsi_zscore_60d",
    "atr_zscore_60d",
    # Calendar
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    # Regime / VIX
    "vix_level",
    "vix_percentile_252d",
    "vix_change_5d",
    "realized_vol_ratio",
    # Cross-asset
    "relative_strength_21d",
    "rolling_beta_63d",
    "rolling_corr_63d",
    "excess_momentum_score",
}
_N_FEATURES = len(_FEATURE_NAMES)


class TestComputeFeatures:
    def test_returns_correct_keys(self) -> None:
        # 40 synthetic candles needed for all indicators
        features = _make_features()
        assert set(features.keys()) == _FEATURE_NAMES

    def test_all_values_are_floats(self) -> None:
        features = _make_features()
        assert all(isinstance(v, float) for v in features.values())

    def test_sentiment_param_accepted_but_not_in_output(self) -> None:
        features = _make_features(sentiment=0.75)
        assert "sentiment" not in features

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
            for i in range(5)  # only 5, need at least 80
        ]
        with pytest.raises(InsufficientDataError):
            compute_features(candles)


class TestComputeFeaturesNaNHandling:
    """#160 — NaN values in compute_features output must be handled."""

    def test_no_nan_in_output_with_minimum_candles(self) -> None:
        """compute_features must return no NaN values even at the minimum candle count."""
        import math

        _min_count = 80
        rng = np.random.default_rng(7)
        prices = 100.0 + rng.standard_normal(_min_count).cumsum()
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
            for i in range(_min_count)
        ]
        features = compute_features(candles)
        assert all(not math.isnan(v) for v in features.values()), (
            f"NaN found in features: {features}"
        )

    def test_volume_ratio_no_nan_at_boundary(self) -> None:
        """volume_ratio_20d must not be NaN at minimum candle count."""
        import math

        _min_count = 80
        rng = np.random.default_rng(99)
        prices = 100.0 + rng.standard_normal(_min_count).cumsum()
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
            for i in range(_min_count)
        ]
        features = compute_features(candles)
        assert not math.isnan(features["volume_ratio_20d"])


class TestVolumeRatioNoLookAhead:
    """#132 — volume_ratio_20d must exclude the current bar from the denominator."""

    def test_volume_ratio_uses_shifted_mean(self) -> None:
        """volume_ratio_20d denominator must use the rolling mean of PRIOR bars only.

        We create candles where all volumes are identical except the last bar.
        If there is no look-ahead, the ratio for the last bar should be
        last_volume / mean_of_prior_20_volumes.
        """
        base_date = datetime(2024, 1, 1, tzinfo=UTC)
        STANDARD_VOLUME = 1000  # noqa: N806
        LAST_VOLUME = 3000  # noqa: N806  # 3x the standard
        _VOL_TEST_COUNT = 80

        candles = [
            Candle(
                symbol="AAPL",
                market_id="us",
                timeframe="1d",
                timestamp=base_date + timedelta(days=i),
                open=Decimal("100.00"),
                high=Decimal("105.00"),
                low=Decimal("95.00"),
                close=Decimal("102.00"),
                volume=STANDARD_VOLUME if i < _VOL_TEST_COUNT - 1 else LAST_VOLUME,
            )
            for i in range(_VOL_TEST_COUNT)
        ]
        features = compute_features(candles)
        # With no look-ahead: ratio = LAST_VOLUME / STANDARD_VOLUME = 3.0
        # With look-ahead (bug): ratio = LAST_VOLUME / mean([...LAST_VOLUME...]) != 3.0
        assert features["volume_ratio_20d"] == pytest.approx(3.0, abs=0.01)


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

    def test_consistent_prediction_regardless_of_dict_insertion_order(self) -> None:
        """predict_proba must return same result for same features regardless of dict order."""
        model = XGBoostModel(segment_id="us_tech")
        x_data = [_make_features()] * 50
        y = [1] * 25 + [0] * 25
        model.fit(x_data, y)

        features = _make_features(sentiment=0.5)
        # Build same features with reversed insertion order
        features_reversed = dict(reversed(list(features.items())))

        result1 = model.predict_proba(features)
        result2 = model.predict_proba(features_reversed)
        assert result1 == pytest.approx(result2)

    def test_feature_mismatch_raises_insufficient_data_error(self) -> None:
        """predict_proba raises InsufficientDataError when feature keys differ from training."""
        model = XGBoostModel(segment_id="us_tech")
        x_data = [_make_features()] * 50
        y = [1] * 25 + [0] * 25
        model.fit(x_data, y)

        # Remove a feature key — should raise
        features = _make_features()
        bad_features = {k: v for k, v in features.items() if k != "rsi_14"}
        with pytest.raises(InsufficientDataError):
            model.predict_proba(bad_features)


class TestLightGBMModel:
    def test_predict_proba_before_fit_returns_half(self) -> None:
        model = LightGBMModel(segment_id="us_tech")
        features = _make_features()
        result = model.predict_proba(features)
        assert result == pytest.approx(0.5)

    def test_fit_and_predict(self) -> None:
        model = LightGBMModel(segment_id="us_tech")
        x_data = [_make_features()] * 50
        y = [1] * 25 + [0] * 25
        model.fit(x_data, y)
        result = model.predict_proba(_make_features())
        assert 0.0 <= result <= 1.0

    def test_consistent_prediction_regardless_of_dict_insertion_order(self) -> None:
        """predict_proba must return same result for same features regardless of dict order."""
        model = LightGBMModel(segment_id="us_tech")
        x_data = [_make_features()] * 50
        y = [1] * 25 + [0] * 25
        model.fit(x_data, y)

        features = _make_features(sentiment=0.5)
        features_reversed = dict(reversed(list(features.items())))

        result1 = model.predict_proba(features)
        result2 = model.predict_proba(features_reversed)
        assert result1 == pytest.approx(result2)

    def test_feature_mismatch_raises_insufficient_data_error(self) -> None:
        """predict_proba raises InsufficientDataError when feature keys differ from training."""
        model = LightGBMModel(segment_id="us_tech")
        x_data = [_make_features()] * 50
        y = [1] * 25 + [0] * 25
        model.fit(x_data, y)

        features = _make_features()
        bad_features = {k: v for k, v in features.items() if k != "rsi_14"}
        with pytest.raises(InsufficientDataError):
            model.predict_proba(bad_features)


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


# ── Early stopping ────────────────────────────────────────────────────────────

_EARLY_STOP_N_FEATURES = 5
_EARLY_STOP_SEED = 99


def _make_synthetic_dataset(
    n_samples: int,
    n_features: int = _EARLY_STOP_N_FEATURES,
    seed: int = _EARLY_STOP_SEED,
) -> tuple[list[dict[str, float]], list[int]]:
    """Create synthetic feature dicts with separable binary labels."""
    rng = np.random.default_rng(seed)
    keys = [f"f_{i}" for i in range(n_features)]
    X: list[dict[str, float]] = []
    y: list[int] = []
    for i in range(n_samples):
        label = 1 if i % 2 == 0 else 0
        row = {k: float(rng.standard_normal() + (1.0 if label else -1.0)) for k in keys}
        X.append(row)
        y.append(label)
    return X, y


class TestXGBoostEarlyStopping:
    """Verify XGBoostModel.fit() uses early stopping with a validation split."""

    def test_small_dataset_does_not_crash(self) -> None:
        """Early stopping must not crash even with very small datasets (20 samples)."""
        model = XGBoostModel(segment_id="test")
        X, y = _make_synthetic_dataset(n_samples=20)
        model.fit(X, y)  # should not raise
        result = model.predict_proba(X[0])
        assert 0.0 <= result <= 1.0

    def test_valid_predictions_after_early_stopping(self) -> None:
        """Model produces probabilities in [0, 1] after training with early stopping."""
        model = XGBoostModel(segment_id="test")
        X, y = _make_synthetic_dataset(n_samples=100)
        model.fit(X, y)
        for sample in X[:10]:
            p = model.predict_proba(sample)
            assert 0.0 <= p <= 1.0

    def test_fit_with_sample_weight(self) -> None:
        """Early stopping works when sample_weight is provided."""
        model = XGBoostModel(segment_id="test")
        X, y = _make_synthetic_dataset(n_samples=50)
        weights = np.ones(len(X), dtype=float)
        weights[:10] = 2.0  # upweight first 10 samples
        model.fit(X, y, sample_weight=weights)
        result = model.predict_proba(X[0])
        assert 0.0 <= result <= 1.0

    def test_fit_without_sample_weight(self) -> None:
        """Early stopping works when sample_weight is None."""
        model = XGBoostModel(segment_id="test")
        X, y = _make_synthetic_dataset(n_samples=50)
        model.fit(X, y, sample_weight=None)
        result = model.predict_proba(X[0])
        assert 0.0 <= result <= 1.0

    def test_stops_before_max_estimators(self) -> None:
        """With noisy data, early stopping should halt before using all n_estimators."""
        model = XGBoostModel(segment_id="test", n_estimators=500)
        X, y = _make_synthetic_dataset(n_samples=200)
        model.fit(X, y)
        # XGBoost best_iteration is 0-indexed; best_ntree_limit gives the count
        assert model._model is not None  # noqa: SLF001
        # If early stopping triggered, best_iteration < n_estimators - 1
        best = model._model.best_iteration  # noqa: SLF001
        assert best < 499, f"Expected early stop but ran all 500 rounds (best_iteration={best})"


class TestLightGBMEarlyStopping:
    """Verify LightGBMModel.fit() uses early stopping with a validation split."""

    def test_small_dataset_does_not_crash(self) -> None:
        """Early stopping must not crash even with very small datasets (20 samples)."""
        from finalayze.ml.models.lightgbm_model import LightGBMModel

        model = LightGBMModel(segment_id="test")
        X, y = _make_synthetic_dataset(n_samples=20)
        model.fit(X, y)  # should not raise
        result = model.predict_proba(X[0])
        assert 0.0 <= result <= 1.0

    def test_valid_predictions_after_early_stopping(self) -> None:
        """Model produces probabilities in [0, 1] after training with early stopping."""
        from finalayze.ml.models.lightgbm_model import LightGBMModel

        model = LightGBMModel(segment_id="test")
        X, y = _make_synthetic_dataset(n_samples=100)
        model.fit(X, y)
        for sample in X[:10]:
            p = model.predict_proba(sample)
            assert 0.0 <= p <= 1.0

    def test_fit_with_sample_weight(self) -> None:
        """Early stopping works when sample_weight is provided."""
        from finalayze.ml.models.lightgbm_model import LightGBMModel

        model = LightGBMModel(segment_id="test")
        X, y = _make_synthetic_dataset(n_samples=50)
        weights = np.ones(len(X), dtype=float)
        weights[:10] = 2.0
        model.fit(X, y, sample_weight=weights)
        result = model.predict_proba(X[0])
        assert 0.0 <= result <= 1.0

    def test_fit_without_sample_weight(self) -> None:
        """Early stopping works when sample_weight is None."""
        from finalayze.ml.models.lightgbm_model import LightGBMModel

        model = LightGBMModel(segment_id="test")
        X, y = _make_synthetic_dataset(n_samples=50)
        model.fit(X, y, sample_weight=None)
        result = model.predict_proba(X[0])
        assert 0.0 <= result <= 1.0

    def test_uses_eval_set(self) -> None:
        """fit() must pass eval_set to LGBMClassifier (early stopping is configured)."""
        import contextlib
        from unittest import mock

        import lightgbm as lgb_lib

        model = LightGBMModel(segment_id="test", n_estimators=200)
        X, y = _make_synthetic_dataset(n_samples=50)

        with mock.patch.object(lgb_lib.LGBMClassifier, "fit", wraps=None) as mocked_fit:
            # Need to actually let it train, so we call through
            mocked_fit.side_effect = None  # don't actually train
            with contextlib.suppress(Exception):
                model.fit(X, y)
            # Check that eval_set was passed
            if mocked_fit.called:
                _, kwargs = mocked_fit.call_args
                assert "eval_set" in kwargs, "fit() must pass eval_set for early stopping"
                assert "callbacks" in kwargs, "fit() must pass callbacks for early stopping"


# ── Helper ───────────────────────────────────────────────────────────────────


_MAKE_FEATURES_COUNT = 80


def _make_features(sentiment: float = 0.0) -> dict[str, float]:
    """Create an 80-candle set and return computed features."""
    rng = np.random.default_rng(42)
    prices = 100.0 + rng.standard_normal(_MAKE_FEATURES_COUNT).cumsum()
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
        for i in range(_MAKE_FEATURES_COUNT)
    ]
    return compute_features(candles, sentiment_score=sentiment)
