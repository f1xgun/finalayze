"""Tests for MLStrategy (Layer 4)."""

from __future__ import annotations

import datetime as dt
from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import MagicMock, patch

from finalayze.core.schemas import Candle, SignalDirection
from finalayze.ml.registry import MLModelRegistry
from finalayze.strategies.ml_strategy import MLStrategy

_PATCH_TARGET = "finalayze.strategies.ml_strategy.compute_features"
_FAKE_FEATURES: dict[str, float] = {"rsi_14": 50.0}


def _make_candles(n: int = 60, base_price: float = 100.0) -> list[Candle]:
    """Create n synthetic candles with small price increments."""
    candles: list[Candle] = []
    for i in range(n):
        price = Decimal(str(base_price + i * 0.1))
        ts = datetime(2025, 1, 1, tzinfo=UTC) + dt.timedelta(days=i)
        candles.append(
            Candle(
                symbol="AAPL",
                market_id="us",
                timeframe="1d",
                timestamp=ts,
                open=price,
                high=price + Decimal(1),
                low=price - Decimal(1),
                close=price,
                volume=1000,
            )
        )
    return candles


class TestMLStrategyName:
    def test_name_returns_ml_ensemble(self) -> None:
        registry = MLModelRegistry()
        strategy = MLStrategy(registry=registry)
        assert strategy.name == "ml_ensemble"


class TestGenerateSignal:
    def test_no_registry_model_returns_none(self) -> None:
        """When no model is registered for the segment, return None."""
        registry = MLModelRegistry()
        strategy = MLStrategy(registry=registry)
        candles = _make_candles(60)
        result = strategy.generate_signal("AAPL", candles, "us_tech")
        assert result is None

    def test_untrained_returns_none(self) -> None:
        """When ensemble returns exactly 0.5 (untrained), return None."""
        registry = MLModelRegistry()
        ensemble = MagicMock()
        ensemble.predict_proba.return_value = 0.5
        registry.register("us_tech", ensemble)

        strategy = MLStrategy(registry=registry)
        candles = _make_candles(60)

        with patch(_PATCH_TARGET, return_value=_FAKE_FEATURES):
            result = strategy.generate_signal("AAPL", candles, "us_tech")
        assert result is None

    def test_buy_above_threshold(self) -> None:
        """Probability 0.7 with default threshold 0.08 → BUY."""
        registry = MLModelRegistry()
        ensemble = MagicMock()
        ensemble.predict_proba.return_value = 0.7
        registry.register("us_tech", ensemble)

        strategy = MLStrategy(registry=registry)
        candles = _make_candles(60)

        with patch(_PATCH_TARGET, return_value=_FAKE_FEATURES):
            result = strategy.generate_signal("AAPL", candles, "us_tech")

        assert result is not None
        assert result.direction == SignalDirection.BUY
        expected_confidence = (0.7 - 0.5) * 2  # 0.4
        assert abs(result.confidence - expected_confidence) < 1e-6

    def test_sell_below_threshold(self) -> None:
        """Probability 0.3 with default threshold 0.08 → SELL."""
        registry = MLModelRegistry()
        ensemble = MagicMock()
        ensemble.predict_proba.return_value = 0.3
        registry.register("us_tech", ensemble)

        strategy = MLStrategy(registry=registry)
        candles = _make_candles(60)

        with patch(_PATCH_TARGET, return_value=_FAKE_FEATURES):
            result = strategy.generate_signal("AAPL", candles, "us_tech")

        assert result is not None
        assert result.direction == SignalDirection.SELL
        expected_confidence = (0.5 - 0.3) * 2  # 0.4
        assert abs(result.confidence - expected_confidence) < 1e-6

    def test_neutral_within_threshold(self) -> None:
        """Probability 0.55, threshold 0.08 → None (deadzone)."""
        registry = MLModelRegistry()
        ensemble = MagicMock()
        ensemble.predict_proba.return_value = 0.55
        registry.register("us_tech", ensemble)

        strategy = MLStrategy(registry=registry)
        candles = _make_candles(60)

        with patch(_PATCH_TARGET, return_value=_FAKE_FEATURES):
            result = strategy.generate_signal("AAPL", candles, "us_tech")

        assert result is None

    def test_confidence_at_or_below_min_rejected(self) -> None:
        """Confidence at or below min_confidence must be rejected.

        prob=0.575 → confidence = (0.575 - 0.5) * 2 = 0.15 <= 0.15 = _DEFAULT_MIN_CONFIDENCE.
        The ``<=`` check ensures this returns None rather than a signal.
        """
        registry = MLModelRegistry()
        ensemble = MagicMock()
        ensemble.predict_proba.return_value = 0.575
        registry.register("us_tech", ensemble)

        strategy = MLStrategy(registry=registry)
        candles = _make_candles(60)

        with patch(_PATCH_TARGET, return_value=_FAKE_FEATURES):
            result = strategy.generate_signal("AAPL", candles, "us_tech")

        assert result is None

    def test_confidence_just_above_min_accepted(self) -> None:
        """Confidence slightly above min_confidence must produce a signal.

        With class defaults (threshold=0.08, min_confidence=0.15):
        prob=0.59 → confidence = (0.59 - 0.5) * 2 = 0.18 > 0.15 → BUY signal.
        """
        registry = MLModelRegistry()
        ensemble = MagicMock()
        ensemble.predict_proba.return_value = 0.59
        registry.register("us_tech", ensemble)

        strategy = MLStrategy(registry=registry)
        candles = _make_candles(60)

        with (
            patch(_PATCH_TARGET, return_value=_FAKE_FEATURES),
            patch.object(strategy, "get_parameters", return_value={}),
        ):
            result = strategy.generate_signal("AAPL", candles, "us_tech")

        assert result is not None
        assert result.direction == SignalDirection.BUY
        expected_confidence = (0.59 - 0.5) * 2  # 0.18
        assert abs(result.confidence - expected_confidence) < 1e-6

    def test_catches_insufficient_data_error(self) -> None:
        """InsufficientDataError from compute_features is caught."""
        from finalayze.core.exceptions import InsufficientDataError

        registry = MLModelRegistry()
        ensemble = MagicMock()
        registry.register("us_tech", ensemble)

        strategy = MLStrategy(registry=registry)
        candles = _make_candles(10)

        with patch(
            _PATCH_TARGET,
            side_effect=InsufficientDataError("too few"),
        ):
            result = strategy.generate_signal("AAPL", candles, "us_tech")

        assert result is None

    def test_catches_predict_error(self) -> None:
        """Exception from predict_proba is caught gracefully."""
        registry = MLModelRegistry()
        ensemble = MagicMock()
        ensemble.predict_proba.side_effect = RuntimeError("model error")
        registry.register("us_tech", ensemble)

        strategy = MLStrategy(registry=registry)
        candles = _make_candles(60)

        with patch(_PATCH_TARGET, return_value=_FAKE_FEATURES):
            result = strategy.generate_signal("AAPL", candles, "us_tech")

        assert result is None

    def test_sentiment_passed_as_zero(self) -> None:
        """compute_features always receives sentiment_score=0.0."""
        registry = MLModelRegistry()
        ensemble = MagicMock()
        ensemble.predict_proba.return_value = 0.8
        registry.register("us_tech", ensemble)

        strategy = MLStrategy(registry=registry)
        candles = _make_candles(60)

        with patch(_PATCH_TARGET, return_value=_FAKE_FEATURES) as mock_cf:
            strategy.generate_signal("AAPL", candles, "us_tech", sentiment_score=0.9)
            mock_cf.assert_called_once_with(
                candles,
                sentiment_score=0.0,
                benchmark_candles=None,
                vix_candles=None,
            )


class TestFeatureFiltering:
    """Feature mismatch fix: MLStrategy filters features using ensemble.selected_features."""

    def test_filters_features_when_selected_features_set(self) -> None:
        """When ensemble has selected_features, only those keys are passed to predict_proba."""
        registry = MLModelRegistry()
        ensemble = MagicMock()
        ensemble.selected_features = ["rsi_14", "atr_14_pct"]
        ensemble.predict_proba.return_value = 0.8
        registry.register("us_tech", ensemble)

        strategy = MLStrategy(registry=registry)
        candles = _make_candles(60)

        all_features = {"rsi_14": 50.0, "atr_14_pct": 0.02, "macd_hist_pct": 0.01, "bb_pct_b": 0.5}
        with patch(_PATCH_TARGET, return_value=all_features):
            strategy.generate_signal("AAPL", candles, "us_tech")

        # predict_proba should receive only the selected features
        called_features = ensemble.predict_proba.call_args[0][0]
        assert set(called_features.keys()) == {"rsi_14", "atr_14_pct"}
        assert called_features["rsi_14"] == 50.0
        assert called_features["atr_14_pct"] == 0.02

    def test_no_filter_when_selected_features_none(self) -> None:
        """When ensemble.selected_features is None, all features pass through (legacy)."""
        registry = MLModelRegistry()
        ensemble = MagicMock()
        ensemble.selected_features = None
        ensemble.predict_proba.return_value = 0.8
        registry.register("us_tech", ensemble)

        strategy = MLStrategy(registry=registry)
        candles = _make_candles(60)

        all_features = {"rsi_14": 50.0, "atr_14_pct": 0.02, "macd_hist_pct": 0.01}
        with patch(_PATCH_TARGET, return_value=all_features):
            strategy.generate_signal("AAPL", candles, "us_tech")

        called_features = ensemble.predict_proba.call_args[0][0]
        assert set(called_features.keys()) == {"rsi_14", "atr_14_pct", "macd_hist_pct"}

    def test_filter_preserves_feature_values(self) -> None:
        """Filtered features retain their exact values."""
        registry = MLModelRegistry()
        ensemble = MagicMock()
        ensemble.selected_features = ["bb_pct_b"]
        ensemble.predict_proba.return_value = 0.8
        registry.register("us_tech", ensemble)

        strategy = MLStrategy(registry=registry)
        candles = _make_candles(60)

        all_features = {"rsi_14": 50.0, "bb_pct_b": 0.42, "volume_ratio_20d": 1.5}
        with patch(_PATCH_TARGET, return_value=all_features):
            strategy.generate_signal("AAPL", candles, "us_tech")

        called_features = ensemble.predict_proba.call_args[0][0]
        assert called_features == {"bb_pct_b": 0.42}


class TestUncertaintyReduction:
    """C5: Ensemble disagreement reduces confidence in MLStrategy."""

    def test_high_uncertainty_reduces_confidence(self) -> None:
        """When prediction_uncertainty > 0.10, confidence is scaled down."""
        registry = MLModelRegistry()
        ensemble = MagicMock()
        ensemble.predict_proba.return_value = 0.8
        ensemble.selected_features = None
        # Set model probas so std = 0.20 (two values 0.2 apart from mean)
        ensemble.last_model_probas = {"XGBoostModel": 0.6, "LightGBMModel": 1.0}
        registry.register("us_tech", ensemble)

        strategy = MLStrategy(registry=registry)
        candles = _make_candles(60)

        with (
            patch(_PATCH_TARGET, return_value=_FAKE_FEATURES),
            patch.object(strategy, "get_parameters", return_value={}),
        ):
            result = strategy.generate_signal("AAPL", candles, "us_tech")

        assert result is not None
        # Raw confidence = (0.8 - 0.5) * 2 = 0.6
        # After uncertainty: 0.6 * (1.0 - 0.20) = 0.48
        expected_confidence = 0.6 * 0.8
        assert abs(result.confidence - expected_confidence) < 1e-6

    def test_low_uncertainty_no_reduction(self) -> None:
        """When prediction_uncertainty <= 0.10, confidence is not reduced."""
        registry = MLModelRegistry()
        ensemble = MagicMock()
        ensemble.predict_proba.return_value = 0.8
        ensemble.selected_features = None
        # Set model probas so std ~ 0.05 (low disagreement, below threshold)
        ensemble.last_model_probas = {"XGBoostModel": 0.75, "LightGBMModel": 0.85}
        registry.register("us_tech", ensemble)

        strategy = MLStrategy(registry=registry)
        candles = _make_candles(60)

        with (
            patch(_PATCH_TARGET, return_value=_FAKE_FEATURES),
            patch.object(strategy, "get_parameters", return_value={}),
        ):
            result = strategy.generate_signal("AAPL", candles, "us_tech")

        assert result is not None
        # Raw confidence = (0.8 - 0.5) * 2 = 0.6, unchanged
        expected_confidence = 0.6
        assert abs(result.confidence - expected_confidence) < 1e-6

    def test_missing_uncertainty_attribute_no_crash(self) -> None:
        """Graceful when ensemble lacks prediction_uncertainty (legacy mocks)."""
        registry = MLModelRegistry()
        ensemble = MagicMock(spec=[])
        ensemble.predict_proba = MagicMock(return_value=0.8)
        ensemble.selected_features = None
        # No prediction_uncertainty attribute → getattr returns 0.0
        registry.register("us_tech", ensemble)

        strategy = MLStrategy(registry=registry)
        candles = _make_candles(60)

        with (
            patch(_PATCH_TARGET, return_value=_FAKE_FEATURES),
            patch.object(strategy, "get_parameters", return_value={}),
        ):
            result = strategy.generate_signal("AAPL", candles, "us_tech")

        assert result is not None


class TestBaseRateCorrection:
    """E2: base_rate parameter shifts direction thresholds."""

    def test_base_rate_correction_buy(self) -> None:
        """base_rate=0.55, prob=0.72, threshold=0.08 -> BUY (0.72 > 0.55+0.08=0.63)."""
        registry = MLModelRegistry()
        ensemble = MagicMock()
        ensemble.predict_proba.return_value = 0.72
        ensemble.selected_features = None
        registry.register("us_tech", ensemble)

        strategy = MLStrategy(registry=registry)
        candles = _make_candles(60)

        params = {"base_rate": 0.55, "threshold": 0.08}
        with (
            patch(_PATCH_TARGET, return_value=_FAKE_FEATURES),
            patch.object(strategy, "get_parameters", return_value=params),
        ):
            result = strategy.generate_signal("AAPL", candles, "us_tech")

        assert result is not None
        assert result.direction == SignalDirection.BUY
        # confidence = (0.72 - 0.55) * 2 = 0.34
        expected_confidence = (0.72 - 0.55) * 2
        assert abs(result.confidence - expected_confidence) < 1e-6

    def test_base_rate_correction_sell(self) -> None:
        """base_rate=0.55, prob=0.40, threshold=0.08 -> SELL (0.40 < 0.55-0.08=0.47)."""
        registry = MLModelRegistry()
        ensemble = MagicMock()
        ensemble.predict_proba.return_value = 0.40
        ensemble.selected_features = None
        registry.register("us_tech", ensemble)

        strategy = MLStrategy(registry=registry)
        candles = _make_candles(60)

        params = {"base_rate": 0.55, "threshold": 0.08}
        with (
            patch(_PATCH_TARGET, return_value=_FAKE_FEATURES),
            patch.object(strategy, "get_parameters", return_value=params),
        ):
            result = strategy.generate_signal("AAPL", candles, "us_tech")

        assert result is not None
        assert result.direction == SignalDirection.SELL
        # confidence = (0.55 - 0.40) * 2 = 0.30
        expected_confidence = (0.55 - 0.40) * 2
        assert abs(result.confidence - expected_confidence) < 1e-6

    def test_base_rate_default_is_05(self) -> None:
        """Without base_rate param, behavior unchanged (uses 0.50 center)."""
        registry = MLModelRegistry()
        ensemble = MagicMock()
        ensemble.predict_proba.return_value = 0.7
        ensemble.selected_features = None
        registry.register("us_tech", ensemble)

        strategy = MLStrategy(registry=registry)
        candles = _make_candles(60)

        # No base_rate in params -> defaults to 0.50
        params: dict[str, object] = {}
        with (
            patch(_PATCH_TARGET, return_value=_FAKE_FEATURES),
            patch.object(strategy, "get_parameters", return_value=params),
        ):
            result = strategy.generate_signal("AAPL", candles, "us_tech")

        assert result is not None
        assert result.direction == SignalDirection.BUY
        # confidence = (0.7 - 0.5) * 2 = 0.4
        expected_confidence = (0.7 - 0.5) * 2
        assert abs(result.confidence - expected_confidence) < 1e-6


class TestSupportedSegments:
    def test_supported_segments_from_yaml(self) -> None:
        """ml_ensemble disabled in all presets (models not production-ready)."""
        registry = MLModelRegistry()
        strategy = MLStrategy(registry=registry)
        segments = strategy.supported_segments()
        assert isinstance(segments, list)
        assert len(segments) == 0
