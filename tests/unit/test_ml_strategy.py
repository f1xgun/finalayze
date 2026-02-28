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
        """Probability 0.8 with default threshold 0.15 → BUY."""
        registry = MLModelRegistry()
        ensemble = MagicMock()
        ensemble.predict_proba.return_value = 0.8
        registry.register("us_tech", ensemble)

        strategy = MLStrategy(registry=registry)
        candles = _make_candles(60)

        with patch(_PATCH_TARGET, return_value=_FAKE_FEATURES):
            result = strategy.generate_signal("AAPL", candles, "us_tech")

        assert result is not None
        assert result.direction == SignalDirection.BUY
        expected_confidence = (0.8 - 0.5) * 2  # 0.6
        assert abs(result.confidence - expected_confidence) < 1e-6

    def test_sell_below_threshold(self) -> None:
        """Probability 0.2 with default threshold 0.15 → SELL."""
        registry = MLModelRegistry()
        ensemble = MagicMock()
        ensemble.predict_proba.return_value = 0.2
        registry.register("us_tech", ensemble)

        strategy = MLStrategy(registry=registry)
        candles = _make_candles(60)

        with patch(_PATCH_TARGET, return_value=_FAKE_FEATURES):
            result = strategy.generate_signal("AAPL", candles, "us_tech")

        assert result is not None
        assert result.direction == SignalDirection.SELL
        expected_confidence = (0.5 - 0.2) * 2  # 0.6
        assert abs(result.confidence - expected_confidence) < 1e-6

    def test_neutral_within_threshold(self) -> None:
        """Probability 0.6, threshold 0.15 → None (deadzone)."""
        registry = MLModelRegistry()
        ensemble = MagicMock()
        ensemble.predict_proba.return_value = 0.6
        registry.register("us_tech", ensemble)

        strategy = MLStrategy(registry=registry)
        candles = _make_candles(60)

        with patch(_PATCH_TARGET, return_value=_FAKE_FEATURES):
            result = strategy.generate_signal("AAPL", candles, "us_tech")

        assert result is None

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
            mock_cf.assert_called_once_with(candles, sentiment_score=0.0)


class TestSupportedSegments:
    def test_supported_segments_from_yaml(self) -> None:
        """Segments with ml_ensemble.enabled: true should be returned."""
        registry = MLModelRegistry()
        strategy = MLStrategy(registry=registry)
        segments = strategy.supported_segments()
        # All 8 presets have ml_ensemble.enabled: true
        assert len(segments) >= 8
        assert "us_tech" in segments
        assert "ru_blue_chips" in segments
