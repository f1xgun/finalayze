"""Tests for MetaLabeler (E1 -- meta-labeling)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from finalayze.core.schemas import Signal, SignalDirection
from finalayze.ml.meta_labeler import MetaLabeler


def _make_signal(
    *,
    direction: SignalDirection = SignalDirection.BUY,
    confidence: float = 0.6,
    symbol: str = "AAPL",
) -> Signal:
    return Signal(
        strategy_name="dual_momentum",
        symbol=symbol,
        market_id="us",
        segment_id="us_tech",
        direction=direction,
        confidence=confidence,
        features={"score": 0.8},
        reasoning="test signal",
    )


class TestMetaLabelerPredict:
    def test_predict_returns_probability(self) -> None:
        """Ensemble returns 0.7, meta-labeler returns 0.7."""
        ensemble = MagicMock()
        ensemble.predict_proba.return_value = 0.7
        meta = MetaLabeler(ensemble)

        signal = _make_signal()
        result = meta.predict(signal, {"rsi_14": 50.0})

        assert result == pytest.approx(0.7)

    def test_predict_adds_signal_features(self) -> None:
        """Verify signal_confidence and signal_direction_buy are added to features."""
        ensemble = MagicMock()
        ensemble.predict_proba.return_value = 0.7
        meta = MetaLabeler(ensemble)

        signal = _make_signal(direction=SignalDirection.BUY, confidence=0.65)
        meta.predict(signal, {"rsi_14": 50.0})

        called_features = ensemble.predict_proba.call_args[0][0]
        assert called_features["signal_confidence"] == pytest.approx(0.65)
        assert called_features["signal_direction_buy"] == pytest.approx(1.0)
        assert called_features["rsi_14"] == pytest.approx(50.0)

    def test_predict_sell_signal_direction_feature(self) -> None:
        """SELL signal sets signal_direction_buy=0.0."""
        ensemble = MagicMock()
        ensemble.predict_proba.return_value = 0.7
        meta = MetaLabeler(ensemble)

        signal = _make_signal(direction=SignalDirection.SELL)
        meta.predict(signal, {"rsi_14": 50.0})

        called_features = ensemble.predict_proba.call_args[0][0]
        assert called_features["signal_direction_buy"] == pytest.approx(0.0)

    def test_predict_untrained_returns_none(self) -> None:
        """Ensemble returns exactly 0.5 (untrained), returns None."""
        ensemble = MagicMock()
        ensemble.predict_proba.return_value = 0.5
        meta = MetaLabeler(ensemble)

        signal = _make_signal()
        result = meta.predict(signal, {"rsi_14": 50.0})

        assert result is None

    def test_predict_exception_returns_none(self) -> None:
        """Ensemble raises, returns None."""
        ensemble = MagicMock()
        ensemble.predict_proba.side_effect = RuntimeError("model error")
        meta = MetaLabeler(ensemble)

        signal = _make_signal()
        result = meta.predict(signal, {"rsi_14": 50.0})

        assert result is None

    def test_predict_does_not_mutate_original_features(self) -> None:
        """Original features dict should not be modified."""
        ensemble = MagicMock()
        ensemble.predict_proba.return_value = 0.7
        meta = MetaLabeler(ensemble)

        original = {"rsi_14": 50.0}
        signal = _make_signal()
        meta.predict(signal, original)

        assert "signal_confidence" not in original


class TestShouldTrade:
    def test_above_threshold(self) -> None:
        """0.60 > 0.40 -> True."""
        ensemble = MagicMock()
        meta = MetaLabeler(ensemble)
        assert meta.should_trade(0.60) is True

    def test_below_threshold(self) -> None:
        """0.30 < 0.40 -> False."""
        ensemble = MagicMock()
        meta = MetaLabeler(ensemble)
        assert meta.should_trade(0.30) is False

    def test_at_threshold(self) -> None:
        """0.40 == 0.40 -> False (strict >)."""
        ensemble = MagicMock()
        meta = MetaLabeler(ensemble)
        assert meta.should_trade(0.40) is False


class TestSizingFactor:
    def test_above_threshold(self) -> None:
        """p=0.70: (0.70 - 0.40) / 0.60 = 0.50."""
        ensemble = MagicMock()
        meta = MetaLabeler(ensemble)
        assert meta.sizing_factor(0.70) == pytest.approx(0.50)

    def test_at_threshold(self) -> None:
        """p=0.40 -> 0.0."""
        ensemble = MagicMock()
        meta = MetaLabeler(ensemble)
        assert meta.sizing_factor(0.40) == pytest.approx(0.0)

    def test_max_capped(self) -> None:
        """p=1.0 -> 1.0."""
        ensemble = MagicMock()
        meta = MetaLabeler(ensemble)
        assert meta.sizing_factor(1.0) == pytest.approx(1.0)

    def test_below_threshold(self) -> None:
        """p=0.20 -> 0.0."""
        ensemble = MagicMock()
        meta = MetaLabeler(ensemble)
        assert meta.sizing_factor(0.20) == pytest.approx(0.0)


class TestCustomThreshold:
    def test_custom_threshold(self) -> None:
        """threshold=0.50, p=0.60 -> should_trade=True, sizing=(0.60-0.50)/0.50=0.20."""
        ensemble = MagicMock()
        meta = MetaLabeler(ensemble, threshold=0.50)

        assert meta.threshold == pytest.approx(0.50)
        assert meta.should_trade(0.60) is True
        assert meta.sizing_factor(0.60) == pytest.approx(0.20)
