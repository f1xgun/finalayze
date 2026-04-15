"""Unit tests for EventDrivenStrategy."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from finalayze.core.schemas import Candle, Signal, SignalDirection
from finalayze.strategies.event_driven import EventDrivenStrategy

_CANDLE = Candle(
    symbol="AAPL",
    market_id="us",
    timeframe="1d",
    timestamp=datetime(2024, 1, 2, tzinfo=UTC),
    open=Decimal(150),
    high=Decimal(155),
    low=Decimal(148),
    close=Decimal(152),
    volume=1000,
)
_CANDLES = [_CANDLE]
_SEGMENT = "us_tech"
_MIN_SENTIMENT = 0.5


class TestEventDrivenStrategy:
    def test_name_is_event_driven(self) -> None:
        strategy = EventDrivenStrategy()
        assert strategy.name == "event_driven"

    def test_high_positive_sentiment_generates_buy(self) -> None:
        strategy = EventDrivenStrategy()
        signal = strategy.generate_signal("AAPL", _CANDLES, _SEGMENT, sentiment_score=0.8)
        assert signal is not None
        assert signal.direction == SignalDirection.BUY

    def test_high_negative_sentiment_generates_sell(self) -> None:
        strategy = EventDrivenStrategy()
        signal = strategy.generate_signal("AAPL", _CANDLES, _SEGMENT, sentiment_score=-0.8)
        assert signal is not None
        assert signal.direction == SignalDirection.SELL

    def test_low_sentiment_returns_none(self) -> None:
        strategy = EventDrivenStrategy()
        signal = strategy.generate_signal("AAPL", _CANDLES, _SEGMENT, sentiment_score=0.1)
        assert signal is None

    def test_zero_sentiment_returns_none(self) -> None:
        strategy = EventDrivenStrategy()
        signal = strategy.generate_signal("AAPL", _CANDLES, _SEGMENT, sentiment_score=0.0)
        assert signal is None

    def test_confidence_scales_with_sentiment(self) -> None:
        strategy = EventDrivenStrategy()
        signal_high = strategy.generate_signal("AAPL", _CANDLES, _SEGMENT, sentiment_score=0.9)
        signal_low = strategy.generate_signal("AAPL", _CANDLES, _SEGMENT, sentiment_score=0.75)
        assert signal_high is not None
        assert signal_low is not None
        assert signal_high.confidence > signal_low.confidence

    def test_get_parameters_returns_dict(self) -> None:
        strategy = EventDrivenStrategy()
        params = strategy.get_parameters(_SEGMENT)
        assert isinstance(params, dict)

    def test_supported_segments_returns_list(self) -> None:
        strategy = EventDrivenStrategy()
        segments = strategy.supported_segments()
        assert isinstance(segments, list)
        # Event-driven is enabled in US presets (us_tech, us_broad, us_finance, us_healthcare)
        assert _SEGMENT in segments


class TestEventTypeCode:
    """Tests for event_type_code embedding in Signal.features (EVNT-01)."""

    def test_event_type_code_cbr_in_features(self) -> None:
        """When event_type_code=1.0 (CBR), Signal.features contains it."""
        strategy = EventDrivenStrategy()
        signal = strategy.generate_signal(
            "SBER", _CANDLES, _SEGMENT, sentiment_score=0.8, event_type_code=1.0
        )
        assert signal is not None
        assert signal.features["event_type_code"] == 1.0

    def test_event_type_code_default_zero(self) -> None:
        """When no event_type_code kwarg passed, Signal.features has 0.0."""
        strategy = EventDrivenStrategy()
        signal = strategy.generate_signal(
            "SBER", _CANDLES, _SEGMENT, sentiment_score=0.8
        )
        assert signal is not None
        assert signal.features["event_type_code"] == 0.0

    def test_credibility_scales_confidence(self) -> None:
        """With credibility=0.7 and sentiment=0.8, confidence = min(1.0, 0.8 * 0.7) = 0.56."""
        strategy = EventDrivenStrategy()
        signal = strategy.generate_signal(
            "SBER", _CANDLES, _SEGMENT, sentiment_score=0.8, credibility=0.7
        )
        assert signal is not None
        assert signal.confidence == pytest.approx(0.56, abs=0.01)


class TestCredibilityInCombiner:
    """Tests for credibility threading from combiner to EventDrivenStrategy."""

    def test_combiner_passes_credibility_to_event_driven(self) -> None:
        """When combiner.generate_signal(credibility=0.8), EventDrivenStrategy gets it."""
        from unittest.mock import MagicMock, patch

        from finalayze.strategies.combiner import StrategyCombiner

        # Create a mock event_driven strategy
        mock_ed = MagicMock()
        mock_ed.name = "event_driven"
        mock_ed.generate_signal.return_value = Signal(
            strategy_name="event_driven",
            symbol="SBER",
            market_id="ru",
            segment_id="ru_blue_chips",
            direction=SignalDirection.BUY,
            confidence=0.8,
            features={"sentiment": 0.8, "credibility": 0.8, "event_type_code": 0.0},
            reasoning="test",
        )

        combiner = StrategyCombiner([mock_ed])
        candles = _CANDLES
        config = {
            "strategies": {
                "event_driven": {"enabled": True, "weight": 1.0},
            }
        }
        with patch.object(combiner, "_load_config", return_value=config):
            combiner.generate_signal(
                "SBER", candles, "ru_blue_chips", credibility=0.8, event_type_code=1.0
            )

        # Verify credibility=0.8 was passed to event_driven strategy
        mock_ed.generate_signal.assert_called_once()
        call_kwargs = mock_ed.generate_signal.call_args
        assert call_kwargs.kwargs.get("credibility") == 0.8 or (
            len(call_kwargs.args) > 5 and call_kwargs.args[5] == 0.8  # noqa: PLR2004
        )

    def test_combiner_does_not_pass_credibility_to_other_strategies(self) -> None:
        """MomentumStrategy.generate_signal() called without credibility kwarg."""
        from unittest.mock import MagicMock, patch

        from finalayze.strategies.combiner import StrategyCombiner

        # Create a mock momentum strategy that only accepts standard args
        mock_momentum = MagicMock()
        mock_momentum.name = "momentum"
        mock_momentum.generate_signal.return_value = Signal(
            strategy_name="momentum",
            symbol="SBER",
            market_id="ru",
            segment_id="ru_blue_chips",
            direction=SignalDirection.BUY,
            confidence=0.9,
            features={"mock_feature": 0.9},
            reasoning="test",
        )

        combiner = StrategyCombiner([mock_momentum])
        candles = _CANDLES
        config = {
            "strategies": {
                "momentum": {"enabled": True, "weight": 1.0},
            }
        }
        with patch.object(combiner, "_load_config", return_value=config):
            combiner.generate_signal(
                "SBER", candles, "ru_blue_chips", credibility=0.8, event_type_code=1.0
            )

        # Verify momentum was NOT called with credibility kwarg
        mock_momentum.generate_signal.assert_called_once()
        call_kwargs = mock_momentum.generate_signal.call_args
        assert "credibility" not in call_kwargs.kwargs
        assert "event_type_code" not in call_kwargs.kwargs
