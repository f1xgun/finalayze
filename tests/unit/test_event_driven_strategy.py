"""Unit tests for EventDrivenStrategy."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any
from unittest.mock import patch

from finalayze.core.schemas import Candle, SignalDirection
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
_CANDLE_RU = Candle(
    symbol="GAZP",
    market_id="moex",
    timeframe="1d",
    timestamp=datetime(2024, 1, 2, tzinfo=UTC),
    open=Decimal(170),
    high=Decimal(175),
    low=Decimal(168),
    close=Decimal(172),
    volume=5000,
)
_CANDLES = [_CANDLE]
_CANDLES_RU = [_CANDLE_RU]
_SEGMENT = "us_tech"
_SEGMENT_RU = "ru_blue_chips"
_MIN_SENTIMENT = 0.5

# Params that include sanctions in event_types (like ru_blue_chips)
_PARAMS_WITH_SANCTIONS: dict[str, Any] = {
    "min_sentiment": 0.5,
    "event_types": ["geopolitical", "sanctions", "cbr_rate", "commodity_price", "earnings"],
}

# Params without sanctions in event_types (like us_tech)
_PARAMS_NO_SANCTIONS: dict[str, Any] = {
    "min_sentiment": 0.5,
    "event_types": ["earnings", "fda", "product_launch"],
}


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
        # Event-driven is disabled in all presets (produces 0 signals without live news feed)
        # Will be re-enabled when real-time news ingestion is wired


class TestSanctionsProximityScoring:
    """Tests for sanctions proximity scoring in EventDrivenStrategy."""

    def test_sanctions_scoring_reduces_confidence_for_gazp(self) -> None:
        """GAZP (proximity=0.8) should have lower confidence than unscored stock."""
        strategy = EventDrivenStrategy()
        sentiment = 0.8

        with patch.object(strategy, "get_parameters", return_value=_PARAMS_WITH_SANCTIONS):
            signal_gazp = strategy.generate_signal(
                "GAZP",
                _CANDLES_RU,
                _SEGMENT_RU,
                sentiment_score=sentiment,
            )
            signal_mgnt = strategy.generate_signal(
                "MGNT",
                _CANDLES_RU,
                _SEGMENT_RU,
                sentiment_score=sentiment,
            )

        assert signal_gazp is not None
        assert signal_mgnt is not None
        # GAZP proximity=0.8 -> confidence * (1 - 0.8*0.5) = confidence * 0.6
        # MGNT proximity=0.2 -> confidence * (1 - 0.2*0.5) = confidence * 0.9
        assert signal_gazp.confidence < signal_mgnt.confidence

    def test_sanctions_scoring_no_effect_on_non_russian_stock(self) -> None:
        """AAPL (not in sanctions dict) should have confidence unaffected."""
        strategy = EventDrivenStrategy()
        sentiment = 0.8
        base_confidence = min(1.0, abs(sentiment) * 1.0)  # credibility=1.0

        with patch.object(strategy, "get_parameters", return_value=_PARAMS_WITH_SANCTIONS):
            signal = strategy.generate_signal(
                "AAPL",
                _CANDLES,
                _SEGMENT_RU,
                sentiment_score=sentiment,
            )

        assert signal is not None
        # AAPL not in dict -> proximity=0.0 -> confidence * 1.0 = unchanged
        assert signal.confidence == base_confidence

    def test_sanctions_proximity_in_features(self) -> None:
        """Signal features should contain the sanctions_proximity key."""
        strategy = EventDrivenStrategy()

        with patch.object(strategy, "get_parameters", return_value=_PARAMS_WITH_SANCTIONS):
            signal = strategy.generate_signal(
                "GAZP",
                _CANDLES_RU,
                _SEGMENT_RU,
                sentiment_score=0.8,
            )

        assert signal is not None
        assert "sanctions_proximity" in signal.features
        assert signal.features["sanctions_proximity"] == 0.8  # noqa: PLR2004

    def test_sanctions_scoring_only_for_sanctions_event_types(self) -> None:
        """Segments without sanctions/geopolitical event_types should not apply scoring."""
        strategy = EventDrivenStrategy()
        sentiment = 0.8
        base_confidence = min(1.0, abs(sentiment) * 1.0)

        with patch.object(strategy, "get_parameters", return_value=_PARAMS_NO_SANCTIONS):
            signal = strategy.generate_signal(
                "GAZP",
                _CANDLES_RU,
                _SEGMENT,
                sentiment_score=sentiment,
            )

        assert signal is not None
        # No sanctions in event_types -> no scaling applied
        assert signal.confidence == base_confidence
        # sanctions_proximity should NOT be in features
        assert "sanctions_proximity" not in signal.features
