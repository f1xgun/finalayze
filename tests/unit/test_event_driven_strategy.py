"""Unit tests for EventDrivenStrategy."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

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
        # Phase 7 Fix 3: event_driven disabled in all presets
        assert _SEGMENT not in segments
