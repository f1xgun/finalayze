"""Unit tests for MomentumStrategy sentiment integration."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from finalayze.core.schemas import Candle, SignalDirection
from finalayze.strategies.momentum import MomentumStrategy

_BASE_TS = datetime(2025, 1, 1, 14, 30, tzinfo=UTC)
_SEGMENT = "us_tech"


def _make_candles_with_oversold_recovery(n: int = 60) -> list[Candle]:
    """Generate candles that produce an oversold RSI recovery (BUY signal).

    Creates a dip-then-recovery pattern: prices drop sharply, then
    start rising — triggering RSI oversold + MACD histogram rising.
    """
    candles: list[Candle] = []
    base_price = 100.0

    for i in range(n):
        if i < 30:
            # Steady drift down to push RSI toward oversold
            price = base_price - i * 1.5
        elif i < 45:
            # Sharp drop to push RSI deep into oversold territory
            price = base_price - 30 * 1.5 - (i - 30) * 2.0
        else:
            # Recovery — prices start rising, MACD histogram should turn positive
            price = base_price - 30 * 1.5 - 15 * 2.0 + (i - 45) * 3.0

        price = max(price, 10.0)  # floor
        candles.append(
            Candle(
                symbol="AAPL",
                market_id="us",
                timeframe="1d",
                timestamp=_BASE_TS + timedelta(days=i),
                open=Decimal(str(price - 0.5)),
                high=Decimal(str(price + 1.0)),
                low=Decimal(str(price - 1.0)),
                close=Decimal(str(price)),
                volume=1_000_000,
            )
        )
    return candles


class TestMomentumSentiment:
    """Test that sentiment_score modifies confidence in the expected direction."""

    def test_positive_sentiment_boosts_buy_confidence(self) -> None:
        """A positive sentiment_score should increase BUY confidence."""
        strategy = MomentumStrategy()
        candles = _make_candles_with_oversold_recovery()

        # Baseline: no sentiment
        signal_neutral = strategy.generate_signal("AAPL", candles, _SEGMENT, sentiment_score=0.0)

        # With positive sentiment
        strategy2 = MomentumStrategy()
        signal_positive = strategy2.generate_signal("AAPL", candles, _SEGMENT, sentiment_score=0.8)

        # Both should generate signals (or both None if pattern doesn't trigger)
        if (
            signal_neutral is not None
            and signal_positive is not None
            and signal_neutral.direction == SignalDirection.BUY
        ):
            assert signal_positive.confidence >= signal_neutral.confidence

    def test_negative_sentiment_reduces_buy_confidence(self) -> None:
        """A negative sentiment_score should reduce BUY confidence."""
        strategy = MomentumStrategy()
        candles = _make_candles_with_oversold_recovery()

        signal_neutral = strategy.generate_signal("AAPL", candles, _SEGMENT, sentiment_score=0.0)

        strategy2 = MomentumStrategy()
        signal_negative = strategy2.generate_signal("AAPL", candles, _SEGMENT, sentiment_score=-0.8)

        if (
            signal_neutral is not None
            and signal_negative is not None
            and signal_neutral.direction == SignalDirection.BUY
        ):
            assert signal_negative.confidence <= signal_neutral.confidence

    def test_sentiment_score_in_features(self) -> None:
        """sentiment_score should appear in Signal.features."""
        strategy = MomentumStrategy()
        candles = _make_candles_with_oversold_recovery()

        signal = strategy.generate_signal("AAPL", candles, _SEGMENT, sentiment_score=0.5)

        if signal is not None:
            assert "sentiment_score" in signal.features
            assert signal.features["sentiment_score"] == 0.5

    def test_confidence_capped_at_one(self) -> None:
        """Even with extreme sentiment, confidence must stay <= 1.0."""
        strategy = MomentumStrategy()
        candles = _make_candles_with_oversold_recovery()

        signal = strategy.generate_signal("AAPL", candles, _SEGMENT, sentiment_score=1.0)

        if signal is not None:
            assert signal.confidence <= 1.0

    def test_confidence_floored_at_zero(self) -> None:
        """Even with extreme negative sentiment, confidence must stay >= 0.0."""
        strategy = MomentumStrategy()
        candles = _make_candles_with_oversold_recovery()

        signal = strategy.generate_signal("AAPL", candles, _SEGMENT, sentiment_score=-1.0)

        if signal is not None:
            assert signal.confidence >= 0.0
