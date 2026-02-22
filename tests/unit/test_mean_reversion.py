"""Unit tests for MeanReversionStrategy (Bollinger Bands)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from finalayze.core.schemas import Candle, SignalDirection
from finalayze.strategies.mean_reversion import MeanReversionStrategy

# Constants (no magic numbers)
BASE_PRICE = Decimal("100")
BAND_PERIOD = 20
VOLUME = 1_000_000
STABLE_HIGH_PRICE = Decimal("200")
CRASH_PRICE = Decimal("50")
SPIKE_PRICE = Decimal("300")
CANDLE_HIGH_OFFSET = Decimal("1")
CANDLE_LOW_OFFSET = Decimal("1")
STABLE_COUNT = 25
EXTRA_CANDLE_COUNT = 5
MIN_CANDLES_INSUFFICIENT = 5


def _candle(price: Decimal, day: int) -> Candle:
    return Candle(
        symbol="AAPL",
        market_id="us",
        timeframe="1d",
        timestamp=datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=day),
        open=price,
        high=price + CANDLE_HIGH_OFFSET,
        low=price - CANDLE_LOW_OFFSET,
        close=price,
        volume=VOLUME,
    )


def _make_stable_candles(price: Decimal, count: int, start_day: int = 0) -> list[Candle]:
    return [_candle(price, start_day + i) for i in range(count)]


class TestMeanReversionStrategy:
    def test_name_returns_string(self) -> None:
        strategy = MeanReversionStrategy()
        assert strategy.name == "mean_reversion"
        assert isinstance(strategy.name, str)

    def test_supported_segments_includes_us_tech(self) -> None:
        strategy = MeanReversionStrategy()
        segments = strategy.supported_segments()
        assert "us_tech" in segments
        assert isinstance(segments, list)

    def test_get_parameters_returns_dict(self) -> None:
        strategy = MeanReversionStrategy()
        params = strategy.get_parameters("us_tech")
        assert isinstance(params, dict)
        assert "bb_period" in params
        assert "bb_std_dev" in params
        assert "min_confidence" in params

    def test_get_parameters_unknown_segment_returns_defaults(self) -> None:
        strategy = MeanReversionStrategy()
        params = strategy.get_parameters("nonexistent_segment_xyz")
        # Unknown segment returns empty dict (no params)
        assert isinstance(params, dict)
        assert len(params) == 0

    def test_no_signal_with_insufficient_candles(self) -> None:
        strategy = MeanReversionStrategy()
        # Less than bb_period + 1 candles -> return None
        candles = [_candle(BASE_PRICE, i) for i in range(MIN_CANDLES_INSUFFICIENT)]
        signal = strategy.generate_signal("AAPL", candles, "us_tech")
        assert signal is None

    def test_buy_signal_when_price_below_lower_band(self) -> None:
        strategy = MeanReversionStrategy()
        # Stable price at high level to establish BB, then sudden crash below lower band
        candles = _make_stable_candles(STABLE_HIGH_PRICE, STABLE_COUNT)
        # Add a final candle that crashes far below the lower band
        crash_candle = _candle(CRASH_PRICE, STABLE_COUNT)
        candles.append(crash_candle)
        signal = strategy.generate_signal("AAPL", candles, "us_tech")
        assert signal is not None, "Expected BUY signal when price crashes below lower Bollinger Band"
        assert signal.direction == SignalDirection.BUY

    def test_no_signal_when_price_near_midline(self) -> None:
        strategy = MeanReversionStrategy()
        # Flat price exactly at middle band -> no signal
        candles = _make_stable_candles(BASE_PRICE, STABLE_COUNT + EXTRA_CANDLE_COUNT)
        signal = strategy.generate_signal("AAPL", candles, "us_tech")
        assert signal is None

    def test_sell_signal_when_price_above_upper_band(self) -> None:
        strategy = MeanReversionStrategy()
        # Stable price at low level to establish BB, then sudden spike above upper band
        candles = _make_stable_candles(BASE_PRICE, STABLE_COUNT)
        # Add a final candle that spikes far above the upper band
        spike_candle = _candle(SPIKE_PRICE, STABLE_COUNT)
        candles.append(spike_candle)
        signal = strategy.generate_signal("AAPL", candles, "us_tech")
        assert signal is not None, "Expected SELL signal when price spikes above upper Bollinger Band"
        assert signal.direction == SignalDirection.SELL

    def test_generate_signal_returns_correct_metadata(self) -> None:
        strategy = MeanReversionStrategy()
        # Stable price then crash to trigger BUY
        candles = _make_stable_candles(STABLE_HIGH_PRICE, STABLE_COUNT)
        candles.append(_candle(CRASH_PRICE, STABLE_COUNT))
        signal = strategy.generate_signal("AAPL", candles, "us_tech")
        assert signal is not None
        assert signal.symbol == "AAPL"
        assert signal.market_id == "us"
        assert signal.segment_id == "us_tech"
        assert signal.strategy_name == "mean_reversion"
        assert 0.0 <= signal.confidence <= 1.0
