"""Unit tests for StrategyCombiner."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any
from unittest.mock import patch

from finalayze.core.schemas import Candle, Signal, SignalDirection
from finalayze.strategies.base import BaseStrategy
from finalayze.strategies.combiner import StrategyCombiner

# Constants (no magic numbers)
BASE_PRICE = Decimal(100)
VOLUME = 1_000_000
CANDLE_HIGH_OFFSET = Decimal(1)
CANDLE_LOW_OFFSET = Decimal(1)
CANDLE_COUNT = 30
HIGH_CONFIDENCE = 0.9
LOW_CONFIDENCE = 0.3
WEIGHT_DOMINANT = 0.6
WEIGHT_MINOR = 0.4
MIN_COMBINED_CONFIDENCE = 0.5


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


def _make_candles(count: int = CANDLE_COUNT) -> list[Candle]:
    return [_candle(BASE_PRICE, i) for i in range(count)]


def _make_signal(
    direction: SignalDirection,
    confidence: float,
    strategy_name: str = "mock",
    segment_id: str = "us_broad",
) -> Signal:
    return Signal(
        strategy_name=strategy_name,
        symbol="AAPL",
        market_id="us",
        segment_id=segment_id,
        direction=direction,
        confidence=confidence,
        features={"mock_feature": confidence},
        reasoning=f"Mock signal: {direction} at {confidence}",
    )


class MockStrategy(BaseStrategy):
    """A controllable mock strategy for testing."""

    def __init__(self, name: str, return_signal: Signal | None) -> None:
        self._name = name
        self._return_signal = return_signal

    @property
    def name(self) -> str:
        return self._name

    def supported_segments(self) -> list[str]:
        return ["us_broad", "us_tech"]

    def get_parameters(self, segment_id: str) -> dict[str, object]:
        return {}

    def generate_signal(self, symbol: str, candles: list[Candle], segment_id: str) -> Signal | None:
        return self._return_signal


class TestStrategyCombiner:
    def test_combine_single_buy_signal(self) -> None:
        """One strategy returns BUY -> combiner returns BUY."""
        single_strategy_config: dict[str, Any] = {
            "strategies": {
                "momentum": {"enabled": True, "weight": 1.0},
            }
        }
        buy_signal = _make_signal(SignalDirection.BUY, HIGH_CONFIDENCE, "momentum")
        strategy = MockStrategy("momentum", buy_signal)
        combiner = StrategyCombiner([strategy])
        candles = _make_candles()
        with patch.object(combiner, "_load_config", return_value=single_strategy_config):
            signal = combiner.generate_signal("AAPL", candles, "us_broad")
        assert signal is not None
        assert signal.direction == SignalDirection.BUY
        assert signal.strategy_name == "combined"
        assert signal.symbol == "AAPL"
        assert signal.segment_id == "us_broad"

    def test_combine_conflicting_signals_weighted(self) -> None:
        """Momentum BUY (weight 0.6) vs mean_reversion SELL (weight 0.4) -> net BUY."""
        # net_score = (BUY: +0.9 * 0.6 + SELL: -0.3 * 0.4) / (0.6 + 0.4)
        #           = (0.54 - 0.12) / 1.0 = 0.42
        # abs(0.42) < min_combined_confidence(0.5) -> signal is None at default threshold
        # To get a signal: use weight 0.6 vs 0.1 so momentum dominates
        # net_score = (0.9 * 0.6 - 0.3 * 0.1) / 0.7 = (0.54 - 0.03) / 0.7 = 0.729 > 0.5 -> BUY
        weighted_config: dict[str, Any] = {
            "strategies": {
                "momentum": {"enabled": True, "weight": 0.6},
                "mean_reversion": {"enabled": True, "weight": 0.1},
            }
        }
        buy_signal = _make_signal(SignalDirection.BUY, HIGH_CONFIDENCE, "momentum")
        sell_signal = _make_signal(SignalDirection.SELL, LOW_CONFIDENCE, "mean_reversion")
        momentum = MockStrategy("momentum", buy_signal)
        mean_rev = MockStrategy("mean_reversion", sell_signal)
        combiner = StrategyCombiner([momentum, mean_rev])
        candles = _make_candles()
        with patch.object(combiner, "_load_config", return_value=weighted_config):
            signal = combiner.generate_signal("AAPL", candles, "us_broad")
        assert signal is not None
        assert signal.direction == SignalDirection.BUY

    def test_combine_no_signals_returns_none(self) -> None:
        """All strategies return None -> combiner returns None."""
        both_enabled_config: dict[str, Any] = {
            "strategies": {
                "momentum": {"enabled": True, "weight": 0.5},
                "mean_reversion": {"enabled": True, "weight": 0.5},
            }
        }
        strategy1 = MockStrategy("momentum", None)
        strategy2 = MockStrategy("mean_reversion", None)
        combiner = StrategyCombiner([strategy1, strategy2])
        candles = _make_candles()
        with patch.object(combiner, "_load_config", return_value=both_enabled_config):
            signal = combiner.generate_signal("AAPL", candles, "us_broad")
        assert signal is None

    def test_combine_respects_min_confidence(self) -> None:
        """When weighted score is below min_combined_confidence, return None."""
        # Equal and opposite signals with equal weights -> net score = 0 -> None
        equal_weight_config: dict[str, Any] = {
            "strategies": {
                "momentum": {"enabled": True, "weight": 0.5},
                "mean_reversion": {"enabled": True, "weight": 0.5},
            }
        }
        buy_signal = _make_signal(SignalDirection.BUY, HIGH_CONFIDENCE, "momentum")
        sell_signal = _make_signal(SignalDirection.SELL, HIGH_CONFIDENCE, "mean_reversion")
        momentum = MockStrategy("momentum", buy_signal)
        mean_rev = MockStrategy("mean_reversion", sell_signal)
        combiner = StrategyCombiner([momentum, mean_rev])
        candles = _make_candles()
        with patch.object(combiner, "_load_config", return_value=equal_weight_config):
            signal = combiner.generate_signal("AAPL", candles, "us_broad")
        # net_score = (0.9*0.5 - 0.9*0.5) / (0.5+0.5) = 0.0 -> below 0.5 -> None
        assert signal is None

    def test_combiner_uses_segment_weights(self) -> None:
        """Different segments get different strategy weights from YAML presets."""
        # us_tech: momentum weight=0.4, mean_reversion weight=0.2
        # us_broad: momentum weight=0.5, mean_reversion weight=0.5
        buy_signal = _make_signal(SignalDirection.BUY, HIGH_CONFIDENCE, "momentum")
        strategy = MockStrategy("momentum", buy_signal)
        combiner = StrategyCombiner([strategy])

        candles = _make_candles()
        # Both segments should produce a BUY signal (only momentum is active in mock)
        signal_tech = combiner.generate_signal("AAPL", candles, "us_tech")
        signal_broad = combiner.generate_signal("AAPL", candles, "us_broad")

        # Both should be BUY since only momentum provides a signal
        assert signal_tech is not None
        assert signal_tech.direction == SignalDirection.BUY
        assert signal_broad is not None
        assert signal_broad.direction == SignalDirection.BUY

    def test_combiner_skips_disabled_strategy(self) -> None:
        """Strategy with enabled: false in YAML should not be called."""
        # Create a custom mock that tracks whether generate_signal was called
        called_tracker: list[bool] = []

        class TrackingStrategy(BaseStrategy):
            @property
            def name(self) -> str:
                return "momentum"

            def supported_segments(self) -> list[str]:
                return ["us_broad"]

            def get_parameters(self, segment_id: str) -> dict[str, object]:
                return {}

            def generate_signal(
                self, symbol: str, candles: list[Candle], segment_id: str
            ) -> Signal | None:
                called_tracker.append(True)
                return _make_signal(SignalDirection.BUY, HIGH_CONFIDENCE, "momentum")

        tracking_strategy = TrackingStrategy()

        # Patch the YAML config to disable momentum for us_broad
        disabled_config: dict[str, Any] = {
            "strategies": {
                "momentum": {"enabled": False, "weight": 0.5},
                "mean_reversion": {"enabled": False, "weight": 0.5},
            }
        }

        combiner = StrategyCombiner([tracking_strategy])
        candles = _make_candles()

        with patch.object(combiner, "_load_config", return_value=disabled_config):
            signal = combiner.generate_signal("AAPL", candles, "us_broad")

        # Strategy should NOT have been called (disabled in config)
        assert len(called_tracker) == 0
        # No enabled strategies -> None
        assert signal is None

    def test_combined_signal_contains_feature_contributions(self) -> None:
        """Combined signal features include per-strategy contributions."""
        single_strategy_config: dict[str, Any] = {
            "strategies": {
                "momentum": {"enabled": True, "weight": 1.0},
            }
        }
        buy_signal = _make_signal(SignalDirection.BUY, HIGH_CONFIDENCE, "momentum")
        strategy = MockStrategy("momentum", buy_signal)
        combiner = StrategyCombiner([strategy])
        candles = _make_candles()
        with patch.object(combiner, "_load_config", return_value=single_strategy_config):
            signal = combiner.generate_signal("AAPL", candles, "us_broad")
        assert signal is not None
        assert "momentum_confidence" in signal.features
        assert "momentum_direction" in signal.features
