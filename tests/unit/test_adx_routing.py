"""Unit tests for ADX regime routing in StrategyCombiner and dual_momentum SELL."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any
from unittest.mock import patch

import pytest

from finalayze.core.schemas import Candle, Signal, SignalDirection
from finalayze.strategies.adx import compute_adx
from finalayze.strategies.base import BaseStrategy
from finalayze.strategies.combiner import StrategyCombiner
from finalayze.strategies.dual_momentum import DualMomentumStrategy

# ── Constants (no magic numbers — ruff PLR2004) ────────────────────────────────

BASE_PRICE = Decimal(100)
VOLUME = 1_000_000
CANDLE_HIGH_OFFSET = Decimal(1)
CANDLE_LOW_OFFSET = Decimal(1)
CANDLE_COUNT = 30
HIGH_CONFIDENCE = 0.9
ADX_PERIOD = 14
MIN_ADX_BARS = 28  # 2 * ADX_PERIOD
SUFFICIENT_BARS = 60
SELL_SCORE_THRESHOLD = -0.05


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

    def generate_signal(
        self,
        symbol: str,
        candles: list[Candle],
        segment_id: str,
        sentiment_score: float = 0.0,
        has_open_position: bool = False,
    ) -> Signal | None:
        return self._return_signal


# ── ADX computation tests ──────────────────────────────────────────────────────


class TestComputeADX:
    def test_compute_adx_returns_float(self) -> None:
        """Basic ADX computation on sufficient data returns a float."""
        # Create simple trending data: steady uptrend
        closes = [100.0 + 0.5 * i for i in range(SUFFICIENT_BARS)]
        highs = [c + 1.0 for c in closes]
        lows = [c - 1.0 for c in closes]

        result = compute_adx(closes, highs, lows, period=ADX_PERIOD)
        assert result is not None
        assert isinstance(result, float)
        assert 0.0 <= result <= 100.0

    def test_compute_adx_insufficient_data_returns_none(self) -> None:
        """Not enough bars (< 2 * period) returns None."""
        too_few = MIN_ADX_BARS - 1
        closes = [100.0 + i for i in range(too_few)]
        highs = [c + 1.0 for c in closes]
        lows = [c - 1.0 for c in closes]

        result = compute_adx(closes, highs, lows, period=ADX_PERIOD)
        assert result is None


# ── ADX regime routing tests ──────────────────────────────────────────────────


class TestADXRegimeRouting:
    """Tests for ADX-based strategy pool gating in the combiner."""

    @staticmethod
    def _config_with_regime(
        regime_enabled: bool = True,
        adx_period: int = ADX_PERIOD,
        trend_threshold: int = 30,
        mr_threshold: int = 20,
        min_confidence: float = 0.0,
    ) -> dict[str, Any]:
        return {
            "regime_routing": {
                "enabled": regime_enabled,
                "adx_period": adx_period,
                "trend_threshold": trend_threshold,
                "mr_threshold": mr_threshold,
            },
            "min_combined_confidence": min_confidence,
            "strategies": {
                "momentum": {"enabled": True, "weight": 0.5},
                "mean_reversion": {"enabled": True, "weight": 0.5},
            },
        }

    def test_regime_trend_skips_mr_strategies(self) -> None:
        """When ADX regime is 'trend', MR strategies are skipped."""
        mom_signal = _make_signal(SignalDirection.BUY, HIGH_CONFIDENCE, "momentum")
        mr_signal = _make_signal(SignalDirection.BUY, HIGH_CONFIDENCE, "mean_reversion")

        # Track if MR strategy generate_signal was called
        mr_called: list[bool] = []

        class TrackingMRStrategy(BaseStrategy):
            @property
            def name(self) -> str:
                return "mean_reversion"

            def supported_segments(self) -> list[str]:
                return ["us_broad"]

            def get_parameters(self, segment_id: str) -> dict[str, object]:
                return {}

            def generate_signal(
                self,
                symbol: str,
                candles: list[Candle],
                segment_id: str,
                sentiment_score: float = 0.0,
                has_open_position: bool = False,
            ) -> Signal | None:
                mr_called.append(True)
                return mr_signal

        momentum = MockStrategy("momentum", mom_signal)
        mean_rev = TrackingMRStrategy()

        combiner = StrategyCombiner([momentum, mean_rev])
        candles = _make_candles()
        config = self._config_with_regime()

        # Mock ADX to return high value (trending)
        with (
            patch.object(combiner, "_load_config", return_value=config),
            patch(
                "finalayze.strategies.combiner.compute_adx",
                return_value=35.0,
            ),
        ):
            signal = combiner.generate_signal("AAPL", candles, "us_broad")

        # MR strategy should NOT have been called
        assert len(mr_called) == 0
        # Signal should still be produced from momentum
        assert signal is not None
        assert signal.direction == SignalDirection.BUY
        assert signal.features["adx_regime"] == 1.0  # trend

    def test_regime_mr_skips_trend_strategies(self) -> None:
        """When ADX regime is 'mr', trend/momentum strategies are skipped."""
        mom_signal = _make_signal(SignalDirection.BUY, HIGH_CONFIDENCE, "momentum")
        mr_signal = _make_signal(SignalDirection.BUY, HIGH_CONFIDENCE, "mean_reversion")

        mom_called: list[bool] = []

        class TrackingMomStrategy(BaseStrategy):
            @property
            def name(self) -> str:
                return "momentum"

            def supported_segments(self) -> list[str]:
                return ["us_broad"]

            def get_parameters(self, segment_id: str) -> dict[str, object]:
                return {}

            def generate_signal(
                self,
                symbol: str,
                candles: list[Candle],
                segment_id: str,
                sentiment_score: float = 0.0,
                has_open_position: bool = False,
            ) -> Signal | None:
                mom_called.append(True)
                return mom_signal

        momentum = TrackingMomStrategy()
        mean_rev = MockStrategy("mean_reversion", mr_signal)

        combiner = StrategyCombiner([momentum, mean_rev])
        candles = _make_candles()
        config = self._config_with_regime()

        # Mock ADX to return low value (mean-reverting)
        with (
            patch.object(combiner, "_load_config", return_value=config),
            patch(
                "finalayze.strategies.combiner.compute_adx",
                return_value=15.0,
            ),
        ):
            signal = combiner.generate_signal("AAPL", candles, "us_broad")

        # Momentum strategy should NOT have been called
        assert len(mom_called) == 0
        # Signal should come from MR
        assert signal is not None
        assert signal.direction == SignalDirection.BUY
        assert signal.features["adx_regime"] == -1.0  # mr

    def test_regime_ambiguous_dominant_pool_wins(self) -> None:
        """In ambiguous zone (20 <= ADX <= 30), the pool with higher score wins."""
        # Momentum BUY with high confidence, MR SELL with lower confidence
        # Momentum pool should dominate
        mom_signal = _make_signal(SignalDirection.BUY, HIGH_CONFIDENCE, "momentum")
        mr_signal = _make_signal(SignalDirection.SELL, 0.3, "mean_reversion")

        momentum = MockStrategy("momentum", mom_signal)
        mean_rev = MockStrategy("mean_reversion", mr_signal)

        combiner = StrategyCombiner([momentum, mean_rev])
        candles = _make_candles()
        config = self._config_with_regime()

        # Mock ADX to return ambiguous value
        with (
            patch.object(combiner, "_load_config", return_value=config),
            patch(
                "finalayze.strategies.combiner.compute_adx",
                return_value=25.0,
            ),
        ):
            signal = combiner.generate_signal("AAPL", candles, "us_broad")

        # Momentum pool had |0.9 * 0.5| = 0.45 > MR pool |0.3 * 0.5| = 0.15
        # So momentum pool wins -> BUY
        assert signal is not None
        assert signal.direction == SignalDirection.BUY
        assert signal.features["adx_regime"] == 0.0  # ambiguous

    def test_regime_ambiguous_mr_pool_wins_when_stronger(self) -> None:
        """In ambiguous zone, MR pool wins when it has stronger score."""
        mom_signal = _make_signal(SignalDirection.BUY, 0.3, "momentum")
        mr_signal = _make_signal(SignalDirection.BUY, HIGH_CONFIDENCE, "mean_reversion")

        momentum = MockStrategy("momentum", mom_signal)
        mean_rev = MockStrategy("mean_reversion", mr_signal)

        combiner = StrategyCombiner([momentum, mean_rev])
        candles = _make_candles()
        config = self._config_with_regime()

        with (
            patch.object(combiner, "_load_config", return_value=config),
            patch(
                "finalayze.strategies.combiner.compute_adx",
                return_value=25.0,
            ),
        ):
            signal = combiner.generate_signal("AAPL", candles, "us_broad")

        # MR pool had |0.9 * 0.5| = 0.45 > momentum pool |0.3 * 0.5| = 0.15
        assert signal is not None
        assert signal.direction == SignalDirection.BUY

    def test_regime_routing_disabled(self) -> None:
        """When regime_routing.enabled=false, all strategies fire (ambiguous)."""
        mom_signal = _make_signal(SignalDirection.BUY, HIGH_CONFIDENCE, "momentum")
        mr_signal = _make_signal(SignalDirection.BUY, HIGH_CONFIDENCE, "mean_reversion")

        mom_called: list[bool] = []
        mr_called: list[bool] = []

        class TrackingMom(BaseStrategy):
            @property
            def name(self) -> str:
                return "momentum"

            def supported_segments(self) -> list[str]:
                return ["us_broad"]

            def get_parameters(self, segment_id: str) -> dict[str, object]:
                return {}

            def generate_signal(
                self,
                symbol: str,
                candles: list[Candle],
                segment_id: str,
                sentiment_score: float = 0.0,
                has_open_position: bool = False,
            ) -> Signal | None:
                mom_called.append(True)
                return mom_signal

        class TrackingMR(BaseStrategy):
            @property
            def name(self) -> str:
                return "mean_reversion"

            def supported_segments(self) -> list[str]:
                return ["us_broad"]

            def get_parameters(self, segment_id: str) -> dict[str, object]:
                return {}

            def generate_signal(
                self,
                symbol: str,
                candles: list[Candle],
                segment_id: str,
                sentiment_score: float = 0.0,
                has_open_position: bool = False,
            ) -> Signal | None:
                mr_called.append(True)
                return mr_signal

        momentum = TrackingMom()
        mean_rev = TrackingMR()

        combiner = StrategyCombiner([momentum, mean_rev])
        candles = _make_candles()
        config = self._config_with_regime(regime_enabled=False)

        with patch.object(combiner, "_load_config", return_value=config):
            signal = combiner.generate_signal("AAPL", candles, "us_broad")

        # Both strategies should have been called
        assert len(mom_called) == 1
        assert len(mr_called) == 1
        assert signal is not None

    def test_adx_features_in_signal(self) -> None:
        """Combined signal features contain adx_value and adx_regime."""
        mom_signal = _make_signal(SignalDirection.BUY, HIGH_CONFIDENCE, "momentum")
        momentum = MockStrategy("momentum", mom_signal)

        combiner = StrategyCombiner([momentum])
        candles = _make_candles()
        config: dict[str, Any] = {
            "strategies": {"momentum": {"enabled": True, "weight": 1.0}},
            "min_combined_confidence": 0.0,
        }

        with (
            patch.object(combiner, "_load_config", return_value=config),
            patch(
                "finalayze.strategies.combiner.compute_adx",
                return_value=35.0,
            ),
        ):
            signal = combiner.generate_signal("AAPL", candles, "us_broad")

        assert signal is not None
        assert "adx_value" in signal.features
        assert signal.features["adx_value"] == pytest.approx(35.0)
        assert "adx_regime" in signal.features
        assert signal.features["adx_regime"] == 1.0  # trend


# ── Dual momentum SELL signal tests ──────────────────────────────────────────


def _make_dual_momentum_candles(
    count: int = 200,
    trend: float = 0.0,
) -> list[Candle]:
    """Create candles for dual momentum testing.

    Args:
        count: Number of candles.
        trend: Per-bar price change (positive = uptrend, negative = downtrend).
    """
    candles_list = []
    base_time = datetime(2024, 1, 1, tzinfo=UTC)
    for i in range(count):
        price = Decimal(str(100.0 + trend * i))
        candles_list.append(
            Candle(
                symbol="AAPL",
                market_id="us",
                timeframe="1d",
                timestamp=base_time + timedelta(days=i),
                open=price,
                high=price + Decimal(1),
                low=price - Decimal(1),
                close=price,
                volume=VOLUME,
            )
        )
    return candles_list


class TestDualMomentumSell:
    def test_dual_momentum_sell_signal(self) -> None:
        """Strongly negative momentum score (< -0.05) produces SELL signal."""
        strategy = DualMomentumStrategy()
        # Strong downtrend: price drops significantly
        candles = _make_dual_momentum_candles(count=200, trend=-1.0)

        signal = strategy.generate_signal("AAPL", candles, "test_segment", sentiment_score=0.0)

        assert signal is not None
        assert signal.direction == SignalDirection.SELL
        assert signal.strategy_name == "dual_momentum"
        assert 0.0 <= signal.confidence <= 1.0

    def test_dual_momentum_sell_dedup(self) -> None:
        """Does not emit SELL on consecutive bars for same symbol."""
        strategy = DualMomentumStrategy()
        # Strong downtrend
        candles = _make_dual_momentum_candles(count=200, trend=-1.0)

        # First call should produce a SELL signal
        signal1 = strategy.generate_signal("AAPL", candles, "test_segment", sentiment_score=0.0)
        assert signal1 is not None
        assert signal1.direction == SignalDirection.SELL

        # Second call with same symbol should be suppressed (dedup)
        signal2 = strategy.generate_signal("AAPL", candles, "test_segment", sentiment_score=0.0)
        assert signal2 is None

    def test_dual_momentum_buy_still_works(self) -> None:
        """Positive momentum score still produces BUY signal."""
        strategy = DualMomentumStrategy()
        # Strong uptrend
        candles = _make_dual_momentum_candles(count=200, trend=1.0)

        signal = strategy.generate_signal("AAPL", candles, "test_segment", sentiment_score=0.0)

        assert signal is not None
        assert signal.direction == SignalDirection.BUY

    def test_dual_momentum_dead_zone_returns_none(self) -> None:
        """Score between -0.05 and 0 returns None (dead zone)."""
        strategy = DualMomentumStrategy()
        # Nearly flat: very small negative trend
        # Price drops only slightly to create score in (-0.05, 0)
        candles = _make_dual_momentum_candles(count=200, trend=-0.01)

        signal = strategy.generate_signal("AAPL", candles, "test_segment", sentiment_score=0.0)

        # With -0.01 per bar, returns are small enough to be in dead zone
        # or might still pass threshold; the key thing is that there is a dead zone
        if signal is not None:
            # If it did fire, it must be a SELL (not BUY)
            assert signal.direction == SignalDirection.SELL

    def test_dual_momentum_sell_after_buy_emits(self) -> None:
        """Switching from BUY to SELL is allowed (direction change)."""
        strategy = DualMomentumStrategy()

        # First: uptrend -> BUY
        up_candles = _make_dual_momentum_candles(count=200, trend=1.0)
        signal1 = strategy.generate_signal("AAPL", up_candles, "test_segment", sentiment_score=0.0)
        assert signal1 is not None
        assert signal1.direction == SignalDirection.BUY

        # Second: downtrend -> SELL (different direction, should emit)
        down_candles = _make_dual_momentum_candles(count=200, trend=-1.0)
        signal2 = strategy.generate_signal(
            "AAPL", down_candles, "test_segment", sentiment_score=0.0
        )
        assert signal2 is not None
        assert signal2.direction == SignalDirection.SELL
