"""Unit tests for Hierarchical Risk Parity (HRP) strategy allocation."""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any
from unittest.mock import patch

import pytest

from finalayze.core.schemas import Candle, Signal, SignalDirection
from finalayze.strategies.base import BaseStrategy
from finalayze.strategies.hrp import compute_hrp_weights

# ---------------------------------------------------------------------------
# Constants (no magic numbers — ruff PLR2004)
# ---------------------------------------------------------------------------
WEIGHT_TOLERANCE = 0.15
EQUAL_WEIGHT_TWO = 0.5
SINGLE_WEIGHT = 1.0
MIN_HISTORY = 20
NUM_STEPS = 60
LOW_VOL = 0.01
HIGH_VOL = 0.10
BASE_PRICE = Decimal(100)
CANDLE_HIGH_OFFSET = Decimal(1)
CANDLE_LOW_OFFSET = Decimal(1)
VOLUME = 1_000_000
HIGH_CONFIDENCE = 0.9
CANDLE_COUNT = 30


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_uncorrelated_returns(n_steps: int, n_strategies: int, vol: float) -> list[list[float]]:
    """Generate pseudo-random uncorrelated return series using a simple LCG.

    Deterministic so tests are reproducible — no numpy.random needed.
    """
    seed = 42
    modulus = 2**31 - 1
    multiplier = 1103515245
    increment = 12345

    result: list[list[float]] = []
    for _ in range(n_strategies):
        series: list[float] = []
        for _ in range(n_steps):
            seed = (multiplier * seed + increment) % modulus
            # Map to roughly standard-normal-ish via simple transform
            uniform = seed / modulus
            normal_approx = (uniform - 0.5) * 2  # range [-1, 1]
            series.append(normal_approx * vol)
        result.append(series)
    return result


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
        strategy_payload={"mock_feature": confidence},
        reasoning=f"Mock signal: {direction} at {confidence}",
    )


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


# ---------------------------------------------------------------------------
# Tests for compute_hrp_weights
# ---------------------------------------------------------------------------
class TestComputeHRPWeights:
    def test_weights_sum_to_one(self) -> None:
        """HRP weights must sum to 1.0 for any valid input."""
        returns = _make_uncorrelated_returns(NUM_STEPS, 3, LOW_VOL)
        names = ["alpha", "beta", "gamma"]
        weights = compute_hrp_weights(returns, names)
        assert len(weights) == 3
        assert sum(weights.values()) == pytest.approx(SINGLE_WEIGHT, abs=1e-9)

    def test_hrp_equal_vol(self) -> None:
        """Two uncorrelated series with equal volatility get roughly equal weight."""
        # Use same vol for both; different seeds via offset
        seed_a, seed_b = 42, 99
        modulus = 2**31 - 1
        multiplier = 1103515245
        increment = 12345
        vol = LOW_VOL

        series: list[list[float]] = []
        for initial_seed in (seed_a, seed_b):
            row: list[float] = []
            current = initial_seed
            for _ in range(NUM_STEPS):
                current = (multiplier * current + increment) % modulus
                uniform = current / modulus
                row.append((uniform - 0.5) * 2 * vol)
            series.append(row)

        weights = compute_hrp_weights(series, ["A", "B"])
        assert weights["A"] == pytest.approx(EQUAL_WEIGHT_TWO, abs=WEIGHT_TOLERANCE)
        assert weights["B"] == pytest.approx(EQUAL_WEIGHT_TWO, abs=WEIGHT_TOLERANCE)

    def test_hrp_favors_low_vol(self) -> None:
        """Lower volatility strategy should receive a higher weight."""
        low_vol_series = [LOW_VOL * ((i % 3) - 1) for i in range(NUM_STEPS)]
        high_vol_series = [HIGH_VOL * ((i % 3) - 1) for i in range(NUM_STEPS)]
        returns = [low_vol_series, high_vol_series]
        names = ["low_vol", "high_vol"]
        weights = compute_hrp_weights(returns, names)
        assert weights["low_vol"] > weights["high_vol"]

    def test_hrp_insufficient_data(self) -> None:
        """With fewer than 20 time steps, return equal weights."""
        short_returns = [[0.01] * 10, [0.02] * 10]
        names = ["A", "B"]
        weights = compute_hrp_weights(short_returns, names)
        assert weights["A"] == pytest.approx(EQUAL_WEIGHT_TWO)
        assert weights["B"] == pytest.approx(EQUAL_WEIGHT_TWO)

    def test_hrp_single_strategy(self) -> None:
        """Single strategy gets weight 1.0."""
        returns = [[0.01] * NUM_STEPS]
        names = ["only"]
        weights = compute_hrp_weights(returns, names)
        assert weights["only"] == pytest.approx(SINGLE_WEIGHT)

    def test_hrp_unequal_row_lengths_truncated_to_min(self) -> None:
        """H-1: rows with different lengths are truncated to the minimum before HRP."""
        # Row A has NUM_STEPS points; row B has NUM_STEPS + 10 extra points.
        # The function must truncate B and still return valid weights summing to 1.
        row_a = _make_uncorrelated_returns(NUM_STEPS, 1, LOW_VOL)[0]
        row_b_long = _make_uncorrelated_returns(NUM_STEPS + 10, 1, LOW_VOL)[0]
        returns = [row_a, row_b_long]
        names = ["short", "long"]
        # Must not raise and must return weights summing to 1
        weights = compute_hrp_weights(returns, names)
        assert set(weights.keys()) == {"short", "long"}
        assert sum(weights.values()) == pytest.approx(SINGLE_WEIGHT, abs=1e-9)


# ---------------------------------------------------------------------------
# Tests for StrategyCombiner HRP integration
# ---------------------------------------------------------------------------
class TestCombinerHRPIntegration:
    def test_combiner_hrp_mode_uses_hrp_weights_after_enough_history(self) -> None:
        """In HRP mode, after recording 20+ returns, weights come from HRP."""
        from finalayze.strategies.combiner import StrategyCombiner

        buy_signal = _make_signal(SignalDirection.BUY, HIGH_CONFIDENCE, "momentum")
        strategy = MockStrategy("momentum", buy_signal)
        combiner = StrategyCombiner([strategy], allocation_mode="hrp")

        # Record enough history for two strategies
        for i in range(MIN_HISTORY):
            combiner.record_strategy_return("momentum", LOW_VOL * ((i % 3) - 1))
            combiner.record_strategy_return("mean_reversion", HIGH_VOL * ((i % 3) - 1))

        # The combiner should now have HRP weights available
        assert combiner._has_hrp_weights()

    def test_combiner_hrp_mode_falls_back_before_enough_history(self) -> None:
        """In HRP mode, before 20 returns, static YAML weights are used."""
        from finalayze.strategies.combiner import StrategyCombiner

        buy_signal = _make_signal(SignalDirection.BUY, HIGH_CONFIDENCE, "momentum")
        strategy = MockStrategy("momentum", buy_signal)
        combiner = StrategyCombiner([strategy], allocation_mode="hrp")

        # Record fewer than MIN_HISTORY returns
        for i in range(10):
            combiner.record_strategy_return("momentum", LOW_VOL * ((i % 3) - 1))

        assert not combiner._has_hrp_weights()

    def test_combiner_static_mode_ignores_recorded_returns(self) -> None:
        """In static mode (default), record_strategy_return is a no-op."""
        from finalayze.strategies.combiner import StrategyCombiner

        buy_signal = _make_signal(SignalDirection.BUY, HIGH_CONFIDENCE, "momentum")
        strategy = MockStrategy("momentum", buy_signal)
        combiner = StrategyCombiner([strategy], allocation_mode="static")

        for i in range(MIN_HISTORY + 5):
            combiner.record_strategy_return("momentum", LOW_VOL * ((i % 3) - 1))

        assert not combiner._has_hrp_weights()

    def test_combiner_hrp_weights_override_yaml(self) -> None:
        """HRP-computed weights replace YAML weights in generate_signal."""
        from finalayze.strategies.combiner import StrategyCombiner

        config: dict[str, Any] = {
            "strategies": {
                "momentum": {"enabled": True, "weight": 0.5},
                "mean_reversion": {"enabled": True, "weight": 0.5},
            },
            "min_combined_confidence": 0.0,
        }

        mom_signal = _make_signal(SignalDirection.BUY, HIGH_CONFIDENCE, "momentum")
        mr_signal = _make_signal(SignalDirection.BUY, HIGH_CONFIDENCE, "mean_reversion")
        momentum = MockStrategy("momentum", mom_signal)
        mean_rev = MockStrategy("mean_reversion", mr_signal)

        combiner = StrategyCombiner([momentum, mean_rev], allocation_mode="hrp")

        # Feed low vol for momentum, high vol for mean_reversion
        for i in range(MIN_HISTORY + 5):
            combiner.record_strategy_return("momentum", LOW_VOL * ((i % 3) - 1))
            combiner.record_strategy_return("mean_reversion", HIGH_VOL * ((i % 3) - 1))

        candles = _make_candles()
        with patch.object(combiner, "_load_config", return_value=config):
            signal = combiner.generate_signal("AAPL", candles, "us_broad")

        assert signal is not None
        # HRP should give momentum higher weight (lower vol)
        assert signal.strategy_payload.get("hrp_weight_momentum", 0) > signal.strategy_payload.get(
            "hrp_weight_mean_reversion", 0
        )
