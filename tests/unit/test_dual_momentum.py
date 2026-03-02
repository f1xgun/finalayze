"""Unit tests for dual momentum strategy."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from finalayze.core.schemas import Candle, SignalDirection
from finalayze.strategies.dual_momentum import DualMomentumStrategy

_MIN_CANDLES = 126
_WEIGHT_1M = 0.4
_WEIGHT_3M = 0.3
_WEIGHT_6M = 0.3
_CONFIDENCE_BASE = 0.4
_CONFIDENCE_SCALE = 1.0
_MAX_CONFIDENCE = 0.95


def _make_candles(
    prices: list[float],
    symbol: str = "AAPL",
    market_id: str = "us",
) -> list[Candle]:
    """Build candles from a list of close prices."""
    base = datetime(2024, 1, 1, 14, 30, tzinfo=UTC)
    candles = []
    for i, price in enumerate(prices):
        p = Decimal(str(price))
        candles.append(
            Candle(
                symbol=symbol,
                market_id=market_id,
                timeframe="1d",
                timestamp=base + timedelta(days=i),
                open=p,
                high=p + Decimal(1),
                low=p - Decimal(1),
                close=p,
                volume=1_000_000,
            )
        )
    return candles


class TestDualMomentum:
    """Tests for DualMomentumStrategy."""

    def test_dual_momentum_buy_signal(self) -> None:
        """Positive momentum score produces a BUY signal."""
        # Steadily rising prices over 126+ bars
        prices = [100.0 + i * 0.5 for i in range(130)]
        candles = _make_candles(prices)

        strategy = DualMomentumStrategy()
        signal = strategy.generate_signal("AAPL", candles, "us_tech")

        assert signal is not None
        assert signal.direction == SignalDirection.BUY
        assert signal.strategy_name == "dual_momentum"
        assert 0.0 <= signal.confidence <= 1.0

    def test_dual_momentum_absolute_gate(self) -> None:
        """Score <= 0 returns None (absolute momentum gate)."""
        # Steadily declining prices
        prices = [200.0 - i * 0.5 for i in range(130)]
        candles = _make_candles(prices)

        strategy = DualMomentumStrategy()
        signal = strategy.generate_signal("AAPL", candles, "us_tech")

        assert signal is None

    def test_dual_momentum_confidence(self) -> None:
        """Verify confidence formula: min(0.95, 0.4 + abs(score) * 1.0)."""
        # Build prices where we can calculate exact score
        n = 130
        base_price = 100.0

        # For score=0.1: confidence should be min(0.95, 0.4 + 0.1*1.0) = 0.5
        # For score=0.5: confidence should be min(0.95, 0.4 + 0.5*1.0) = 0.9

        # Create upward trend to get known returns
        # ret_1m: (p[-1] - p[-21]) / p[-21]
        # ret_3m: (p[-1] - p[-63]) / p[-63]
        # ret_6m: (p[-1] - p[-126]) / p[-126]
        # score = ret_1m * 0.4 + ret_3m * 0.3 + ret_6m * 0.3

        # Simple linear growth: price[i] = 100 + i * growth
        growth = 0.5
        prices = [base_price + i * growth for i in range(n)]
        candles = _make_candles(prices)

        last = prices[-1]
        p_21 = prices[-21]
        p_63 = prices[-63]
        p_126 = prices[-126]

        ret_1m = (last - p_21) / p_21
        ret_3m = (last - p_63) / p_63
        ret_6m = (last - p_126) / p_126
        expected_score = ret_1m * _WEIGHT_1M + ret_3m * _WEIGHT_3M + ret_6m * _WEIGHT_6M
        expected_confidence = min(
            _MAX_CONFIDENCE, _CONFIDENCE_BASE + abs(expected_score) * _CONFIDENCE_SCALE
        )

        strategy = DualMomentumStrategy()
        signal = strategy.generate_signal("AAPL", candles, "us_tech")

        assert signal is not None
        assert abs(signal.confidence - expected_confidence) < 1e-6

    def test_dual_momentum_insufficient_data(self) -> None:
        """Less than 126 candles returns None."""
        prices = [100.0 + i * 0.5 for i in range(125)]
        candles = _make_candles(prices)

        strategy = DualMomentumStrategy()
        signal = strategy.generate_signal("AAPL", candles, "us_tech")

        assert signal is None

    def test_dual_momentum_weighted_scoring(self) -> None:
        """Verify the 40/30/30 weighting is applied correctly."""
        n = 130
        # Create prices where we know exact returns
        prices = [100.0] * n
        # Make specific changes at the lookback points
        prices[-1] = 120.0  # current
        prices[-21] = 100.0  # 1m ago
        prices[-63] = 110.0  # 3m ago
        prices[-126] = 105.0  # 6m ago

        candles = _make_candles(prices)
        strategy = DualMomentumStrategy()
        signal = strategy.generate_signal("AAPL", candles, "us_tech")

        ret_1m = (120.0 - 100.0) / 100.0  # 0.2
        ret_3m = (120.0 - 110.0) / 110.0  # ~0.0909
        ret_6m = (120.0 - 105.0) / 105.0  # ~0.1429

        expected_score = ret_1m * 0.4 + ret_3m * 0.3 + ret_6m * 0.3

        assert signal is not None
        features = signal.features
        assert abs(features["score_1m"] - ret_1m) < 1e-6
        assert abs(features["score_3m"] - ret_3m) < 1e-6
        assert abs(features["score_6m"] - ret_6m) < 1e-6

        expected_confidence = min(0.95, 0.4 + abs(expected_score) * 1.0)
        assert abs(signal.confidence - expected_confidence) < 1e-6

    def test_dual_momentum_supported_segments(self) -> None:
        """All standard segments should be supported."""
        strategy = DualMomentumStrategy()
        segments = strategy.supported_segments()
        assert "us_tech" in segments
        assert "ru_blue_chips" in segments

    def test_dual_momentum_name(self) -> None:
        """Strategy name is 'dual_momentum'."""
        strategy = DualMomentumStrategy()
        assert strategy.name == "dual_momentum"
