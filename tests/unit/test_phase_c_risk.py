"""Tests for Phase C risk/portfolio features.

Covers: Adaptive Combiner, Turnover Budget, Drawdown Monitor.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from finalayze.core.schemas import Candle, Signal, SignalDirection
from finalayze.risk.drawdown_monitor import DrawdownMonitor
from finalayze.risk.turnover_budget import TurnoverBudget
from finalayze.strategies.adaptive_combiner import AdaptiveStrategyCombiner
from finalayze.strategies.base import BaseStrategy
from finalayze.strategies.combiner import StrategyCombiner

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE_TS = datetime(2024, 6, 1, 14, 30, tzinfo=UTC)
_BUY_CONFIDENCE = 0.8
_SELL_CONFIDENCE = 0.7


class _StubStrategy(BaseStrategy):
    """Minimal strategy that returns a fixed signal for testing."""

    def __init__(
        self,
        strategy_name: str,
        direction: SignalDirection = SignalDirection.BUY,
        confidence: float = _BUY_CONFIDENCE,
    ) -> None:
        self._name = strategy_name
        self._direction = direction
        self._confidence = confidence

    @property
    def name(self) -> str:
        return self._name

    def supported_segments(self) -> list[str]:
        return ["us_tech"]

    def generate_signal(
        self,
        symbol: str,
        candles: list[Candle],
        segment_id: str,
        sentiment_score: float = 0.0,
        has_open_position: bool = False,
    ) -> Signal | None:
        return Signal(
            strategy_name=self._name,
            symbol=symbol,
            market_id=candles[0].market_id,
            segment_id=segment_id,
            direction=self._direction,
            confidence=self._confidence,
            features={},
            reasoning="stub",
        )

    def get_parameters(self, segment_id: str) -> dict[str, object]:
        return {}


def _make_candles(n: int = 1, symbol: str = "AAPL") -> list[Candle]:
    """Create a minimal list of candles for testing."""
    return [
        Candle(
            symbol=symbol,
            market_id="us",
            timeframe="1d",
            timestamp=_BASE_TS + timedelta(days=i),
            open=Decimal(100),
            high=Decimal(101),
            low=Decimal(99),
            close=Decimal(100),
            volume=1_000_000,
        )
        for i in range(n)
    ]


# ===========================================================================
# Adaptive Combiner tests
# ===========================================================================


class TestAdaptiveCombiner:
    """Tests for AdaptiveStrategyCombiner."""

    def _make_combiner(
        self,
        strategies: list[BaseStrategy] | None = None,
    ) -> AdaptiveStrategyCombiner:
        if strategies is None:
            strategies = [
                _StubStrategy("momentum"),
                _StubStrategy("mean_reversion"),
            ]
        combiner = AdaptiveStrategyCombiner(strategies, segment_id="us_tech")
        # Point presets dir to a temp location so YAML loading returns {}
        combiner._presets_dir = combiner._presets_dir / "__nonexistent__"
        return combiner

    def test_adaptive_weights_applied(self) -> None:
        """Dynamic weights should override YAML weights after rebalance."""
        combiner = self._make_combiner()

        # Record good but varied returns for momentum, poor for mean_reversion
        good_returns = [0.03, 0.05, 0.04, 0.06, 0.02, 0.07, 0.03, 0.05, 0.04, 0.06]
        for ret in good_returns:
            combiner.record_trade_result("momentum", ret)
            combiner.record_trade_result("mean_reversion", -0.02)

        combiner._recompute_weights()
        assert combiner._dynamic_weights  # weights were computed
        # Momentum should get higher weight than mean_reversion
        assert combiner._dynamic_weights["momentum"] > combiner._dynamic_weights.get(
            "mean_reversion", Decimal(0)
        )

    def test_paused_strategy_minimum_weight(self) -> None:
        """Strategy with 0 Sharpe should retain at least 5% weight."""
        combiner = self._make_combiner()

        for _ in range(10):
            combiner.record_trade_result("momentum", 0.05)
            # mean_reversion returns that produce ~0 Sharpe
            combiner.record_trade_result("mean_reversion", -0.05)

        combiner._recompute_weights()
        assert combiner._dynamic_weights
        min_weight = Decimal("0.05")
        mr_weight = combiner._dynamic_weights.get("mean_reversion", Decimal(0))
        # mean_reversion has 0 Sharpe (clamped) so gets the floor
        assert mr_weight >= min_weight

    def test_rebalance_frequency(self) -> None:
        """Weights should update every 21 bars."""
        combiner = self._make_combiner()
        candles = _make_candles(1)

        for _ in range(10):
            combiner.record_trade_result("momentum", 0.05)
            combiner.record_trade_result("mean_reversion", 0.01)

        # Generate 20 signals -- should NOT rebalance yet (bar 1..20)
        for _ in range(20):
            combiner.generate_signal("AAPL", candles, "us_tech")
        assert combiner._dynamic_weights == {}

        # 21st bar triggers rebalance
        combiner.generate_signal("AAPL", candles, "us_tech")
        assert combiner._dynamic_weights != {}


class TestCombinerWeightOverrides:
    """Tests that the parent StrategyCombiner accepts weight_overrides."""

    def test_weight_overrides_in_combiner(self, tmp_path: object) -> None:
        """Parent combiner should use weight_overrides when provided."""
        strat_a = _StubStrategy("strat_a", confidence=0.9)
        strat_b = _StubStrategy("strat_b", confidence=0.9)

        combiner = StrategyCombiner([strat_a, strat_b])
        combiner._presets_dir = combiner._presets_dir / "__nonexistent__"

        candles = _make_candles(1)

        # Without overrides -- no YAML config means no strategies_cfg, returns None
        result_no_cfg = combiner.generate_signal("AAPL", candles, "us_tech")
        assert result_no_cfg is None  # no YAML = no strategy configs

        # Now supply a minimal YAML-like config by patching _load_config
        original_load = combiner._load_config

        def _patched_config(seg_id: str) -> dict[str, object]:
            return {
                "strategies": {
                    "strat_a": {"enabled": True, "weight": "0.5"},
                    "strat_b": {"enabled": True, "weight": "0.5"},
                },
            }

        combiner._load_config = _patched_config  # type: ignore[assignment]

        # With weight_overrides -- strat_a gets 90%, strat_b gets 10%
        overrides = {
            "strat_a": Decimal("0.9"),
            "strat_b": Decimal("0.1"),
        }
        result = combiner.generate_signal("AAPL", candles, "us_tech", weight_overrides=overrides)
        assert result is not None
        assert result.direction == SignalDirection.BUY

        combiner._load_config = original_load  # type: ignore[assignment]


# ===========================================================================
# Turnover Budget tests
# ===========================================================================

_JAN_15 = datetime(2024, 1, 15, 10, 0, tzinfo=UTC)
_JAN_20 = datetime(2024, 1, 20, 10, 0, tzinfo=UTC)
_JAN_25 = datetime(2024, 1, 25, 10, 0, tzinfo=UTC)
_FEB_01 = datetime(2024, 2, 1, 10, 0, tzinfo=UTC)


class TestTurnoverBudget:
    """Tests for TurnoverBudget."""

    def test_turnover_allows_first_trade(self) -> None:
        """First trade for a symbol should always be allowed."""
        budget = TurnoverBudget()
        assert budget.can_trade("AAPL", _JAN_15) is True

    def test_turnover_blocks_after_2(self) -> None:
        """Third round-trip in the same month should be blocked."""
        budget = TurnoverBudget(max_round_trips=2)
        budget.record_round_trip("AAPL", _JAN_15)
        assert budget.can_trade("AAPL", _JAN_20) is True

        budget.record_round_trip("AAPL", _JAN_20)
        assert budget.can_trade("AAPL", _JAN_25) is False

    def test_turnover_resets_monthly(self) -> None:
        """New month should reset the counter, allowing trades again."""
        budget = TurnoverBudget(max_round_trips=2)
        budget.record_round_trip("AAPL", _JAN_15)
        budget.record_round_trip("AAPL", _JAN_20)
        assert budget.can_trade("AAPL", _JAN_25) is False

        # February -- fresh month
        assert budget.can_trade("AAPL", _FEB_01) is True

    def test_turnover_independent_per_symbol(self) -> None:
        """Different symbols should have independent budgets."""
        budget = TurnoverBudget(max_round_trips=2)
        budget.record_round_trip("AAPL", _JAN_15)
        budget.record_round_trip("AAPL", _JAN_20)

        # AAPL is exhausted
        assert budget.can_trade("AAPL", _JAN_25) is False
        # GOOG is untouched
        assert budget.can_trade("GOOG", _JAN_25) is True


# ===========================================================================
# Drawdown Monitor tests
# ===========================================================================


class TestDrawdownMonitor:
    """Tests for DrawdownMonitor."""

    def test_drawdown_triggers_at_12pct(self) -> None:
        """12% drop from peak should trigger."""
        monitor = DrawdownMonitor()
        monitor.update(Decimal(100_000))
        triggered = monitor.update(Decimal(88_000))
        assert triggered is True
        assert monitor.triggered is True

    def test_drawdown_not_triggered_at_10pct(self) -> None:
        """10% drop should NOT trigger (threshold is 12%)."""
        monitor = DrawdownMonitor()
        monitor.update(Decimal(100_000))
        triggered = monitor.update(Decimal(90_000))
        assert triggered is False
        assert monitor.triggered is False

    def test_drawdown_resets_on_new_peak(self) -> None:
        """New equity peak should reset the triggered flag."""
        monitor = DrawdownMonitor()
        monitor.update(Decimal(100_000))
        monitor.update(Decimal(87_000))  # triggers
        assert monitor.triggered is True

        # New peak resets trigger
        monitor.update(Decimal(110_000))
        assert monitor.triggered is False

        # 10% from new peak = 99000, not triggered
        triggered = monitor.update(Decimal(99_000))
        assert triggered is False

    def test_drawdown_rolling_not_calendar(self) -> None:
        """Drawdown should track rolling peak, not reset on day boundaries.

        Simulate equity across multiple 'days' -- peak should persist.
        """
        monitor = DrawdownMonitor()

        # Day 1: peak at 100k
        monitor.update(Decimal(100_000))
        # Day 2: dip
        monitor.update(Decimal(95_000))
        assert monitor.triggered is False
        # Day 3: further dip, still tracks from 100k peak
        triggered = monitor.update(Decimal(87_500))
        assert triggered is True
        # Peak is still 100k, drawdown = 12.5%

    def test_drawdown_current_pct(self) -> None:
        """current_drawdown property should return correct percentage."""
        monitor = DrawdownMonitor()
        monitor.update(Decimal(100_000))
        monitor.update(Decimal(90_000))
        expected = Decimal(10_000) / Decimal(100_000)  # 0.10
        assert monitor.current_drawdown == expected

    def test_drawdown_current_pct_no_data(self) -> None:
        """current_drawdown should return 0 when no data has been recorded."""
        monitor = DrawdownMonitor()
        assert monitor.current_drawdown == Decimal(0)

    def test_drawdown_custom_threshold(self) -> None:
        """Custom threshold should be respected."""
        monitor = DrawdownMonitor(threshold=Decimal("0.05"))
        monitor.update(Decimal(100_000))
        triggered = monitor.update(Decimal(94_000))
        assert triggered is True

    def test_drawdown_reset(self) -> None:
        """reset() should clear all state."""
        monitor = DrawdownMonitor()
        monitor.update(Decimal(100_000))
        monitor.update(Decimal(85_000))
        assert monitor.triggered is True

        monitor.reset()
        assert monitor.triggered is False
        assert monitor.current_drawdown == Decimal(0)
