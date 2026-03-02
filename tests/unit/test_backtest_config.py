"""Unit tests for BacktestConfig dataclass."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from finalayze.backtest.config import BacktestConfig
from finalayze.backtest.engine import BacktestEngine
from finalayze.core.schemas import Candle, Signal, SignalDirection
from finalayze.strategies.base import BaseStrategy

INITIAL_CASH = Decimal(100000)
CANDLE_COUNT = 40
TRADE_DAY_BUY = 30
TRADE_DAY_SELL = 35


class _StubStrategy(BaseStrategy):
    """Emits BUY at candle index TRADE_DAY_BUY, SELL at TRADE_DAY_SELL."""

    @property
    def name(self) -> str:
        return "stub"

    def supported_segments(self) -> list[str]:
        return ["us_large_cap"]

    def generate_signal(  # type: ignore[override]
        self,
        symbol: str,
        candles: list[Candle],
        segment_id: str,
        **kwargs: object,
    ) -> Signal | None:
        idx = len(candles) - 1
        if idx == TRADE_DAY_BUY:
            return Signal(
                strategy_name=self.name,
                symbol=symbol,
                market_id="us",
                segment_id=segment_id,
                direction=SignalDirection.BUY,
                confidence=0.8,
                features={"momentum": 1.0},
                reasoning="Test buy signal",
            )
        if idx == TRADE_DAY_SELL:
            return Signal(
                strategy_name=self.name,
                symbol=symbol,
                market_id="us",
                segment_id=segment_id,
                direction=SignalDirection.SELL,
                confidence=0.8,
                features={"momentum": -1.0},
                reasoning="Test sell signal",
            )
        return None

    def get_parameters(self, segment_id: str) -> dict[str, object]:
        return {}


def _make_candles(count: int = CANDLE_COUNT) -> list[Candle]:
    """Create an upward-trending candle series."""
    base_price = Decimal(100)
    candles: list[Candle] = []
    for i in range(count):
        price = base_price + Decimal(i)
        candles.append(
            Candle(
                symbol="TEST",
                market_id="us",
                timeframe="1d",
                timestamp=datetime(2024, 1, 1, 14, 30, tzinfo=UTC) + timedelta(days=i),
                open=price,
                high=price + Decimal(2),
                low=price - Decimal(2),
                close=price + Decimal(1),
                volume=1_000_000,
            )
        )
    return candles


class TestBacktestConfigDefaults:
    """Verify all defaults match original engine defaults."""

    def test_backtest_config_defaults(self) -> None:
        cfg = BacktestConfig()

        assert cfg.initial_cash == Decimal(100000)
        assert cfg.max_position_pct == Decimal("0.20")
        assert cfg.max_positions == 10
        assert cfg.kelly_fraction == Decimal("0.5")
        assert cfg.atr_multiplier == Decimal("3.0")
        assert cfg.transaction_costs is None
        assert cfg.trail_activation_atr == Decimal("1.0")
        assert cfg.trail_distance_atr == Decimal("1.5")
        assert cfg.circuit_breaker is None
        assert cfg.rolling_kelly is None
        assert cfg.loss_limits is None
        assert cfg.target_vol is None
        assert cfg.decision_journal is None
        assert cfg.profit_target_atr == Decimal("5.0")
        assert cfg.max_hold_bars == 30

    def test_new_fields_defaults(self) -> None:
        cfg = BacktestConfig()

        assert cfg.stop_loss_mode == "trailing"
        assert cfg.trend_filter_enabled is False
        assert cfg.trend_sma_period == 200


class TestEngineWithConfig:
    """Engine runs correctly when constructed with a BacktestConfig."""

    def test_engine_with_config(self) -> None:
        cfg = BacktestConfig(initial_cash=INITIAL_CASH)
        engine = BacktestEngine(strategy=_StubStrategy(), config=cfg)
        candles = _make_candles()

        trades, snapshots = engine.run(symbol="TEST", segment_id="us_large_cap", candles=candles)

        assert len(snapshots) == CANDLE_COUNT
        assert len(trades) >= 1

    def test_engine_with_config_custom_params(self) -> None:
        cfg = BacktestConfig(
            initial_cash=Decimal(50000),
            max_positions=5,
            kelly_fraction=Decimal("0.25"),
        )
        engine = BacktestEngine(strategy=_StubStrategy(), config=cfg)

        assert engine._initial_cash == Decimal(50000)
        assert engine._max_positions == 5
        assert engine._kelly_fraction == Decimal("0.25")


class TestEngineBackwardCompat:
    """Engine still works with old-style keyword arguments."""

    def test_engine_backward_compat(self) -> None:
        engine = BacktestEngine(strategy=_StubStrategy(), initial_cash=INITIAL_CASH)
        candles = _make_candles()

        trades, snapshots = engine.run(symbol="TEST", segment_id="us_large_cap", candles=candles)

        assert len(snapshots) == CANDLE_COUNT
        assert len(trades) >= 1

    def test_engine_backward_compat_stores_config(self) -> None:
        engine = BacktestEngine(
            strategy=_StubStrategy(),
            initial_cash=Decimal(75000),
            max_positions=3,
        )

        assert engine._config.initial_cash == Decimal(75000)
        assert engine._config.max_positions == 3


class TestConfigFrozen:
    """Verify BacktestConfig immutability."""

    def test_config_frozen(self) -> None:
        cfg = BacktestConfig()

        with pytest.raises(FrozenInstanceError):
            cfg.initial_cash = Decimal(50000)  # type: ignore[misc]

    def test_config_frozen_new_fields(self) -> None:
        cfg = BacktestConfig()

        with pytest.raises(FrozenInstanceError):
            cfg.stop_loss_mode = "fixed"  # type: ignore[misc]

        with pytest.raises(FrozenInstanceError):
            cfg.trend_filter_enabled = True  # type: ignore[misc]
