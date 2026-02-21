"""Unit tests for BacktestEngine."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from finalayze.backtest.engine import BacktestEngine
from finalayze.core.schemas import Candle, Signal, SignalDirection
from finalayze.strategies.base import BaseStrategy

INITIAL_CASH = Decimal(100000)
CANDLE_COUNT = 40
TRADE_DAY_BUY = 30
TRADE_DAY_SELL = 35


def _make_candle_series(count: int = CANDLE_COUNT) -> list[Candle]:
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


class StubStrategy(BaseStrategy):
    """Emits BUY at candle index TRADE_DAY_BUY, SELL at TRADE_DAY_SELL."""

    @property
    def name(self) -> str:
        return "stub"

    def supported_segments(self) -> list[str]:
        return ["us_large_cap"]

    def generate_signal(
        self, symbol: str, candles: list[Candle], segment_id: str
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


class SilentStrategy(BaseStrategy):
    """Always returns None -- no signals."""

    @property
    def name(self) -> str:
        return "silent"

    def supported_segments(self) -> list[str]:
        return ["us_large_cap"]

    def generate_signal(
        self, symbol: str, candles: list[Candle], segment_id: str
    ) -> Signal | None:
        return None

    def get_parameters(self, segment_id: str) -> dict[str, object]:
        return {}


class TestBacktestEngineRunsToCompletion:
    """Engine with StubStrategy should produce trades and full snapshots."""

    def test_engine_runs_to_completion(self) -> None:
        engine = BacktestEngine(strategy=StubStrategy(), initial_cash=INITIAL_CASH)
        candles = _make_candle_series()

        trades, snapshots = engine.run(symbol="TEST", segment_id="us_large_cap", candles=candles)

        assert len(snapshots) == CANDLE_COUNT
        assert len(trades) >= 1


class TestBacktestEngineNoSignals:
    """Engine with SilentStrategy should produce zero trades."""

    def test_engine_no_signals_no_trades(self) -> None:
        engine = BacktestEngine(strategy=SilentStrategy(), initial_cash=INITIAL_CASH)
        candles = _make_candle_series()

        trades, snapshots = engine.run(symbol="TEST", segment_id="us_large_cap", candles=candles)

        assert len(trades) == 0
        assert len(snapshots) == CANDLE_COUNT


class TestBacktestEnginePreservesInitialCash:
    """When no trades happen, equity should equal initial cash."""

    def test_engine_preserves_initial_cash_when_no_trades(self) -> None:
        engine = BacktestEngine(strategy=SilentStrategy(), initial_cash=INITIAL_CASH)
        candles = _make_candle_series()

        _trades, snapshots = engine.run(symbol="TEST", segment_id="us_large_cap", candles=candles)

        assert snapshots[-1].equity == INITIAL_CASH
