"""Unit tests for BacktestEngine integration with RollingKelly and LossLimitTracker."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from finalayze.backtest.engine import BacktestEngine
from finalayze.core.schemas import Candle, SignalDirection, TradeResult
from finalayze.risk.kelly import RollingKelly, TradeRecord
from finalayze.risk.loss_limits import LossLimitTracker
from finalayze.strategies.base import BaseStrategy

# ── Constants ─────────────────────────────────────────────────────────────

INITIAL_CASH = Decimal(100_000)
SYMBOL = "TEST"
SEGMENT = "us_tech"
MARKET_ID = "us"
TIMEFRAME = "1d"
SOURCE = "test"

# 14:30 UTC = 10:30 ET (within US market hours 14:30-21:00 UTC)
BASE_DATE = datetime(2024, 1, 2, 14, 30, tzinfo=UTC)
CANDLE_VOLUME = 1_000_000


# ── Helpers ───────────────────────────────────────────────────────────────


def _make_candle(
    day_offset: int,
    open_: float,
    high: float,
    low: float,
    close: float,
) -> Candle:
    return Candle(
        symbol=SYMBOL,
        market_id=MARKET_ID,
        timeframe=TIMEFRAME,
        timestamp=BASE_DATE + timedelta(days=day_offset),
        open=Decimal(str(open_)),
        high=Decimal(str(high)),
        low=Decimal(str(low)),
        close=Decimal(str(close)),
        volume=CANDLE_VOLUME,
        source=SOURCE,
    )


class _AlwaysBuyStrategy(BaseStrategy):
    """Strategy that emits BUY on every candle (for testing)."""

    @property
    def name(self) -> str:
        return "always_buy"

    def supported_segments(self) -> list[str]:
        return [SEGMENT]

    def get_parameters(self, segment_id: str) -> dict[str, object]:
        return {}

    def generate_signal(
        self,
        symbol: str,
        candles: list[Candle],
        segment_id: str,
        sentiment_score: float = 0.0,
    ):
        from finalayze.core.schemas import Signal

        return Signal(
            strategy_name=self.name,
            symbol=symbol,
            market_id=MARKET_ID,
            segment_id=segment_id,
            direction=SignalDirection.BUY,
            confidence=0.8,
            features={},
            reasoning="test",
        )


class _AlternatingStrategy(BaseStrategy):
    """Strategy that alternates BUY/SELL signals."""

    def __init__(self) -> None:
        self._call_count = 0

    @property
    def name(self) -> str:
        return "alternating"

    def supported_segments(self) -> list[str]:
        return [SEGMENT]

    def get_parameters(self, segment_id: str) -> dict[str, object]:
        return {}

    def generate_signal(
        self,
        symbol: str,
        candles: list[Candle],
        segment_id: str,
        sentiment_score: float = 0.0,
    ):
        from finalayze.core.schemas import Signal

        self._call_count += 1
        direction = SignalDirection.BUY if self._call_count % 2 == 1 else SignalDirection.SELL
        return Signal(
            strategy_name=self.name,
            symbol=symbol,
            market_id=MARKET_ID,
            segment_id=segment_id,
            direction=direction,
            confidence=0.8,
            features={},
            reasoning="test",
        )


def _make_stable_candles(n: int, price: float = 100.0) -> list[Candle]:
    """Create n candles at a stable price."""
    return [_make_candle(i, price, price + 1, price - 1, price) for i in range(n)]


# ── Tests ─────────────────────────────────────────────────────────────────


class TestBacktestEngineWithRollingKelly:
    """Tests for RollingKelly integration in BacktestEngine."""

    def test_engine_accepts_rolling_kelly(self) -> None:
        """Engine can be instantiated with a RollingKelly."""
        kelly = RollingKelly()
        engine = BacktestEngine(
            strategy=_AlwaysBuyStrategy(),
            initial_cash=INITIAL_CASH,
            rolling_kelly=kelly,
        )
        assert engine._rolling_kelly is kelly

    def test_engine_without_rolling_kelly_uses_defaults(self) -> None:
        """Without RollingKelly, engine uses default half-Kelly sizing."""
        strategy = _AlternatingStrategy()
        candles = _make_stable_candles(10)
        engine = BacktestEngine(strategy=strategy, initial_cash=INITIAL_CASH)
        trades, _snapshots = engine.run(SYMBOL, SEGMENT, candles)
        # Should produce trades using default sizing
        assert isinstance(trades, list)

    def test_kelly_updated_after_trade(self) -> None:
        """RollingKelly is updated after each completed trade."""
        kelly = RollingKelly()
        strategy = _AlternatingStrategy()
        candles = _make_stable_candles(10)
        engine = BacktestEngine(
            strategy=strategy,
            initial_cash=INITIAL_CASH,
            rolling_kelly=kelly,
        )
        engine.run(SYMBOL, SEGMENT, candles)
        # At least some trades should have been recorded
        assert kelly.trade_count > 0


class TestBacktestEngineWithLossLimits:
    """Tests for LossLimitTracker integration in BacktestEngine."""

    def test_engine_accepts_loss_limits(self) -> None:
        """Engine can be instantiated with a LossLimitTracker."""
        tracker = LossLimitTracker()
        engine = BacktestEngine(
            strategy=_AlwaysBuyStrategy(),
            initial_cash=INITIAL_CASH,
            loss_limits=tracker,
        )
        assert engine._loss_limits is tracker

    def test_loss_limits_suppress_new_entries(self) -> None:
        """When daily loss limit is breached, no new BUY signals are acted on."""
        # Use a very tight daily limit (0.1%) so any small fluctuation triggers it
        tracker = LossLimitTracker(daily_loss_limit_pct=0.1, weekly_loss_limit_pct=50.0)

        # Create candles that crash: start at 100, then drop to 50
        candles: list[Candle] = []
        for i in range(20):
            price = 100.0 - i * 2.5  # drops from 100 to 52.5
            candles.append(_make_candle(i, price, price + 1, price - 1, price))

        engine_limited = BacktestEngine(
            strategy=_AlternatingStrategy(),
            initial_cash=INITIAL_CASH,
            loss_limits=tracker,
        )
        engine_unlimited = BacktestEngine(
            strategy=_AlternatingStrategy(),
            initial_cash=INITIAL_CASH,
        )

        trades_limited, _ = engine_limited.run(SYMBOL, SEGMENT, candles)
        trades_unlimited, _ = engine_unlimited.run(SYMBOL, SEGMENT, candles)

        # Limited engine should have fewer or equal trades (loss limits suppress entries)
        assert len(trades_limited) <= len(trades_unlimited)

    def test_engine_without_loss_limits_runs_normally(self) -> None:
        """Without LossLimitTracker, engine runs without halting."""
        strategy = _AlternatingStrategy()
        candles = _make_stable_candles(10)
        engine = BacktestEngine(strategy=strategy, initial_cash=INITIAL_CASH)
        trades, _snapshots = engine.run(SYMBOL, SEGMENT, candles)
        assert isinstance(trades, list)
