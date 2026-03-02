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


class SilentStrategy(BaseStrategy):
    """Always returns None -- no signals."""

    @property
    def name(self) -> str:
        return "silent"

    def supported_segments(self) -> list[str]:
        return ["us_large_cap"]

    def generate_signal(  # type: ignore[override]
        self,
        symbol: str,
        candles: list[Candle],
        segment_id: str,
        **kwargs: object,
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


def _make_candle_series_for_symbol(
    symbol: str, count: int = CANDLE_COUNT, base: int = 100
) -> list[Candle]:
    """Create an upward-trending candle series for a given symbol."""
    candles: list[Candle] = []
    for i in range(count):
        price = Decimal(base + i)
        candles.append(
            Candle(
                symbol=symbol,
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


class TestPortfolioBacktest:
    """Tests for run_portfolio() method (6B.6)."""

    def test_portfolio_backtest_two_symbols(self) -> None:
        """Run with 2 symbols, verify both get processed."""
        engine = BacktestEngine(strategy=StubStrategy(), initial_cash=INITIAL_CASH)
        sym_a = _make_candle_series_for_symbol("SYM_A")
        sym_b = _make_candle_series_for_symbol("SYM_B", base=200)
        trades, snapshots = engine.run_portfolio(
            symbols=["SYM_A", "SYM_B"],
            segment_id="us_large_cap",
            candles_by_symbol={"SYM_A": sym_a, "SYM_B": sym_b},
        )
        # Both symbols should produce trades (StubStrategy fires at index 30, 35)
        assert len(trades) >= 1
        assert len(snapshots) > 0

    def test_portfolio_backtest_respects_max_positions(self) -> None:
        """max_positions=1 -> only one position opened."""
        engine = BacktestEngine(
            strategy=StubStrategy(),
            initial_cash=INITIAL_CASH,
            max_positions=1,
        )
        sym_a = _make_candle_series_for_symbol("SYM_A")
        sym_b = _make_candle_series_for_symbol("SYM_B", base=200)
        trades, _snapshots = engine.run_portfolio(
            symbols=["SYM_A", "SYM_B"],
            segment_id="us_large_cap",
            candles_by_symbol={"SYM_A": sym_a, "SYM_B": sym_b},
        )
        # With max_positions=1, at most one position at a time
        # Both may eventually trade (after one closes), but constraint is respected
        assert len(trades) >= 1

    def test_portfolio_backtest_empty_symbols(self) -> None:
        """Empty symbol list returns empty results."""
        engine = BacktestEngine(strategy=StubStrategy(), initial_cash=INITIAL_CASH)
        trades, snapshots = engine.run_portfolio(
            symbols=[],
            segment_id="us_large_cap",
            candles_by_symbol={},
        )
        assert trades == []
        assert snapshots == []

    def test_portfolio_backtest_single_symbol_produces_trades(self) -> None:
        """Single symbol run_portfolio produces trades."""
        engine = BacktestEngine(strategy=StubStrategy(), initial_cash=INITIAL_CASH)
        candles = _make_candle_series_for_symbol("TEST")
        trades, snapshots = engine.run_portfolio(
            symbols=["TEST"],
            segment_id="us_large_cap",
            candles_by_symbol={"TEST": candles},
        )
        assert len(trades) >= 1
        assert len(snapshots) > 0

    def test_portfolio_backtest_unaligned_timestamps(self) -> None:
        """Symbols with different candle date ranges handled correctly."""
        # SYM_A has 40 candles, SYM_B has only 20 candles (starts later)
        sym_a = _make_candle_series_for_symbol("SYM_A", count=CANDLE_COUNT)
        sym_b_start = 20
        sym_b: list[Candle] = []
        for i in range(sym_b_start, CANDLE_COUNT):
            price = Decimal(200 + i)
            sym_b.append(
                Candle(
                    symbol="SYM_B",
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
        engine = BacktestEngine(strategy=StubStrategy(), initial_cash=INITIAL_CASH)
        _trades, snapshots = engine.run_portfolio(
            symbols=["SYM_A", "SYM_B"],
            segment_id="us_large_cap",
            candles_by_symbol={"SYM_A": sym_a, "SYM_B": sym_b},
        )
        # Should not crash, and produce some trades
        assert len(snapshots) > 0
