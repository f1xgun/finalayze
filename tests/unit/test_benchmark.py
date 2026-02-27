"""Unit tests for benchmark comparison in PerformanceAnalyzer."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from finalayze.backtest.performance import PerformanceAnalyzer
from finalayze.core.schemas import Candle, PortfolioState, TradeResult

_QUANTIZE_4DP = Decimal("0.0001")


def _snapshot(equity: Decimal, day: int) -> PortfolioState:
    """Create a portfolio snapshot at the given day offset."""
    return PortfolioState(
        cash=equity,
        positions={},
        equity=equity,
        timestamp=datetime(2024, 1, 1, 14, 30, tzinfo=UTC) + timedelta(days=day),
    )


def _candle(price: Decimal, day: int, symbol: str = "SPY") -> Candle:
    """Create a simple candle."""
    return Candle(
        symbol=symbol,
        market_id="us",
        timeframe="1d",
        timestamp=datetime(2024, 1, 1, 14, 30, tzinfo=UTC) + timedelta(days=day),
        open=price,
        high=price + Decimal(1),
        low=price - Decimal(1),
        close=price,
        volume=1_000_000,
    )


def _trade(entry: Decimal, exit_price: Decimal, qty: Decimal = Decimal(10)) -> TradeResult:
    """Create a simple trade result."""
    from uuid import uuid4

    pnl = (exit_price - entry) * qty
    pnl_pct = (exit_price - entry) / entry if entry != 0 else Decimal(0)
    return TradeResult(
        signal_id=uuid4(),
        symbol="TEST",
        side="SELL",
        quantity=qty,
        entry_price=entry,
        exit_price=exit_price,
        pnl=pnl,
        pnl_pct=pnl_pct,
    )


class TestBenchmarkFields:
    """BacktestResult includes optional benchmark fields."""

    def test_backtest_result_has_benchmark_fields(self) -> None:
        from finalayze.core.schemas import BacktestResult

        result = BacktestResult(
            sharpe=Decimal(0),
            max_drawdown=Decimal(0),
            win_rate=Decimal(0),
            profit_factor=Decimal(0),
            total_return=Decimal(0),
            total_trades=0,
            alpha=Decimal("0.05"),
            beta=Decimal("1.1"),
            information_ratio=Decimal("0.3"),
            max_relative_drawdown=Decimal("0.02"),
            benchmark_return=Decimal("0.10"),
        )
        assert result.alpha == Decimal("0.05")
        assert result.beta == Decimal("1.1")
        assert result.information_ratio == Decimal("0.3")
        assert result.max_relative_drawdown == Decimal("0.02")
        assert result.benchmark_return == Decimal("0.10")

    def test_benchmark_fields_default_none(self) -> None:
        from finalayze.core.schemas import BacktestResult

        result = BacktestResult(
            sharpe=Decimal(0),
            max_drawdown=Decimal(0),
            win_rate=Decimal(0),
            profit_factor=Decimal(0),
            total_return=Decimal(0),
            total_trades=0,
        )
        assert result.alpha is None
        assert result.beta is None
        assert result.information_ratio is None
        assert result.max_relative_drawdown is None
        assert result.benchmark_return is None


class TestAlphaBetaComputation:
    """PerformanceAnalyzer computes alpha, beta, IR when benchmark provided."""

    def test_alpha_positive_when_strategy_beats_benchmark(self) -> None:
        """Strategy returning 20% vs benchmark 10% => alpha ~10%."""
        analyzer = PerformanceAnalyzer()
        # Strategy equity: 100k -> 120k over 10 days (linearly)
        snapshots = [_snapshot(Decimal(100000 + i * 2000), i) for i in range(11)]
        # Benchmark: 100 -> 110 over same period
        benchmark = [_candle(Decimal(100 + i), i) for i in range(11)]
        trades = [_trade(Decimal(100), Decimal(120))]

        result = analyzer.analyze(trades, snapshots, benchmark_candles=benchmark)

        assert result.alpha is not None
        assert result.alpha > Decimal(0)
        assert result.benchmark_return is not None
        assert result.benchmark_return > Decimal(0)

    def test_beta_computation(self) -> None:
        """Beta is cov(strategy, benchmark) / var(benchmark)."""
        analyzer = PerformanceAnalyzer()
        # Strategy and benchmark move in same direction
        snapshots = [_snapshot(Decimal(100000 + i * 1000), i) for i in range(21)]
        benchmark = [_candle(Decimal(100 + i), i) for i in range(21)]
        trades = [_trade(Decimal(100), Decimal(110))]

        result = analyzer.analyze(trades, snapshots, benchmark_candles=benchmark)

        assert result.beta is not None
        # Both moving linearly up, beta should be positive
        assert result.beta > Decimal(0)

    def test_information_ratio_computed(self) -> None:
        """IR = alpha / tracking error."""
        analyzer = PerformanceAnalyzer()
        snapshots = [_snapshot(Decimal(100000 + i * 1500), i) for i in range(21)]
        benchmark = [_candle(Decimal(100 + i), i) for i in range(21)]
        trades = [_trade(Decimal(100), Decimal(115))]

        result = analyzer.analyze(trades, snapshots, benchmark_candles=benchmark)

        assert result.information_ratio is not None

    def test_no_benchmark_returns_none(self) -> None:
        """Without benchmark, benchmark fields stay None."""
        analyzer = PerformanceAnalyzer()
        snapshots = [_snapshot(Decimal(100000 + i * 1000), i) for i in range(11)]
        trades = [_trade(Decimal(100), Decimal(110))]

        result = analyzer.analyze(trades, snapshots)

        assert result.alpha is None
        assert result.beta is None
        assert result.information_ratio is None


class TestMaxRelativeDrawdown:
    """Max relative drawdown: worst underperformance vs benchmark."""

    def test_max_relative_drawdown_computed(self) -> None:
        analyzer = PerformanceAnalyzer()
        # Strategy flat at 100k, benchmark goes up => underperformance
        snapshots = [_snapshot(Decimal(100000), i) for i in range(11)]
        benchmark = [_candle(Decimal(100 + i * 2), i) for i in range(11)]
        trades: list[TradeResult] = []

        result = analyzer.analyze(trades, snapshots, benchmark_candles=benchmark)

        assert result.max_relative_drawdown is not None
        assert result.max_relative_drawdown > Decimal(0)
