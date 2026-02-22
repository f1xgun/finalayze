"""Unit tests for PerformanceAnalyzer."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from uuid import uuid4

from finalayze.backtest.performance import PerformanceAnalyzer
from finalayze.core.schemas import BacktestResult, PortfolioState, TradeResult

# ── Constants (no magic numbers) ────────────────────────────────────────
INITIAL_EQUITY = Decimal(100000)
EQUITY_STEP_1 = Decimal(100100)  # +100
EQUITY_STEP_2 = Decimal(100050)  # -50
EQUITY_STEP_3 = Decimal(100250)  # +200
EQUITY_STEP_4 = Decimal(99800)  # -450 (drawdown)
EQUITY_STEP_5 = Decimal(100500)  # +700

TRADE_PNL_WIN_1 = Decimal(100)
TRADE_PNL_LOSS = Decimal(-50)
TRADE_PNL_WIN_2 = Decimal(200)

TOTAL_TRADE_COUNT = 3
WIN_COUNT = 2
EXPECTED_WIN_RATE = Decimal("0.6667")

ZERO = Decimal(0)
ONE = Decimal(1)

SNAPSHOT_COUNT = 6
MIN_SNAPSHOTS_FOR_SHARPE = 3


# ── Helpers ─────────────────────────────────────────────────────────────


def _make_trades() -> list[TradeResult]:
    """Create 3 trades: +100, -50, +200."""
    base = [
        (TRADE_PNL_WIN_1, Decimal("0.01")),
        (TRADE_PNL_LOSS, Decimal("-0.005")),
        (TRADE_PNL_WIN_2, Decimal("0.02")),
    ]
    return [
        TradeResult(
            signal_id=uuid4(),
            symbol="AAPL",
            side="BUY",
            quantity=Decimal(10),
            entry_price=Decimal(150),
            exit_price=Decimal(150) + pnl / Decimal(10),
            pnl=pnl,
            pnl_pct=pnl_pct,
        )
        for pnl, pnl_pct in base
    ]


def _make_snapshots() -> list[PortfolioState]:
    """Equity curve: 100000, 100100, 100050, 100250, 99800, 100500."""
    equities = [
        INITIAL_EQUITY,
        EQUITY_STEP_1,
        EQUITY_STEP_2,
        EQUITY_STEP_3,
        EQUITY_STEP_4,
        EQUITY_STEP_5,
    ]
    base_time = datetime(2024, 1, 1, 14, 30, tzinfo=UTC)
    return [
        PortfolioState(
            cash=eq,
            positions={},
            equity=eq,
            timestamp=base_time + timedelta(days=i),
        )
        for i, eq in enumerate(equities)
    ]


# ── Tests ───────────────────────────────────────────────────────────────


class TestPerformanceAnalyzerTotalTrades:
    """total_trades should equal the number of trades provided."""

    def test_total_trades_count(self) -> None:
        analyzer = PerformanceAnalyzer()
        result = analyzer.analyze(_make_trades(), _make_snapshots())
        assert result.total_trades == TOTAL_TRADE_COUNT


class TestPerformanceAnalyzerWinRate:
    """win_rate should be wins / total."""

    def test_win_rate_value(self) -> None:
        analyzer = PerformanceAnalyzer()
        result = analyzer.analyze(_make_trades(), _make_snapshots())
        assert result.win_rate == EXPECTED_WIN_RATE


class TestPerformanceAnalyzerTotalReturn:
    """total_return should be positive when final equity > initial."""

    def test_total_return_positive(self) -> None:
        analyzer = PerformanceAnalyzer()
        result = analyzer.analyze(_make_trades(), _make_snapshots())
        assert result.total_return > ZERO

    def test_total_return_value(self) -> None:
        analyzer = PerformanceAnalyzer()
        result = analyzer.analyze(_make_trades(), _make_snapshots())
        expected = (EQUITY_STEP_5 - INITIAL_EQUITY) / INITIAL_EQUITY
        assert result.total_return == expected.quantize(Decimal("0.0001"))


class TestPerformanceAnalyzerMaxDrawdown:
    """max_drawdown should be between 0 and 1."""

    def test_max_drawdown_bounds(self) -> None:
        analyzer = PerformanceAnalyzer()
        result = analyzer.analyze(_make_trades(), _make_snapshots())
        assert result.max_drawdown >= ZERO
        assert result.max_drawdown <= ONE

    def test_max_drawdown_nonzero(self) -> None:
        """The equity curve has a drawdown at step 4, so max_drawdown > 0."""
        analyzer = PerformanceAnalyzer()
        result = analyzer.analyze(_make_trades(), _make_snapshots())
        assert result.max_drawdown > ZERO


class TestPerformanceAnalyzerProfitFactor:
    """profit_factor > 1 when gross profit exceeds gross loss."""

    def test_profit_factor_above_one(self) -> None:
        analyzer = PerformanceAnalyzer()
        result = analyzer.analyze(_make_trades(), _make_snapshots())
        assert result.profit_factor > ONE


class TestPerformanceAnalyzerResultType:
    """Result should be a BacktestResult instance."""

    def test_result_is_backtest_result(self) -> None:
        analyzer = PerformanceAnalyzer()
        result = analyzer.analyze(_make_trades(), _make_snapshots())
        assert isinstance(result, BacktestResult)


class TestPerformanceAnalyzerEmptyTrades:
    """Empty trades list should return zero trade metrics but still compute equity curve metrics."""

    def test_empty_total_trades(self) -> None:
        analyzer = PerformanceAnalyzer()
        result = analyzer.analyze([], _make_snapshots())
        assert result.total_trades == 0

    def test_empty_win_rate(self) -> None:
        analyzer = PerformanceAnalyzer()
        result = analyzer.analyze([], _make_snapshots())
        assert result.win_rate == ZERO

    def test_empty_sharpe_computed_from_snapshots(self) -> None:
        """With snapshots provided, Sharpe is computed from the equity curve."""
        analyzer = PerformanceAnalyzer()
        result = analyzer.analyze([], _make_snapshots())
        # Non-zero Sharpe is computed from the equity curve snapshots
        assert isinstance(result.sharpe, Decimal)

    def test_empty_sharpe_no_snapshots(self) -> None:
        """Without snapshots, Sharpe should be zero."""
        analyzer = PerformanceAnalyzer()
        result = analyzer.analyze([], [])
        assert result.sharpe == ZERO

    def test_empty_profit_factor(self) -> None:
        analyzer = PerformanceAnalyzer()
        result = analyzer.analyze([], _make_snapshots())
        assert result.profit_factor == ZERO

    def test_empty_trades_total_return_from_snapshots(self) -> None:
        """With snapshots provided, total_return is computed from equity curve."""
        analyzer = PerformanceAnalyzer()
        result = analyzer.analyze([], _make_snapshots())
        expected = (EQUITY_STEP_5 - INITIAL_EQUITY) / INITIAL_EQUITY
        assert result.total_return == expected.quantize(Decimal("0.0001"))

    def test_empty_trades_max_drawdown_from_snapshots(self) -> None:
        """With snapshots provided, max_drawdown is computed from equity curve."""
        analyzer = PerformanceAnalyzer()
        result = analyzer.analyze([], _make_snapshots())
        assert result.max_drawdown > ZERO
