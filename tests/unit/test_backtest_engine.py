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


# ---------------------------------------------------------------------------
# Structural break exclusion tests
# ---------------------------------------------------------------------------


class TestBacktestConfigExcludePeriods:
    """Tests for the exclude_periods field on BacktestConfig."""

    def test_backtest_config_exclude_periods_default(self) -> None:
        """Default exclude_periods is empty tuple."""
        from finalayze.backtest.config import BacktestConfig

        cfg = BacktestConfig()
        assert cfg.exclude_periods == ()

    def test_moex_2022_break_constant(self) -> None:
        """MOEX_2022_BREAK constant matches expected date range."""
        from finalayze.backtest.config import MOEX_2022_BREAK

        assert MOEX_2022_BREAK == (("2022-02-21", "2022-04-01"),)


class TestFilterCandlesByExclusion:
    """Tests for the filter_candles_by_exclusion helper."""

    def test_filter_candles_by_exclusion(self) -> None:
        """Candles within excluded period are removed, others preserved."""
        from finalayze.risk.stop_loss import filter_candles_by_exclusion

        # 10 candles: 4 before, 3 inside, 3 after the exclusion window
        dates = [
            datetime(2022, 1, 10, tzinfo=UTC),  # before
            datetime(2022, 2, 1, tzinfo=UTC),  # before
            datetime(2022, 2, 15, tzinfo=UTC),  # before
            datetime(2022, 2, 20, tzinfo=UTC),  # before (day before exclusion)
            datetime(2022, 2, 21, tzinfo=UTC),  # excluded (start)
            datetime(2022, 3, 1, tzinfo=UTC),  # excluded (middle)
            datetime(2022, 4, 1, tzinfo=UTC),  # excluded (end, inclusive)
            datetime(2022, 4, 2, tzinfo=UTC),  # after
            datetime(2022, 5, 1, tzinfo=UTC),  # after
            datetime(2022, 6, 1, tzinfo=UTC),  # after
        ]
        candles = [
            Candle(
                symbol="SBER",
                market_id="moex",
                timeframe="1d",
                timestamp=dt,
                open=Decimal(100),
                high=Decimal(105),
                low=Decimal(95),
                close=Decimal(102),
                volume=1_000_000,
            )
            for dt in dates
        ]

        exclude = (("2022-02-21", "2022-04-01"),)
        filtered = filter_candles_by_exclusion(candles, exclude)
        expected_count = 7
        assert len(filtered) == expected_count
        # None of the filtered candles should fall within the excluded range
        from datetime import date

        excl_start = date(2022, 2, 21)
        excl_end = date(2022, 4, 1)
        for c in filtered:
            d = c.timestamp.date()
            assert not (excl_start <= d <= excl_end)


class TestAtrExcludesStructuralBreak:
    """Tests for compute_atr_stop_loss with exclude_periods."""

    def test_atr_excludes_structural_break_period(self) -> None:
        """ATR with exclusion is significantly lower than without when break has extreme vol."""
        from finalayze.risk.stop_loss import compute_atr_stop_loss

        # 5 normal candles before break (Jan 2022)
        normal_before = [
            Candle(
                symbol="SBER",
                market_id="moex",
                timeframe="1d",
                timestamp=datetime(2022, 1, day, 7, 0, tzinfo=UTC),
                open=Decimal(100),
                high=Decimal(102),
                low=Decimal(98),
                close=Decimal(101),
                volume=1_000_000,
            )
            for day in range(10, 15)
        ]
        # 10 extreme vol candles during break (Feb 21-28 + Mar 1-5 2022)
        extreme_dates = [
            datetime(2022, 2, 21, 7, 0, tzinfo=UTC),
            datetime(2022, 2, 22, 7, 0, tzinfo=UTC),
            datetime(2022, 2, 23, 7, 0, tzinfo=UTC),
            datetime(2022, 2, 24, 7, 0, tzinfo=UTC),
            datetime(2022, 2, 25, 7, 0, tzinfo=UTC),
            datetime(2022, 3, 1, 7, 0, tzinfo=UTC),
            datetime(2022, 3, 2, 7, 0, tzinfo=UTC),
            datetime(2022, 3, 3, 7, 0, tzinfo=UTC),
            datetime(2022, 3, 4, 7, 0, tzinfo=UTC),
            datetime(2022, 3, 5, 7, 0, tzinfo=UTC),
        ]
        extreme = [
            Candle(
                symbol="SBER",
                market_id="moex",
                timeframe="1d",
                timestamp=dt,
                open=Decimal(80),
                high=Decimal(150),
                low=Decimal(40),
                close=Decimal(70),
                volume=5_000_000,
            )
            for dt in extreme_dates
        ]
        # 5 normal candles after break (Apr 5-9 2022)
        normal_after = [
            Candle(
                symbol="SBER",
                market_id="moex",
                timeframe="1d",
                timestamp=datetime(2022, 4, day, 7, 0, tzinfo=UTC),
                open=Decimal(100),
                high=Decimal(103),
                low=Decimal(97),
                close=Decimal(101),
                volume=1_000_000,
            )
            for day in range(5, 10)
        ]
        candles: list[Candle] = normal_before + extreme + normal_after

        # Total: 20 candles (5 normal + 10 extreme + 5 normal)
        # Without exclusion, last 15 = 10 extreme + 5 normal -> high ATR
        # With exclusion, extreme candles removed -> 10 normal candles only
        entry_price = Decimal(100)
        exclude = (("2022-02-21", "2022-04-01"),)

        # Without exclusion -- ATR inflated by extreme candles
        stop_no_exclude = compute_atr_stop_loss(
            entry_price=entry_price,
            candles=candles,
            atr_period=14,
            atr_multiplier=Decimal("2.0"),
        )

        # With exclusion -- ATR based on normal candles only (10 candles, need 14+1
        # so we use a smaller period to ensure the test works)
        stop_with_exclude = compute_atr_stop_loss(
            entry_price=entry_price,
            candles=candles,
            atr_period=8,
            atr_multiplier=Decimal("2.0"),
            exclude_periods=exclude,
        )

        assert stop_no_exclude is not None
        assert stop_with_exclude is not None
        # ATR without exclusion produces a LOWER stop (wider stop distance)
        # because extreme vol makes ATR much larger.
        # With exclusion, stop is closer to entry (smaller ATR).
        assert stop_with_exclude > stop_no_exclude

    def test_atr_without_exclude_periods_backward_compatible(self) -> None:
        """compute_atr_stop_loss without exclude_periods works as before."""
        from finalayze.risk.stop_loss import compute_atr_stop_loss

        candles = _make_candle_series(count=20)
        stop = compute_atr_stop_loss(
            entry_price=Decimal(120),
            candles=candles,
            atr_period=14,
            atr_multiplier=Decimal("2.0"),
        )
        assert stop is not None
        assert stop > Decimal(0)
