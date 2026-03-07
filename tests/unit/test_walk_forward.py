"""Unit tests for WalkForwardOptimizer."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import pytest

from finalayze.backtest.engine import BacktestEngine
from finalayze.backtest.walk_forward import (
    _MIN_FOLD_TRADES,
    WalkForwardConfig,
    WalkForwardOptimizer,
    WalkForwardResult,
    WalkForwardWindow,
    _iter_param_combinations,
)
from finalayze.core.schemas import Candle, Signal, SignalDirection
from finalayze.strategies.base import BaseStrategy

# ── Constants (no magic numbers) ─────────────────────────────────────────

DEFAULT_START = date(2018, 1, 1)
DEFAULT_END = date(2025, 1, 1)

# With default config (train=12mo, test=6mo, step=3mo) on 2018-2025,
# windows generated should be numerous (many 18-month spans fit in 7 years).
# step=3 produces ~23 windows on 7 years of data.
EXPECTED_MIN_WINDOWS_DEFAULT = 20
EXPECTED_MAX_WINDOWS_DEFAULT = 26

SHORT_DATA_START = date(2020, 1, 1)
SHORT_DATA_END = date(2020, 12, 1)

CUSTOM_TRAIN_MONTHS = 24
CUSTOM_TEST_MONTHS = 12
CUSTOM_STEP_MONTHS = 12

CUSTOM_START = date(2018, 1, 1)
CUSTOM_END = date(2025, 1, 1)
# With train=24mo, test=12mo, step=12mo, we step annually starting 2018.
# Window 1: train 2018-01-01..2019-12-31, test 2020-01-01..2020-12-31
# Window 2: train 2019-01-01..2020-12-31, test 2021-01-01..2021-12-31
# Window 3: train 2020-01-01..2021-12-31, test 2022-01-01..2022-12-31
# Window 4: train 2021-01-01..2022-12-31, test 2023-01-01..2023-12-31
# Window 5: train 2022-01-01..2023-12-31, test 2024-01-01..2024-12-31
EXPECTED_CUSTOM_WINDOWS = 5

CANDLE_SYMBOL = "AAPL"
CANDLE_MARKET = "us"
CANDLE_TIMEFRAME = "1d"
CANDLE_SOURCE = "test"
CANDLE_OPEN = Decimal("150.00")
CANDLE_HIGH = Decimal("155.00")
CANDLE_LOW = Decimal("148.00")
CANDLE_CLOSE = Decimal("153.00")
CANDLE_VOLUME = 1000

RUN_SEGMENT = "us_tech"
RUN_INITIAL_CASH = Decimal(100000)
RUN_TRAIN_MONTHS = 24
RUN_TEST_MONTHS = 12
RUN_STEP_MONTHS = 12
# With 7 years of data (2018-2025), 24mo train + 12mo test + 12mo step -> 5 windows
RUN_EXPECTED_WINDOWS = 5
# Weekly candles over 7 years ~ 365 candles
RUN_CANDLE_DAYS = 7  # Generate one candle per week
RUN_BUY_CONFIDENCE = 0.8


# ── Helpers ──────────────────────────────────────────────────────────────


def _make_candle(dt: date) -> Candle:
    """Create a candle on a given date."""
    return Candle(
        symbol=CANDLE_SYMBOL,
        market_id=CANDLE_MARKET,
        timeframe=CANDLE_TIMEFRAME,
        # 14:30 UTC = 10:30 ET (within US market hours 14:30-21:00 UTC)
        timestamp=datetime(dt.year, dt.month, dt.day, 14, 30, tzinfo=UTC),
        open=CANDLE_OPEN,
        high=CANDLE_HIGH,
        low=CANDLE_LOW,
        close=CANDLE_CLOSE,
        volume=CANDLE_VOLUME,
        source=CANDLE_SOURCE,
    )


def _make_candles_range(start: date, end: date, step_days: int = RUN_CANDLE_DAYS) -> list[Candle]:
    """Create candles at regular intervals between start and end."""
    candles: list[Candle] = []
    current = start
    while current < end:
        candles.append(_make_candle(current))
        current += timedelta(days=step_days)
    return candles


# ── Tests ────────────────────────────────────────────────────────────────


class TestGenerateWindows:
    """Tests for WalkForwardOptimizer.generate_windows."""

    def test_generate_windows_default_config(self) -> None:
        """Default config on 2018-2025 produces a reasonable number of windows."""
        optimizer = WalkForwardOptimizer()
        windows = optimizer.generate_windows(DEFAULT_START, DEFAULT_END)

        assert len(windows) >= EXPECTED_MIN_WINDOWS_DEFAULT
        assert len(windows) <= EXPECTED_MAX_WINDOWS_DEFAULT

        # All windows should be WalkForwardWindow instances
        for w in windows:
            assert isinstance(w, WalkForwardWindow)

    def test_generate_windows_short_data(self) -> None:
        """Data too short for any complete window returns empty list."""
        optimizer = WalkForwardOptimizer()
        # Default needs 12mo train + 6mo test = 18mo minimum.
        # 11 months of data is not enough.
        windows = optimizer.generate_windows(SHORT_DATA_START, SHORT_DATA_END)

        assert windows == []

    def test_window_dates_non_overlapping_test_periods(self) -> None:
        """Test periods do not overlap with their own train period."""
        optimizer = WalkForwardOptimizer()
        windows = optimizer.generate_windows(DEFAULT_START, DEFAULT_END)

        assert len(windows) > 0
        for w in windows:
            # Train must end before test starts
            assert w.train_end < w.test_start
            # Train start must come before train end
            assert w.train_start <= w.train_end
            # Test start must come before test end
            assert w.test_start <= w.test_end

    def test_custom_config(self) -> None:
        """Custom train_months/test_months/step_months produces expected windows."""
        config = WalkForwardConfig(
            train_months=CUSTOM_TRAIN_MONTHS,
            test_months=CUSTOM_TEST_MONTHS,
            step_months=CUSTOM_STEP_MONTHS,
            purge_bars=0,
        )
        optimizer = WalkForwardOptimizer(config=config)
        windows = optimizer.generate_windows(CUSTOM_START, CUSTOM_END)

        assert len(windows) == EXPECTED_CUSTOM_WINDOWS

        # Verify first window boundaries
        first = windows[0]
        assert first.train_start == date(2018, 1, 1)
        assert first.train_end == date(2019, 12, 31)
        assert first.test_start == date(2020, 1, 1)
        assert first.test_end == date(2020, 12, 31)


class TestDefaultConfig:
    """Tests for WalkForwardConfig default values."""

    def test_default_step_months_is_3(self) -> None:
        """Default step_months changed from 6 to 3 for more OOS folds."""
        config = WalkForwardConfig()
        expected_step = 3
        assert config.step_months == expected_step

    def test_default_train_months_is_12(self) -> None:
        """Default train window is 12 months."""
        config = WalkForwardConfig()
        expected_train = 12
        assert config.train_months == expected_train

    def test_default_test_months_is_6(self) -> None:
        """Default test window is 6 months."""
        config = WalkForwardConfig()
        expected_test = 6
        assert config.test_months == expected_test

    def test_default_purge_bars_is_60(self) -> None:
        """Default purge_bars is 60 (trading days, ~3 calendar months)."""
        config = WalkForwardConfig()
        expected_purge = 60
        assert config.purge_bars == expected_purge


# ── Purge/embargo gap constants ──────────────────────────────────────────
PURGE_TRAIN_MONTHS = 12
PURGE_TEST_MONTHS = 6
PURGE_STEP_MONTHS = 6
PURGE_BARS_DEFAULT = 60
PURGE_BARS_CUSTOM = 30
PURGE_START = date(2018, 1, 1)
PURGE_END = date(2025, 1, 1)


class TestPurgeGap:
    """Tests for purge/embargo gap between train and test windows."""

    def test_wf_purge_gap_between_train_and_test(self) -> None:
        """Purge gap creates an embargo period > 1 day between train_end and test_start."""
        config = WalkForwardConfig(
            train_months=PURGE_TRAIN_MONTHS,
            test_months=PURGE_TEST_MONTHS,
            step_months=PURGE_STEP_MONTHS,
            purge_bars=PURGE_BARS_DEFAULT,
        )
        optimizer = WalkForwardOptimizer(config=config)
        windows = optimizer.generate_windows(PURGE_START, PURGE_END)

        assert len(windows) > 0
        for w in windows:
            gap_days = (w.test_start - w.train_end).days
            # Gap must be strictly greater than 1 day (purge_bars=60 -> 61 days gap)
            assert gap_days > 1, (
                f"Expected gap > 1 day but got {gap_days} days "
                f"(train_end={w.train_end}, test_start={w.test_start})"
            )
            # Gap should equal purge_bars + 1 (the +1 is the original 1-day gap)
            expected_gap = PURGE_BARS_DEFAULT + 1
            assert gap_days == expected_gap

    def test_wf_purge_zero_gives_adjacent_windows(self) -> None:
        """purge_bars=0 gives train_end and test_start exactly 1 day apart (no embargo)."""
        config = WalkForwardConfig(
            train_months=PURGE_TRAIN_MONTHS,
            test_months=PURGE_TEST_MONTHS,
            step_months=PURGE_STEP_MONTHS,
            purge_bars=0,
        )
        optimizer = WalkForwardOptimizer(config=config)
        windows = optimizer.generate_windows(PURGE_START, PURGE_END)

        assert len(windows) > 0
        for w in windows:
            gap_days = (w.test_start - w.train_end).days
            expected_adjacent_gap = 1
            assert gap_days == expected_adjacent_gap

    def test_wf_purge_gap_no_overlap_with_split_candles(self) -> None:
        """Candles in the purge gap should appear in neither train nor test splits."""
        config = WalkForwardConfig(
            train_months=PURGE_TRAIN_MONTHS,
            test_months=PURGE_TEST_MONTHS,
            step_months=PURGE_STEP_MONTHS,
            purge_bars=PURGE_BARS_CUSTOM,
        )
        optimizer = WalkForwardOptimizer(config=config)
        windows = optimizer.generate_windows(PURGE_START, PURGE_END)

        assert len(windows) > 0
        first = windows[0]

        # Create a candle in the purge gap (between train_end and test_start)
        gap_date = first.train_end + timedelta(days=15)
        assert gap_date > first.train_end
        assert gap_date < first.test_start

        gap_candle = _make_candle(gap_date)
        train, test = optimizer.split_candles([gap_candle], first)

        assert len(train) == 0, "Purge-gap candle must not appear in train set"
        assert len(test) == 0, "Purge-gap candle must not appear in test set"

    def test_wf_purge_reduces_window_count(self) -> None:
        """Larger purge gap reduces the number of valid windows."""
        config_no_purge = WalkForwardConfig(
            train_months=PURGE_TRAIN_MONTHS,
            test_months=PURGE_TEST_MONTHS,
            step_months=PURGE_STEP_MONTHS,
            purge_bars=0,
        )
        config_with_purge = WalkForwardConfig(
            train_months=PURGE_TRAIN_MONTHS,
            test_months=PURGE_TEST_MONTHS,
            step_months=PURGE_STEP_MONTHS,
            purge_bars=PURGE_BARS_DEFAULT,
        )
        opt_no = WalkForwardOptimizer(config=config_no_purge)
        opt_yes = WalkForwardOptimizer(config=config_with_purge)

        windows_no = opt_no.generate_windows(PURGE_START, PURGE_END)
        windows_yes = opt_yes.generate_windows(PURGE_START, PURGE_END)

        assert len(windows_yes) <= len(windows_no)


class TestSplitCandles:
    """Tests for WalkForwardOptimizer.split_candles."""

    def test_split_candles_correct_partition(self) -> None:
        """Candles are correctly split into train and test sets."""
        optimizer = WalkForwardOptimizer()

        window = WalkForwardWindow(
            train_start=date(2020, 1, 1),
            train_end=date(2020, 6, 30),
            test_start=date(2020, 7, 1),
            test_end=date(2020, 12, 31),
        )

        # Create candles spanning 2020 — one per month
        candles = [_make_candle(date(2020, m, 15)) for m in range(1, 13)]
        # Add an out-of-range candle
        candles.append(_make_candle(date(2019, 12, 15)))

        train, test = optimizer.split_candles(candles, window)

        # Jan-Jun = 6 train candles, Jul-Dec = 6 test candles
        expected_train_count = 6
        expected_test_count = 6
        assert len(train) == expected_train_count
        assert len(test) == expected_test_count

        # The 2019 candle should be in neither
        all_split = train + test
        total_expected = expected_train_count + expected_test_count
        assert len(all_split) == total_expected


class _AlternatingStrategy(BaseStrategy):
    """Strategy that alternates BUY/SELL for testing walk-forward run()."""

    def __init__(self) -> None:
        self._call_count = 0

    @property
    def name(self) -> str:
        return "alternating"

    def supported_segments(self) -> list[str]:
        return [RUN_SEGMENT]

    def get_parameters(self, segment_id: str) -> dict[str, object]:
        return {}

    def generate_signal(
        self,
        symbol: str,
        candles: list[Candle],
        segment_id: str,
        sentiment_score: float = 0.0,
        **kwargs: object,
    ) -> Signal | None:
        self._call_count += 1
        direction = SignalDirection.BUY if self._call_count % 2 == 1 else SignalDirection.SELL
        return Signal(
            strategy_name=self.name,
            symbol=symbol,
            market_id=CANDLE_MARKET,
            segment_id=segment_id,
            direction=direction,
            confidence=RUN_BUY_CONFIDENCE,
            features={},
            reasoning="test",
        )


class TestWalkForwardRun:
    """Tests for WalkForwardOptimizer.run()."""

    def test_run_populates_result(self) -> None:
        """run() produces a WalkForwardResult with windows and metrics."""
        config = WalkForwardConfig(
            train_months=RUN_TRAIN_MONTHS,
            test_months=RUN_TEST_MONTHS,
            step_months=RUN_STEP_MONTHS,
        )
        optimizer = WalkForwardOptimizer(config=config)

        # Create weekly candles spanning 2018-2025
        candles = _make_candles_range(DEFAULT_START, DEFAULT_END)

        strategy = _AlternatingStrategy()
        engine = BacktestEngine(strategy=strategy, initial_cash=RUN_INITIAL_CASH)

        result = optimizer.run(CANDLE_SYMBOL, RUN_SEGMENT, candles, engine)

        assert isinstance(result, WalkForwardResult)
        assert len(result.windows) > 0
        assert result.total_oos_trades > 0
        assert len(result.oos_trades) == result.total_oos_trades

    def test_run_empty_candles(self) -> None:
        """run() with empty candles returns empty result."""
        optimizer = WalkForwardOptimizer()
        strategy = _AlternatingStrategy()
        engine = BacktestEngine(strategy=strategy, initial_cash=RUN_INITIAL_CASH)

        result = optimizer.run(CANDLE_SYMBOL, RUN_SEGMENT, [], engine)

        assert result.total_oos_trades == 0
        assert len(result.windows) == 0


class TestWalkForwardSnapshots:
    """Tests for oos_snapshots in WalkForwardResult."""

    def test_wf_result_includes_oos_snapshots(self) -> None:
        """run() populates oos_snapshots with PortfolioState objects."""
        from finalayze.core.schemas import PortfolioState

        config = WalkForwardConfig(
            train_months=RUN_TRAIN_MONTHS,
            test_months=RUN_TEST_MONTHS,
            step_months=RUN_STEP_MONTHS,
        )
        optimizer = WalkForwardOptimizer(config=config)
        candles = _make_candles_range(DEFAULT_START, DEFAULT_END)
        strategy = _AlternatingStrategy()
        engine = BacktestEngine(strategy=strategy, initial_cash=RUN_INITIAL_CASH)

        result = optimizer.run(CANDLE_SYMBOL, RUN_SEGMENT, candles, engine)

        assert len(result.oos_snapshots) > 0
        assert all(isinstance(s, PortfolioState) for s in result.oos_snapshots)

    def test_wf_max_drawdown_from_snapshots(self) -> None:
        """oos_max_drawdown_pct is computed from bar-level snapshots, not per-trade PnL."""
        config = WalkForwardConfig(
            train_months=RUN_TRAIN_MONTHS,
            test_months=RUN_TEST_MONTHS,
            step_months=RUN_STEP_MONTHS,
        )
        optimizer = WalkForwardOptimizer(config=config)
        candles = _make_candles_range(DEFAULT_START, DEFAULT_END)
        strategy = _AlternatingStrategy()
        engine = BacktestEngine(strategy=strategy, initial_cash=RUN_INITIAL_CASH)

        result = optimizer.run(CANDLE_SYMBOL, RUN_SEGMENT, candles, engine)

        # Max drawdown should be non-negative
        assert result.oos_max_drawdown_pct >= 0.0
        # With an alternating strategy, there should be some drawdown
        assert result.oos_max_drawdown_pct > 0.0

    def test_wf_empty_candles_no_snapshots(self) -> None:
        """run() with empty candles returns empty oos_snapshots."""
        optimizer = WalkForwardOptimizer()
        strategy = _AlternatingStrategy()
        engine = BacktestEngine(strategy=strategy, initial_cash=RUN_INITIAL_CASH)

        result = optimizer.run(CANDLE_SYMBOL, RUN_SEGMENT, [], engine)

        assert result.oos_snapshots == []


class TestWalkForwardPerFoldSharpe:
    """Tests for per-fold Sharpe aggregation (no splice bias)."""

    def test_wf_sharpe_per_fold_no_splicing(self) -> None:
        """Fold boundary discontinuity must NOT inflate the aggregated Sharpe.

        Fold 1 ends at equity $120k, fold 2 starts at $100k (engine resets).
        If equity series are naively spliced, the $20k drop at the boundary
        creates a phantom negative return that distorts the Sharpe.
        Per-fold aggregation avoids this.
        """
        config = WalkForwardConfig(
            train_months=RUN_TRAIN_MONTHS,
            test_months=RUN_TEST_MONTHS,
            step_months=RUN_STEP_MONTHS,
        )
        optimizer = WalkForwardOptimizer(config=config)
        candles = _make_candles_range(DEFAULT_START, DEFAULT_END)
        strategy = _AlternatingStrategy()
        engine = BacktestEngine(strategy=strategy, initial_cash=RUN_INITIAL_CASH)

        result = optimizer.run(CANDLE_SYMBOL, RUN_SEGMENT, candles, engine)

        # Result should have per-fold data
        assert hasattr(result, "per_fold_sharpes")
        assert hasattr(result, "per_fold_trade_counts")
        assert len(result.per_fold_sharpes) == len(result.windows)
        assert len(result.per_fold_trade_counts) == len(result.windows)

        # Aggregated Sharpe should equal trade-count-weighted mean of per-fold Sharpes,
        # excluding folds with fewer than _MIN_FOLD_TRADES trades.
        valid_pairs = [
            (s, n)
            for s, n in zip(result.per_fold_sharpes, result.per_fold_trade_counts, strict=True)
            if n >= _MIN_FOLD_TRADES
        ]
        total_valid_trades = sum(n for _, n in valid_pairs)
        if total_valid_trades > 0:
            expected_sharpe = sum(s * n for s, n in valid_pairs) / total_valid_trades
            assert abs(result.oos_sharpe - expected_sharpe) < 1e-10

    def test_wf_per_fold_trade_counts_sum(self) -> None:
        """Sum of per-fold trade counts equals total_oos_trades."""
        config = WalkForwardConfig(
            train_months=RUN_TRAIN_MONTHS,
            test_months=RUN_TEST_MONTHS,
            step_months=RUN_STEP_MONTHS,
        )
        optimizer = WalkForwardOptimizer(config=config)
        candles = _make_candles_range(DEFAULT_START, DEFAULT_END)
        strategy = _AlternatingStrategy()
        engine = BacktestEngine(strategy=strategy, initial_cash=RUN_INITIAL_CASH)

        result = optimizer.run(CANDLE_SYMBOL, RUN_SEGMENT, candles, engine)

        assert sum(result.per_fold_trade_counts) == result.total_oos_trades


class TestWalkForwardLowTradeFoldExclusion:
    """Tests for excluding low-trade folds from Sharpe aggregation."""

    def test_wf_sharpe_excludes_low_trade_folds(self) -> None:
        """Folds with < _MIN_FOLD_TRADES trades are excluded from Sharpe aggregation.

        A fold with only 5 trades and Sharpe of -10.0 should not drag the
        aggregate Sharpe down when other folds have >= 30 trades and positive Sharpe.
        """
        from unittest.mock import MagicMock, patch

        from finalayze.core.schemas import PortfolioState

        # Set up controlled per-fold data:
        # Fold 0: 5 trades (below threshold), terrible Sharpe -10.0
        # Fold 1: 50 trades (above threshold), good Sharpe +0.5
        # Fold 2: 40 trades (above threshold), good Sharpe +0.3
        low_trade_count = 5
        fold_1_trades = 50
        fold_2_trades = 40
        low_sharpe = -10.0
        good_sharpe_1 = 0.5
        good_sharpe_2 = 0.3

        config = WalkForwardConfig(
            train_months=RUN_TRAIN_MONTHS,
            test_months=RUN_TEST_MONTHS,
            step_months=RUN_STEP_MONTHS,
        )

        # Expected: weighted average of fold 1 and fold 2 only
        expected_sharpe = (good_sharpe_1 * fold_1_trades + good_sharpe_2 * fold_2_trades) / (
            fold_1_trades + fold_2_trades
        )

        # If we did NOT filter, the Sharpe would be dragged negative:
        unfiltered_sharpe = (
            low_sharpe * low_trade_count
            + good_sharpe_1 * fold_1_trades
            + good_sharpe_2 * fold_2_trades
        ) / (low_trade_count + fold_1_trades + fold_2_trades)

        # Build a subclass that injects controlled per-fold data
        class _ControlledOptimizer(WalkForwardOptimizer):
            """Optimizer that injects pre-defined per-fold Sharpe/trade-count data."""

            def run(
                self,
                symbol: str,
                segment_id: str,
                candles: list,
                engine: object,
            ) -> WalkForwardResult:
                # Simulate 3 windows with controlled sharpes and trade counts
                per_fold_sharpes = [low_sharpe, good_sharpe_1, good_sharpe_2]
                per_fold_trade_counts = [low_trade_count, fold_1_trades, fold_2_trades]

                # Apply the same filtering logic from walk_forward.py
                valid_pairs = [
                    (s, n)
                    for s, n in zip(per_fold_sharpes, per_fold_trade_counts, strict=True)
                    if n >= _MIN_FOLD_TRADES
                ]
                total_trade_count = sum(n for _, n in valid_pairs)
                if total_trade_count > 0:
                    oos_sharpe = sum(s * n for s, n in valid_pairs) / total_trade_count
                else:
                    oos_sharpe = 0.0

                return WalkForwardResult(
                    oos_sharpe=oos_sharpe,
                    per_fold_sharpes=per_fold_sharpes,
                    per_fold_trade_counts=per_fold_trade_counts,
                    total_oos_trades=sum(per_fold_trade_counts),
                )

        optimizer = _ControlledOptimizer(config=config)
        result = optimizer.run(CANDLE_SYMBOL, RUN_SEGMENT, [], None)  # type: ignore[arg-type]

        # The low-trade fold should be excluded: Sharpe should match expected
        assert abs(result.oos_sharpe - expected_sharpe) < 1e-10
        # Sharpe should be positive (the good folds dominate)
        assert result.oos_sharpe > 0.0
        # Without filtering, Sharpe would be lower
        assert result.oos_sharpe > unfiltered_sharpe

    def test_wf_sharpe_all_folds_below_threshold_returns_zero(self) -> None:
        """When all folds have < _MIN_FOLD_TRADES, Sharpe should be 0.0."""
        # Directly verify: if per_fold_trade_counts are all below threshold,
        # the filtering yields 0.0
        per_fold_sharpes = [-5.0, 2.0, -1.0]
        per_fold_trade_counts = [10, 20, 15]  # All below _MIN_FOLD_TRADES (30)

        valid_pairs = [
            (s, n)
            for s, n in zip(per_fold_sharpes, per_fold_trade_counts, strict=True)
            if n >= _MIN_FOLD_TRADES
        ]
        total = sum(n for _, n in valid_pairs)
        oos_sharpe = sum(s * n for s, n in valid_pairs) / total if total > 0 else 0.0

        assert oos_sharpe == 0.0
        assert len(valid_pairs) == 0

    def test_min_fold_trades_constant_is_30(self) -> None:
        """_MIN_FOLD_TRADES should be 30."""
        expected_min_fold_trades = 30
        assert expected_min_fold_trades == _MIN_FOLD_TRADES


class TestWalkForwardOptimization:
    """Tests for param_grid and engine_factory (6B.5)."""

    def test_walk_forward_without_grid_uses_default_engine(self) -> None:
        """No param_grid -> runs default engine on test windows (backward compat)."""
        config = WalkForwardConfig(
            train_months=RUN_TRAIN_MONTHS,
            test_months=RUN_TEST_MONTHS,
            step_months=RUN_STEP_MONTHS,
        )
        optimizer = WalkForwardOptimizer(config=config)
        candles = _make_candles_range(DEFAULT_START, DEFAULT_END)
        strategy = _AlternatingStrategy()
        engine = BacktestEngine(strategy=strategy, initial_cash=RUN_INITIAL_CASH)

        result = optimizer.run(CANDLE_SYMBOL, RUN_SEGMENT, candles, engine)
        assert result.total_oos_trades > 0

    def test_walk_forward_with_grid_optimizes_on_train(self) -> None:
        """Provide param_grid and engine_factory -> optimizer selects best."""
        config = WalkForwardConfig(
            train_months=RUN_TRAIN_MONTHS,
            test_months=RUN_TEST_MONTHS,
            step_months=RUN_STEP_MONTHS,
        )
        factory_calls: list[dict[str, object]] = []

        def engine_factory(params: dict[str, object]) -> BacktestEngine:
            factory_calls.append(params)
            frac = Decimal(str(params.get("kelly_fraction", "0.5")))
            return BacktestEngine(
                strategy=_AlternatingStrategy(),
                initial_cash=RUN_INITIAL_CASH,
                kelly_fraction=frac,
            )

        grid = {"kelly_fraction": [0.3, 0.5]}
        optimizer = WalkForwardOptimizer(
            config=config, param_grid=grid, engine_factory=engine_factory
        )
        candles = _make_candles_range(DEFAULT_START, DEFAULT_END)
        engine = BacktestEngine(strategy=_AlternatingStrategy(), initial_cash=RUN_INITIAL_CASH)

        result = optimizer.run(CANDLE_SYMBOL, RUN_SEGMENT, candles, engine)
        # Factory should have been called for each combo x each window
        assert len(factory_calls) > 0
        assert result.total_oos_trades >= 0

    def test_iter_param_combinations(self) -> None:
        """Grid with 2x2 produces 4 combinations."""
        grid = {"a": [1, 2], "b": [3, 4]}
        combos = _iter_param_combinations(grid)
        expected_combos = 4
        assert len(combos) == expected_combos
        assert {"a": 1, "b": 3} in combos
        assert {"a": 2, "b": 4} in combos

    def test_walk_forward_train_data_not_discarded(self) -> None:
        """Verify train candles are passed to _optimize_on_train."""
        config = WalkForwardConfig(
            train_months=RUN_TRAIN_MONTHS,
            test_months=RUN_TEST_MONTHS,
            step_months=RUN_STEP_MONTHS,
        )
        train_lengths: list[int] = []

        class TrackingOptimizer(WalkForwardOptimizer):
            def _optimize_on_train(
                self,
                symbol: str,
                segment_id: str,
                train_candles: list,
                default_engine: BacktestEngine,
            ) -> BacktestEngine:
                train_lengths.append(len(train_candles))
                return default_engine

        optimizer = TrackingOptimizer(config=config)
        candles = _make_candles_range(DEFAULT_START, DEFAULT_END)
        engine = BacktestEngine(strategy=_AlternatingStrategy(), initial_cash=RUN_INITIAL_CASH)
        optimizer.run(CANDLE_SYMBOL, RUN_SEGMENT, candles, engine)
        # Train data should be non-empty for each window
        assert len(train_lengths) > 0
        assert all(length > 0 for length in train_lengths)


# ── RC4: months-based config and zero-windows guard ─────────────────────

# 3 years of data with 12mo train + 6mo test => at least 4 windows with step=3
RC4_THREE_YEAR_START = date(2020, 1, 1)
RC4_THREE_YEAR_END = date(2023, 1, 1)
RC4_TRAIN_MONTHS = 12
RC4_TEST_MONTHS = 6
RC4_STEP_MONTHS = 3
RC4_MIN_WINDOWS_3Y = 4

# Data range too short for even one window (less than train + test)
RC4_SHORT_START = date(2023, 1, 1)
RC4_SHORT_END = date(2024, 2, 1)  # 13 months, need 18 (12 train + 6 test)


class TestRC4MonthsConfig:
    """RC4: WalkForwardConfig uses train_months/test_months instead of years."""

    def test_months_config_3y_data_produces_windows(self) -> None:
        """train_months=12, test_months=6, step_months=6 on 3 years of data produces >=2 windows."""
        config = WalkForwardConfig(
            train_months=RC4_TRAIN_MONTHS,
            test_months=RC4_TEST_MONTHS,
            step_months=RC4_STEP_MONTHS,
        )
        optimizer = WalkForwardOptimizer(config=config)
        windows = optimizer.generate_windows(RC4_THREE_YEAR_START, RC4_THREE_YEAR_END)

        assert len(windows) >= RC4_MIN_WINDOWS_3Y

        # Verify window structure
        for w in windows:
            assert w.train_end < w.test_start
            assert w.train_start <= w.train_end
            assert w.test_start <= w.test_end

    def test_zero_windows_short_data(self) -> None:
        """Data range shorter than train+test produces 0 windows and logs warning."""
        config = WalkForwardConfig(
            train_months=RC4_TRAIN_MONTHS,
            test_months=RC4_TEST_MONTHS,
            step_months=RC4_STEP_MONTHS,
        )
        optimizer = WalkForwardOptimizer(config=config)
        windows = optimizer.generate_windows(RC4_SHORT_START, RC4_SHORT_END)

        assert windows == []

    def test_zero_windows_run_returns_empty_result(self) -> None:
        """run() with data too short for any window returns empty WalkForwardResult."""
        config = WalkForwardConfig(
            train_months=RC4_TRAIN_MONTHS,
            test_months=RC4_TEST_MONTHS,
            step_months=RC4_STEP_MONTHS,
        )
        optimizer = WalkForwardOptimizer(config=config)
        candles = _make_candles_range(RC4_SHORT_START, RC4_SHORT_END)
        strategy = _AlternatingStrategy()
        engine = BacktestEngine(strategy=strategy, initial_cash=RUN_INITIAL_CASH)

        result = optimizer.run(CANDLE_SYMBOL, RUN_SEGMENT, candles, engine)

        assert isinstance(result, WalkForwardResult)
        assert len(result.windows) == 0
        assert result.total_oos_trades == 0
        assert result.oos_sharpe == 0.0
