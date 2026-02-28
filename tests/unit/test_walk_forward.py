"""Unit tests for WalkForwardOptimizer."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import pytest

from finalayze.backtest.engine import BacktestEngine
from finalayze.backtest.walk_forward import (
    WalkForwardConfig,
    WalkForwardOptimizer,
    WalkForwardResult,
    WalkForwardWindow,
)
from finalayze.core.schemas import Candle, Signal, SignalDirection
from finalayze.strategies.base import BaseStrategy

# ── Constants (no magic numbers) ─────────────────────────────────────────

DEFAULT_START = date(2018, 1, 1)
DEFAULT_END = date(2025, 1, 1)

# With default config (train=3yr, test=1yr, step=6mo) on 2018-2025,
# windows generated should be approximately 7.
EXPECTED_MIN_WINDOWS_DEFAULT = 5
EXPECTED_MAX_WINDOWS_DEFAULT = 9

SHORT_DATA_START = date(2020, 1, 1)
SHORT_DATA_END = date(2022, 6, 1)

CUSTOM_TRAIN_YEARS = 2
CUSTOM_TEST_YEARS = 1
CUSTOM_STEP_MONTHS = 12

CUSTOM_START = date(2018, 1, 1)
CUSTOM_END = date(2025, 1, 1)
# With train=2yr, test=1yr, step=12mo, we step annually starting 2018.
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
RUN_TRAIN_YEARS = 2
RUN_TEST_YEARS = 1
RUN_STEP_MONTHS = 12
# With 7 years of data (2018-2025), 2yr train + 1yr test + 12mo step -> 5 windows
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
        timestamp=datetime(dt.year, dt.month, dt.day, tzinfo=UTC),
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
        # Default needs 3yr train + 1yr test = 4yr minimum.
        # 2.5 years of data is not enough.
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
        """Custom train_years/test_years/step_months produces expected windows."""
        config = WalkForwardConfig(
            train_years=CUSTOM_TRAIN_YEARS,
            test_years=CUSTOM_TEST_YEARS,
            step_months=CUSTOM_STEP_MONTHS,
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
            train_years=RUN_TRAIN_YEARS,
            test_years=RUN_TEST_YEARS,
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
