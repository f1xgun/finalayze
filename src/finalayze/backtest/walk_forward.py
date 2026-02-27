"""Walk-forward optimization framework for out-of-sample validation.

Splits historical data into rolling train/test windows so that strategy
performance can be evaluated on truly unseen data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date  # noqa: TC003
from typing import TYPE_CHECKING

from dateutil.relativedelta import relativedelta

if TYPE_CHECKING:
    from finalayze.core.schemas import BacktestResult, Candle


@dataclass(frozen=True, slots=True)
class WalkForwardWindow:
    """A single train/test window."""

    train_start: date
    train_end: date
    test_start: date
    test_end: date


@dataclass(frozen=True, slots=True)
class WalkForwardConfig:
    """Configuration for walk-forward window generation."""

    train_years: int = 3
    test_years: int = 1
    step_months: int = 6


@dataclass
class WalkForwardResult:
    """Aggregate results from all out-of-sample windows."""

    windows: list[WalkForwardWindow] = field(default_factory=list)
    oos_results: list[BacktestResult] = field(default_factory=list)
    total_oos_trades: int = 0
    oos_sharpe: float = 0.0
    oos_total_return_pct: float = 0.0
    oos_win_rate: float = 0.0
    oos_max_drawdown_pct: float = 0.0


class WalkForwardOptimizer:
    """Split data into rolling train/test windows for out-of-sample validation."""

    def __init__(self, config: WalkForwardConfig | None = None) -> None:
        self._config = config or WalkForwardConfig()

    def generate_windows(self, start_date: date, end_date: date) -> list[WalkForwardWindow]:
        """Generate rolling train/test windows.

        Example with default config (train=3yr, test=1yr, step=6mo) on 2018-2025:
        Window 1: Train 2018-01 to 2020-12, Test 2021-01 to 2021-12
        Window 2: Train 2018-07 to 2021-06, Test 2021-07 to 2022-06
        ...etc
        """
        windows: list[WalkForwardWindow] = []
        current_start = start_date

        while True:
            train_end = (
                current_start
                + relativedelta(years=self._config.train_years)
                - relativedelta(days=1)
            )
            test_start = train_end + relativedelta(days=1)
            test_end = (
                test_start + relativedelta(years=self._config.test_years) - relativedelta(days=1)
            )

            if test_end > end_date:
                break

            windows.append(
                WalkForwardWindow(
                    train_start=current_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                )
            )

            current_start = current_start + relativedelta(months=self._config.step_months)

        return windows

    def split_candles(
        self, candles: list[Candle], window: WalkForwardWindow
    ) -> tuple[list[Candle], list[Candle]]:
        """Split candles into train and test sets for a given window."""
        train = [c for c in candles if window.train_start <= c.timestamp.date() <= window.train_end]
        test = [c for c in candles if window.test_start <= c.timestamp.date() <= window.test_end]
        return train, test
