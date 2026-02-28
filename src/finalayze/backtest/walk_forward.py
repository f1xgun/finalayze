"""Walk-forward optimization framework for out-of-sample validation.

Splits historical data into rolling train/test windows so that strategy
performance can be evaluated on truly unseen data.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from datetime import date  # noqa: TC003
from typing import TYPE_CHECKING

from dateutil.relativedelta import relativedelta

if TYPE_CHECKING:
    from finalayze.backtest.engine import BacktestEngine
    from finalayze.core.schemas import Candle, TradeResult

# ── Constants ────────────────────────────────────────────────────────────
_ANNUALIZATION_FACTOR = 252  # Trading days per year
_PERCENT = 100.0


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
    oos_trades: list[TradeResult] = field(default_factory=list)
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

    def run(
        self,
        symbol: str,
        segment_id: str,
        candles: list[Candle],
        engine: BacktestEngine,
    ) -> WalkForwardResult:
        """Run walk-forward validation over all windows.

        For each window, splits candles into train/test, runs the backtest
        engine on the test (OOS) candles, and aggregates metrics.

        Args:
            symbol: Ticker symbol (e.g. "AAPL").
            segment_id: Market segment identifier.
            candles: Full candle history spanning all windows.
            engine: BacktestEngine configured with a strategy.

        Returns:
            WalkForwardResult with aggregated out-of-sample metrics.
        """
        if not candles:
            return WalkForwardResult()

        start_date = min(c.timestamp.date() for c in candles)
        end_date = max(c.timestamp.date() for c in candles)
        windows = self.generate_windows(start_date, end_date)

        all_trades: list[TradeResult] = []

        for window in windows:
            _train, test = self.split_candles(candles, window)
            if not test:
                continue
            trades, _snapshots = engine.run(symbol, segment_id, test)
            all_trades.extend(trades)

        pnl_pcts = [float(t.pnl_pct) * _PERCENT for t in all_trades]

        return WalkForwardResult(
            windows=windows,
            oos_trades=all_trades,
            total_oos_trades=len(all_trades),
            oos_sharpe=_compute_sharpe(pnl_pcts),
            oos_total_return_pct=_compute_total_return(pnl_pcts),
            oos_win_rate=_compute_win_rate(pnl_pcts),
            oos_max_drawdown_pct=_compute_max_drawdown(pnl_pcts),
        )


# ── Private metric helpers ───────────────────────────────────────────────


def _compute_sharpe(pnl_pcts: list[float]) -> float:
    """Annualized Sharpe ratio from per-trade pnl_pct values."""
    if len(pnl_pcts) < 2:  # noqa: PLR2004
        return 0.0
    mean = statistics.mean(pnl_pcts)
    stdev = statistics.stdev(pnl_pcts)
    if stdev == 0.0:
        return 0.0
    return mean / stdev * math.sqrt(_ANNUALIZATION_FACTOR)


def _compute_total_return(pnl_pcts: list[float]) -> float:
    """Compounded total return from per-trade pnl_pct values."""
    if not pnl_pcts:
        return 0.0
    equity = 1.0
    for pct in pnl_pcts:
        equity *= 1.0 + pct / _PERCENT
    return (equity - 1.0) * _PERCENT


def _compute_win_rate(pnl_pcts: list[float]) -> float:
    """Win rate as a percentage of profitable trades."""
    if not pnl_pcts:
        return 0.0
    wins = sum(1 for p in pnl_pcts if p > 0)
    return wins / len(pnl_pcts) * _PERCENT


def _compute_max_drawdown(pnl_pcts: list[float]) -> float:
    """Maximum peak-to-trough drawdown from per-trade equity curve."""
    if not pnl_pcts:
        return 0.0
    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    for pct in pnl_pcts:
        equity *= 1.0 + pct / _PERCENT
        peak = max(peak, equity)
        dd = (peak - equity) / peak * _PERCENT
        max_dd = max(max_dd, dd)
    return max_dd
