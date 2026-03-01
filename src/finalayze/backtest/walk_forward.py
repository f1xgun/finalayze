"""Walk-forward optimization framework for out-of-sample validation.

Splits historical data into rolling train/test windows so that strategy
performance can be evaluated on truly unseen data.
"""

from __future__ import annotations

import itertools
import math
import statistics
from dataclasses import dataclass, field
from datetime import date  # noqa: TC003
from typing import TYPE_CHECKING, Callable

from dateutil.relativedelta import relativedelta

if TYPE_CHECKING:
    from finalayze.backtest.engine import BacktestEngine
    from finalayze.core.schemas import Candle, TradeResult

ParameterGrid = dict[str, list[object]]

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
    per_window_params: list[dict[str, object]] = field(default_factory=list)


class WalkForwardOptimizer:
    """Split data into rolling train/test windows for out-of-sample validation."""

    def __init__(
        self,
        config: WalkForwardConfig | None = None,
        param_grid: ParameterGrid | None = None,
        engine_factory: Callable[[dict[str, object]], BacktestEngine] | None = None,
    ) -> None:
        self._config = config or WalkForwardConfig()
        self._param_grid = param_grid
        self._engine_factory = engine_factory

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
        all_equities: list[float] = []

        for window in windows:
            train, test = self.split_candles(candles, window)
            if not test:
                continue
            optimized_engine = self._optimize_on_train(symbol, segment_id, train, engine)
            trades, snapshots = optimized_engine.run(symbol, segment_id, test)
            all_trades.extend(trades)
            # Collect bar-level equity values for Sharpe computation (#131)
            all_equities.extend(float(s.equity) for s in snapshots)

        pnl_pcts = [float(t.pnl_pct) * _PERCENT for t in all_trades]

        return WalkForwardResult(
            windows=windows,
            oos_trades=all_trades,
            total_oos_trades=len(all_trades),
            # Use bar-level equity for Sharpe (statistically correct, fixes #131)
            oos_sharpe=_compute_sharpe_from_snapshots(all_equities),
            oos_total_return_pct=_compute_total_return(pnl_pcts),
            oos_win_rate=_compute_win_rate(pnl_pcts),
            oos_max_drawdown_pct=_compute_max_drawdown(pnl_pcts),
        )

    def _optimize_on_train(
        self,
        symbol: str,
        segment_id: str,
        train_candles: list[Candle],
        default_engine: BacktestEngine,
    ) -> BacktestEngine:
        """Find best parameters on training data via grid search.

        Returns the engine with the best-performing parameter set,
        or the default engine if no param_grid/engine_factory is configured.
        """
        if not self._param_grid or not self._engine_factory:
            return default_engine

        best_sharpe = float("-inf")
        best_engine = default_engine

        for combo in _iter_param_combinations(self._param_grid):
            engine = self._engine_factory(combo)
            _trades, snapshots = engine.run(symbol, segment_id, train_candles)
            equities = [float(s.equity) for s in snapshots]
            sharpe = _compute_sharpe_from_snapshots(equities)
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_engine = engine

        return best_engine


def _iter_param_combinations(grid: ParameterGrid) -> list[dict[str, object]]:
    """Generate all combinations from a parameter grid."""
    keys = list(grid.keys())
    values = list(grid.values())
    combos = []
    for combo in itertools.product(*values):
        combos.append(dict(zip(keys, combo, strict=False)))
    return combos


# ── Private metric helpers ───────────────────────────────────────────────


def _compute_sharpe_from_snapshots(equities: list[float]) -> float:
    """Compute annualised Sharpe ratio from bar-level equity values.

    This is statistically correct — Sharpe should be computed on the
    underlying return distribution (one return per bar) rather than on
    per-trade P&L, which can span very different holding periods.
    """
    if len(equities) < 2:  # noqa: PLR2004
        return 0.0
    returns = [
        (equities[i] - equities[i - 1]) / equities[i - 1]
        for i in range(1, len(equities))
        if equities[i - 1] > 0
    ]
    if len(returns) < 2:  # noqa: PLR2004
        return 0.0
    mean = statistics.mean(returns)
    stdev = statistics.stdev(returns)
    if stdev == 0.0:
        return 0.0
    return mean / stdev * math.sqrt(_ANNUALIZATION_FACTOR)


def _compute_sharpe(pnl_pcts: list[float]) -> float:
    """Annualized Sharpe ratio from per-trade pnl_pct values.

    .. deprecated::
        Use :func:`_compute_sharpe_from_snapshots` which operates on bar-level
        returns for statistical correctness.  This function is retained for
        backward compatibility with callers that only have trade P&L data.
    """
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
