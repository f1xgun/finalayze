"""Walk-forward validation for bond strategies.

Bond-appropriate parameters:
- 24-month train window (to capture full CBR cycles)
- 12-month test window (bonds move slowly)
- 6-month step (overlapping folds for more data points)
- Minimum 5 trades per fold (event strategies are low-frequency)

All Sharpe ratios are computed as EXCESS over RUONIA (risk-free rate),
because a 15% return on RUB bonds with 15% risk-free is Sharpe ~ 0.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from dateutil.relativedelta import relativedelta

if TYPE_CHECKING:
    from datetime import date

# ── Constants ─────────────────────────────────────────────────────────────────
_DEFAULT_TRAIN_MONTHS = 24
_DEFAULT_TEST_MONTHS = 12
_DEFAULT_STEP_MONTHS = 6
_DEFAULT_MIN_TRADES_PER_FOLD = 5
_DEFAULT_TRADING_DAYS_PER_YEAR = 252
_DEFAULT_RUONIA_ANNUAL_PCT = 15.0
_PERCENT = 100
_MIN_RETURNS_FOR_SHARPE = 2
_CONSISTENCY_THRESHOLD = 0.5


@dataclass(frozen=True)
class BondWalkForwardFold:
    """Result from a single walk-forward fold."""

    fold_idx: int
    train_start: date
    train_end: date
    test_start: date
    test_end: date

    # In-sample metrics
    is_trades: int
    is_return_pct: float
    is_excess_sharpe: float

    # Out-of-sample metrics
    oos_trades: int
    oos_return_pct: float
    oos_excess_sharpe: float

    # Fold quality
    sufficient_trades: bool  # True if oos_trades >= min_trades


@dataclass(frozen=True)
class BondWalkForwardResult:
    """Aggregated walk-forward result across all folds."""

    folds: list[BondWalkForwardFold] = field(default_factory=list)

    # Aggregated OOS metrics
    avg_oos_excess_sharpe: float = 0.0
    median_oos_excess_sharpe: float = 0.0
    min_oos_excess_sharpe: float = 0.0
    max_oos_excess_sharpe: float = 0.0

    # Consistency
    n_positive_folds: int = 0  # folds with oos_excess_sharpe > 0
    n_total_folds: int = 0
    consistency_ratio: float = 0.0  # n_positive / n_total

    # Quality
    n_sufficient_trade_folds: int = 0  # folds meeting min trade count
    total_oos_trades: int = 0

    # Verdict
    passes_validation: bool = False  # avg OOS excess Sharpe > 0 AND consistency > 50%


def generate_wf_windows(
    start_date: date,
    end_date: date,
    train_months: int = _DEFAULT_TRAIN_MONTHS,
    test_months: int = _DEFAULT_TEST_MONTHS,
    step_months: int = _DEFAULT_STEP_MONTHS,
) -> list[tuple[date, date, date, date]]:
    """Generate walk-forward windows (train_start, train_end, test_start, test_end).

    Each window advances by step_months. Windows that would extend past end_date
    are skipped. train_end is the day before test_start (no gap, no overlap).
    """
    windows: list[tuple[date, date, date, date]] = []
    current_start = start_date

    while True:
        train_end = current_start + relativedelta(months=train_months) - relativedelta(days=1)
        test_start = train_end + relativedelta(days=1)
        test_end = test_start + relativedelta(months=test_months) - relativedelta(days=1)

        if test_end > end_date:
            break

        windows.append((current_start, train_end, test_start, test_end))
        current_start = current_start + relativedelta(months=step_months)

    return windows


def run_bond_walk_forward(
    equity_curve: list[float],
    dates: list[date],
    trades: list[Any],
    train_months: int = _DEFAULT_TRAIN_MONTHS,
    test_months: int = _DEFAULT_TEST_MONTHS,
    step_months: int = _DEFAULT_STEP_MONTHS,
    min_trades_per_fold: int = _DEFAULT_MIN_TRADES_PER_FOLD,
    risk_free_annual_pct: float = _DEFAULT_RUONIA_ANNUAL_PCT,
) -> BondWalkForwardResult:
    """Run walk-forward validation on a bond backtest result.

    This takes pre-computed equity curve and trades (from BondBacktestEngine)
    and slices them into IS/OOS windows for validation.

    It does NOT re-run the engine -- it analyzes the existing results using
    time-based slicing. This is "walk-forward analysis" not "walk-forward optimization".

    Steps:
    1. Generate WF windows from the date range
    2. For each fold: slice equity_curve and trades into train/test periods
    3. Compute IS and OOS excess Sharpe for each fold
    4. Aggregate results
    """
    if not equity_curve or not dates:
        return BondWalkForwardResult()

    if len(equity_curve) != len(dates):
        msg = "equity_curve and dates must have the same length"
        raise ValueError(msg)

    start_date = dates[0]
    end_date = dates[-1]

    windows = generate_wf_windows(start_date, end_date, train_months, test_months, step_months)
    if not windows:
        return BondWalkForwardResult()

    folds: list[BondWalkForwardFold] = []
    for fold_idx, (train_start, train_end, test_start, test_end) in enumerate(windows):
        fold = _process_fold(
            fold_idx=fold_idx,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            equity_curve=equity_curve,
            dates=dates,
            trades=trades,
            min_trades_per_fold=min_trades_per_fold,
            risk_free_annual_pct=risk_free_annual_pct,
        )
        folds.append(fold)

    return _aggregate_folds(folds)


# ── Private helpers ──────────────────────────────────────────────────────────


def _process_fold(
    *,
    fold_idx: int,
    train_start: date,
    train_end: date,
    test_start: date,
    test_end: date,
    equity_curve: list[float],
    dates: list[date],
    trades: list[Any],
    min_trades_per_fold: int,
    risk_free_annual_pct: float,
) -> BondWalkForwardFold:
    """Process a single walk-forward fold."""
    # Slice equity curve for IS and OOS periods
    is_indices = [i for i, d in enumerate(dates) if train_start <= d <= train_end]
    oos_indices = [i for i, d in enumerate(dates) if test_start <= d <= test_end]

    is_equity = [equity_curve[i] for i in is_indices]
    oos_equity = [equity_curve[i] for i in oos_indices]

    # Slice trades by exit_date (trade belongs to the period where it was closed)
    is_trades = _filter_trades(trades, train_start, train_end)
    oos_trades = _filter_trades(trades, test_start, test_end)

    # Compute metrics
    is_return_pct = _compute_return_pct(is_equity)
    oos_return_pct = _compute_return_pct(oos_equity)

    is_excess_sharpe = _compute_excess_sharpe_from_equity(is_equity, risk_free_annual_pct)
    oos_excess_sharpe = _compute_excess_sharpe_from_equity(oos_equity, risk_free_annual_pct)

    return BondWalkForwardFold(
        fold_idx=fold_idx,
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
        is_trades=len(is_trades),
        is_return_pct=is_return_pct,
        is_excess_sharpe=is_excess_sharpe,
        oos_trades=len(oos_trades),
        oos_return_pct=oos_return_pct,
        oos_excess_sharpe=oos_excess_sharpe,
        sufficient_trades=len(oos_trades) >= min_trades_per_fold,
    )


def _filter_trades(trades: list[Any], start: date, end: date) -> list[Any]:
    """Filter trades whose exit_date falls within [start, end]."""
    result: list[Any] = []
    for t in trades:
        exit_date = getattr(t, "exit_date", None)
        if exit_date is not None and start <= exit_date <= end:
            result.append(t)
    return result


def _compute_return_pct(equity: list[float]) -> float:
    """Compute total return as a percentage from an equity sub-curve."""
    if len(equity) < _MIN_RETURNS_FOR_SHARPE:
        return 0.0
    if equity[0] <= 0:
        return 0.0
    return (equity[-1] / equity[0] - 1.0) * _PERCENT


def _compute_excess_sharpe_from_equity(
    equity: list[float],
    risk_free_annual_pct: float,
    trading_days_per_year: int = _DEFAULT_TRADING_DAYS_PER_YEAR,
) -> float:
    """Compute annualised excess Sharpe from an equity sub-curve.

    Excess returns = daily returns - daily risk-free rate.
    Sharpe = mean(excess) / std(excess) * sqrt(252).
    """
    if len(equity) < _MIN_RETURNS_FOR_SHARPE + 1:
        return 0.0

    # Daily returns
    daily_returns = [
        equity[i] / equity[i - 1] - 1.0 for i in range(1, len(equity)) if equity[i - 1] > 0
    ]

    if len(daily_returns) < _MIN_RETURNS_FOR_SHARPE:
        return 0.0

    # Daily risk-free rate (continuous compounding approximation)
    daily_rf = (1 + risk_free_annual_pct / _PERCENT) ** (1 / trading_days_per_year) - 1.0

    # Excess daily returns
    excess = [r - daily_rf for r in daily_returns]

    mean_excess = statistics.mean(excess)
    if len(excess) < _MIN_RETURNS_FOR_SHARPE:
        return 0.0

    std_excess = statistics.stdev(excess)
    if std_excess <= 0:
        return 0.0

    return float(mean_excess / std_excess * math.sqrt(trading_days_per_year))


def _aggregate_folds(folds: list[BondWalkForwardFold]) -> BondWalkForwardResult:
    """Aggregate individual fold results into a single walk-forward result."""
    if not folds:
        return BondWalkForwardResult()

    n_total = len(folds)
    oos_sharpes = [f.oos_excess_sharpe for f in folds]
    n_positive = sum(1 for s in oos_sharpes if s > 0)
    consistency = n_positive / n_total if n_total > 0 else 0.0

    avg_sharpe = statistics.mean(oos_sharpes)
    median_sharpe = statistics.median(oos_sharpes)

    n_sufficient = sum(1 for f in folds if f.sufficient_trades)
    total_oos_trades = sum(f.oos_trades for f in folds)

    passes = avg_sharpe > 0 and consistency > _CONSISTENCY_THRESHOLD

    return BondWalkForwardResult(
        folds=folds,
        avg_oos_excess_sharpe=avg_sharpe,
        median_oos_excess_sharpe=median_sharpe,
        min_oos_excess_sharpe=min(oos_sharpes),
        max_oos_excess_sharpe=max(oos_sharpes),
        n_positive_folds=n_positive,
        n_total_folds=n_total,
        consistency_ratio=consistency,
        n_sufficient_trade_folds=n_sufficient,
        total_oos_trades=total_oos_trades,
        passes_validation=passes,
    )
