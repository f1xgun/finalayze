"""Monte Carlo bootstrap for computing confidence intervals on backtest metrics.

Resamples trade returns with replacement to estimate the distribution of
key performance metrics (total return, Sharpe, drawdown, win rate, profit factor).
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from finalayze.core.schemas import PortfolioState

# Annualisation factor for daily returns.
_TRADING_DAYS_PER_YEAR = 252
_MAX_PROFIT_FACTOR = 100.0


@dataclass(frozen=True, slots=True)
class BootstrapCI:
    """Confidence interval for a metric."""

    point_estimate: float
    lower: float
    upper: float
    confidence_level: float = 0.95


@dataclass(frozen=True, slots=True)
class BootstrapResult:
    """Bootstrap confidence intervals for key metrics."""

    total_return: BootstrapCI
    sharpe_ratio: BootstrapCI
    max_drawdown: BootstrapCI
    win_rate: BootstrapCI
    profit_factor: BootstrapCI
    n_simulations: int
    n_trades: int


def _compute_sample_metrics(
    sample: list[float],
) -> tuple[float, float, float, float, float]:
    """Compute (total_return, sharpe, max_drawdown, win_rate, profit_factor) for a sample."""
    n = len(sample)

    # Total return (compounded) and max drawdown
    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    for r in sample:
        equity *= 1 + r / 100
        peak = max(peak, equity)
        dd = (peak - equity) / peak
        max_dd = max(max_dd, dd)

    total_ret = (equity - 1) * 100
    max_dd_pct = max_dd * 100

    # Sharpe (annualised, assuming ~252 trading days)
    mean_r = statistics.mean(sample)
    if n > 1:
        std_r = statistics.stdev(sample)
        sharpe = (mean_r / std_r * (_TRADING_DAYS_PER_YEAR**0.5)) if std_r > 0 else 0.0
    else:
        sharpe = 0.0

    # Win rate
    wins = sum(1 for r in sample if r > 0)
    win_rate = wins / n * 100

    # Profit factor (capped at 100.0 to avoid inf in CI computation)
    gross_profit = sum(r for r in sample if r > 0)
    gross_loss = abs(sum(r for r in sample if r < 0))
    pf = (
        min(gross_profit / gross_loss, _MAX_PROFIT_FACTOR) if gross_loss > 0 else _MAX_PROFIT_FACTOR
    )

    return total_ret, sharpe, max_dd_pct, win_rate, pf


def _make_ci(values: list[float], point: float, confidence_level: float) -> BootstrapCI:
    """Build a BootstrapCI from a list of bootstrap samples."""
    alpha = (1 - confidence_level) / 2
    sorted_v = sorted(values)
    lo_idx = int(alpha * len(sorted_v))
    hi_idx = int((1 - alpha) * len(sorted_v)) - 1
    return BootstrapCI(
        point_estimate=point,
        lower=sorted_v[lo_idx],
        upper=sorted_v[hi_idx],
        confidence_level=confidence_level,
    )


def bootstrap_metrics(
    trade_returns: list[float],
    n_simulations: int = 10_000,
    confidence_level: float = 0.95,
    seed: int | None = None,
) -> BootstrapResult:
    """Resample trade returns with replacement, compute metrics for each sample.

    Args:
        trade_returns: List of per-trade return percentages.
        n_simulations: Number of bootstrap samples.
        confidence_level: CI level (default 0.95).
        seed: Random seed for reproducibility.

    Returns:
        BootstrapResult with CIs for total return, Sharpe, max DD, win rate,
        profit factor.
    """
    # Use numpy's default_rng for reproducibility and independence from global RNG state
    rng = np.random.default_rng(seed=seed)

    n = len(trade_returns)
    if n == 0:
        zero_ci = BootstrapCI(0.0, 0.0, 0.0, confidence_level)
        return BootstrapResult(
            total_return=zero_ci,
            sharpe_ratio=zero_ci,
            max_drawdown=zero_ci,
            win_rate=zero_ci,
            profit_factor=zero_ci,
            n_simulations=n_simulations,
            n_trades=0,
        )

    pnl_array = np.array(trade_returns, dtype=np.float64)
    # Draw all bootstrap samples at once for efficiency
    indices = rng.choice(n, size=(n_simulations, n), replace=True)

    total_returns: list[float] = []
    sharpes: list[float] = []
    max_drawdowns: list[float] = []
    win_rates: list[float] = []
    profit_factors: list[float] = []

    for sim_indices in indices:
        sample = pnl_array[sim_indices].tolist()
        total_ret, sharpe, max_dd, win_rate, pf = _compute_sample_metrics(sample)
        total_returns.append(total_ret)
        sharpes.append(sharpe)
        max_drawdowns.append(max_dd)
        win_rates.append(win_rate)
        profit_factors.append(pf)

    # Point estimates from actual data
    actual_total, actual_sharpe, actual_dd, actual_wr, actual_pf = _compute_sample_metrics(
        trade_returns
    )

    return BootstrapResult(
        total_return=_make_ci(total_returns, actual_total, confidence_level),
        sharpe_ratio=_make_ci(sharpes, actual_sharpe, confidence_level),
        max_drawdown=_make_ci(max_drawdowns, actual_dd, confidence_level),
        win_rate=_make_ci(win_rates, actual_wr, confidence_level),
        profit_factor=_make_ci(profit_factors, actual_pf, confidence_level),
        n_simulations=n_simulations,
        n_trades=n,
    )


def bootstrap_from_snapshots(
    snapshots: list[PortfolioState],
    n_simulations: int = 10_000,
    confidence_level: float = 0.95,
    seed: int | None = None,
) -> BootstrapResult:
    """Bootstrap confidence intervals from bar-level equity snapshots.

    Extracts daily returns from the equity curve and resamples those
    (not per-trade PnL), making ``sqrt(252)`` annualisation correct.

    Args:
        snapshots: Bar-level PortfolioState snapshots with equity values.
        n_simulations: Number of bootstrap samples.
        confidence_level: CI level (default 0.95).
        seed: Random seed for reproducibility.

    Returns:
        BootstrapResult with CIs computed from daily return distribution.
    """
    if len(snapshots) < 2:  # noqa: PLR2004
        zero_ci = BootstrapCI(0.0, 0.0, 0.0, confidence_level)
        return BootstrapResult(
            total_return=zero_ci,
            sharpe_ratio=zero_ci,
            max_drawdown=zero_ci,
            win_rate=zero_ci,
            profit_factor=zero_ci,
            n_simulations=n_simulations,
            n_trades=0,
        )

    # Extract daily percentage returns from equity curve
    equities = [float(s.equity) for s in snapshots]
    daily_returns = [
        (equities[i] - equities[i - 1]) / equities[i - 1] * 100
        for i in range(1, len(equities))
        if equities[i - 1] > 0
    ]

    if not daily_returns:
        zero_ci = BootstrapCI(0.0, 0.0, 0.0, confidence_level)
        return BootstrapResult(
            total_return=zero_ci,
            sharpe_ratio=zero_ci,
            max_drawdown=zero_ci,
            win_rate=zero_ci,
            profit_factor=zero_ci,
            n_simulations=n_simulations,
            n_trades=0,
        )

    # Delegate to bootstrap_metrics which handles the resampling
    return bootstrap_metrics(
        daily_returns,
        n_simulations=n_simulations,
        confidence_level=confidence_level,
        seed=seed,
    )
