"""Bond-specific performance metrics (Layer 5).

All returns are computed as EXCESS over RUONIA (or provided risk-free rate).
This is critical for RUB bonds where the risk-free rate is ~15%.
A portfolio returning 15% has excess Sharpe ~ 0, not 1.5.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

# ── Constants ────────────────────────────────────────────────────────────────
_TRADING_DAYS_PER_YEAR = 252
_PERCENT = 100
_DEFAULT_RUONIA_ANNUAL_PCT = 15.0
_MIN_EQUITY_POINTS = 2
_MIN_RETURNS_FOR_STD = 1  # ddof=1, need at least 2 values → len-1 >= 1


@dataclass(frozen=True)
class BondPerformanceMetrics:
    """Bond backtest performance metrics."""

    # Absolute returns
    total_return_pct: float
    annualized_return_pct: float

    # Excess returns over risk-free
    excess_return_pct: float  # total excess over RUONIA
    annualized_excess_return_pct: float
    excess_sharpe: float  # Sharpe of excess returns

    # Risk metrics
    max_drawdown_pct: float
    annualized_volatility_pct: float

    # Trade metrics
    trade_count: int
    win_rate: float
    profit_factor: float
    avg_hold_bars: float

    # Bond-specific
    total_coupon_income_gross: float
    total_coupon_income_net: float  # after NDFL tax
    coupon_contribution_pct: float  # % of total return from coupons
    avg_portfolio_duration: float  # average modified duration during backtest


def compute_bond_metrics(
    equity_curve: list[float],
    dates: list[Any],  # noqa: ARG001
    trades: list[Any],
    coupon_income_gross: float,
    coupon_income_net: float,
    initial_cash: float,  # noqa: ARG001
    risk_free_annual_pct: float = _DEFAULT_RUONIA_ANNUAL_PCT,
    trading_days_per_year: int = _TRADING_DAYS_PER_YEAR,
) -> BondPerformanceMetrics:
    """Compute bond-specific performance metrics.

    Args:
        equity_curve: Daily portfolio values.
        dates: Corresponding dates.
        trades: List of completed trades (must have ``pnl`` and ``hold_bars`` attrs).
        coupon_income_gross: Total gross coupon income.
        coupon_income_net: Total net coupon income (after tax).
        initial_cash: Starting capital.
        risk_free_annual_pct: Annual risk-free rate as % (default 15% for RUONIA).
        trading_days_per_year: Trading days per year for annualisation.

    Returns:
        BondPerformanceMetrics with all computed values.
    """
    if len(equity_curve) < _MIN_EQUITY_POINTS:
        return _empty_metrics()

    daily_returns = _compute_daily_returns(equity_curve)

    # Total return
    total_return = equity_curve[-1] / equity_curve[0] - 1.0
    n_days = len(equity_curve)
    n_years = n_days / trading_days_per_year
    annualized_return = (1 + total_return) ** (1.0 / n_years) - 1.0 if n_years > 0 else 0.0

    # Risk-free rate (daily)
    daily_rf = (1 + risk_free_annual_pct / _PERCENT) ** (1 / trading_days_per_year) - 1.0

    # Excess returns
    excess_daily = [r - daily_rf for r in daily_returns]
    excess_return = total_return - (risk_free_annual_pct / _PERCENT) * n_years
    annualized_excess = annualized_return - risk_free_annual_pct / _PERCENT

    # Excess Sharpe
    excess_sharpe = _compute_excess_sharpe(excess_daily, trading_days_per_year)

    # Volatility
    annualized_vol = _compute_annualized_volatility(daily_returns, trading_days_per_year)

    # Max drawdown
    max_dd = _compute_max_drawdown(equity_curve)

    # Trade metrics
    trade_count = len(trades)
    win_rate, profit_factor, avg_hold = _compute_trade_metrics(trades, trade_count)

    # Coupon contribution
    total_pnl = equity_curve[-1] - equity_curve[0]
    coupon_contribution = (coupon_income_net / total_pnl * _PERCENT) if total_pnl > 0 else 0.0

    return BondPerformanceMetrics(
        total_return_pct=total_return * _PERCENT,
        annualized_return_pct=annualized_return * _PERCENT,
        excess_return_pct=excess_return * _PERCENT,
        annualized_excess_return_pct=annualized_excess * _PERCENT,
        excess_sharpe=excess_sharpe,
        max_drawdown_pct=max_dd * _PERCENT,
        annualized_volatility_pct=annualized_vol,
        trade_count=trade_count,
        win_rate=win_rate,
        profit_factor=profit_factor,
        avg_hold_bars=avg_hold,
        total_coupon_income_gross=coupon_income_gross,
        total_coupon_income_net=coupon_income_net,
        coupon_contribution_pct=coupon_contribution,
        avg_portfolio_duration=0.0,  # computed externally
    )


# ── Private helpers ──────────────────────────────────────────────────────────


def _compute_daily_returns(equity_curve: list[float]) -> list[float]:
    """Compute daily simple returns from an equity curve."""
    returns: list[float] = []
    for i in range(1, len(equity_curve)):
        if equity_curve[i - 1] > 0:
            returns.append(equity_curve[i] / equity_curve[i - 1] - 1.0)
        else:
            returns.append(0.0)
    return returns


def _compute_excess_sharpe(excess_daily: list[float], trading_days_per_year: int) -> float:
    """Compute annualised Sharpe ratio from excess daily returns."""
    if not excess_daily:
        return 0.0

    mean_excess = sum(excess_daily) / len(excess_daily)
    n = len(excess_daily)
    if n < _MIN_RETURNS_FOR_STD + 1:
        return 0.0
    var_excess = sum((r - mean_excess) ** 2 for r in excess_daily) / (n - 1)
    std_excess = math.sqrt(var_excess)

    if std_excess <= 0:
        return 0.0

    return mean_excess / std_excess * math.sqrt(trading_days_per_year)


def _compute_annualized_volatility(
    daily_returns: list[float],
    trading_days_per_year: int,
) -> float:
    """Compute annualised volatility in percent."""
    if not daily_returns:
        return 0.0
    n = len(daily_returns)
    if n < _MIN_RETURNS_FOR_STD + 1:
        return 0.0
    mean_ret = sum(daily_returns) / n
    var_ret = sum((r - mean_ret) ** 2 for r in daily_returns) / (n - 1)
    return math.sqrt(var_ret) * math.sqrt(trading_days_per_year) * _PERCENT


def _compute_max_drawdown(equity_curve: list[float]) -> float:
    """Compute maximum peak-to-trough drawdown as a fraction (0-1)."""
    peak = equity_curve[0]
    max_dd = 0.0
    for val in equity_curve:
        peak = max(peak, val)
        dd = (peak - val) / peak if peak > 0 else 0.0
        max_dd = max(max_dd, dd)
    return max_dd


def _compute_trade_metrics(
    trades: list[Any],
    trade_count: int,
) -> tuple[float, float, float]:
    """Compute win_rate, profit_factor, avg_hold_bars from trades."""
    if trade_count == 0:
        return 0.0, 0.0, 0.0

    wins = sum(1 for t in trades if float(getattr(t, "pnl", 0)) > 0)
    win_rate = wins / trade_count

    gross_profit = sum(
        float(getattr(t, "pnl", 0)) for t in trades if float(getattr(t, "pnl", 0)) > 0
    )
    gross_loss = abs(
        sum(float(getattr(t, "pnl", 0)) for t in trades if float(getattr(t, "pnl", 0)) < 0)
    )
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    avg_hold = sum(getattr(t, "hold_bars", 0) or 0 for t in trades) / trade_count

    return win_rate, profit_factor, avg_hold


def _empty_metrics() -> BondPerformanceMetrics:
    """Return zero-valued metrics for empty backtests."""
    return BondPerformanceMetrics(
        total_return_pct=0.0,
        annualized_return_pct=0.0,
        excess_return_pct=0.0,
        annualized_excess_return_pct=0.0,
        excess_sharpe=0.0,
        max_drawdown_pct=0.0,
        annualized_volatility_pct=0.0,
        trade_count=0,
        win_rate=0.0,
        profit_factor=0.0,
        avg_hold_bars=0.0,
        total_coupon_income_gross=0.0,
        total_coupon_income_net=0.0,
        coupon_contribution_pct=0.0,
        avg_portfolio_duration=0.0,
    )
