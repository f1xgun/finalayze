"""Portfolio backtest orchestrator -- merges bond and equity results.

Combines pre-computed bond and equity backtest results into a unified
portfolio with configurable allocation (default 40/60), monthly rebalancing
when drift exceeds a threshold, and a USDRUB crisis brake.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from finalayze.backtest.bond_walk_forward import (
    _compute_excess_sharpe_from_equity,
    generate_wf_windows,
)

if TYPE_CHECKING:
    from datetime import date

    from finalayze.backtest.bond_engine import BondBacktestResult
    from finalayze.core.schemas import PortfolioState, TradeResult

# Minimum daily returns needed for meaningful Sharpe computation.
_MIN_RETURNS_FOR_SHARPE = 5

# Default RUONIA rate (%) used for excess Sharpe.
_DEFAULT_RISK_FREE_PCT = 15.0

# Trading days per year for annualisation.
_TRADING_DAYS_PER_YEAR = 252

# Percentage conversion factor.
_PERCENT = 100.0

# Minimum curve length for metrics computation.
_MIN_CURVE_LEN = 2


@dataclass
class PortfolioBacktestResult:
    """Aggregated result from a combined bond + equity backtest."""

    bond_equity_curve: list[float]
    equity_equity_curve: list[float]
    merged_equity_curve: list[float]
    dates: list[date]
    bond_trades: list[TradeResult]
    equity_trades: list[TradeResult]
    sharpe: float
    max_drawdown_pct: float
    profit_factor: float
    total_return_pct: float
    bond_weight_series: list[float]
    equity_weight_series: list[float]
    crisis_brake_active_dates: list[date]
    wf_sharpe: float = 0.0


class PortfolioBacktestOrchestrator:
    """Merges bond and equity backtest results into a portfolio.

    Engines already receive their share of capital (e.g. 40% bond, 60% equity).
    The merged curve is the SUM of the two raw curves. Rebalancing applies
    scaling for future bars only at month boundaries.
    """

    def __init__(
        self,
        bond_weight: float = 0.40,
        equity_weight: float = 0.60,
        rebalance_threshold: float = 0.05,
        crisis_usdrub_threshold: float = 0.15,
        crisis_usdrub_window: int = 20,
        crisis_bond_weight: float = 0.80,
    ) -> None:
        self._bond_weight = bond_weight
        self._equity_weight = equity_weight
        self._rebalance_threshold = rebalance_threshold
        self._crisis_threshold = crisis_usdrub_threshold
        self._crisis_window = crisis_usdrub_window
        self._crisis_bond_weight = crisis_bond_weight
        self._crisis_equity_weight = 1.0 - crisis_bond_weight

    def run(
        self,
        bond_result: BondBacktestResult,
        equity_snapshots: list[PortfolioState],
        usdrub_series: list[tuple[date, float]],
        total_capital: float,  # noqa: ARG002
        equity_trades: list[TradeResult] | None = None,
    ) -> PortfolioBacktestResult:
        """Merge bond and equity results into a portfolio result."""
        common_dates, bond_curve, equity_curve = self._align_and_normalize(
            bond_result, equity_snapshots
        )

        usdrub_lookup = dict(usdrub_series)
        sorted_fx_dates = [d for d, _ in usdrub_series]

        (
            bond_weight_series,
            equity_weight_series,
            merged_curve,
            crisis_dates,
        ) = self._apply_allocation_and_rebalancing(
            common_dates,
            bond_curve,
            equity_curve,
            usdrub_lookup,
            sorted_fx_dates,
        )

        sharpe, max_dd, pf, total_return = self._compute_metrics(merged_curve)

        return PortfolioBacktestResult(
            bond_equity_curve=bond_curve,
            equity_equity_curve=equity_curve,
            merged_equity_curve=merged_curve,
            dates=common_dates,
            bond_trades=list(bond_result.trades),
            equity_trades=equity_trades or [],
            sharpe=sharpe,
            max_drawdown_pct=max_dd,
            profit_factor=pf,
            total_return_pct=total_return,
            bond_weight_series=bond_weight_series,
            equity_weight_series=equity_weight_series,
            crisis_brake_active_dates=crisis_dates,
        )

    def compute_walk_forward_sharpe(
        self,
        result: PortfolioBacktestResult,
        train_months: int = 12,
        test_months: int = 6,
        step_months: int = 3,
        risk_free_annual_pct: float = _DEFAULT_RISK_FREE_PCT,
    ) -> float:
        """Compute walk-forward Sharpe on the pre-computed merged equity curve.

        Slices the merged curve into 12mo-train / 6mo-test windows and computes
        excess Sharpe on each OOS slice.  Does NOT re-run engines -- purely
        analytical.

        Returns the average OOS excess Sharpe across folds, or 0.0 if the curve
        is too short for even one fold.  Also sets ``result.wf_sharpe``.
        """
        if len(result.dates) < _MIN_CURVE_LEN:
            result.wf_sharpe = 0.0
            return 0.0

        windows = generate_wf_windows(
            result.dates[0],
            result.dates[-1],
            train_months=train_months,
            test_months=test_months,
            step_months=step_months,
        )

        if not windows:
            result.wf_sharpe = 0.0
            return 0.0

        # Build date -> index lookup for fast slicing
        date_to_idx: dict[date, int] = {d: i for i, d in enumerate(result.dates)}

        oos_sharpes: list[float] = []
        for _train_start, _train_end, test_start, test_end in windows:
            # Find indices for the test period
            test_indices = [
                date_to_idx[d]
                for d in result.dates
                if test_start <= d <= test_end and d in date_to_idx
            ]
            if len(test_indices) < _MIN_CURVE_LEN:
                continue

            test_equity = [result.merged_equity_curve[i] for i in test_indices]
            oos_sharpe = _compute_excess_sharpe_from_equity(test_equity, risk_free_annual_pct)
            oos_sharpes.append(oos_sharpe)

        if not oos_sharpes:
            result.wf_sharpe = 0.0
            return 0.0

        avg_sharpe = sum(oos_sharpes) / len(oos_sharpes)
        result.wf_sharpe = avg_sharpe
        return avg_sharpe

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _align_and_normalize(
        self,
        bond_result: BondBacktestResult,
        equity_snapshots: list[PortfolioState],
    ) -> tuple[list[date], list[float], list[float]]:
        """Align bond and equity curves to common dates via forward-fill.

        Converts bond Decimal curve to float and extracts equity float curve
        from PortfolioState.equity.
        """
        # Build date->value lookups
        bond_lookup: dict[date, float] = {}
        for d, v in zip(bond_result.dates, bond_result.equity_curve, strict=False):
            bond_lookup[d] = float(v)

        equity_lookup: dict[date, float] = {}
        for snap in equity_snapshots:
            d = snap.timestamp.date()
            equity_lookup[d] = float(snap.equity)

        # Union of all dates, sorted
        all_dates = sorted(set(bond_lookup.keys()) | set(equity_lookup.keys()))

        # Forward-fill both curves
        bond_curve: list[float] = []
        equity_curve: list[float] = []
        last_bond = next(iter(bond_lookup.values())) if bond_lookup else 0.0
        last_equity = next(iter(equity_lookup.values())) if equity_lookup else 0.0

        for d in all_dates:
            if d in bond_lookup:
                last_bond = bond_lookup[d]
            bond_curve.append(last_bond)

            if d in equity_lookup:
                last_equity = equity_lookup[d]
            equity_curve.append(last_equity)

        return all_dates, bond_curve, equity_curve

    def _apply_allocation_and_rebalancing(
        self,
        dates: list[date],
        bond_curve: list[float],
        equity_curve: list[float],
        usdrub_lookup: dict[date, float],
        sorted_fx_dates: list[date],
    ) -> tuple[list[float], list[float], list[float], list[date]]:
        """Walk through dates applying allocation, rebalancing, and crisis brake.

        Returns (bond_weight_series, equity_weight_series, merged_curve, crisis_dates).
        """
        if not dates:
            return [], [], [], []

        bond_weight_series: list[float] = []
        equity_weight_series: list[float] = []
        merged_curve: list[float] = []
        crisis_dates: list[date] = []

        # Scale factors -- start at 1.0 (engines already have their capital share)
        bond_scale = 1.0
        equity_scale = 1.0

        active_bond_weight = self._bond_weight

        for i, d in enumerate(dates):
            # Check crisis brake
            is_crisis = self._is_crisis(i, d, usdrub_lookup, sorted_fx_dates)
            if is_crisis:
                active_bond_weight = self._crisis_bond_weight
                crisis_dates.append(d)
            else:
                active_bond_weight = self._bond_weight

            # Check monthly rebalancing
            if i > 0:
                prev_date = dates[i - 1]
                if d.month != prev_date.month:
                    # Month boundary -- check drift and rebalance if needed
                    bond_val = bond_curve[i] * bond_scale
                    equity_val = equity_curve[i] * equity_scale
                    total = bond_val + equity_val
                    if total > 0:
                        current_bond_pct = bond_val / total
                        drift = abs(current_bond_pct - active_bond_weight)
                        if drift > self._rebalance_threshold:
                            target_bond = total * active_bond_weight
                            target_equity = total * (1.0 - active_bond_weight)
                            if bond_curve[i] > 0:
                                bond_scale = target_bond / bond_curve[i]
                            if equity_curve[i] > 0:
                                equity_scale = target_equity / equity_curve[i]

            bond_val = bond_curve[i] * bond_scale
            equity_val = equity_curve[i] * equity_scale

            bond_weight_series.append(active_bond_weight)
            equity_weight_series.append(1.0 - active_bond_weight)
            merged_curve.append(bond_val + equity_val)

        return bond_weight_series, equity_weight_series, merged_curve, crisis_dates

    def _is_crisis(
        self,
        bar_idx: int,
        current_date: date,
        usdrub_lookup: dict[date, float],
        sorted_fx_dates: list[date],
    ) -> bool:
        """Check if USDRUB 20-bar return exceeds crisis threshold."""
        if bar_idx < self._crisis_window:
            return False

        current_rate = usdrub_lookup.get(current_date)
        if current_rate is None:
            return False

        # Find the rate `crisis_window` bars back in the FX series
        lookback_idx = bar_idx - self._crisis_window
        if lookback_idx >= len(sorted_fx_dates):
            return False

        lookback_date = sorted_fx_dates[lookback_idx]
        lookback_rate = usdrub_lookup.get(lookback_date)
        if lookback_rate is None or lookback_rate <= 0:
            return False

        fx_return = (current_rate / lookback_rate) - 1.0
        return fx_return > self._crisis_threshold

    def _compute_metrics(
        self,
        merged_curve: list[float],
    ) -> tuple[float, float, float, float]:
        """Compute (sharpe, max_drawdown_pct, profit_factor, total_return_pct)."""
        if len(merged_curve) < _MIN_CURVE_LEN:
            return 0.0, 0.0, 0.0, 0.0

        # Daily returns
        daily_returns: list[float] = []
        for i in range(1, len(merged_curve)):
            if merged_curve[i - 1] > 0:
                daily_returns.append(merged_curve[i] / merged_curve[i - 1] - 1.0)
            else:
                daily_returns.append(0.0)

        sharpe = self._compute_sharpe(daily_returns)
        max_dd = self._compute_max_drawdown(merged_curve)
        pf = self._compute_profit_factor(daily_returns)

        total_return = 0.0
        if merged_curve[0] > 0:
            total_return = (merged_curve[-1] / merged_curve[0] - 1.0) * _PERCENT

        return sharpe, max_dd, pf, total_return

    def _compute_sharpe(self, daily_returns: list[float]) -> float:
        """Annualised excess Sharpe using RUONIA as risk-free rate."""
        if len(daily_returns) < _MIN_RETURNS_FOR_SHARPE:
            return 0.0

        daily_rf = (1 + _DEFAULT_RISK_FREE_PCT / _PERCENT) ** (1 / _TRADING_DAYS_PER_YEAR) - 1.0
        excess = [r - daily_rf for r in daily_returns]

        mean_excess = sum(excess) / len(excess)
        variance = sum((r - mean_excess) ** 2 for r in excess) / (len(excess) - 1)
        std = math.sqrt(variance)

        if std <= 0:
            return 0.0

        return mean_excess / std * math.sqrt(_TRADING_DAYS_PER_YEAR)

    @staticmethod
    def _compute_max_drawdown(curve: list[float]) -> float:
        """Peak-to-trough max drawdown as percentage."""
        if len(curve) < _MIN_CURVE_LEN:
            return 0.0

        peak = curve[0]
        max_dd = 0.0

        for val in curve[1:]:
            peak = max(peak, val)
            dd = (peak - val) / peak * _PERCENT if peak > 0 else 0.0
            max_dd = max(max_dd, dd)

        return max_dd

    @staticmethod
    def _compute_profit_factor(daily_returns: list[float]) -> float:
        """Sum of gains / abs(sum of losses) from daily returns."""
        gains = sum(r for r in daily_returns if r > 0)
        losses = abs(sum(r for r in daily_returns if r < 0))

        if losses <= 0:
            return float("inf") if gains > 0 else 0.0

        return gains / losses
