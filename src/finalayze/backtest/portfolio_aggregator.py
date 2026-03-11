"""Portfolio aggregator -- combines per-layer backtest results (Layer 5).

Each layer runs independently (own broker, own engine). The aggregator:
1. Aligns equity curves to common date index
2. Computes combined equity curve (sum of per-layer values)
3. Computes portfolio-level metrics (Sharpe, DD, etc.)
4. Checks portfolio-level circuit breaker (-10% DD)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import date

# ── Constants ────────────────────────────────────────────────────────────────
_DEFAULT_DD_LIMIT = 0.10  # 10%
_DEFAULT_RUONIA_ANNUAL_PCT = 15.0
_DEFAULT_TRADING_DAYS = 252
_MIN_RETURNS_FOR_SHARPE = 1  # ddof=1 requires at least 2 values

# Phase 4 exit criteria thresholds
_STRATEGIC_DD_LIMIT_PCT = 8.0  # strategic layer DD must be < 8%
_CORE_MIN_RETURN_PCT = 0.0  # core return must be positive (> 0%)
_CORE_CALMAR_THRESHOLD = 1.5  # core Calmar ratio > 1.5
_ABSOLUTE_SHARPE_THRESHOLD = 0.3  # portfolio absolute Sharpe > 0.3
_SHORT_PF_THRESHOLD = 0.8  # short equity profit factor > 0.8
_SOFT_GATES_MIN_PASS = 2  # minimum soft gates to pass (out of 3)


@dataclass(frozen=True)
class LayerResult:
    """Result from a single portfolio layer."""

    layer_id: str  # "core", "strategic", "tactical", "short"
    equity_curve: list[float]  # daily equity values
    dates: list[date]
    trades: list  # TradeResult list  # type: ignore[type-arg]
    total_return_pct: float
    max_drawdown_pct: float
    coupon_income_net: float = 0.0
    sharpe: float = 0.0
    profit_factor: float = 0.0


@dataclass(frozen=True)
class PortfolioResult:
    """Combined portfolio result across all layers."""

    # Per-layer breakdown
    layer_results: dict[str, LayerResult]

    # Portfolio-level metrics
    combined_equity_curve: list[float]
    combined_dates: list[date]
    total_return_pct: float
    annualized_return_pct: float
    excess_return_pct: float  # over RUONIA
    excess_sharpe: float
    max_drawdown_pct: float

    # Portfolio DD breach info
    portfolio_dd_breach: bool  # True if DD > 10%
    portfolio_dd_breach_date: date | None

    # Aggregate trade metrics
    total_trades: int
    total_coupon_income_net: float

    # Per-layer contribution
    layer_return_contribution: dict[str, float]  # layer_id -> % contribution to total

    # ── Phase 4 multi-tier exit criteria ─────────────────────────────────
    # Absolute Sharpe (risk-free=0, measures raw risk-adjusted return)
    absolute_sharpe: float = 0.0

    # Per-layer tier validation
    core_return_vs_ruonia: float = 0.0  # core return minus cumulative RUONIA (in %)
    strategic_dd_ok: bool = True  # True if strategic layer DD < 8%
    tactical_has_trades: bool = False  # True if tactical layer has >= 1 trade

    # Tier results
    hard_gates_passed: int = 0  # out of 4
    hard_gates_total: int = 4
    soft_gates_passed: int = 0  # out of 3
    soft_gates_total: int = 3
    phase4_exit_ok: bool = False  # True if all hard gates pass AND >= 2/3 soft gates


class PortfolioAggregator:
    """Aggregates per-layer backtest results into portfolio metrics."""

    def __init__(
        self,
        portfolio_dd_limit: float = _DEFAULT_DD_LIMIT,
        risk_free_annual_pct: float = _DEFAULT_RUONIA_ANNUAL_PCT,
        trading_days_per_year: int = _DEFAULT_TRADING_DAYS,
    ) -> None:
        self._dd_limit = portfolio_dd_limit
        self._rf = risk_free_annual_pct
        self._tdays = trading_days_per_year

    def aggregate(self, layer_results: list[LayerResult]) -> PortfolioResult:
        """Combine layer results into portfolio result.

        Args:
            layer_results: Results from each layer's independent backtest.

        Returns:
            PortfolioResult with combined metrics.
        """
        if not layer_results:
            return self._empty_result()

        # Build common date index (union of all layer dates)
        all_dates_set: set[date] = set()
        for lr in layer_results:
            all_dates_set.update(lr.dates)
        common_dates = sorted(all_dates_set)

        if not common_dates:
            return self._empty_result()

        # Interpolate each layer's equity to common dates (forward-fill)
        layer_curves = self._align_curves(layer_results, common_dates)

        # Combined equity curve = sum of all layers
        combined = [
            sum(layer_curves[lr.layer_id][i] for lr in layer_results)
            for i in range(len(common_dates))
        ]

        # Compute portfolio metrics
        initial_value = combined[0] if combined else 0.0
        final_value = combined[-1] if combined else 0.0

        total_return = (final_value / initial_value - 1.0) if initial_value > 0 else 0.0
        n_years = len(combined) / self._tdays
        ann_return = (1 + total_return) ** (1.0 / n_years) - 1.0 if n_years > 0 else 0.0
        excess_return = total_return - (self._rf / 100) * n_years

        # Excess Sharpe from daily returns
        excess_sharpe = self._compute_excess_sharpe(combined)

        # Absolute Sharpe (no risk-free subtraction)
        absolute_sharpe = self._compute_absolute_sharpe(combined)

        # Max drawdown and DD breach
        max_dd, dd_breach, dd_breach_date = self._compute_drawdown(combined, common_dates)

        # Per-layer contribution
        layer_contribution = self._compute_layer_contributions(
            layer_results, initial_value, final_value
        )

        # Aggregate trade and coupon totals
        total_trades = sum(len(lr.trades) for lr in layer_results)
        total_coupon = sum(lr.coupon_income_net for lr in layer_results)

        # Build layer lookup for tier checks
        layer_map = {lr.layer_id: lr for lr in layer_results}

        # ── Phase 4 multi-tier exit criteria ────────────────────────────
        # Per-layer tier metrics
        core_return_vs_ruonia = self._compute_core_return_vs_ruonia(
            layer_map,
            n_years,
        )
        strategic_dd_ok = self._check_strategic_dd(layer_map)
        tactical_has_trades = self._check_tactical_has_trades(layer_map)

        # Hard gates (all 4 must pass)
        hard_1_dd = max_dd * 100 < self._dd_limit * 100  # portfolio DD < 10%
        core_lr = layer_map.get("core")
        core_return_pct = core_lr.total_return_pct if core_lr else 0.0
        hard_2_core = core_return_pct > _CORE_MIN_RETURN_PCT  # core absolute return > 0%
        hard_3_strategic_dd = strategic_dd_ok
        hard_4_tactical_trades = tactical_has_trades
        hard_gates = [hard_1_dd, hard_2_core, hard_3_strategic_dd, hard_4_tactical_trades]
        hard_gates_passed = sum(hard_gates)

        # Soft gates (2 of 3 must pass)
        core_calmar = self._compute_core_calmar(layer_map, n_years)
        short_pf = self._get_short_profit_factor(layer_map)

        soft_1_calmar = core_calmar > _CORE_CALMAR_THRESHOLD
        soft_2_sharpe = absolute_sharpe > _ABSOLUTE_SHARPE_THRESHOLD
        soft_3_short_pf = short_pf > _SHORT_PF_THRESHOLD
        soft_gates = [soft_1_calmar, soft_2_sharpe, soft_3_short_pf]
        soft_gates_passed = sum(soft_gates)

        phase4_exit_ok = (
            hard_gates_passed == len(hard_gates) and soft_gates_passed >= _SOFT_GATES_MIN_PASS
        )

        return PortfolioResult(
            layer_results=layer_map,
            combined_equity_curve=combined,
            combined_dates=common_dates,
            total_return_pct=total_return * 100,
            annualized_return_pct=ann_return * 100,
            excess_return_pct=excess_return * 100,
            excess_sharpe=excess_sharpe,
            max_drawdown_pct=max_dd * 100,
            portfolio_dd_breach=dd_breach,
            portfolio_dd_breach_date=dd_breach_date,
            total_trades=total_trades,
            total_coupon_income_net=total_coupon,
            layer_return_contribution=layer_contribution,
            absolute_sharpe=absolute_sharpe,
            core_return_vs_ruonia=core_return_vs_ruonia,
            strategic_dd_ok=strategic_dd_ok,
            tactical_has_trades=tactical_has_trades,
            hard_gates_passed=hard_gates_passed,
            hard_gates_total=len(hard_gates),
            soft_gates_passed=soft_gates_passed,
            soft_gates_total=len(soft_gates),
            phase4_exit_ok=phase4_exit_ok,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _align_curves(
        layer_results: list[LayerResult],
        common_dates: list[date],
    ) -> dict[str, list[float]]:
        """Interpolate each layer's equity to common dates via forward-fill."""
        layer_curves: dict[str, list[float]] = {}
        for lr in layer_results:
            date_to_equity = dict(zip(lr.dates, lr.equity_curve, strict=False))
            curve: list[float] = []
            last_val = lr.equity_curve[0] if lr.equity_curve else 0.0
            for d in common_dates:
                if d in date_to_equity:
                    last_val = date_to_equity[d]
                curve.append(last_val)
            layer_curves[lr.layer_id] = curve
        return layer_curves

    def _compute_excess_sharpe(self, combined: list[float]) -> float:
        """Compute annualised excess Sharpe from combined equity curve."""
        daily_returns: list[float] = []
        for i in range(1, len(combined)):
            if combined[i - 1] > 0:
                daily_returns.append(combined[i] / combined[i - 1] - 1.0)
            else:
                daily_returns.append(0.0)

        daily_rf = (1 + self._rf / 100) ** (1 / self._tdays) - 1.0
        excess_daily = [r - daily_rf for r in daily_returns]

        if len(excess_daily) <= _MIN_RETURNS_FOR_SHARPE:
            return 0.0

        mean_excess = sum(excess_daily) / len(excess_daily)
        var = sum((r - mean_excess) ** 2 for r in excess_daily) / (len(excess_daily) - 1)
        std = math.sqrt(var)

        if std <= 0:
            return 0.0

        return mean_excess / std * math.sqrt(self._tdays)

    def _compute_absolute_sharpe(self, combined: list[float]) -> float:
        """Compute annualised absolute Sharpe (no risk-free subtraction)."""
        daily_returns: list[float] = []
        for i in range(1, len(combined)):
            if combined[i - 1] > 0:
                daily_returns.append(combined[i] / combined[i - 1] - 1.0)
            else:
                daily_returns.append(0.0)

        if len(daily_returns) <= _MIN_RETURNS_FOR_SHARPE:
            return 0.0

        mean_ret = sum(daily_returns) / len(daily_returns)
        var = sum((r - mean_ret) ** 2 for r in daily_returns) / (len(daily_returns) - 1)
        std = math.sqrt(var)

        if std <= 0:
            return 0.0

        return mean_ret / std * math.sqrt(self._tdays)

    def _compute_core_return_vs_ruonia(
        self,
        layer_map: dict[str, LayerResult],
        n_years: float,
    ) -> float:
        """Compute core layer return minus cumulative RUONIA (in %).

        Returns the difference: core_return_pct - cumulative_ruonia_pct.
        If no core layer exists, returns 0.0.
        """
        core = layer_map.get("core")
        if core is None:
            return 0.0
        cumulative_ruonia = self._rf * n_years  # already in %
        return core.total_return_pct - cumulative_ruonia

    @staticmethod
    def _check_strategic_dd(layer_map: dict[str, LayerResult]) -> bool:
        """Check that strategic layer DD is below 8%. True if OK or layer absent."""
        strategic = layer_map.get("strategic")
        if strategic is None:
            return True
        return strategic.max_drawdown_pct < _STRATEGIC_DD_LIMIT_PCT

    @staticmethod
    def _check_tactical_has_trades(layer_map: dict[str, LayerResult]) -> bool:
        """Check that tactical layer has at least 1 trade."""
        tactical = layer_map.get("tactical")
        if tactical is None:
            return False
        return len(tactical.trades) >= 1

    def _compute_core_calmar(
        self,
        layer_map: dict[str, LayerResult],
        n_years: float,
    ) -> float:
        """Compute Core Calmar ratio: annualised return / max DD.

        Uses total return over the period divided by max DD.
        Returns 0.0 if no core layer or DD is zero.
        """
        core = layer_map.get("core")
        if core is None or core.max_drawdown_pct <= 0 or n_years <= 0:
            return 0.0
        ann_return: float = ((1 + core.total_return_pct / 100) ** (1.0 / n_years) - 1.0) * 100
        return ann_return / core.max_drawdown_pct

    @staticmethod
    def _get_short_profit_factor(layer_map: dict[str, LayerResult]) -> float:
        """Get profit factor from short layer. Returns 0.0 if absent."""
        short = layer_map.get("short")
        if short is None:
            return 0.0
        return short.profit_factor

    def _compute_drawdown(
        self,
        combined: list[float],
        common_dates: list[date],
    ) -> tuple[float, bool, date | None]:
        """Compute max drawdown and check for DD limit breach.

        Returns:
            Tuple of (max_dd_fraction, dd_breach, dd_breach_date).
        """
        if not combined:
            return 0.0, False, None

        peak = combined[0]
        max_dd = 0.0
        dd_breach = False
        dd_breach_date: date | None = None

        for i, val in enumerate(combined):
            peak = max(peak, val)
            dd = (peak - val) / peak if peak > 0 else 0.0
            max_dd = max(max_dd, dd)
            if dd > self._dd_limit and not dd_breach:
                dd_breach = True
                dd_breach_date = common_dates[i]

        return max_dd, dd_breach, dd_breach_date

    @staticmethod
    def _compute_layer_contributions(
        layer_results: list[LayerResult],
        initial_value: float,  # noqa: ARG004
        final_value: float,  # noqa: ARG004
    ) -> dict[str, float]:
        """Compute per-layer percentage contribution to total PnL."""
        layer_contribution: dict[str, float] = {}
        total_pnl = sum(
            (lr.equity_curve[-1] - lr.equity_curve[0]) if lr.equity_curve else 0.0
            for lr in layer_results
        )
        for lr in layer_results:
            lr_pnl = (lr.equity_curve[-1] - lr.equity_curve[0]) if lr.equity_curve else 0.0
            layer_contribution[lr.layer_id] = (lr_pnl / total_pnl * 100) if total_pnl != 0 else 0.0
        return layer_contribution

    def _empty_result(self) -> PortfolioResult:
        """Return PortfolioResult with zero values."""
        return PortfolioResult(
            layer_results={},
            combined_equity_curve=[],
            combined_dates=[],
            total_return_pct=0.0,
            annualized_return_pct=0.0,
            excess_return_pct=0.0,
            excess_sharpe=0.0,
            max_drawdown_pct=0.0,
            portfolio_dd_breach=False,
            portfolio_dd_breach_date=None,
            total_trades=0,
            total_coupon_income_net=0.0,
            layer_return_contribution={},
            absolute_sharpe=0.0,
            core_return_vs_ruonia=0.0,
            strategic_dd_ok=True,
            tactical_has_trades=False,
            hard_gates_passed=0,
            hard_gates_total=4,
            soft_gates_passed=0,
            soft_gates_total=3,
            phase4_exit_ok=False,
        )
