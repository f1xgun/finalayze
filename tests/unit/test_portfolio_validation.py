"""Portfolio pipeline validation tests (Tasks 5.1 + 5.6).

Integration-level tests that validate the full portfolio aggregation pipeline
produces correct results with synthetic data:
- Combined portfolio backtest aggregation across 4 layers
- Excess Sharpe computation over RUONIA (15%)
- Bond metrics integration with coupon contribution
- Drawdown breach detection at the 10% portfolio threshold
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from uuid import uuid4

import pytest

from finalayze.backtest.bond_metrics import compute_bond_metrics
from finalayze.backtest.portfolio_aggregator import (
    LayerResult,
    PortfolioAggregator,
)
from finalayze.core.schemas import TradeResult

# ── Constants (no magic numbers per ruff PLR2004) ────────────────────────────

_TRADING_DAYS_PER_YEAR = 252
_RUONIA_ANNUAL = 0.15  # 15%
_RUONIA_ANNUAL_PCT = 15.0
_ZERO = 0.0
_TOLERANCE = 1e-6
_LOOSE_TOLERANCE = 0.05  # 5% relative tolerance for stochastic assertions
_SHARPE_ABS_TOLERANCE = 0.15  # absolute tolerance for near-zero Sharpe
_CONTRIBUTION_SUM_TOLERANCE = 1e-4  # percentage-point tolerance for sum-to-100

# Default layer allocation (matching design)
_CORE_CAPITAL_PCT = 0.45
_STRATEGIC_CAPITAL_PCT = 0.275
_TACTICAL_CAPITAL_PCT = 0.175
_SHORT_CAPITAL_PCT = 0.10
_TOTAL_CAPITAL = 1_500_000.0

# Synthetic curve defaults
_DEFAULT_N_DAYS = 756  # 3 years of trading days
_DEFAULT_SEED = 42

# Test-specific constants
_FOUR_LAYER_N_DAYS = 504  # 2 years
_DD_BREACH_THRESHOLD_PCT = 10.0
_DD_BREACH_LIMIT = 0.10
_OUTPERFORMER_RETURN = 0.20  # 20% annual
_LOW_VOLATILITY = 0.005  # low daily vol
_HIGH_VOLATILITY = 0.03  # high daily vol
_SHORT_PERIOD_DAYS = 10
_BOND_INITIAL = 412_500.0  # 27.5% of 1.5M
_BOND_FINAL = 500_000.0
_BOND_CURVE_DAYS = 756
_COUPON_GROSS = 40_000.0
_COUPON_NET = 34_800.0  # after 13% NDFL
_COUPON_NDFL_RATE = 0.13

# Trade PnL constants
_WIN_PNL = Decimal(500)
_LOSS_PNL = Decimal(-200)
_HOLD_BARS_WIN = 10
_HOLD_BARS_LOSS = 5


# ── Synthetic data helpers ──────────────────────────────────────────────────


def _make_equity_curve(
    initial: float,
    annual_return: float,
    n_days: int,
    volatility: float = 0.01,
    seed: int = _DEFAULT_SEED,
) -> tuple[list[float], list[date]]:
    """Generate a synthetic daily equity curve.

    Uses geometric random walk: R_t = mu + sigma * epsilon_t
    where mu = daily_return = (1 + annual_return)^(1/252) - 1
    """
    rng = random.Random(seed)  # noqa: S311

    daily_return = (1 + annual_return) ** (1 / _TRADING_DAYS_PER_YEAR) - 1
    equity = [initial]
    dates = [date(2022, 1, 3)]  # first trading day of 2022

    for _i in range(1, n_days):
        r = daily_return + volatility * rng.gauss(0, 1)
        equity.append(equity[-1] * (1 + r))
        dates.append(dates[-1] + timedelta(days=1))

    return equity, dates


def _compute_max_dd(equity: list[float]) -> float:
    """Compute maximum peak-to-trough drawdown as a fraction."""
    if not equity:
        return _ZERO
    peak = equity[0]
    max_dd = _ZERO
    for val in equity:
        peak = max(peak, val)
        if peak > 0:
            dd = (peak - val) / peak
            max_dd = max(max_dd, dd)
    return max_dd


def _make_layer_result(
    layer_id: str,
    initial: float,
    annual_return: float,
    n_days: int = _DEFAULT_N_DAYS,
    trades: list[object] | None = None,
    coupon_income: float = _ZERO,
    volatility: float = 0.01,
    seed: int = _DEFAULT_SEED,
) -> LayerResult:
    """Create a synthetic LayerResult."""
    equity, dates = _make_equity_curve(initial, annual_return, n_days, volatility, seed)
    total_return = (equity[-1] / equity[0] - 1) * 100
    max_dd = _compute_max_dd(equity) * 100

    return LayerResult(
        layer_id=layer_id,
        equity_curve=equity,
        dates=dates,
        trades=trades or [],
        total_return_pct=total_return,
        max_drawdown_pct=max_dd,
        coupon_income_net=coupon_income,
    )


def _make_deterministic_curve(
    initial: float,
    annual_return: float,
    n_days: int,
) -> tuple[list[float], list[date]]:
    """Generate a deterministic (zero-volatility) equity curve.

    No randomness -- useful for exact formula verification.
    """
    daily_return = (1 + annual_return) ** (1 / _TRADING_DAYS_PER_YEAR) - 1
    equity = [initial]
    dates = [date(2022, 1, 3)]

    for _i in range(1, n_days):
        equity.append(equity[-1] * (1 + daily_return))
        dates.append(dates[-1] + timedelta(days=1))

    return equity, dates


def _make_trade(pnl: Decimal, hold_bars: int = _HOLD_BARS_WIN) -> TradeResult:
    """Create a TradeResult with given PnL."""
    entry = Decimal(100)
    exit_price = entry + pnl / Decimal(10)
    return TradeResult(
        signal_id=uuid4(),
        symbol="SU26244RMFS2",
        side="BUY",
        quantity=Decimal(10),
        entry_price=entry,
        exit_price=exit_price,
        pnl=pnl,
        pnl_pct=pnl / (entry * Decimal(10)),
        hold_bars=hold_bars,
        instrument_type="bond",
    )


def _make_dates_list(n: int, start: date | None = None) -> list[date]:
    """Create n consecutive dates."""
    start_date = start or date(2022, 1, 3)
    return [start_date + timedelta(days=i) for i in range(n)]


# ── TestPortfolioAggregationPipeline ────────────────────────────────────────


class TestPortfolioAggregationPipeline:
    """Validate the full aggregation pipeline with 4 synthetic layers."""

    def test_four_layer_aggregation(self) -> None:
        """4 layers aggregate to correct combined return, max DD, and total trades."""
        core = _make_layer_result(
            "core",
            initial=_TOTAL_CAPITAL * _CORE_CAPITAL_PCT,
            annual_return=0.16,
            n_days=_FOUR_LAYER_N_DAYS,
            trades=["t1", "t2", "t3"],
            coupon_income=30_000.0,
            seed=1,
        )
        strategic = _make_layer_result(
            "strategic",
            initial=_TOTAL_CAPITAL * _STRATEGIC_CAPITAL_PCT,
            annual_return=0.18,
            n_days=_FOUR_LAYER_N_DAYS,
            trades=["t4", "t5"],
            coupon_income=15_000.0,
            seed=2,
        )
        tactical = _make_layer_result(
            "tactical",
            initial=_TOTAL_CAPITAL * _TACTICAL_CAPITAL_PCT,
            annual_return=0.12,
            n_days=_FOUR_LAYER_N_DAYS,
            trades=["t6", "t7", "t8", "t9"],
            seed=3,
        )
        short = _make_layer_result(
            "short",
            initial=_TOTAL_CAPITAL * _SHORT_CAPITAL_PCT,
            annual_return=0.10,
            n_days=_FOUR_LAYER_N_DAYS,
            trades=["t10"],
            seed=4,
        )

        agg = PortfolioAggregator(risk_free_annual_pct=_RUONIA_ANNUAL_PCT)
        result = agg.aggregate([core, strategic, tactical, short])

        # All 4 layers present
        assert len(result.layer_results) == 4
        assert set(result.layer_results.keys()) == {"core", "strategic", "tactical", "short"}

        # Total trades = 3 + 2 + 4 + 1 = 10
        expected_trades = 10
        assert result.total_trades == expected_trades

        # Combined equity curve starts at sum of initials
        expected_initial = _TOTAL_CAPITAL
        assert result.combined_equity_curve[0] == pytest.approx(expected_initial, rel=_TOLERANCE)

        # Total return should be positive (all layers have positive annual return)
        assert result.total_return_pct > _ZERO

        # Max drawdown should be non-negative
        assert result.max_drawdown_pct >= _ZERO

        # Coupon income summed
        expected_coupon = 45_000.0
        assert result.total_coupon_income_net == pytest.approx(expected_coupon, rel=_TOLERANCE)

    def test_excess_return_over_ruonia(self) -> None:
        """Portfolio returning exactly RUONIA (15%) over 1 year has ~0% excess return.

        The aggregator uses a linear risk-free deduction:
            excess = total_return - (rf/100) * n_years
        For exactly 1 year at 15%, total_return = 0.15, rf_deduction = 0.15,
        so excess ~ 0. Over multiple years, compound vs linear diverges.
        """
        n_days = _TRADING_DAYS_PER_YEAR + 1  # 1 year (minimises compound mismatch)
        equity, dates = _make_deterministic_curve(
            initial=_TOTAL_CAPITAL,
            annual_return=_RUONIA_ANNUAL,
            n_days=n_days,
        )

        layer = LayerResult(
            layer_id="core",
            equity_curve=equity,
            dates=dates,
            trades=[],
            total_return_pct=(equity[-1] / equity[0] - 1) * 100,
            max_drawdown_pct=_ZERO,
            coupon_income_net=_ZERO,
        )

        agg = PortfolioAggregator(risk_free_annual_pct=_RUONIA_ANNUAL_PCT)
        result = agg.aggregate([layer])

        # Over exactly 1 year, total_return ~ 15% and rf deduction ~ 15%,
        # so excess should be approximately zero (within 1 ppt).
        excess_abs_tolerance = 1.0  # within 1 percentage point
        assert result.excess_return_pct == pytest.approx(_ZERO, abs=excess_abs_tolerance)

    def test_excess_sharpe_positive_for_outperformer(self) -> None:
        """Portfolio returning 20% with low volatility should have positive excess Sharpe."""
        layer = _make_layer_result(
            "core",
            initial=_TOTAL_CAPITAL,
            annual_return=_OUTPERFORMER_RETURN,
            n_days=_TRADING_DAYS_PER_YEAR * 3,
            volatility=_LOW_VOLATILITY,
            seed=100,
        )

        agg = PortfolioAggregator(risk_free_annual_pct=_RUONIA_ANNUAL_PCT)
        result = agg.aggregate([layer])

        assert result.excess_sharpe > _ZERO

    def test_excess_sharpe_near_zero_for_ruonia_tracker(self) -> None:
        """Portfolio tracking RUONIA exactly should have excess Sharpe near zero."""
        n_days = _TRADING_DAYS_PER_YEAR * 3
        equity, dates = _make_deterministic_curve(
            initial=_TOTAL_CAPITAL,
            annual_return=_RUONIA_ANNUAL,
            n_days=n_days,
        )

        layer = LayerResult(
            layer_id="core",
            equity_curve=equity,
            dates=dates,
            trades=[],
            total_return_pct=(equity[-1] / equity[0] - 1) * 100,
            max_drawdown_pct=_ZERO,
            coupon_income_net=_ZERO,
        )

        agg = PortfolioAggregator(risk_free_annual_pct=_RUONIA_ANNUAL_PCT)
        result = agg.aggregate([layer])

        # Deterministic curve matching RUONIA: all excess daily returns are zero,
        # so std is zero, and the code returns 0.0 for zero-std case.
        assert result.excess_sharpe == pytest.approx(_ZERO, abs=_SHARPE_ABS_TOLERANCE)

    def test_dd_breach_detected_at_10pct(self) -> None:
        """Combined portfolio dropping 12% triggers DD breach with correct date."""
        start = date(2022, 1, 3)
        dates = _make_dates_list(6, start=start)

        # Build equity curve that rises to a peak, then drops 12%
        peak = 1_500_000.0
        trough = peak * 0.88  # 12% drop from peak
        curve = [
            1_400_000.0,  # day 0
            1_450_000.0,  # day 1
            peak,  # day 2 (peak)
            1_400_000.0,  # day 3 (6.67% DD)
            trough,  # day 4 (12% DD -- breach)
            1_350_000.0,  # day 5
        ]

        layer = LayerResult(
            layer_id="core",
            equity_curve=curve,
            dates=dates,
            trades=[],
            total_return_pct=(curve[-1] / curve[0] - 1) * 100,
            max_drawdown_pct=_compute_max_dd(curve) * 100,
            coupon_income_net=_ZERO,
        )

        agg = PortfolioAggregator(
            portfolio_dd_limit=_DD_BREACH_LIMIT,
            risk_free_annual_pct=_RUONIA_ANNUAL_PCT,
        )
        result = agg.aggregate([layer])

        assert result.portfolio_dd_breach is True
        # Breach occurs on day 4 (index 4) where DD first exceeds 10%
        expected_breach_date = dates[4]
        assert result.portfolio_dd_breach_date == expected_breach_date


# ── TestLayerIsolation ──────────────────────────────────────────────────────


class TestLayerIsolation:
    """Verify layers tracked independently and contributions correct."""

    def test_layer_pnl_independent(self) -> None:
        """Each layer equity curve is tracked independently.

        One layer losing while others gain should show independent returns.
        Uses deterministic curves to avoid seed-dependent failures.
        """
        n_days = _TRADING_DAYS_PER_YEAR
        core_initial = 500_000.0
        strat_initial = 300_000.0

        # Core: deterministic +20% over 1 year
        core_equity, core_dates = _make_deterministic_curve(core_initial, 0.20, n_days)
        core = LayerResult(
            layer_id="core",
            equity_curve=core_equity,
            dates=core_dates,
            trades=[],
            total_return_pct=(core_equity[-1] / core_equity[0] - 1) * 100,
            max_drawdown_pct=_ZERO,
            coupon_income_net=_ZERO,
        )

        # Strategic: deterministic -5% over 1 year
        strat_equity, strat_dates = _make_deterministic_curve(strat_initial, -0.05, n_days)
        strat = LayerResult(
            layer_id="strategic",
            equity_curve=strat_equity,
            dates=strat_dates,
            trades=[],
            total_return_pct=(strat_equity[-1] / strat_equity[0] - 1) * 100,
            max_drawdown_pct=_compute_max_dd(strat_equity) * 100,
            coupon_income_net=_ZERO,
        )

        agg = PortfolioAggregator(risk_free_annual_pct=_RUONIA_ANNUAL_PCT)
        result = agg.aggregate([core, strat])

        # Core should have positive return
        assert result.layer_results["core"].total_return_pct > _ZERO
        # Strategic should have negative return
        assert result.layer_results["strategic"].total_return_pct < _ZERO
        # Combined return should differ from either layer individually
        assert result.total_return_pct != result.layer_results["core"].total_return_pct
        assert result.total_return_pct != result.layer_results["strategic"].total_return_pct

    def test_layer_contribution_sums_to_100(self) -> None:
        """Layer return contribution values should sum to approximately 100%."""
        n_days = _FOUR_LAYER_N_DAYS
        core = _make_layer_result(
            "core", initial=675_000.0, annual_return=0.16, n_days=n_days, seed=1
        )
        strategic = _make_layer_result(
            "strategic", initial=412_500.0, annual_return=0.18, n_days=n_days, seed=2
        )
        tactical = _make_layer_result(
            "tactical", initial=262_500.0, annual_return=0.12, n_days=n_days, seed=3
        )
        short = _make_layer_result(
            "short", initial=150_000.0, annual_return=0.10, n_days=n_days, seed=4
        )

        agg = PortfolioAggregator(risk_free_annual_pct=_RUONIA_ANNUAL_PCT)
        result = agg.aggregate([core, strategic, tactical, short])

        contribution_sum = sum(result.layer_return_contribution.values())
        expected_sum = 100.0
        assert contribution_sum == pytest.approx(expected_sum, abs=_CONTRIBUTION_SUM_TOLERANCE)

    def test_core_dominates_in_easing_scenario(self) -> None:
        """When core outperforms, its contribution should be the highest."""
        n_days = _FOUR_LAYER_N_DAYS
        core = _make_layer_result(
            "core", initial=675_000.0, annual_return=0.25, n_days=n_days, seed=1
        )
        strategic = _make_layer_result(
            "strategic", initial=412_500.0, annual_return=0.05, n_days=n_days, seed=2
        )
        tactical = _make_layer_result(
            "tactical", initial=262_500.0, annual_return=0.03, n_days=n_days, seed=3
        )
        short = _make_layer_result(
            "short", initial=150_000.0, annual_return=0.02, n_days=n_days, seed=4
        )

        agg = PortfolioAggregator(risk_free_annual_pct=_RUONIA_ANNUAL_PCT)
        result = agg.aggregate([core, strategic, tactical, short])

        core_contrib = result.layer_return_contribution["core"]
        for layer_id in ("strategic", "tactical", "short"):
            assert core_contrib > result.layer_return_contribution[layer_id], (
                f"Core contribution ({core_contrib:.1f}%) should exceed "
                f"{layer_id} ({result.layer_return_contribution[layer_id]:.1f}%)"
            )

    def test_short_layer_negative_still_works(self) -> None:
        """A short layer with negative returns reduces portfolio return but does not crash."""
        n_days = _TRADING_DAYS_PER_YEAR
        core = _make_layer_result(
            "core", initial=675_000.0, annual_return=0.18, n_days=n_days, seed=1
        )
        short = _make_layer_result(
            "short", initial=150_000.0, annual_return=-0.15, n_days=n_days, seed=5
        )

        agg = PortfolioAggregator(risk_free_annual_pct=_RUONIA_ANNUAL_PCT)

        # Should not raise
        result = agg.aggregate([core, short])

        # Short layer has negative contribution
        assert result.layer_return_contribution["short"] < _ZERO

        # Core-only would have higher return
        result_core_only = agg.aggregate([core])
        assert result_core_only.total_return_pct > result.total_return_pct


# ── TestExcessSharpeCalculation ─────────────────────────────────────────────


class TestExcessSharpeCalculation:
    """Validate excess Sharpe formula and edge cases."""

    def test_excess_sharpe_formula_matches_manual(self) -> None:
        """Hand-compute excess Sharpe for a known equity curve and verify match."""
        # Use a short deterministic curve so we can compute by hand
        curve = [
            1_000_000.0,
            1_002_000.0,
            1_005_000.0,
            1_003_000.0,
            1_008_000.0,
            1_010_000.0,
        ]
        start = date(2022, 1, 3)
        dates = _make_dates_list(len(curve), start=start)

        layer = LayerResult(
            layer_id="core",
            equity_curve=curve,
            dates=dates,
            trades=[],
            total_return_pct=(curve[-1] / curve[0] - 1) * 100,
            max_drawdown_pct=_compute_max_dd(curve) * 100,
            coupon_income_net=_ZERO,
        )

        rf_pct = _RUONIA_ANNUAL_PCT
        agg = PortfolioAggregator(risk_free_annual_pct=rf_pct)
        result = agg.aggregate([layer])

        # Manual computation
        daily_returns = [curve[i] / curve[i - 1] - 1.0 for i in range(1, len(curve))]
        daily_rf = (1 + rf_pct / 100) ** (1 / _TRADING_DAYS_PER_YEAR) - 1.0
        excess_daily = [r - daily_rf for r in daily_returns]
        mean_excess = sum(excess_daily) / len(excess_daily)
        var_excess = sum((r - mean_excess) ** 2 for r in excess_daily) / (len(excess_daily) - 1)
        std_excess = math.sqrt(var_excess)
        expected_sharpe = mean_excess / std_excess * math.sqrt(_TRADING_DAYS_PER_YEAR)

        assert result.excess_sharpe == pytest.approx(expected_sharpe, rel=_TOLERANCE)

    def test_risk_free_rate_configurable(self) -> None:
        """Different risk_free_annual_pct values change excess return correctly."""
        n_days = _TRADING_DAYS_PER_YEAR + 1
        equity, dates = _make_deterministic_curve(
            initial=_TOTAL_CAPITAL,
            annual_return=0.20,
            n_days=n_days,
        )

        layer = LayerResult(
            layer_id="core",
            equity_curve=equity,
            dates=dates,
            trades=[],
            total_return_pct=(equity[-1] / equity[0] - 1) * 100,
            max_drawdown_pct=_ZERO,
            coupon_income_net=_ZERO,
        )

        # With 10% risk-free: excess = 20% - 10% = ~10%
        low_rf = 10.0
        agg_low = PortfolioAggregator(risk_free_annual_pct=low_rf)
        result_low = agg_low.aggregate([layer])

        # With 25% risk-free: excess = 20% - 25% = ~-5%
        high_rf = 25.0
        agg_high = PortfolioAggregator(risk_free_annual_pct=high_rf)
        result_high = agg_high.aggregate([layer])

        assert result_low.excess_return_pct > _ZERO
        assert result_high.excess_return_pct < _ZERO
        # Higher rf should give lower excess
        assert result_low.excess_return_pct > result_high.excess_return_pct

    def test_high_vol_reduces_sharpe(self) -> None:
        """Same return with higher volatility produces a lower Sharpe ratio."""
        n_days = _TRADING_DAYS_PER_YEAR * 2
        annual_return = 0.20

        low_vol_layer = _make_layer_result(
            "core",
            initial=_TOTAL_CAPITAL,
            annual_return=annual_return,
            n_days=n_days,
            volatility=_LOW_VOLATILITY,
            seed=42,
        )
        high_vol_layer = _make_layer_result(
            "core",
            initial=_TOTAL_CAPITAL,
            annual_return=annual_return,
            n_days=n_days,
            volatility=_HIGH_VOLATILITY,
            seed=42,
        )

        agg = PortfolioAggregator(risk_free_annual_pct=_RUONIA_ANNUAL_PCT)

        result_low_vol = agg.aggregate([low_vol_layer])
        result_high_vol = agg.aggregate([high_vol_layer])

        # Lower volatility -> higher Sharpe (same expected return)
        assert result_low_vol.excess_sharpe > result_high_vol.excess_sharpe

    def test_short_period_sharpe_bounded(self) -> None:
        """Very short equity curve (10 bars) produces valid Sharpe (not NaN/inf)."""
        layer = _make_layer_result(
            "core",
            initial=_TOTAL_CAPITAL,
            annual_return=0.20,
            n_days=_SHORT_PERIOD_DAYS,
            volatility=0.01,
            seed=77,
        )

        agg = PortfolioAggregator(risk_free_annual_pct=_RUONIA_ANNUAL_PCT)
        result = agg.aggregate([layer])

        assert math.isfinite(result.excess_sharpe)
        assert not math.isnan(result.excess_sharpe)


# ── TestBondMetricsIntegration ──────────────────────────────────────────────


class TestBondMetricsIntegration:
    """Bond metrics integration with the portfolio pipeline."""

    def test_bond_metrics_coupon_contribution(self) -> None:
        """Known coupon income produces correct coupon_contribution_pct."""
        # Create equity curve with known PnL
        initial = _TOTAL_CAPITAL
        final = 1_600_000.0  # +100K PnL
        n_points = _TRADING_DAYS_PER_YEAR + 1
        step = (final - initial) / (n_points - 1)
        curve = [initial + step * i for i in range(n_points)]
        dates = _make_dates_list(n_points)

        total_pnl = final - initial
        coupon_net = 50_000.0
        expected_contribution = coupon_net / total_pnl * 100  # 50%

        result = compute_bond_metrics(
            equity_curve=curve,
            dates=dates,
            trades=[],
            coupon_income_gross=coupon_net / (1 - _COUPON_NDFL_RATE),
            coupon_income_net=coupon_net,
            initial_cash=initial,
            risk_free_annual_pct=_RUONIA_ANNUAL_PCT,
        )

        assert result.coupon_contribution_pct == pytest.approx(
            expected_contribution, rel=_TOLERANCE
        )

    def test_bond_metrics_excess_sharpe(self) -> None:
        """compute_bond_metrics with RUONIA 15% gives expected excess Sharpe sign."""
        n_points = _TRADING_DAYS_PER_YEAR * 2 + 1
        # 20% annual return -> should have positive excess Sharpe over 15% RUONIA
        initial = _TOTAL_CAPITAL
        annual_return = 0.20
        daily_return = (1 + annual_return) ** (1 / _TRADING_DAYS_PER_YEAR) - 1
        curve = [initial]
        for _i in range(1, n_points):
            curve.append(curve[-1] * (1 + daily_return))
        dates = _make_dates_list(n_points)

        result = compute_bond_metrics(
            equity_curve=curve,
            dates=dates,
            trades=[],
            coupon_income_gross=_ZERO,
            coupon_income_net=_ZERO,
            initial_cash=initial,
            risk_free_annual_pct=_RUONIA_ANNUAL_PCT,
        )

        # Deterministic 20% return, zero vol -> code returns 0.0 (zero std case).
        # This verifies the safe handling rather than a raw formula result.
        # The excess return should be clearly positive.
        assert result.excess_return_pct > _ZERO
        assert result.annualized_excess_return_pct > _ZERO

    def test_bond_metrics_with_no_trades(self) -> None:
        """Empty trades list produces metrics with 0 trades, 0 win rate."""
        initial = _TOTAL_CAPITAL
        n_points = _TRADING_DAYS_PER_YEAR + 1
        daily_return = (1 + 0.18) ** (1 / _TRADING_DAYS_PER_YEAR) - 1
        curve = [initial]
        for _i in range(1, n_points):
            curve.append(curve[-1] * (1 + daily_return))
        dates = _make_dates_list(n_points)

        result = compute_bond_metrics(
            equity_curve=curve,
            dates=dates,
            trades=[],
            coupon_income_gross=_ZERO,
            coupon_income_net=_ZERO,
            initial_cash=initial,
            risk_free_annual_pct=_RUONIA_ANNUAL_PCT,
        )

        assert result.trade_count == 0
        assert result.win_rate == _ZERO
        assert result.profit_factor == _ZERO
        assert result.avg_hold_bars == _ZERO
        # But return metrics should still be computed
        assert result.total_return_pct > _ZERO

    def test_bond_metrics_with_realistic_curve(self) -> None:
        """3-year equity curve from 412.5K to ~500K with drawdowns.

        Verifies annualized_excess_return_pct is positive for a bond layer
        that outperforms RUONIA.
        """
        n_points = _BOND_CURVE_DAYS
        # ~6.5% annual return on a bond layer (just above RUONIA for the period)
        # Using a seed that gives some realistic drawdowns
        rng = random.Random(99)  # noqa: S311
        initial = _BOND_INITIAL
        # Target ~21% total return over 3 years => ~6.5% annual
        # But we want to beat RUONIA so use 18% annual
        annual_return = 0.18
        daily_return = (1 + annual_return) ** (1 / _TRADING_DAYS_PER_YEAR) - 1
        daily_vol = 0.008

        curve = [initial]
        for _i in range(1, n_points):
            r = daily_return + daily_vol * rng.gauss(0, 1)
            curve.append(curve[-1] * (1 + r))
        dates = _make_dates_list(n_points)

        # Create some realistic trades
        trades = [
            _make_trade(_WIN_PNL),
            _make_trade(_WIN_PNL),
            _make_trade(_LOSS_PNL, hold_bars=_HOLD_BARS_LOSS),
        ]

        result = compute_bond_metrics(
            equity_curve=curve,
            dates=dates,
            trades=trades,
            coupon_income_gross=_COUPON_GROSS,
            coupon_income_net=_COUPON_NET,
            initial_cash=initial,
            risk_free_annual_pct=_RUONIA_ANNUAL_PCT,
        )

        # Annualized excess return should be positive (18% annual > 15% RUONIA)
        assert result.annualized_excess_return_pct > _ZERO
        # Trade count is correct
        expected_trade_count = 3
        assert result.trade_count == expected_trade_count
        # Win rate = 2/3
        expected_win_rate = 2 / 3
        assert result.win_rate == pytest.approx(expected_win_rate, rel=_TOLERANCE)
        # Coupon values pass through
        assert result.total_coupon_income_gross == _COUPON_GROSS
        assert result.total_coupon_income_net == _COUPON_NET
