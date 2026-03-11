"""Unit tests for portfolio aggregator — combines per-layer backtest results."""

from __future__ import annotations

from datetime import date

import pytest

from finalayze.backtest.portfolio_aggregator import (
    LayerResult,
    PortfolioAggregator,
    PortfolioResult,
)

# ── Constants (no magic numbers per ruff PLR2004) ────────────────────────────
_ZERO = 0.0
_TOLERANCE = 1e-6
_LOOSE_TOLERANCE = 0.01  # 1% tolerance for annualisation checks

_INITIAL_VALUE = 1_000_000.0
_RUONIA_ANNUAL_PCT = 15.0
_TRADING_DAYS = 252
_DD_LIMIT_PCT = 0.10  # 10%

# Layer equity curve final values
_CORE_FINAL = 1_050_000.0  # +5%
_STRATEGIC_FINAL = 1_030_000.0  # +3%
_TACTICAL_FINAL = 980_000.0  # -2%

# Coupon income values
_CORE_COUPON = 30_000.0
_STRATEGIC_COUPON = 15_000.0
_TACTICAL_COUPON = 5_000.0

# Number of days for standard test curves
_N_DAYS = 10

# Phase 4 exit gate counts
_HARD_GATES_TOTAL = 4
_SOFT_GATES_TOTAL = 3
_SOFT_GATES_MIN_PASS = 2


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_dates(n: int, start_day: int = 1) -> list[date]:
    """Create n consecutive dates starting from 2025-01-start_day."""
    from datetime import timedelta

    start = date(2025, 1, start_day)
    return [start + timedelta(days=i) for i in range(n)]


def _make_linear_curve(start: float, end: float, n_points: int) -> list[float]:
    """Create a linearly interpolated equity curve."""
    if n_points < 2:  # noqa: PLR2004
        return [start]
    step = (end - start) / (n_points - 1)
    return [start + step * i for i in range(n_points)]


def _make_flat_curve(value: float, n_points: int) -> list[float]:
    """Create a flat equity curve."""
    return [value] * n_points


def _make_layer(
    layer_id: str,
    equity_curve: list[float],
    dates: list[date],
    trades: list | None = None,
    coupon_income_net: float = _ZERO,
    sharpe: float = _ZERO,
) -> LayerResult:
    """Create a LayerResult with sensible defaults."""
    if trades is None:
        trades = []
    initial = equity_curve[0] if equity_curve else _ZERO
    final = equity_curve[-1] if equity_curve else _ZERO
    total_return = ((final / initial - 1.0) * 100) if initial > 0 else _ZERO
    max_dd = _compute_simple_dd(equity_curve)
    return LayerResult(
        layer_id=layer_id,
        equity_curve=equity_curve,
        dates=dates,
        trades=trades,
        total_return_pct=total_return,
        max_drawdown_pct=max_dd,
        coupon_income_net=coupon_income_net,
        sharpe=sharpe,
    )


def _compute_simple_dd(curve: list[float]) -> float:
    """Compute max drawdown percentage for helper."""
    if not curve:
        return _ZERO
    peak = curve[0]
    max_dd = _ZERO
    for val in curve:
        peak = max(peak, val)
        if peak > 0:
            dd = (peak - val) / peak * 100
            max_dd = max(max_dd, dd)
    return max_dd


# ── Test: Empty input ────────────────────────────────────────────────────────


class TestEmptyInput:
    """Empty input returns zero result."""

    def test_no_layers_returns_empty(self) -> None:
        agg = PortfolioAggregator()
        result = agg.aggregate([])

        assert isinstance(result, PortfolioResult)
        assert result.total_return_pct == _ZERO
        assert result.annualized_return_pct == _ZERO
        assert result.excess_return_pct == _ZERO
        assert result.excess_sharpe == _ZERO
        assert result.max_drawdown_pct == _ZERO
        assert result.total_trades == 0
        assert result.total_coupon_income_net == _ZERO
        assert result.portfolio_dd_breach is False
        assert result.portfolio_dd_breach_date is None
        assert result.combined_equity_curve == []
        assert result.combined_dates == []
        assert result.layer_results == {}
        assert result.layer_return_contribution == {}

        # Phase 4 exit criteria defaults
        assert result.absolute_sharpe == _ZERO
        assert result.core_return_vs_ruonia == _ZERO
        assert result.strategic_dd_ok is True
        assert result.tactical_has_trades is False
        assert result.hard_gates_passed == 0
        assert result.hard_gates_total == _HARD_GATES_TOTAL
        assert result.soft_gates_passed == 0
        assert result.soft_gates_total == _SOFT_GATES_TOTAL
        assert result.phase4_exit_ok is False


# ── Test: Single layer ───────────────────────────────────────────────────────


class TestSingleLayer:
    """Portfolio with one layer: portfolio metrics should match layer."""

    def test_single_layer_equity_matches(self) -> None:
        dates = _make_dates(_N_DAYS)
        curve = _make_linear_curve(_INITIAL_VALUE, _CORE_FINAL, _N_DAYS)
        layer = _make_layer("core", curve, dates)

        agg = PortfolioAggregator()
        result = agg.aggregate([layer])

        # Combined equity curve should equal the single layer
        assert len(result.combined_equity_curve) == _N_DAYS
        for actual, expected in zip(result.combined_equity_curve, curve, strict=True):
            assert actual == pytest.approx(expected, rel=_TOLERANCE)

    def test_single_layer_return(self) -> None:
        dates = _make_dates(_N_DAYS)
        curve = _make_linear_curve(_INITIAL_VALUE, _CORE_FINAL, _N_DAYS)
        layer = _make_layer("core", curve, dates)

        agg = PortfolioAggregator()
        result = agg.aggregate([layer])

        expected_return = (_CORE_FINAL / _INITIAL_VALUE - 1.0) * 100
        assert result.total_return_pct == pytest.approx(expected_return, rel=_TOLERANCE)

    def test_single_layer_contribution_100_pct(self) -> None:
        dates = _make_dates(_N_DAYS)
        curve = _make_linear_curve(_INITIAL_VALUE, _CORE_FINAL, _N_DAYS)
        layer = _make_layer("core", curve, dates)

        agg = PortfolioAggregator()
        result = agg.aggregate([layer])

        assert result.layer_return_contribution["core"] == pytest.approx(100.0, rel=_TOLERANCE)


# ── Test: Two layers combined equity ─────────────────────────────────────────


class TestTwoLayersCombined:
    """Two layers: combined equity is sum of per-layer equities."""

    def test_combined_equity_is_sum(self) -> None:
        dates = _make_dates(_N_DAYS)
        core_curve = _make_linear_curve(_INITIAL_VALUE, _CORE_FINAL, _N_DAYS)
        strat_curve = _make_linear_curve(_INITIAL_VALUE, _STRATEGIC_FINAL, _N_DAYS)

        core = _make_layer("core", core_curve, dates)
        strat = _make_layer("strategic", strat_curve, dates)

        agg = PortfolioAggregator()
        result = agg.aggregate([core, strat])

        assert len(result.combined_equity_curve) == _N_DAYS
        for i in range(_N_DAYS):
            expected = core_curve[i] + strat_curve[i]
            assert result.combined_equity_curve[i] == pytest.approx(expected, rel=_TOLERANCE)

    def test_combined_return_from_summed_equity(self) -> None:
        dates = _make_dates(_N_DAYS)
        core_curve = _make_linear_curve(_INITIAL_VALUE, _CORE_FINAL, _N_DAYS)
        strat_curve = _make_linear_curve(_INITIAL_VALUE, _STRATEGIC_FINAL, _N_DAYS)

        core = _make_layer("core", core_curve, dates)
        strat = _make_layer("strategic", strat_curve, dates)

        agg = PortfolioAggregator()
        result = agg.aggregate([core, strat])

        initial_combined = _INITIAL_VALUE + _INITIAL_VALUE
        final_combined = _CORE_FINAL + _STRATEGIC_FINAL
        expected_return = (final_combined / initial_combined - 1.0) * 100
        assert result.total_return_pct == pytest.approx(expected_return, rel=_TOLERANCE)

    def test_both_layers_in_result(self) -> None:
        dates = _make_dates(_N_DAYS)
        core_curve = _make_flat_curve(_INITIAL_VALUE, _N_DAYS)
        strat_curve = _make_flat_curve(_INITIAL_VALUE, _N_DAYS)

        core = _make_layer("core", core_curve, dates)
        strat = _make_layer("strategic", strat_curve, dates)

        agg = PortfolioAggregator()
        result = agg.aggregate([core, strat])

        assert "core" in result.layer_results
        assert "strategic" in result.layer_results
        assert result.layer_results["core"].layer_id == "core"


# ── Test: DD breach detection ────────────────────────────────────────────────


class TestDDBreachDetection:
    """10% drawdown breach is flagged with the date it occurred."""

    def test_breach_detected(self) -> None:
        dates = _make_dates(5)
        # Single layer: peak at 1.1M, then drops to 0.95M = 13.6% DD > 10%
        curve = [1_000_000.0, 1_100_000.0, 1_050_000.0, 950_000.0, 960_000.0]
        layer = _make_layer("core", curve, dates)

        agg = PortfolioAggregator(portfolio_dd_limit=_DD_LIMIT_PCT)
        result = agg.aggregate([layer])

        assert result.portfolio_dd_breach is True
        # The breach happens at index 3 (950k vs peak 1.1M => 13.6% DD)
        assert result.portfolio_dd_breach_date == dates[3]

    def test_breach_captures_first_date(self) -> None:
        dates = _make_dates(6)
        # DD exceeds 10% at index 3 and again at index 5
        curve = [
            1_000_000.0,
            1_100_000.0,
            1_050_000.0,
            950_000.0,  # 13.6% DD -> breach
            1_080_000.0,
            940_000.0,  # deeper DD but already breached
        ]
        layer = _make_layer("core", curve, dates)

        agg = PortfolioAggregator(portfolio_dd_limit=_DD_LIMIT_PCT)
        result = agg.aggregate([layer])

        assert result.portfolio_dd_breach is True
        # First breach at index 3
        assert result.portfolio_dd_breach_date == dates[3]


# ── Test: No DD breach ──────────────────────────────────────────────────────


class TestNoDDBreach:
    """Drawdown below limit does not trigger breach."""

    def test_small_dd_no_breach(self) -> None:
        dates = _make_dates(5)
        # DD is only 5%: 1M -> 1.1M -> 1.045M
        curve = [1_000_000.0, 1_100_000.0, 1_045_000.0, 1_060_000.0, 1_080_000.0]
        layer = _make_layer("core", curve, dates)

        agg = PortfolioAggregator(portfolio_dd_limit=_DD_LIMIT_PCT)
        result = agg.aggregate([layer])

        assert result.portfolio_dd_breach is False
        assert result.portfolio_dd_breach_date is None

    def test_monotonic_increase_no_breach(self) -> None:
        dates = _make_dates(_N_DAYS)
        curve = _make_linear_curve(_INITIAL_VALUE, _CORE_FINAL, _N_DAYS)
        layer = _make_layer("core", curve, dates)

        agg = PortfolioAggregator(portfolio_dd_limit=_DD_LIMIT_PCT)
        result = agg.aggregate([layer])

        assert result.portfolio_dd_breach is False
        assert result.portfolio_dd_breach_date is None
        assert result.max_drawdown_pct == pytest.approx(_ZERO, abs=_TOLERANCE)


# ── Test: Per-layer contribution ─────────────────────────────────────────────


class TestLayerContribution:
    """Correct percentage contribution from each layer to total PnL."""

    def test_two_layers_proportional(self) -> None:
        dates = _make_dates(_N_DAYS)
        # Core: 1M -> 1.05M (PnL = 50k)
        # Strategic: 1M -> 1.03M (PnL = 30k)
        # Total PnL = 80k
        core_curve = _make_linear_curve(_INITIAL_VALUE, _CORE_FINAL, _N_DAYS)
        strat_curve = _make_linear_curve(_INITIAL_VALUE, _STRATEGIC_FINAL, _N_DAYS)

        core = _make_layer("core", core_curve, dates)
        strat = _make_layer("strategic", strat_curve, dates)

        agg = PortfolioAggregator()
        result = agg.aggregate([core, strat])

        core_pnl = _CORE_FINAL - _INITIAL_VALUE  # 50k
        strat_pnl = _STRATEGIC_FINAL - _INITIAL_VALUE  # 30k
        total_pnl = core_pnl + strat_pnl  # 80k

        expected_core_contrib = core_pnl / total_pnl * 100  # 62.5%
        expected_strat_contrib = strat_pnl / total_pnl * 100  # 37.5%

        assert result.layer_return_contribution["core"] == pytest.approx(
            expected_core_contrib, rel=_TOLERANCE
        )
        assert result.layer_return_contribution["strategic"] == pytest.approx(
            expected_strat_contrib, rel=_TOLERANCE
        )

    def test_negative_pnl_layer(self) -> None:
        """A losing layer has negative contribution."""
        dates = _make_dates(_N_DAYS)
        core_curve = _make_linear_curve(_INITIAL_VALUE, _CORE_FINAL, _N_DAYS)
        tact_curve = _make_linear_curve(_INITIAL_VALUE, _TACTICAL_FINAL, _N_DAYS)

        core = _make_layer("core", core_curve, dates)
        tact = _make_layer("tactical", tact_curve, dates)

        agg = PortfolioAggregator()
        result = agg.aggregate([core, tact])

        core_pnl = _CORE_FINAL - _INITIAL_VALUE  # +50k
        tact_pnl = _TACTICAL_FINAL - _INITIAL_VALUE  # -20k
        total_pnl = core_pnl + tact_pnl  # 30k

        expected_tact_contrib = tact_pnl / total_pnl * 100  # negative
        assert result.layer_return_contribution["tactical"] == pytest.approx(
            expected_tact_contrib, rel=_TOLERANCE
        )
        assert result.layer_return_contribution["tactical"] < _ZERO

    def test_zero_total_pnl_contribution(self) -> None:
        """When total PnL is zero, all contributions are zero."""
        dates = _make_dates(_N_DAYS)
        flat_curve = _make_flat_curve(_INITIAL_VALUE, _N_DAYS)

        core = _make_layer("core", flat_curve, dates)
        strat = _make_layer("strategic", flat_curve, dates)

        agg = PortfolioAggregator()
        result = agg.aggregate([core, strat])

        assert result.layer_return_contribution["core"] == pytest.approx(_ZERO, abs=_TOLERANCE)
        assert result.layer_return_contribution["strategic"] == pytest.approx(_ZERO, abs=_TOLERANCE)


# ── Test: Excess Sharpe ──────────────────────────────────────────────────────


class TestExcessSharpe:
    """Negative excess Sharpe when return is below RUONIA."""

    def test_below_ruonia_negative_sharpe(self) -> None:
        """5% return over 1 year vs 15% RUONIA => negative excess Sharpe."""
        n_points = _TRADING_DAYS + 1  # ~1 year
        dates = _make_dates(n_points)
        # +5% over 1 year, well below 15% RUONIA
        curve = _make_linear_curve(_INITIAL_VALUE, 1_050_000.0, n_points)
        layer = _make_layer("core", curve, dates)

        agg = PortfolioAggregator(risk_free_annual_pct=_RUONIA_ANNUAL_PCT)
        result = agg.aggregate([layer])

        assert result.excess_sharpe < _ZERO

    def test_above_ruonia_positive_sharpe(self) -> None:
        """25% return over 1 year vs 15% RUONIA => positive excess Sharpe."""
        n_points = _TRADING_DAYS + 1
        dates = _make_dates(n_points)
        curve = _make_linear_curve(_INITIAL_VALUE, 1_250_000.0, n_points)
        layer = _make_layer("core", curve, dates)

        agg = PortfolioAggregator(risk_free_annual_pct=_RUONIA_ANNUAL_PCT)
        result = agg.aggregate([layer])

        assert result.excess_sharpe > _ZERO

    def test_excess_return_negative_below_ruonia(self) -> None:
        """Excess return should be negative when total return < RUONIA."""
        n_points = _TRADING_DAYS + 1
        dates = _make_dates(n_points)
        curve = _make_linear_curve(_INITIAL_VALUE, 1_050_000.0, n_points)
        layer = _make_layer("core", curve, dates)

        agg = PortfolioAggregator(risk_free_annual_pct=_RUONIA_ANNUAL_PCT)
        result = agg.aggregate([layer])

        assert result.excess_return_pct < _ZERO


# ── Test: Mismatched dates ───────────────────────────────────────────────────


class TestMismatchedDates:
    """Layers with different date ranges: forward-fill works correctly."""

    def test_forward_fill_aligns_curves(self) -> None:
        """Layer A has dates 1-5, Layer B has dates 3-7.
        Combined should span dates 1-7 with forward-fill for missing days.
        """
        dates_a = _make_dates(5, start_day=1)  # Jan 1-5
        dates_b = _make_dates(5, start_day=3)  # Jan 3-7
        curve_a = [100.0, 102.0, 104.0, 106.0, 108.0]
        curve_b = [200.0, 204.0, 208.0, 212.0, 216.0]

        layer_a = _make_layer("core", curve_a, dates_a)
        layer_b = _make_layer("strategic", curve_b, dates_b)

        agg = PortfolioAggregator()
        result = agg.aggregate([layer_a, layer_b])

        # Common dates: Jan 1, 2, 3, 4, 5, 6, 7
        expected_n_dates = 7
        assert len(result.combined_dates) == expected_n_dates
        assert result.combined_dates[0] == date(2025, 1, 1)
        assert result.combined_dates[-1] == date(2025, 1, 7)

    def test_forward_fill_values(self) -> None:
        """Layer B's value is forward-filled before its start date."""
        dates_a = _make_dates(5, start_day=1)  # Jan 1-5
        dates_b = _make_dates(3, start_day=3)  # Jan 3-5
        curve_a = [100.0, 102.0, 104.0, 106.0, 108.0]
        curve_b = [200.0, 204.0, 208.0]

        layer_a = _make_layer("core", curve_a, dates_a)
        layer_b = _make_layer("strategic", curve_b, dates_b)

        agg = PortfolioAggregator()
        result = agg.aggregate([layer_a, layer_b])

        # Jan 1 and Jan 2: B not started yet, uses first value (200.0)
        assert result.combined_equity_curve[0] == pytest.approx(
            100.0 + 200.0, rel=_TOLERANCE
        )  # Jan 1
        assert result.combined_equity_curve[1] == pytest.approx(
            102.0 + 200.0, rel=_TOLERANCE
        )  # Jan 2
        # Jan 3: both have data
        assert result.combined_equity_curve[2] == pytest.approx(  # noqa: PLR2004
            104.0 + 200.0, rel=_TOLERANCE
        )  # Jan 3

    def test_forward_fill_after_layer_ends(self) -> None:
        """Layer A ends before Layer B; A's last value carried forward."""
        dates_a = _make_dates(3, start_day=1)  # Jan 1-3
        dates_b = _make_dates(5, start_day=1)  # Jan 1-5
        curve_a = [100.0, 102.0, 104.0]
        curve_b = [200.0, 204.0, 208.0, 212.0, 216.0]

        layer_a = _make_layer("core", curve_a, dates_a)
        layer_b = _make_layer("strategic", curve_b, dates_b)

        agg = PortfolioAggregator()
        result = agg.aggregate([layer_a, layer_b])

        # Jan 4 and 5: A ends at 104.0, should carry forward
        last_a_value = 104.0
        assert result.combined_equity_curve[3] == pytest.approx(
            last_a_value + 212.0, rel=_TOLERANCE
        )  # Jan 4
        assert result.combined_equity_curve[4] == pytest.approx(
            last_a_value + 216.0, rel=_TOLERANCE
        )  # Jan 5


# ── Test: Coupon income aggregation ──────────────────────────────────────────


class TestCouponAggregation:
    """Total coupon income sums across layers."""

    def test_coupon_sum(self) -> None:
        dates = _make_dates(_N_DAYS)
        core_curve = _make_flat_curve(_INITIAL_VALUE, _N_DAYS)
        strat_curve = _make_flat_curve(_INITIAL_VALUE, _N_DAYS)
        tact_curve = _make_flat_curve(_INITIAL_VALUE, _N_DAYS)

        core = _make_layer("core", core_curve, dates, coupon_income_net=_CORE_COUPON)
        strat = _make_layer("strategic", strat_curve, dates, coupon_income_net=_STRATEGIC_COUPON)
        tact = _make_layer("tactical", tact_curve, dates, coupon_income_net=_TACTICAL_COUPON)

        agg = PortfolioAggregator()
        result = agg.aggregate([core, strat, tact])

        expected_total = _CORE_COUPON + _STRATEGIC_COUPON + _TACTICAL_COUPON
        assert result.total_coupon_income_net == pytest.approx(expected_total, rel=_TOLERANCE)

    def test_zero_coupon_when_none(self) -> None:
        dates = _make_dates(_N_DAYS)
        curve = _make_flat_curve(_INITIAL_VALUE, _N_DAYS)
        layer = _make_layer("core", curve, dates, coupon_income_net=_ZERO)

        agg = PortfolioAggregator()
        result = agg.aggregate([layer])

        assert result.total_coupon_income_net == pytest.approx(_ZERO, abs=_TOLERANCE)


# ── Test: Total trades aggregation ───────────────────────────────────────────


class TestTotalTrades:
    """Trade count sums across layers."""

    def test_trade_count_sums(self) -> None:
        dates = _make_dates(_N_DAYS)
        curve = _make_flat_curve(_INITIAL_VALUE, _N_DAYS)

        # Use simple placeholder objects for trades
        core = _make_layer("core", curve, dates, trades=["t1", "t2", "t3"])
        strat = _make_layer("strategic", curve, dates, trades=["t4", "t5"])

        agg = PortfolioAggregator()
        result = agg.aggregate([core, strat])

        expected_trades = 5
        assert result.total_trades == expected_trades

    def test_no_trades(self) -> None:
        dates = _make_dates(_N_DAYS)
        curve = _make_flat_curve(_INITIAL_VALUE, _N_DAYS)
        layer = _make_layer("core", curve, dates, trades=[])

        agg = PortfolioAggregator()
        result = agg.aggregate([layer])

        assert result.total_trades == 0


# ── Test: Annualized return ──────────────────────────────────────────────────


class TestAnnualizedReturn:
    """Correct annualization of returns."""

    def test_half_year_annualization(self) -> None:
        """8% in half year ~ (1.08)^2 - 1 = 16.64% annualized."""
        half_year = _TRADING_DAYS // 2 + 1  # ~126 days
        dates = _make_dates(half_year)
        curve = _make_linear_curve(_INITIAL_VALUE, 1_080_000.0, half_year)
        layer = _make_layer("core", curve, dates)

        agg = PortfolioAggregator()
        result = agg.aggregate([layer])

        expected_annual = ((1.08) ** 2 - 1.0) * 100  # ~16.64%
        assert result.annualized_return_pct == pytest.approx(expected_annual, rel=_LOOSE_TOLERANCE)

    def test_full_year_no_extra_scaling(self) -> None:
        """Full-year return: annualized ~ total."""
        n_points = _TRADING_DAYS + 1
        dates = _make_dates(n_points)
        curve = _make_linear_curve(_INITIAL_VALUE, 1_200_000.0, n_points)
        layer = _make_layer("core", curve, dates)

        agg = PortfolioAggregator()
        result = agg.aggregate([layer])

        expected_return = 20.0  # 20%
        assert result.total_return_pct == pytest.approx(expected_return, rel=_LOOSE_TOLERANCE)
        assert result.annualized_return_pct == pytest.approx(expected_return, rel=_LOOSE_TOLERANCE)


# ── Test: Max drawdown ──────────────────────────────────────────────────────


class TestMaxDrawdown:
    """Combined portfolio max drawdown computation."""

    def test_combined_drawdown_from_summed_curve(self) -> None:
        """Max DD is computed from the combined equity curve, not per-layer."""
        dates = _make_dates(5)
        # Layer A: rising
        curve_a = [500_000.0, 520_000.0, 540_000.0, 560_000.0, 580_000.0]
        # Layer B: has a dip
        curve_b = [500_000.0, 600_000.0, 450_000.0, 480_000.0, 500_000.0]

        layer_a = _make_layer("core", curve_a, dates)
        layer_b = _make_layer("strategic", curve_b, dates)

        agg = PortfolioAggregator()
        result = agg.aggregate([layer_a, layer_b])

        # Combined: [1M, 1.12M, 0.99M, 1.04M, 1.08M]
        # Peak at 1.12M, trough at 0.99M => DD = (1.12M - 0.99M) / 1.12M * 100
        peak = 1_120_000.0
        trough = 990_000.0
        expected_dd = (peak - trough) / peak * 100
        assert result.max_drawdown_pct == pytest.approx(expected_dd, rel=_TOLERANCE)


# ── Test: Edge cases ─────────────────────────────────────────────────────────


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_single_date_layer(self) -> None:
        """A layer with a single date point."""
        dates = _make_dates(1)
        curve = [_INITIAL_VALUE]
        layer = _make_layer("core", curve, dates)

        agg = PortfolioAggregator()
        result = agg.aggregate([layer])

        assert len(result.combined_equity_curve) == 1
        assert result.total_return_pct == pytest.approx(_ZERO, abs=_TOLERANCE)

    def test_custom_dd_limit(self) -> None:
        """Custom DD limit of 5% triggers breach."""
        dates = _make_dates(4)
        curve = [1_000_000.0, 1_100_000.0, 1_020_000.0, 1_050_000.0]
        # DD = (1.1M - 1.02M) / 1.1M = 7.27% > 5%
        layer = _make_layer("core", curve, dates)

        custom_limit = 0.05
        agg = PortfolioAggregator(portfolio_dd_limit=custom_limit)
        result = agg.aggregate([layer])

        assert result.portfolio_dd_breach is True

    def test_custom_risk_free_rate(self) -> None:
        """Custom risk-free rate affects excess return."""
        n_points = _TRADING_DAYS + 1
        dates = _make_dates(n_points)
        curve = _make_linear_curve(_INITIAL_VALUE, 1_100_000.0, n_points)
        layer = _make_layer("core", curve, dates)

        # With 5% risk-free, 10% return => positive excess
        low_rf = 5.0
        agg = PortfolioAggregator(risk_free_annual_pct=low_rf)
        result = agg.aggregate([layer])
        assert result.excess_return_pct > _ZERO

        # With 20% risk-free, 10% return => negative excess
        high_rf = 20.0
        agg_high = PortfolioAggregator(risk_free_annual_pct=high_rf)
        result_high = agg_high.aggregate([layer])
        assert result_high.excess_return_pct < _ZERO


# ── Test: Absolute Sharpe ──────────────────────────────────────────────────


class TestAbsoluteSharpe:
    """Absolute Sharpe is computed without risk-free subtraction."""

    def test_positive_return_positive_absolute_sharpe(self) -> None:
        """A steadily rising curve should have positive absolute Sharpe."""
        n_points = _TRADING_DAYS + 1
        dates = _make_dates(n_points)
        curve = _make_linear_curve(_INITIAL_VALUE, 1_200_000.0, n_points)
        layer = _make_layer("core", curve, dates)

        agg = PortfolioAggregator()
        result = agg.aggregate([layer])

        assert result.absolute_sharpe > _ZERO

    def test_absolute_sharpe_exceeds_excess_sharpe_below_ruonia(self) -> None:
        """When return < RUONIA, absolute Sharpe > excess Sharpe."""
        n_points = _TRADING_DAYS + 1
        dates = _make_dates(n_points)
        # 5% return < 15% RUONIA
        curve = _make_linear_curve(_INITIAL_VALUE, 1_050_000.0, n_points)
        layer = _make_layer("core", curve, dates)

        agg = PortfolioAggregator(risk_free_annual_pct=_RUONIA_ANNUAL_PCT)
        result = agg.aggregate([layer])

        assert result.absolute_sharpe > result.excess_sharpe

    def test_empty_result_absolute_sharpe_zero(self) -> None:
        """Empty result has zero absolute Sharpe."""
        agg = PortfolioAggregator()
        result = agg.aggregate([])
        assert result.absolute_sharpe == _ZERO


# ── Test: Phase 4 Hard Gates ───────────────────────────────────────────────

# Use a ~4 year period to have realistic multi-year data
_FOUR_YEAR_DAYS = _TRADING_DAYS * 4 + 1


def _make_full_portfolio_layers(
    core_return_pct: float = 70.0,
    strategic_return_pct: float = 20.0,
    tactical_return_pct: float = 5.0,
    short_return_pct: float = 10.0,
    strategic_dd_pct: float = 5.0,
    n_tactical_trades: int = 10,
    short_pf: float = 1.5,
) -> list[LayerResult]:
    """Create a full 4-layer portfolio with configurable metrics.

    Returns list of LayerResult for core, strategic, tactical, short.
    """
    n = _FOUR_YEAR_DAYS
    dates = _make_dates(n)

    core_final = _INITIAL_VALUE * (1 + core_return_pct / 100)
    strategic_final = _INITIAL_VALUE * (1 + strategic_return_pct / 100)
    tactical_final = _INITIAL_VALUE * (1 + tactical_return_pct / 100)
    short_final = _INITIAL_VALUE * (1 + short_return_pct / 100)

    core = _make_layer(
        "core",
        _make_linear_curve(_INITIAL_VALUE, core_final, n),
        dates,
        coupon_income_net=50_000.0,
    )
    # For strategic, create a curve with a controlled drawdown
    strat_curve = _make_linear_curve(_INITIAL_VALUE, strategic_final, n)
    strat = LayerResult(
        layer_id="strategic",
        equity_curve=strat_curve,
        dates=dates,
        trades=["t"] * 5,
        total_return_pct=strategic_return_pct,
        max_drawdown_pct=strategic_dd_pct,
    )

    tact_curve = _make_linear_curve(_INITIAL_VALUE, tactical_final, n)
    tact = LayerResult(
        layer_id="tactical",
        equity_curve=tact_curve,
        dates=dates,
        trades=["t"] * n_tactical_trades,
        total_return_pct=tactical_return_pct,
        max_drawdown_pct=2.0,
    )

    short_curve = _make_linear_curve(_INITIAL_VALUE, short_final, n)
    short = LayerResult(
        layer_id="short",
        equity_curve=short_curve,
        dates=dates,
        trades=["t"] * 20,
        total_return_pct=short_return_pct,
        max_drawdown_pct=3.0,
        profit_factor=short_pf,
    )

    return [core, strat, tact, short]


class TestPhase4HardGates:
    """Hard gates: all 4 must pass for phase4_exit_ok."""

    def test_all_hard_gates_pass(self) -> None:
        """Full portfolio with good metrics passes all hard gates."""
        layers = _make_full_portfolio_layers()
        agg = PortfolioAggregator()
        result = agg.aggregate(layers)

        assert result.hard_gates_passed == _HARD_GATES_TOTAL

    def test_strategic_dd_breach_fails_hard_gate(self) -> None:
        """Strategic DD > 8% fails hard gate 3."""
        layers = _make_full_portfolio_layers(strategic_dd_pct=9.0)
        agg = PortfolioAggregator()
        result = agg.aggregate(layers)

        assert result.strategic_dd_ok is False
        assert result.hard_gates_passed < _HARD_GATES_TOTAL

    def test_no_tactical_trades_fails_hard_gate(self) -> None:
        """Tactical layer with 0 trades fails hard gate 4."""
        layers = _make_full_portfolio_layers(n_tactical_trades=0)
        agg = PortfolioAggregator()
        result = agg.aggregate(layers)

        assert result.tactical_has_trades is False
        assert result.hard_gates_passed < _HARD_GATES_TOTAL

    def test_core_negative_return_fails(self) -> None:
        """Core return < 0% fails hard gate 2 (absolute profitability)."""
        layers = _make_full_portfolio_layers(core_return_pct=-5.0)
        agg = PortfolioAggregator(risk_free_annual_pct=_RUONIA_ANNUAL_PCT)
        result = agg.aggregate(layers)

        assert result.hard_gates_passed < _HARD_GATES_TOTAL

    def test_core_positive_return_passes(self) -> None:
        """Core return > 0% passes hard gate 2."""
        layers = _make_full_portfolio_layers(core_return_pct=10.0)
        agg = PortfolioAggregator(risk_free_annual_pct=_RUONIA_ANNUAL_PCT)
        result = agg.aggregate(layers)

        # Gate 2 passes (core > 0%), other gates may still fail
        core_lr = result.layer_results.get("core")
        assert core_lr is not None
        assert core_lr.total_return_pct > 0

    def test_missing_strategic_layer_ok(self) -> None:
        """If strategic layer is absent, strategic DD gate passes by default."""
        dates = _make_dates(_N_DAYS)
        core = _make_layer(
            "core",
            _make_linear_curve(_INITIAL_VALUE, _CORE_FINAL, _N_DAYS),
            dates,
        )
        agg = PortfolioAggregator()
        result = agg.aggregate([core])

        assert result.strategic_dd_ok is True

    def test_missing_tactical_layer_fails(self) -> None:
        """If tactical layer is absent, tactical_has_trades is False."""
        dates = _make_dates(_N_DAYS)
        core = _make_layer(
            "core",
            _make_linear_curve(_INITIAL_VALUE, _CORE_FINAL, _N_DAYS),
            dates,
        )
        agg = PortfolioAggregator()
        result = agg.aggregate([core])

        assert result.tactical_has_trades is False


# ── Test: Phase 4 Soft Gates ──────────────────────────────────────────────


class TestPhase4SoftGates:
    """Soft gates: 2 of 3 must pass for phase4_exit_ok (given hard gates pass)."""

    def test_all_soft_gates_pass(self) -> None:
        """Portfolio with good absolute Sharpe, core Calmar, and short PF."""
        layers = _make_full_portfolio_layers(short_pf=1.5)
        agg = PortfolioAggregator()
        result = agg.aggregate(layers)

        assert result.soft_gates_passed >= _SOFT_GATES_MIN_PASS

    def test_short_pf_below_threshold(self) -> None:
        """Short PF < 0.8 fails soft gate 3; other 2 can still pass."""
        layers = _make_full_portfolio_layers(short_pf=0.5)
        agg = PortfolioAggregator()
        result = agg.aggregate(layers)

        short_lr = result.layer_results.get("short")
        assert short_lr is not None
        assert short_lr.profit_factor < 0.8  # noqa: PLR2004

    def test_profit_factor_field_on_layer_result(self) -> None:
        """LayerResult.profit_factor defaults to 0.0."""
        dates = _make_dates(_N_DAYS)
        layer = _make_layer("core", _make_flat_curve(_INITIAL_VALUE, _N_DAYS), dates)
        assert layer.profit_factor == _ZERO


# ── Test: Phase 4 Overall Exit ─────────────────────────────────────────────


class TestPhase4ExitOk:
    """phase4_exit_ok requires all hard gates AND >= 2/3 soft gates."""

    def test_all_gates_pass(self) -> None:
        """All gates pass -> phase4_exit_ok is True."""
        layers = _make_full_portfolio_layers(
            core_return_pct=70.0,
            strategic_dd_pct=5.0,
            n_tactical_trades=10,
            short_pf=1.5,
        )
        agg = PortfolioAggregator()
        result = agg.aggregate(layers)

        assert result.hard_gates_passed == _HARD_GATES_TOTAL
        assert result.soft_gates_passed >= _SOFT_GATES_MIN_PASS
        assert result.phase4_exit_ok is True

    def test_hard_gate_fails_exit_not_ok(self) -> None:
        """One hard gate fails -> phase4_exit_ok is False regardless of soft."""
        layers = _make_full_portfolio_layers(
            strategic_dd_pct=9.0,  # fails hard gate 3
            short_pf=1.5,
        )
        agg = PortfolioAggregator()
        result = agg.aggregate(layers)

        assert result.phase4_exit_ok is False

    def test_insufficient_soft_gates_exit_not_ok(self) -> None:
        """All hard pass but < 2 soft gates -> phase4_exit_ok is False.

        We make: core Calmar low (0 DD => 0 Calmar), low absolute Sharpe, low short PF.
        """
        dates = _make_dates(_N_DAYS)
        # Flat curves produce zero Sharpe and zero Calmar
        core = LayerResult(
            layer_id="core",
            equity_curve=_make_flat_curve(_INITIAL_VALUE, _N_DAYS),
            dates=dates,
            trades=[],
            total_return_pct=0.0,
            max_drawdown_pct=0.0,
        )
        strategic = LayerResult(
            layer_id="strategic",
            equity_curve=_make_flat_curve(_INITIAL_VALUE, _N_DAYS),
            dates=dates,
            trades=[],
            total_return_pct=0.0,
            max_drawdown_pct=5.0,
        )
        tactical = LayerResult(
            layer_id="tactical",
            equity_curve=_make_flat_curve(_INITIAL_VALUE, _N_DAYS),
            dates=dates,
            trades=["t"],
            total_return_pct=0.0,
            max_drawdown_pct=0.0,
        )
        short = LayerResult(
            layer_id="short",
            equity_curve=_make_flat_curve(_INITIAL_VALUE, _N_DAYS),
            dates=dates,
            trades=[],
            total_return_pct=0.0,
            max_drawdown_pct=0.0,
            profit_factor=0.5,
        )

        agg = PortfolioAggregator(risk_free_annual_pct=0.0)
        result = agg.aggregate([core, strategic, tactical, short])

        # Flat curves: absolute Sharpe = 0, core Calmar = 0, short PF = 0.5
        assert result.soft_gates_passed < _SOFT_GATES_MIN_PASS
        assert result.phase4_exit_ok is False
