"""Unit tests for bond-specific performance metrics."""

from __future__ import annotations

from decimal import Decimal
from uuid import uuid4

import pytest

from finalayze.backtest.bond_metrics import BondPerformanceMetrics, compute_bond_metrics
from finalayze.core.schemas import TradeResult

# ── Constants (no magic numbers per ruff PLR2004) ────────────────────────────
_INITIAL_CASH = 1_000_000.0
_RUONIA_ANNUAL_PCT = 15.0
_TRADING_DAYS = 252
_ZERO = 0.0
_TOLERANCE = 1e-6
_LOOSE_TOLERANCE = 0.01  # 1% tolerance for annualisation checks

# Equity curve values for various test scenarios
_FLAT_VALUE = 1_000_000.0
_FLAT_CURVE_LENGTH = 10

# Positive return below risk-free: 5% total return over 252 days
_BELOW_RF_INITIAL = 1_000_000.0
_BELOW_RF_FINAL = 1_050_000.0  # +5%, well below 15% RUONIA

# Positive return above risk-free: 25% total return over 252 days
_ABOVE_RF_INITIAL = 1_000_000.0
_ABOVE_RF_FINAL = 1_250_000.0  # +25%, above 15% RUONIA

# Drawdown scenario
_DD_PEAK = 1_100_000.0
_DD_TROUGH = 990_000.0
_DD_EXPECTED_PCT = (_DD_PEAK - _DD_TROUGH) / _DD_PEAK * 100  # ~10%

# Trade PnL values
_WIN_PNL_1 = Decimal(500)
_WIN_PNL_2 = Decimal(300)
_LOSS_PNL = Decimal(-200)
_WIN_HOLD_BARS = 10
_LOSS_HOLD_BARS = 5

# Coupon values
_COUPON_GROSS = 50_000.0
_COUPON_NET = 43_500.0  # after 13% NDFL
_NDFL_RATE = 0.13

# Annualisation test: half-year equity curve
_HALF_YEAR_DAYS = 126  # ~0.5 year
_HALF_YEAR_RETURN = 0.08  # 8% in half year

# Expected values for trade metrics test
_EXPECTED_WIN_RATE = 2 / 3  # 2 wins out of 3
_EXPECTED_GROSS_PROFIT = 800.0  # 500 + 300
_EXPECTED_GROSS_LOSS = 200.0  # abs(-200)
_EXPECTED_PROFIT_FACTOR = _EXPECTED_GROSS_PROFIT / _EXPECTED_GROSS_LOSS  # 4.0
_EXPECTED_AVG_HOLD = (10 + 10 + 5) / 3


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_linear_curve(start: float, end: float, n_points: int) -> list[float]:
    """Create a linearly interpolated equity curve."""
    if n_points < 2:  # noqa: PLR2004
        return [start]
    step = (end - start) / (n_points - 1)
    return [start + step * i for i in range(n_points)]


def _make_flat_curve(value: float, n_points: int) -> list[float]:
    """Create a flat equity curve."""
    return [value] * n_points


def _make_trade(pnl: Decimal, hold_bars: int = _WIN_HOLD_BARS) -> TradeResult:
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


def _make_dates(n: int) -> list[str]:
    """Create a list of placeholder date strings."""
    return [f"2025-01-{i + 1:02d}" for i in range(n)]


# ── Test classes ─────────────────────────────────────────────────────────────


class TestEmptyEquityCurve:
    """Empty or singleton equity curve returns zero metrics."""

    def test_empty_list_returns_zeros(self) -> None:
        result = compute_bond_metrics(
            equity_curve=[],
            dates=[],
            trades=[],
            coupon_income_gross=_ZERO,
            coupon_income_net=_ZERO,
            initial_cash=_INITIAL_CASH,
        )
        assert isinstance(result, BondPerformanceMetrics)
        assert result.total_return_pct == _ZERO
        assert result.excess_sharpe == _ZERO
        assert result.max_drawdown_pct == _ZERO
        assert result.trade_count == 0

    def test_single_point_returns_zeros(self) -> None:
        result = compute_bond_metrics(
            equity_curve=[_INITIAL_CASH],
            dates=_make_dates(1),
            trades=[],
            coupon_income_gross=_ZERO,
            coupon_income_net=_ZERO,
            initial_cash=_INITIAL_CASH,
        )
        assert result.total_return_pct == _ZERO
        assert result.excess_sharpe == _ZERO


class TestFlatEquityCurve:
    """Flat equity curve: zero return, zero Sharpe."""

    def test_flat_curve_zero_return(self) -> None:
        curve = _make_flat_curve(_FLAT_VALUE, _FLAT_CURVE_LENGTH)
        result = compute_bond_metrics(
            equity_curve=curve,
            dates=_make_dates(len(curve)),
            trades=[],
            coupon_income_gross=_ZERO,
            coupon_income_net=_ZERO,
            initial_cash=_FLAT_VALUE,
        )
        assert result.total_return_pct == pytest.approx(_ZERO, abs=_TOLERANCE)
        assert result.excess_sharpe == pytest.approx(_ZERO, abs=_TOLERANCE)
        assert result.annualized_volatility_pct == pytest.approx(_ZERO, abs=_TOLERANCE)

    def test_flat_curve_negative_excess(self) -> None:
        """A flat curve has 0% return, so excess over RUONIA is negative."""
        curve = _make_flat_curve(_FLAT_VALUE, _FLAT_CURVE_LENGTH)
        result = compute_bond_metrics(
            equity_curve=curve,
            dates=_make_dates(len(curve)),
            trades=[],
            coupon_income_gross=_ZERO,
            coupon_income_net=_ZERO,
            initial_cash=_FLAT_VALUE,
        )
        assert result.excess_return_pct < _ZERO


class TestReturnBelowRiskFree:
    """Positive return below risk-free rate produces negative excess Sharpe."""

    def test_negative_excess_sharpe(self) -> None:
        n_points = _TRADING_DAYS + 1  # 1 year of daily data
        curve = _make_linear_curve(_BELOW_RF_INITIAL, _BELOW_RF_FINAL, n_points)
        result = compute_bond_metrics(
            equity_curve=curve,
            dates=_make_dates(n_points),
            trades=[],
            coupon_income_gross=_ZERO,
            coupon_income_net=_ZERO,
            initial_cash=_BELOW_RF_INITIAL,
            risk_free_annual_pct=_RUONIA_ANNUAL_PCT,
        )
        # 5% return vs 15% RUONIA => negative excess
        assert result.excess_return_pct < _ZERO
        assert result.excess_sharpe < _ZERO

    def test_positive_total_return(self) -> None:
        """Total return is still positive even though excess is negative."""
        n_points = _TRADING_DAYS + 1
        curve = _make_linear_curve(_BELOW_RF_INITIAL, _BELOW_RF_FINAL, n_points)
        result = compute_bond_metrics(
            equity_curve=curve,
            dates=_make_dates(n_points),
            trades=[],
            coupon_income_gross=_ZERO,
            coupon_income_net=_ZERO,
            initial_cash=_BELOW_RF_INITIAL,
            risk_free_annual_pct=_RUONIA_ANNUAL_PCT,
        )
        assert result.total_return_pct > _ZERO


class TestReturnAboveRiskFree:
    """Positive return above risk-free rate produces positive excess Sharpe."""

    def test_positive_excess_sharpe(self) -> None:
        n_points = _TRADING_DAYS + 1  # 1 year
        curve = _make_linear_curve(_ABOVE_RF_INITIAL, _ABOVE_RF_FINAL, n_points)
        result = compute_bond_metrics(
            equity_curve=curve,
            dates=_make_dates(n_points),
            trades=[],
            coupon_income_gross=_ZERO,
            coupon_income_net=_ZERO,
            initial_cash=_ABOVE_RF_INITIAL,
            risk_free_annual_pct=_RUONIA_ANNUAL_PCT,
        )
        # 25% return vs 15% RUONIA => positive excess
        assert result.excess_return_pct > _ZERO
        assert result.excess_sharpe > _ZERO

    def test_annualized_excess_positive(self) -> None:
        n_points = _TRADING_DAYS + 1
        curve = _make_linear_curve(_ABOVE_RF_INITIAL, _ABOVE_RF_FINAL, n_points)
        result = compute_bond_metrics(
            equity_curve=curve,
            dates=_make_dates(n_points),
            trades=[],
            coupon_income_gross=_ZERO,
            coupon_income_net=_ZERO,
            initial_cash=_ABOVE_RF_INITIAL,
            risk_free_annual_pct=_RUONIA_ANNUAL_PCT,
        )
        assert result.annualized_excess_return_pct > _ZERO


class TestMaxDrawdown:
    """Max drawdown computation: correct peak-to-trough."""

    def test_drawdown_value(self) -> None:
        # Curve: 1M -> 1.1M -> 0.99M -> 1.05M
        curve = [_INITIAL_CASH, _DD_PEAK, _DD_TROUGH, 1_050_000.0]
        result = compute_bond_metrics(
            equity_curve=curve,
            dates=_make_dates(len(curve)),
            trades=[],
            coupon_income_gross=_ZERO,
            coupon_income_net=_ZERO,
            initial_cash=_INITIAL_CASH,
        )
        assert result.max_drawdown_pct == pytest.approx(_DD_EXPECTED_PCT, rel=_TOLERANCE)

    def test_no_drawdown_on_monotonic_increase(self) -> None:
        curve = [1_000_000.0, 1_010_000.0, 1_020_000.0, 1_030_000.0]
        result = compute_bond_metrics(
            equity_curve=curve,
            dates=_make_dates(len(curve)),
            trades=[],
            coupon_income_gross=_ZERO,
            coupon_income_net=_ZERO,
            initial_cash=_INITIAL_CASH,
        )
        assert result.max_drawdown_pct == pytest.approx(_ZERO, abs=_TOLERANCE)

    def test_drawdown_at_end(self) -> None:
        """Drawdown at the end of the curve is captured."""
        curve = [1_000_000.0, 1_100_000.0, 1_050_000.0]
        expected_dd = (1_100_000.0 - 1_050_000.0) / 1_100_000.0 * 100
        result = compute_bond_metrics(
            equity_curve=curve,
            dates=_make_dates(len(curve)),
            trades=[],
            coupon_income_gross=_ZERO,
            coupon_income_net=_ZERO,
            initial_cash=_INITIAL_CASH,
        )
        assert result.max_drawdown_pct == pytest.approx(expected_dd, rel=_TOLERANCE)


class TestTradeMetrics:
    """Win rate, profit factor from trade list."""

    def test_win_rate(self) -> None:
        trades = [
            _make_trade(_WIN_PNL_1),
            _make_trade(_WIN_PNL_2),
            _make_trade(_LOSS_PNL, hold_bars=_LOSS_HOLD_BARS),
        ]
        curve = _make_linear_curve(_INITIAL_CASH, 1_050_000.0, _FLAT_CURVE_LENGTH)
        result = compute_bond_metrics(
            equity_curve=curve,
            dates=_make_dates(len(curve)),
            trades=trades,
            coupon_income_gross=_ZERO,
            coupon_income_net=_ZERO,
            initial_cash=_INITIAL_CASH,
        )
        assert result.win_rate == pytest.approx(_EXPECTED_WIN_RATE, rel=_TOLERANCE)

    def test_profit_factor(self) -> None:
        trades = [
            _make_trade(_WIN_PNL_1),
            _make_trade(_WIN_PNL_2),
            _make_trade(_LOSS_PNL, hold_bars=_LOSS_HOLD_BARS),
        ]
        curve = _make_linear_curve(_INITIAL_CASH, 1_050_000.0, _FLAT_CURVE_LENGTH)
        result = compute_bond_metrics(
            equity_curve=curve,
            dates=_make_dates(len(curve)),
            trades=trades,
            coupon_income_gross=_ZERO,
            coupon_income_net=_ZERO,
            initial_cash=_INITIAL_CASH,
        )
        assert result.profit_factor == pytest.approx(_EXPECTED_PROFIT_FACTOR, rel=_TOLERANCE)

    def test_avg_hold_bars(self) -> None:
        trades = [
            _make_trade(_WIN_PNL_1),
            _make_trade(_WIN_PNL_2),
            _make_trade(_LOSS_PNL, hold_bars=_LOSS_HOLD_BARS),
        ]
        curve = _make_linear_curve(_INITIAL_CASH, 1_050_000.0, _FLAT_CURVE_LENGTH)
        result = compute_bond_metrics(
            equity_curve=curve,
            dates=_make_dates(len(curve)),
            trades=trades,
            coupon_income_gross=_ZERO,
            coupon_income_net=_ZERO,
            initial_cash=_INITIAL_CASH,
        )
        assert result.avg_hold_bars == pytest.approx(_EXPECTED_AVG_HOLD, rel=_TOLERANCE)

    def test_no_trades_zero_metrics(self) -> None:
        curve = _make_linear_curve(_INITIAL_CASH, 1_050_000.0, _FLAT_CURVE_LENGTH)
        result = compute_bond_metrics(
            equity_curve=curve,
            dates=_make_dates(len(curve)),
            trades=[],
            coupon_income_gross=_ZERO,
            coupon_income_net=_ZERO,
            initial_cash=_INITIAL_CASH,
        )
        assert result.trade_count == 0
        assert result.win_rate == _ZERO
        assert result.profit_factor == _ZERO

    def test_all_winning_trades_infinite_pf(self) -> None:
        """Profit factor is inf when there are no losing trades."""
        trades = [_make_trade(_WIN_PNL_1), _make_trade(_WIN_PNL_2)]
        curve = _make_linear_curve(_INITIAL_CASH, 1_050_000.0, _FLAT_CURVE_LENGTH)
        result = compute_bond_metrics(
            equity_curve=curve,
            dates=_make_dates(len(curve)),
            trades=trades,
            coupon_income_gross=_ZERO,
            coupon_income_net=_ZERO,
            initial_cash=_INITIAL_CASH,
        )
        assert result.profit_factor == float("inf")


class TestCouponContribution:
    """Correct percentage of total PnL from coupons."""

    def test_coupon_contribution_pct(self) -> None:
        # Total PnL = 1_050_000 - 1_000_000 = 50_000
        # Net coupon = 43_500 => 43_500 / 50_000 * 100 = 87%
        curve = _make_linear_curve(_INITIAL_CASH, 1_050_000.0, _FLAT_CURVE_LENGTH)
        total_pnl = 1_050_000.0 - _INITIAL_CASH
        expected_contrib = _COUPON_NET / total_pnl * 100
        result = compute_bond_metrics(
            equity_curve=curve,
            dates=_make_dates(len(curve)),
            trades=[],
            coupon_income_gross=_COUPON_GROSS,
            coupon_income_net=_COUPON_NET,
            initial_cash=_INITIAL_CASH,
        )
        assert result.coupon_contribution_pct == pytest.approx(expected_contrib, rel=_TOLERANCE)
        assert result.total_coupon_income_gross == _COUPON_GROSS
        assert result.total_coupon_income_net == _COUPON_NET

    def test_coupon_zero_when_no_pnl(self) -> None:
        """Coupon contribution is 0 when total PnL is zero or negative."""
        curve = _make_flat_curve(_INITIAL_CASH, _FLAT_CURVE_LENGTH)
        result = compute_bond_metrics(
            equity_curve=curve,
            dates=_make_dates(len(curve)),
            trades=[],
            coupon_income_gross=_COUPON_GROSS,
            coupon_income_net=_COUPON_NET,
            initial_cash=_INITIAL_CASH,
        )
        assert result.coupon_contribution_pct == _ZERO

    def test_coupon_values_passed_through(self) -> None:
        """Gross and net coupon values are stored in result as-is."""
        curve = _make_linear_curve(_INITIAL_CASH, 1_050_000.0, _FLAT_CURVE_LENGTH)
        result = compute_bond_metrics(
            equity_curve=curve,
            dates=_make_dates(len(curve)),
            trades=[],
            coupon_income_gross=_COUPON_GROSS,
            coupon_income_net=_COUPON_NET,
            initial_cash=_INITIAL_CASH,
        )
        assert result.total_coupon_income_gross == _COUPON_GROSS
        assert result.total_coupon_income_net == _COUPON_NET


class TestAnnualisation:
    """Correct scaling for partial years."""

    def test_half_year_annualisation(self) -> None:
        """Half-year return annualises to roughly (1+r)^2 - 1."""
        n_points = _HALF_YEAR_DAYS + 1  # half year
        start = _INITIAL_CASH
        end = start * (1 + _HALF_YEAR_RETURN)
        curve = _make_linear_curve(start, end, n_points)

        result = compute_bond_metrics(
            equity_curve=curve,
            dates=_make_dates(n_points),
            trades=[],
            coupon_income_gross=_ZERO,
            coupon_income_net=_ZERO,
            initial_cash=start,
        )

        # Total return should be ~8%
        assert result.total_return_pct == pytest.approx(
            _HALF_YEAR_RETURN * 100, rel=_LOOSE_TOLERANCE
        )

        # Annualised return should be roughly (1.08)^2 - 1 = ~16.64%
        expected_annual = ((1 + _HALF_YEAR_RETURN) ** 2 - 1) * 100
        assert result.annualized_return_pct == pytest.approx(expected_annual, rel=_LOOSE_TOLERANCE)

    def test_full_year_no_scaling(self) -> None:
        """Full year: annualised return ~ total return."""
        n_points = _TRADING_DAYS + 1
        total_return_pct = 20.0
        start = _INITIAL_CASH
        end = start * (1 + total_return_pct / 100)
        curve = _make_linear_curve(start, end, n_points)

        result = compute_bond_metrics(
            equity_curve=curve,
            dates=_make_dates(n_points),
            trades=[],
            coupon_income_gross=_ZERO,
            coupon_income_net=_ZERO,
            initial_cash=start,
        )

        assert result.total_return_pct == pytest.approx(total_return_pct, rel=_LOOSE_TOLERANCE)
        # Annualised should be very close to total for exactly 1 year
        assert result.annualized_return_pct == pytest.approx(total_return_pct, rel=_LOOSE_TOLERANCE)

    def test_excess_return_uses_proportional_rf(self) -> None:
        """Excess return subtracts proportional risk-free rate for the period."""
        n_points = _HALF_YEAR_DAYS + 1
        total_return = 0.08  # 8% in half year
        start = _INITIAL_CASH
        end = start * (1 + total_return)
        curve = _make_linear_curve(start, end, n_points)

        result = compute_bond_metrics(
            equity_curve=curve,
            dates=_make_dates(n_points),
            trades=[],
            coupon_income_gross=_ZERO,
            coupon_income_net=_ZERO,
            initial_cash=start,
            risk_free_annual_pct=_RUONIA_ANNUAL_PCT,
        )

        # n_years = (126+1)/252 ~ 0.504
        n_years = n_points / _TRADING_DAYS
        expected_excess = (total_return - (_RUONIA_ANNUAL_PCT / 100) * n_years) * 100
        assert result.excess_return_pct == pytest.approx(expected_excess, rel=_LOOSE_TOLERANCE)
