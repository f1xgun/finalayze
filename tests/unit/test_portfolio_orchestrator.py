"""Tests for PortfolioBacktestOrchestrator -- merged bond+equity portfolio."""

from __future__ import annotations

import datetime as dt
from decimal import Decimal

import pytest

from finalayze.backtest.portfolio_orchestrator import (
    PortfolioBacktestOrchestrator,
    PortfolioBacktestResult,
)
from finalayze.core.schemas import PortfolioState

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_bond_result(
    equity_curve: list[float],
    dates: list[dt.date],
) -> dict:
    """Build a minimal BondBacktestResult-like dict for testing."""
    from finalayze.backtest.bond_engine import BondBacktestResult

    return BondBacktestResult(
        trades=[],
        equity_curve=[Decimal(str(v)) for v in equity_curve],
        dates=dates,
        total_coupon_income_gross=Decimal(0),
        total_coupon_income_net=Decimal(0),
        total_tax_paid=Decimal(0),
        total_return_pct=Decimal(0),
        max_drawdown_pct=Decimal(0),
        sharpe_ratio=Decimal(0),
        trade_count=0,
        win_rate=Decimal(0),
        profit_factor=Decimal(0),
    )


def _make_snapshots(
    equity_values: list[float],
    dates: list[dt.date],
) -> list[PortfolioState]:
    """Build PortfolioState snapshots from equity values and dates."""
    return [
        PortfolioState(
            cash=Decimal(0),
            positions={},
            equity=Decimal(str(v)),
            timestamp=dt.datetime(d.year, d.month, d.day, tzinfo=dt.UTC),
        )
        for v, d in zip(equity_values, dates, strict=True)
    ]


def _make_usdrub(dates: list[dt.date], rates: list[float]) -> list[tuple[dt.date, float]]:
    return list(zip(dates, rates, strict=True))


# ---------------------------------------------------------------------------
# PORT-01: Merged curve and aggregate metrics
# ---------------------------------------------------------------------------


class TestPortfolioOrchestrator:
    """Tests for merged equity curve and aggregate metrics (PORT-01)."""

    def test_merged_curve_is_sum(self) -> None:
        """Bond + equity curves sum to merged curve."""
        dates = [dt.date(2024, 1, 1), dt.date(2024, 1, 2), dt.date(2024, 1, 3)]
        bond = _make_bond_result([100, 102, 104], dates)
        snapshots = _make_snapshots([100, 101, 103], dates)
        usdrub = _make_usdrub(dates, [90.0, 90.0, 90.0])

        orch = PortfolioBacktestOrchestrator()
        result = orch.run(
            bond_result=bond,
            equity_snapshots=snapshots,
            usdrub_series=usdrub,
            total_capital=200.0,
        )

        assert result.merged_equity_curve == pytest.approx([200.0, 203.0, 207.0])

    def test_date_alignment_forward_fill(self) -> None:
        """Misaligned dates are unified via forward-fill."""
        d1 = dt.date(2024, 1, 1)
        d2 = dt.date(2024, 1, 2)
        d3 = dt.date(2024, 1, 3)
        d4 = dt.date(2024, 1, 4)

        bond = _make_bond_result([100, 102, 104], [d1, d2, d4])
        snapshots = _make_snapshots([100, 101, 103], [d1, d3, d4])
        usdrub = _make_usdrub([d1, d2, d3, d4], [90.0] * 4)

        orch = PortfolioBacktestOrchestrator()
        result = orch.run(
            bond_result=bond,
            equity_snapshots=snapshots,
            usdrub_series=usdrub,
            total_capital=200.0,
        )

        # Common dates: d1, d2, d3, d4
        assert result.dates == [d1, d2, d3, d4]
        # d1: 100+100=200, d2: 102+100(ff)=202, d3: 102(ff)+101=203, d4: 104+103=207
        assert result.merged_equity_curve == pytest.approx([200.0, 202.0, 203.0, 207.0])

    def test_aggregate_sharpe_computed(self) -> None:
        """Sharpe is computed on merged curve."""
        # 20 days of data so we have enough returns
        dates = [dt.date(2024, 1, i + 1) for i in range(20)]
        bond_vals = [100.0 + i * 0.5 for i in range(20)]
        equity_vals = [100.0 + i * 0.3 for i in range(20)]
        bond = _make_bond_result(bond_vals, dates)
        snapshots = _make_snapshots(equity_vals, dates)
        usdrub = _make_usdrub(dates, [90.0] * 20)

        orch = PortfolioBacktestOrchestrator()
        result = orch.run(
            bond_result=bond,
            equity_snapshots=snapshots,
            usdrub_series=usdrub,
            total_capital=200.0,
        )

        # Should be positive (steadily rising curve)
        assert result.sharpe > 0

    def test_aggregate_max_drawdown(self) -> None:
        """Max drawdown computed from merged curve peak-to-trough."""
        dates = [dt.date(2024, 1, i + 1) for i in range(4)]
        # Bond flat, equity drops then recovers
        bond = _make_bond_result([50, 55, 47.5, 52.5], dates)
        snapshots = _make_snapshots([50, 55, 47.5, 52.5], dates)
        usdrub = _make_usdrub(dates, [90.0] * 4)

        orch = PortfolioBacktestOrchestrator()
        result = orch.run(
            bond_result=bond,
            equity_snapshots=snapshots,
            usdrub_series=usdrub,
            total_capital=200.0,
        )

        # Merged: [100, 110, 95, 105]. Peak=110, trough=95, DD = 15/110 = 13.636...%
        assert result.max_drawdown_pct == pytest.approx(15.0 / 110 * 100, rel=1e-2)

    def test_aggregate_profit_factor(self) -> None:
        """Profit factor = sum gains / abs(sum losses) from daily returns."""
        dates = [dt.date(2024, 1, i + 1) for i in range(5)]
        # Merged will be: [200, 210, 205, 215, 220]
        bond = _make_bond_result([100, 105, 102.5, 107.5, 110], dates)
        snapshots = _make_snapshots([100, 105, 102.5, 107.5, 110], dates)
        usdrub = _make_usdrub(dates, [90.0] * 5)

        orch = PortfolioBacktestOrchestrator()
        result = orch.run(
            bond_result=bond,
            equity_snapshots=snapshots,
            usdrub_series=usdrub,
            total_capital=200.0,
        )

        # PF from percentage daily returns; gains dominate losses -> PF > 1
        assert result.profit_factor > 4.0

    def test_result_dataclass_fields(self) -> None:
        """PortfolioBacktestResult has all required fields."""
        required = {
            "bond_equity_curve",
            "equity_equity_curve",
            "merged_equity_curve",
            "dates",
            "bond_trades",
            "equity_trades",
            "sharpe",
            "max_drawdown_pct",
            "profit_factor",
            "total_return_pct",
            "bond_weight_series",
            "equity_weight_series",
            "crisis_brake_active_dates",
            "wf_sharpe",
        }
        actual = {f.name for f in PortfolioBacktestResult.__dataclass_fields__.values()}
        assert required <= actual, f"Missing fields: {required - actual}"


# ---------------------------------------------------------------------------
# PORT-02: Allocation, rebalancing, crisis brake
# ---------------------------------------------------------------------------


class TestRebalancing:
    """Tests for 40/60 allocation and monthly rebalancing (PORT-02)."""

    def test_initial_capital_split(self) -> None:
        """40/60 split reflected in starting curve values."""
        dates = [dt.date(2024, 1, 1)]
        # Each engine receives its share of capital
        bond = _make_bond_result([400_000], dates)
        snapshots = _make_snapshots([600_000], dates)
        usdrub = _make_usdrub(dates, [90.0])

        orch = PortfolioBacktestOrchestrator()
        result = orch.run(
            bond_result=bond,
            equity_snapshots=snapshots,
            usdrub_series=usdrub,
            total_capital=1_000_000,
        )

        assert result.merged_equity_curve[0] == pytest.approx(1_000_000.0)
        assert result.bond_equity_curve[0] == pytest.approx(400_000.0)
        assert result.equity_equity_curve[0] == pytest.approx(600_000.0)

    def test_monthly_rebalance_triggers_on_drift(self) -> None:
        """Rebalance happens at month boundary when drift > 5%."""
        # Bond grows 20% in Jan, equity flat -> drift > 5%
        jan_dates = [dt.date(2024, 1, d) for d in range(1, 32)]
        feb_dates = [dt.date(2024, 2, d) for d in range(1, 6)]
        all_dates = jan_dates + feb_dates

        bond_vals = [400.0] * 31
        bond_vals[-1] = 480.0  # +20% on last day of Jan
        equity_vals = [600.0] * 36  # flat

        bond = _make_bond_result(bond_vals + [480.0] * 5, all_dates)
        snapshots = _make_snapshots(equity_vals, all_dates)
        usdrub = _make_usdrub(all_dates, [90.0] * len(all_dates))

        orch = PortfolioBacktestOrchestrator()
        result = orch.run(
            bond_result=bond,
            equity_snapshots=snapshots,
            usdrub_series=usdrub,
            total_capital=1000.0,
        )

        # At Feb 1: total=1080, target bond=432, equity=648
        # Bond weight should shift back towards 40%
        # Check that weight series changes at month boundary
        feb_start_idx = 31  # index of Feb 1
        # After rebalance, bond weight should be close to 0.40
        assert result.bond_weight_series[feb_start_idx] == pytest.approx(0.40, abs=0.01)

    def test_no_rebalance_below_threshold(self) -> None:
        """No rebalance when drift < 5%."""
        jan_dates = [dt.date(2024, 1, d) for d in range(1, 32)]
        feb_dates = [dt.date(2024, 2, 1)]
        all_dates = jan_dates + feb_dates

        # Bond grows 2%, equity grows 1% -> drift < 5%
        bond_vals = [400.0 + i * 0.26 for i in range(32)]
        equity_vals = [600.0 + i * 0.19 for i in range(32)]

        bond = _make_bond_result(bond_vals, all_dates)
        snapshots = _make_snapshots(equity_vals, all_dates)
        usdrub = _make_usdrub(all_dates, [90.0] * len(all_dates))

        orch = PortfolioBacktestOrchestrator()
        result = orch.run(
            bond_result=bond,
            equity_snapshots=snapshots,
            usdrub_series=usdrub,
            total_capital=1000.0,
        )

        # Weights should remain stable (no rebalance)
        assert result.bond_weight_series[-1] == pytest.approx(
            result.bond_weight_series[0], abs=0.02
        )

    def test_no_rebalance_mid_month(self) -> None:
        """Even with large drift, no rebalance mid-month."""
        dates = [dt.date(2024, 1, d) for d in range(1, 16)]
        # Bond jumps 30% on day 10 -> large drift, but mid-month
        bond_vals = [400.0] * 15
        bond_vals[9] = 520.0
        for i in range(10, 15):
            bond_vals[i] = 520.0
        equity_vals = [600.0] * 15

        bond = _make_bond_result(bond_vals, dates)
        snapshots = _make_snapshots(equity_vals, dates)
        usdrub = _make_usdrub(dates, [90.0] * 15)

        orch = PortfolioBacktestOrchestrator()
        result = orch.run(
            bond_result=bond,
            equity_snapshots=snapshots,
            usdrub_series=usdrub,
            total_capital=1000.0,
        )

        # No rebalance mid-month; initial scale stays
        assert result.bond_weight_series[14] == pytest.approx(
            result.bond_weight_series[0], abs=0.01
        )


class TestCrisisBrake:
    """Tests for USDRUB crisis brake (PORT-02)."""

    def test_crisis_brake_activates(self) -> None:
        """Crisis activates when USDRUB 20-bar return > 15%."""
        dates = [dt.date(2024, 1, d + 1) for d in range(25)]
        bond = _make_bond_result([400.0] * 25, dates)
        snapshots = _make_snapshots([600.0] * 25, dates)

        # USDRUB: stable for 20 bars, then spikes 16%
        usdrub_rates = [90.0] * 20 + [104.4] * 5  # 104.4/90 - 1 = 16%
        usdrub = _make_usdrub(dates, usdrub_rates)

        orch = PortfolioBacktestOrchestrator()
        result = orch.run(
            bond_result=bond,
            equity_snapshots=snapshots,
            usdrub_series=usdrub,
            total_capital=1000.0,
        )

        # Crisis should be active from bar 20 onward
        assert len(result.crisis_brake_active_dates) > 0

    def test_crisis_brake_allocation_shift(self) -> None:
        """During crisis, weights shift to 80/20."""
        dates = [dt.date(2024, 1, d + 1) for d in range(25)]
        bond = _make_bond_result([400.0] * 25, dates)
        snapshots = _make_snapshots([600.0] * 25, dates)

        usdrub_rates = [90.0] * 20 + [104.4] * 5
        usdrub = _make_usdrub(dates, usdrub_rates)

        orch = PortfolioBacktestOrchestrator()
        result = orch.run(
            bond_result=bond,
            equity_snapshots=snapshots,
            usdrub_series=usdrub,
            total_capital=1000.0,
        )

        # After crisis activates (bar 20+), bond weight should be 0.80
        crisis_idx = 20  # first crisis bar
        assert result.bond_weight_series[crisis_idx] == pytest.approx(0.80, abs=0.01)

    def test_crisis_brake_deactivates(self) -> None:
        """Crisis deactivates when USDRUB return drops below 15%."""
        base = dt.date(2024, 1, 1)
        dates = [base + dt.timedelta(days=i) for i in range(45)]
        bond = _make_bond_result([400.0] * 45, dates)
        snapshots = _make_snapshots([600.0] * 45, dates)

        # Spike then revert
        usdrub_rates = [90.0] * 20 + [104.4] * 5 + [90.0] * 20
        usdrub = _make_usdrub(dates, usdrub_rates)

        orch = PortfolioBacktestOrchestrator()
        result = orch.run(
            bond_result=bond,
            equity_snapshots=snapshots,
            usdrub_series=usdrub,
            total_capital=1000.0,
        )

        # After revert (bar 40+), crisis should be off, weight back to 0.40
        assert result.bond_weight_series[44] == pytest.approx(0.40, abs=0.01)

    def test_crisis_brake_not_triggered_below_threshold(self) -> None:
        """No crisis when USDRUB rise < 15%."""
        dates = [dt.date(2024, 1, d + 1) for d in range(25)]
        bond = _make_bond_result([400.0] * 25, dates)
        snapshots = _make_snapshots([600.0] * 25, dates)

        # 10% rise -- below 15% threshold
        usdrub_rates = [90.0] * 20 + [99.0] * 5
        usdrub = _make_usdrub(dates, usdrub_rates)

        orch = PortfolioBacktestOrchestrator()
        result = orch.run(
            bond_result=bond,
            equity_snapshots=snapshots,
            usdrub_series=usdrub,
            total_capital=1000.0,
        )

        assert len(result.crisis_brake_active_dates) == 0

    def test_crisis_brake_active_dates_tracked(self) -> None:
        """crisis_brake_active_dates contains the correct dates."""
        dates = [dt.date(2024, 1, d + 1) for d in range(25)]
        bond = _make_bond_result([400.0] * 25, dates)
        snapshots = _make_snapshots([600.0] * 25, dates)

        usdrub_rates = [90.0] * 20 + [104.4] * 5
        usdrub = _make_usdrub(dates, usdrub_rates)

        orch = PortfolioBacktestOrchestrator()
        result = orch.run(
            bond_result=bond,
            equity_snapshots=snapshots,
            usdrub_series=usdrub,
            total_capital=1000.0,
        )

        # Dates from bar 20 onward should be crisis dates
        expected_crisis_dates = dates[20:]
        assert result.crisis_brake_active_dates == expected_crisis_dates
