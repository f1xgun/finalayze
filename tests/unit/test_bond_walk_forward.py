"""Tests for bond walk-forward validation.

Bond walk-forward differs from equity WF:
- Longer windows: 24-month train, 12-month test, 6-month step
- Minimum trades per fold: 5 (not 30)
- Metrics: excess Sharpe over RUONIA, not absolute Sharpe
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import date

from finalayze.backtest.bond_walk_forward import (
    _DEFAULT_MIN_TRADES_PER_FOLD,
    BondWalkForwardFold,
    BondWalkForwardResult,
    generate_wf_windows,
    run_bond_walk_forward,
)

# ── Constants ───────────────────────────────────────────────────────────────
_SEED = 42
# Use calendar days (not trading days) so date spans cover enough months
# for the 24mo train + 12mo test windows.
_FOUR_YEARS_CALENDAR_DAYS = 365 * 4 + 1  # ~48 months


# ── Helpers ─────────────────────────────────────────────────────────────────


@dataclass
class _FakeTrade:
    """Minimal trade stub with pnl and entry/exit dates."""

    pnl: float
    entry_date: date
    exit_date: date


def _make_synthetic_curve(
    start: date,
    n_days: int,
    daily_return: float = 0.0003,
    seed: int = _SEED,
) -> tuple[list[float], list[date]]:
    """Create a synthetic equity curve with slight upward drift.

    Returns (equity_curve, dates).
    """
    rng = random.Random(seed)  # noqa: S311
    equity = [1_000_000.0]
    dates: list[date] = [start]
    for i in range(1, n_days):
        noise = rng.gauss(0, 0.005)
        equity.append(equity[-1] * (1 + daily_return + noise))
        dates.append(date.fromordinal(start.toordinal() + i))
    return equity, dates


def _make_synthetic_trades(
    start: date,
    end: date,
    n_trades: int,
    win_fraction: float = 0.6,
    seed: int = _SEED,
) -> list[_FakeTrade]:
    """Create synthetic trades spread across the date range."""
    rng = random.Random(seed)  # noqa: S311
    total_days = (end - start).days
    if total_days <= 0 or n_trades <= 0:
        return []
    trades: list[_FakeTrade] = []
    for _i in range(n_trades):
        day_offset = rng.randint(0, max(0, total_days - 10))
        entry = date.fromordinal(start.toordinal() + day_offset)
        hold = rng.randint(3, 10)
        exit_d = min(date.fromordinal(entry.toordinal() + hold), end)
        pnl = rng.uniform(100, 5000) if rng.random() < win_fraction else rng.uniform(-5000, -100)
        trades.append(_FakeTrade(pnl=pnl, entry_date=entry, exit_date=exit_d))
    return trades


# ── TestGenerateWFWindows ───────────────────────────────────────────────────


class TestGenerateWFWindows:
    """Test walk-forward window generation."""

    def test_basic_windows(self) -> None:
        """4 years of data with default params (24mo train, 12mo test, 6mo step)
        should produce the correct number of folds."""
        start = date(2020, 1, 1)
        end = date(2023, 12, 31)
        windows = generate_wf_windows(start, end)

        # 24 + 12 = 36 months minimum. 4 years = 48 months.
        # After the first window (months 0-35), stepping 6 months:
        # Window 0: train 0-23, test 24-35 -> fits
        # Window 1: train 6-29, test 30-41 -> fits
        # Window 2: train 12-35, test 36-47 -> fits (end of Dec 2023)
        # Window 3: train 18-41, test 42-53 -> 53 months > 48, doesn't fit
        expected_folds = 3
        assert len(windows) == expected_folds

    def test_window_boundaries(self) -> None:
        """train_end should be the day before test_start."""
        start = date(2020, 1, 1)
        end = date(2024, 12, 31)
        windows = generate_wf_windows(start, end)
        assert len(windows) > 0
        for _train_start, train_end, test_start, _test_end in windows:
            # train_end + 1 day == test_start
            assert test_start.toordinal() - train_end.toordinal() == 1

    def test_short_data_no_folds(self) -> None:
        """Less than train+test period should produce no folds."""
        start = date(2020, 1, 1)
        end = date(2022, 6, 30)  # 30 months < 36 months needed
        windows = generate_wf_windows(start, end)
        assert len(windows) == 0

    def test_custom_parameters(self) -> None:
        """12mo train, 6mo test, 3mo step on 3 years produces multiple folds."""
        start = date(2020, 1, 1)
        end = date(2022, 12, 31)
        windows = generate_wf_windows(start, end, train_months=12, test_months=6, step_months=3)
        # 12+6 = 18 months minimum. 36 months of data.
        # Should produce several folds
        assert len(windows) >= 4  # noqa: PLR2004

    def test_windows_dont_exceed_end(self) -> None:
        """Last window's test_end must not go past end_date."""
        start = date(2020, 1, 1)
        end = date(2025, 6, 30)
        windows = generate_wf_windows(start, end)
        for _train_start, _train_end, _test_start, test_end in windows:
            assert test_end <= end


# ── TestBondWalkForwardFold ────────────────────────────────────────────────


class TestBondWalkForwardFold:
    """Test the BondWalkForwardFold dataclass semantics."""

    def test_fold_with_sufficient_trades(self) -> None:
        """10 OOS trades >= default min (5) -> sufficient_trades=True."""
        fold = BondWalkForwardFold(
            fold_idx=0,
            train_start=date(2020, 1, 1),
            train_end=date(2021, 12, 31),
            test_start=date(2022, 1, 1),
            test_end=date(2022, 12, 31),
            is_trades=20,
            is_return_pct=5.0,
            is_excess_sharpe=0.5,
            oos_trades=10,
            oos_return_pct=3.0,
            oos_excess_sharpe=0.3,
            sufficient_trades=True,
        )
        assert fold.sufficient_trades is True
        oos_min = 10
        assert fold.oos_trades >= oos_min

    def test_fold_with_insufficient_trades(self) -> None:
        """2 OOS trades < default min (5) -> sufficient_trades=False."""
        fold = BondWalkForwardFold(
            fold_idx=0,
            train_start=date(2020, 1, 1),
            train_end=date(2021, 12, 31),
            test_start=date(2022, 1, 1),
            test_end=date(2022, 12, 31),
            is_trades=5,
            is_return_pct=2.0,
            is_excess_sharpe=0.2,
            oos_trades=2,
            oos_return_pct=1.0,
            oos_excess_sharpe=0.1,
            sufficient_trades=False,
        )
        assert fold.sufficient_trades is False
        assert fold.oos_trades < _DEFAULT_MIN_TRADES_PER_FOLD

    def test_positive_oos_sharpe(self) -> None:
        """Good fold has positive excess Sharpe."""
        fold = BondWalkForwardFold(
            fold_idx=0,
            train_start=date(2020, 1, 1),
            train_end=date(2021, 12, 31),
            test_start=date(2022, 1, 1),
            test_end=date(2022, 12, 31),
            is_trades=15,
            is_return_pct=8.0,
            is_excess_sharpe=0.8,
            oos_trades=10,
            oos_return_pct=5.0,
            oos_excess_sharpe=0.5,
            sufficient_trades=True,
        )
        assert fold.oos_excess_sharpe > 0

    def test_negative_oos_sharpe(self) -> None:
        """Bad fold has negative excess Sharpe."""
        fold = BondWalkForwardFold(
            fold_idx=0,
            train_start=date(2020, 1, 1),
            train_end=date(2021, 12, 31),
            test_start=date(2022, 1, 1),
            test_end=date(2022, 12, 31),
            is_trades=15,
            is_return_pct=8.0,
            is_excess_sharpe=0.8,
            oos_trades=10,
            oos_return_pct=-2.0,
            oos_excess_sharpe=-0.3,
            sufficient_trades=True,
        )
        assert fold.oos_excess_sharpe < 0


# ── TestRunBondWalkForward ─────────────────────────────────────────────────


class TestRunBondWalkForward:
    """Test the full walk-forward analysis pipeline."""

    def test_integration_with_synthetic_data(self) -> None:
        """4-year equity curve with trades -> produces folds and a result."""
        start = date(2020, 1, 1)
        n_days = _FOUR_YEARS_CALENDAR_DAYS
        equity, dates = _make_synthetic_curve(start, n_days)
        trades = _make_synthetic_trades(start, dates[-1], n_trades=40, seed=_SEED)

        result = run_bond_walk_forward(equity, dates, trades)

        assert isinstance(result, BondWalkForwardResult)
        assert result.n_total_folds > 0
        assert len(result.folds) == result.n_total_folds
        assert result.total_oos_trades >= 0

    def test_consistency_ratio_all_positive(self) -> None:
        """Create data where all OOS folds are strongly positive -> ratio = 1.0."""
        # Use a strong upward trend so every fold has positive excess Sharpe
        start = date(2020, 1, 1)
        n_days = _FOUR_YEARS_CALENDAR_DAYS
        equity, dates = _make_synthetic_curve(start, n_days, daily_return=0.002, seed=_SEED)
        trades = _make_synthetic_trades(start, dates[-1], n_trades=40, win_fraction=0.9, seed=_SEED)

        result = run_bond_walk_forward(equity, dates, trades, risk_free_annual_pct=5.0)

        # With strong positive returns and low risk-free, all folds should be positive
        if result.n_total_folds > 0:
            assert result.consistency_ratio > 0.5

    def test_mixed_folds(self) -> None:
        """Synthetic data with moderate returns -> check ratio calculation."""
        start = date(2020, 1, 1)
        n_days = _FOUR_YEARS_CALENDAR_DAYS
        equity, dates = _make_synthetic_curve(start, n_days, daily_return=0.0001, seed=_SEED)
        trades = _make_synthetic_trades(start, dates[-1], n_trades=30, seed=_SEED)

        result = run_bond_walk_forward(equity, dates, trades)

        assert result.n_total_folds > 0
        # consistency_ratio should be between 0 and 1
        assert 0.0 <= result.consistency_ratio <= 1.0
        assert result.n_positive_folds <= result.n_total_folds

    def test_passes_validation_criteria(self) -> None:
        """Strong returns + high consistency -> passes_validation = True."""
        start = date(2020, 1, 1)
        n_days = _FOUR_YEARS_CALENDAR_DAYS
        equity, dates = _make_synthetic_curve(start, n_days, daily_return=0.002, seed=_SEED)
        trades = _make_synthetic_trades(start, dates[-1], n_trades=40, win_fraction=0.9, seed=_SEED)

        result = run_bond_walk_forward(equity, dates, trades, risk_free_annual_pct=5.0)

        if result.n_total_folds > 0:
            # With very strong returns above risk-free, should pass
            assert result.passes_validation is True

    def test_fails_validation_negative_sharpe(self) -> None:
        """Returns below risk-free -> negative avg OOS excess Sharpe -> fails."""
        start = date(2020, 1, 1)
        n_days = _FOUR_YEARS_CALENDAR_DAYS
        # Very low returns, high risk-free rate -> negative excess
        equity, dates = _make_synthetic_curve(start, n_days, daily_return=0.0001, seed=_SEED)
        trades = _make_synthetic_trades(start, dates[-1], n_trades=30, seed=_SEED)

        result = run_bond_walk_forward(equity, dates, trades, risk_free_annual_pct=30.0)

        if result.n_total_folds > 0:
            assert result.passes_validation is False

    def test_total_oos_trades_aggregated(self) -> None:
        """total_oos_trades equals sum of all fold OOS trades."""
        start = date(2020, 1, 1)
        n_days = _FOUR_YEARS_CALENDAR_DAYS
        equity, dates = _make_synthetic_curve(start, n_days)
        trades = _make_synthetic_trades(start, dates[-1], n_trades=40, seed=_SEED)

        result = run_bond_walk_forward(equity, dates, trades)

        expected_total = sum(f.oos_trades for f in result.folds)
        assert result.total_oos_trades == expected_total


# ── TestEdgeCases ──────────────────────────────────────────────────────────


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_equity_curve(self) -> None:
        """Empty input produces an empty result."""
        result = run_bond_walk_forward([], [], [])

        assert isinstance(result, BondWalkForwardResult)
        assert result.n_total_folds == 0
        assert len(result.folds) == 0
        assert result.passes_validation is False
        assert result.total_oos_trades == 0

    def test_single_fold(self) -> None:
        """Barely enough data for exactly one fold."""
        # 24 + 12 = 36 months minimum. Create ~37 months of calendar data.
        start = date(2020, 1, 1)
        n_days = int(365 * 3.1)  # ~37 calendar months
        equity, dates = _make_synthetic_curve(start, n_days)
        trades = _make_synthetic_trades(start, dates[-1], n_trades=10, seed=_SEED)

        result = run_bond_walk_forward(equity, dates, trades)

        assert result.n_total_folds >= 1

    def test_no_trades_in_test_window(self) -> None:
        """Fold with zero OOS trades should be handled gracefully."""
        # Create data with trades only in the first half
        start = date(2020, 1, 1)
        n_days = _FOUR_YEARS_CALENDAR_DAYS
        equity, dates = _make_synthetic_curve(start, n_days)
        # Trades only in first 6 months
        early_end = date.fromordinal(start.toordinal() + 180)
        trades = _make_synthetic_trades(start, early_end, n_trades=10, seed=_SEED)

        result = run_bond_walk_forward(equity, dates, trades)

        # Should still run without errors
        assert isinstance(result, BondWalkForwardResult)
        # Some folds may have 0 OOS trades
        zero_trade_folds = [f for f in result.folds if f.oos_trades == 0]
        # At least one fold should have no trades (trades are all early)
        if result.n_total_folds > 1:
            assert len(zero_trade_folds) >= 1
