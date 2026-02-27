"""Unit tests for Monte Carlo bootstrap metrics."""

from __future__ import annotations

import math

import pytest

from finalayze.backtest.monte_carlo import BootstrapCI, BootstrapResult, bootstrap_metrics

# ── Constants (no magic numbers) ─────────────────────────────────────────

SEED = 42
N_SIMULATIONS = 5000
N_SIMULATIONS_SMALL = 1000

CONFIDENCE_95 = 0.95
CONFIDENCE_99 = 0.99

# Winning strategy returns: all positive
WINNING_RETURNS = [2.0, 1.5, 3.0, 0.5, 2.5, 1.0, 4.0, 0.8, 1.2, 3.5]

# Losing strategy returns: mostly negative
LOSING_RETURNS = [-3.0, -2.0, 0.5, -4.0, -1.5, -2.5, 0.2, -3.5, -1.0, -2.0]

# Mixed returns for CI width tests
NARROW_RETURNS = [1.0] * 100  # Uniform returns → very narrow CIs
WIDE_RETURNS = [10.0, -10.0] * 25  # High variance → wide CIs

# Larger trade set for CI-width-narrows test
MANY_TRADES = [1.0, -0.5, 2.0, -1.0, 0.5] * 40  # 200 trades
FEW_TRADES = [1.0, -0.5, 2.0, -1.0, 0.5]  # 5 trades


# ── Tests ────────────────────────────────────────────────────────────────


class TestBootstrapMetrics:
    """Tests for the bootstrap_metrics function."""

    def test_bootstrap_zero_trades(self) -> None:
        """Empty trade list returns zero CIs."""
        result = bootstrap_metrics([], n_simulations=N_SIMULATIONS, seed=SEED)

        assert isinstance(result, BootstrapResult)
        assert result.n_trades == 0
        assert result.total_return.point_estimate == 0.0
        assert result.sharpe_ratio.point_estimate == 0.0
        assert result.max_drawdown.point_estimate == 0.0
        assert result.win_rate.point_estimate == 0.0
        assert result.profit_factor.point_estimate == 0.0

    def test_bootstrap_deterministic_with_seed(self) -> None:
        """Same seed produces identical results."""
        result1 = bootstrap_metrics(WINNING_RETURNS, n_simulations=N_SIMULATIONS, seed=SEED)
        result2 = bootstrap_metrics(WINNING_RETURNS, n_simulations=N_SIMULATIONS, seed=SEED)

        assert result1.total_return.lower == result2.total_return.lower
        assert result1.total_return.upper == result2.total_return.upper
        assert result1.sharpe_ratio.lower == result2.sharpe_ratio.lower
        assert result1.sharpe_ratio.upper == result2.sharpe_ratio.upper

    def test_bootstrap_positive_returns(self) -> None:
        """Winning strategy has positive CI bounds for total return and win rate."""
        result = bootstrap_metrics(WINNING_RETURNS, n_simulations=N_SIMULATIONS, seed=SEED)

        # All returns are positive, so total return CI lower bound should be > 0
        assert result.total_return.lower > 0.0
        assert result.total_return.point_estimate > 0.0

        # Win rate should be 100% (all positive)
        expected_win_rate = 100.0
        assert result.win_rate.point_estimate == expected_win_rate

        # Sharpe should be positive
        assert result.sharpe_ratio.point_estimate > 0.0

    def test_bootstrap_ci_width(self) -> None:
        """CIs narrow with more trades (lower variance in bootstrap).

        We compare win_rate CIs because, as a proportion, the bootstrap
        variance decreases with sample size.  Compounded total-return CIs
        can actually widen with more trades due to compounding effects.
        """
        result_few = bootstrap_metrics(FEW_TRADES, n_simulations=N_SIMULATIONS, seed=SEED)
        result_many = bootstrap_metrics(MANY_TRADES, n_simulations=N_SIMULATIONS, seed=SEED)

        few_width = result_few.win_rate.upper - result_few.win_rate.lower
        many_width = result_many.win_rate.upper - result_many.win_rate.lower

        # More trades should yield narrower win-rate CI
        assert many_width < few_width

    def test_bootstrap_losing_strategy(self) -> None:
        """Losing strategy has negative Sharpe lower CI bound."""
        result = bootstrap_metrics(LOSING_RETURNS, n_simulations=N_SIMULATIONS, seed=SEED)

        # Sharpe lower bound should be negative for a losing strategy
        assert result.sharpe_ratio.lower < 0.0

        # Total return should be negative
        assert result.total_return.point_estimate < 0.0

    def test_confidence_level_affects_width(self) -> None:
        """99% CI is wider than 95% CI."""
        result_95 = bootstrap_metrics(
            WINNING_RETURNS,
            n_simulations=N_SIMULATIONS,
            confidence_level=CONFIDENCE_95,
            seed=SEED,
        )
        result_99 = bootstrap_metrics(
            WINNING_RETURNS,
            n_simulations=N_SIMULATIONS,
            confidence_level=CONFIDENCE_99,
            seed=SEED,
        )

        width_95 = result_95.total_return.upper - result_95.total_return.lower
        width_99 = result_99.total_return.upper - result_99.total_return.lower

        assert width_99 > width_95

        # Also check sharpe
        sharpe_width_95 = result_95.sharpe_ratio.upper - result_95.sharpe_ratio.lower
        sharpe_width_99 = result_99.sharpe_ratio.upper - result_99.sharpe_ratio.lower

        assert sharpe_width_99 > sharpe_width_95
