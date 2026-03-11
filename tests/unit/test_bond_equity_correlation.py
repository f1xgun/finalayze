"""Tests for bond-equity correlation tracker (Task 5.5).

Validates the BondEquityCorrelationTracker that monitors trailing 60-day
correlation between MOEX bond (OFZ) and equity (IMOEX) returns, triggering
position limit reductions and cash buffer increases during stress periods
when bond-equity correlation rises.
"""

from __future__ import annotations

import math
import random

import pytest

from finalayze.risk.bond_equity_correlation import (
    BondEquityCorrelationTracker,
    CorrelationStatus,
    compute_pearson_correlation,
)

# ---------------------------------------------------------------------------
# Constants (avoid magic numbers — ruff PLR2004)
# ---------------------------------------------------------------------------

_PERFECT_POSITIVE = 1.0
_PERFECT_NEGATIVE = -1.0
_ZERO = 0.0
_ABS_TOL = 1e-9
_LOOSE_TOL = 0.05

_NORMAL_MULTIPLIER = 1.0
_HIGH_MULTIPLIER = 0.75
_CRITICAL_MULTIPLIER = 0.5

_NO_BUFFER = 0.0
_HIGH_BUFFER = 0.05
_CRITICAL_BUFFER = 0.10

_DEFAULT_WINDOW = 60
_MIN_OBSERVATIONS = 20


# ---------------------------------------------------------------------------
# TestPearsonCorrelation
# ---------------------------------------------------------------------------


class TestPearsonCorrelation:
    """Tests for the pure-Python Pearson correlation function."""

    def test_perfect_positive_correlation(self) -> None:
        """Identical series should yield r = 1.0."""
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert compute_pearson_correlation(x, y) == pytest.approx(_PERFECT_POSITIVE, abs=_ABS_TOL)

    def test_perfect_negative_correlation(self) -> None:
        """Perfectly inversely related series should yield r = -1.0."""
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [5.0, 4.0, 3.0, 2.0, 1.0]
        assert compute_pearson_correlation(x, y) == pytest.approx(_PERFECT_NEGATIVE, abs=_ABS_TOL)

    def test_zero_correlation_orthogonal(self) -> None:
        """Orthogonal series should yield r close to 0.0.

        Use a carefully constructed pair where sum of cross-products is zero.
        """
        x = [1.0, -1.0, 1.0, -1.0]
        y = [1.0, 1.0, -1.0, -1.0]
        assert compute_pearson_correlation(x, y) == pytest.approx(_ZERO, abs=_ABS_TOL)

    def test_known_correlation_value(self) -> None:
        """Hand-computed correlation for a small dataset.

        x = [10, 20, 30, 40, 50], y = [12, 24, 28, 42, 48]
        mean_x = 30, mean_y = 30.8
        cov = 180, var_x = 200, var_y = 165.76
        r = 180 / sqrt(200 * 165.76) = 180 / 182.077 = 0.9886
        """
        x = [10.0, 20.0, 30.0, 40.0, 50.0]
        y = [12.0, 24.0, 28.0, 42.0, 48.0]
        expected = 0.9886
        assert compute_pearson_correlation(x, y) == pytest.approx(expected, abs=0.001)

    def test_insufficient_data_returns_zero(self) -> None:
        """Fewer than 2 data points should return 0.0."""
        assert compute_pearson_correlation([], []) == pytest.approx(_ZERO, abs=_ABS_TOL)
        assert compute_pearson_correlation([1.0], [2.0]) == pytest.approx(_ZERO, abs=_ABS_TOL)

    def test_zero_variance_returns_zero(self) -> None:
        """Constant series (zero variance) should return 0.0, not NaN."""
        x = [5.0, 5.0, 5.0, 5.0, 5.0]
        y = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = compute_pearson_correlation(x, y)
        assert result == pytest.approx(_ZERO, abs=_ABS_TOL)
        assert math.isfinite(result)


# ---------------------------------------------------------------------------
# TestCorrelationTracker
# ---------------------------------------------------------------------------


class TestCorrelationTracker:
    """Tests for the BondEquityCorrelationTracker state machine."""

    def test_initial_state_no_observations(self) -> None:
        """No observations: correlation 0.0, not high, normal limits."""
        tracker = BondEquityCorrelationTracker(window=_DEFAULT_WINDOW)
        status = tracker.status()

        assert status.correlation == pytest.approx(_ZERO, abs=_ABS_TOL)
        assert status.is_high is False
        assert status.is_critical is False
        assert status.position_limit_multiplier == pytest.approx(_NORMAL_MULTIPLIER, abs=_ABS_TOL)
        assert status.cash_buffer_pct == pytest.approx(_NO_BUFFER, abs=_ABS_TOL)

    def test_below_threshold_normal_limits(self) -> None:
        """Low correlation keeps normal position limits and zero cash buffer."""
        tracker = BondEquityCorrelationTracker(window=_DEFAULT_WINDOW)
        rng = random.Random(42)  # noqa: S311 — not for crypto

        # Feed uncorrelated noise for enough observations
        for _ in range(_DEFAULT_WINDOW):
            tracker.update(
                bond_return=rng.gauss(0, 0.01),
                equity_return=rng.gauss(0, 0.01),
            )

        status = tracker.status()
        assert status.is_high is False
        assert status.is_critical is False
        assert status.position_limit_multiplier == pytest.approx(_NORMAL_MULTIPLIER, abs=_ABS_TOL)
        assert status.cash_buffer_pct == pytest.approx(_NO_BUFFER, abs=_ABS_TOL)

    def test_above_high_threshold(self) -> None:
        """Correlation > 0.5 triggers high regime: multiplier 0.75, 5% buffer."""
        tracker = BondEquityCorrelationTracker(window=_DEFAULT_WINDOW)

        # Feed moderately correlated returns: noise=0.018 vs signal=0.02 -> corr ~0.64
        rng = random.Random(123)  # noqa: S311
        for _ in range(_DEFAULT_WINDOW):
            base = rng.gauss(0, 0.02)
            tracker.update(
                bond_return=base + rng.gauss(0, 0.018),
                equity_return=base + rng.gauss(0, 0.018),
            )

        status = tracker.status()
        assert status.correlation > 0.5
        assert status.correlation <= 0.7  # High but not critical
        assert status.is_high is True
        assert status.is_critical is False
        assert status.position_limit_multiplier == pytest.approx(_HIGH_MULTIPLIER, abs=_ABS_TOL)
        assert status.cash_buffer_pct == pytest.approx(_HIGH_BUFFER, abs=_ABS_TOL)

    def test_above_critical_threshold(self) -> None:
        """Correlation > 0.7 triggers critical regime: multiplier 0.5, 10% buffer."""
        tracker = BondEquityCorrelationTracker(window=_DEFAULT_WINDOW)

        # Feed near-identical returns to get correlation > 0.7
        rng = random.Random(456)  # noqa: S311
        for _ in range(_DEFAULT_WINDOW):
            base = rng.gauss(0, 0.03)
            # Very small noise -> near-perfect correlation
            tracker.update(
                bond_return=base + rng.gauss(0, 0.001),
                equity_return=base + rng.gauss(0, 0.001),
            )

        status = tracker.status()
        assert status.correlation > 0.7
        assert status.is_critical is True
        assert status.position_limit_multiplier == pytest.approx(_CRITICAL_MULTIPLIER, abs=_ABS_TOL)
        assert status.cash_buffer_pct == pytest.approx(_CRITICAL_BUFFER, abs=_ABS_TOL)

    def test_window_slides_correctly(self) -> None:
        """Old observations are dropped when window is exceeded."""
        window = 30
        tracker = BondEquityCorrelationTracker(window=window)

        # Phase 1: fill with highly correlated data
        rng = random.Random(789)  # noqa: S311
        for _ in range(window):
            base = rng.gauss(0, 0.02)
            tracker.update(bond_return=base, equity_return=base)

        high_corr = tracker.correlation()
        assert high_corr > 0.9  # Near-perfect correlation

        # Phase 2: overwrite with uncorrelated noise
        for _ in range(window):
            tracker.update(
                bond_return=rng.gauss(0, 0.02),
                equity_return=rng.gauss(0, 0.02),
            )

        # Now the window should contain only uncorrelated data
        low_corr = tracker.correlation()
        assert abs(low_corr) < abs(high_corr)

    def test_reset_clears_state(self) -> None:
        """Reset should restore initial state."""
        tracker = BondEquityCorrelationTracker(window=_DEFAULT_WINDOW)

        # Add some data
        for i in range(30):
            tracker.update(bond_return=0.01 * i, equity_return=0.01 * i)

        tracker.reset()
        status = tracker.status()

        assert status.correlation == pytest.approx(_ZERO, abs=_ABS_TOL)
        assert status.is_high is False
        assert status.is_critical is False
        assert status.window_size == 0

    def test_transition_high_to_normal(self) -> None:
        """As stress subsides, correlation should drop and limits should normalize."""
        window = 30
        tracker = BondEquityCorrelationTracker(window=window)

        # Phase 1: stress period with high correlation
        rng = random.Random(101)  # noqa: S311
        for _ in range(window):
            base = rng.gauss(-0.02, 0.01)  # Stress: both declining
            tracker.update(
                bond_return=base + rng.gauss(0, 0.001),
                equity_return=base + rng.gauss(0, 0.001),
            )

        stress_status = tracker.status()
        assert stress_status.is_high is True

        # Phase 2: normal period replaces stress window
        for _ in range(window):
            tracker.update(
                bond_return=rng.gauss(0, 0.01),
                equity_return=rng.gauss(0, 0.01),
            )

        normal_status = tracker.status()
        assert normal_status.is_high is False
        assert normal_status.position_limit_multiplier == pytest.approx(
            _NORMAL_MULTIPLIER, abs=_ABS_TOL
        )

    def test_insufficient_window_returns_zero(self) -> None:
        """Fewer than 20 observations should return 0.0 correlation."""
        tracker = BondEquityCorrelationTracker(window=_DEFAULT_WINDOW)

        # Add 19 observations (below minimum of 20)
        insufficient_count = _MIN_OBSERVATIONS - 1
        for i in range(insufficient_count):
            tracker.update(bond_return=0.01 * i, equity_return=0.01 * i)

        status = tracker.status()
        assert status.correlation == pytest.approx(_ZERO, abs=_ABS_TOL)
        assert status.is_high is False
        assert status.window_size == insufficient_count


# ---------------------------------------------------------------------------
# TestCorrelationScenarios — financial regime tests
# ---------------------------------------------------------------------------


class TestCorrelationScenarios:
    """Scenario tests based on real MOEX bond-equity dynamics."""

    def test_normal_market_low_correlation(self) -> None:
        """Normal market: random uncorrelated returns produce low correlation."""
        tracker = BondEquityCorrelationTracker(window=_DEFAULT_WINDOW)
        rng = random.Random(2026)  # noqa: S311

        for _ in range(_DEFAULT_WINDOW):
            tracker.update(
                bond_return=rng.gauss(0.0002, 0.005),  # Small positive drift, low vol
                equity_return=rng.gauss(0.0005, 0.015),  # Higher vol, higher drift
            )

        status = tracker.status()
        # Random uncorrelated series should have low absolute correlation
        assert abs(status.correlation) < 0.5
        assert status.is_high is False

    def test_stress_period_high_correlation(self) -> None:
        """Stress: both bonds and equities sell off together (sanctions/CBR shock)."""
        tracker = BondEquityCorrelationTracker(window=_DEFAULT_WINDOW)
        rng = random.Random(2022)  # noqa: S311

        for _ in range(_DEFAULT_WINDOW):
            # Common stress factor drives both down
            stress_shock = rng.gauss(-0.01, 0.005)
            tracker.update(
                bond_return=stress_shock + rng.gauss(0, 0.002),
                equity_return=stress_shock * 1.5 + rng.gauss(0, 0.003),  # Equities drop harder
            )

        status = tracker.status()
        assert status.correlation > 0.5
        assert status.is_high is True

    def test_cbr_easing_moderate_correlation(self) -> None:
        """CBR easing: rate cuts benefit both bonds (price up) and equities."""
        tracker = BondEquityCorrelationTracker(window=_DEFAULT_WINDOW)
        rng = random.Random(2024)  # noqa: S311

        for _ in range(_DEFAULT_WINDOW):
            # Easing environment: positive drift in both, partially correlated
            easing_factor = rng.gauss(0.005, 0.003)
            tracker.update(
                bond_return=easing_factor * 0.5 + rng.gauss(0, 0.003),
                equity_return=easing_factor + rng.gauss(0, 0.008),
            )

        status = tracker.status()
        # Should be moderately positive
        assert status.correlation > 0.0

    def test_mixed_regime_negative_correlation(self) -> None:
        """Flight to quality: bonds up while equities fall (negative correlation)."""
        tracker = BondEquityCorrelationTracker(window=_DEFAULT_WINDOW)
        rng = random.Random(2023)  # noqa: S311

        for _ in range(_DEFAULT_WINDOW):
            # Flight to quality: bonds rally as equities sell off
            risk_factor = rng.gauss(0.01, 0.005)
            tracker.update(
                bond_return=risk_factor + rng.gauss(0, 0.002),  # Bonds rally
                equity_return=-risk_factor + rng.gauss(0, 0.003),  # Equities fall
            )

        status = tracker.status()
        assert status.correlation < -0.3
        assert status.is_high is False  # Negative correlation is not "high"


# ---------------------------------------------------------------------------
# TestCorrelationStatusDataclass
# ---------------------------------------------------------------------------


class TestCorrelationStatusDataclass:
    """Verify CorrelationStatus is frozen and correctly constructed."""

    def test_frozen(self) -> None:
        """CorrelationStatus should be immutable."""
        status = CorrelationStatus(
            correlation=0.6,
            is_high=True,
            is_critical=False,
            position_limit_multiplier=0.75,
            cash_buffer_pct=0.05,
            window_size=60,
        )
        with pytest.raises(AttributeError):
            status.correlation = 0.0  # type: ignore[misc]

    def test_fields(self) -> None:
        """All fields should be accessible."""
        status = CorrelationStatus(
            correlation=0.8,
            is_high=True,
            is_critical=True,
            position_limit_multiplier=0.5,
            cash_buffer_pct=0.10,
            window_size=45,
        )
        assert status.correlation == pytest.approx(0.8, abs=_ABS_TOL)
        assert status.is_high is True
        assert status.is_critical is True
        assert status.position_limit_multiplier == pytest.approx(0.5, abs=_ABS_TOL)
        assert status.cash_buffer_pct == pytest.approx(0.10, abs=_ABS_TOL)
        expected_window = 45
        assert status.window_size == expected_window
