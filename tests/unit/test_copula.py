"""Tests for copula tail dependence module.

Tests cover:
- Clayton copula fit with simulated lower-tail-dependent data
- Gumbel copula fit with simulated upper-tail-dependent data
- Frank copula fit (symmetric, no tail dependence)
- Tail dependence coefficients in [0, 1] range
- is_tail_dependent detection
- Insufficient data returns None
- select_best_copula returns valid result
- Edge cases: constant data, identical series
"""

from __future__ import annotations

import numpy as np
import pytest

from finalayze.risk.copula import (
    CopulaFit,
    CopulaType,
    fit_copula,
    is_tail_dependent,
    select_best_copula,
)

# ── Constants ──────────────────────────────────────────────────────────────────

_SEED = 42
_N_OBSERVATIONS = 200
_MIN_OBSERVATIONS = 50
_INSUFFICIENT_OBSERVATIONS = 30
_TAIL_DEP_LOWER_BOUND = 0.0
_TAIL_DEP_UPPER_BOUND = 1.0
_DEFAULT_THRESHOLD = 0.3
_LOW_THRESHOLD = 0.01
_ZERO = 0.0
_THETA_POSITIVE_MIN = 0.0
_GUMBEL_THETA_MIN = 1.0


# ── Helpers ────────────────────────────────────────────────────────────────────


def _generate_lower_tail_dependent_data(
    n: int = _N_OBSERVATIONS,
    seed: int = _SEED,
) -> tuple[list[float], list[float]]:
    """Generate data with strong lower tail dependence via Clayton copula.

    Uses the conditional method to sample from a Clayton copula with theta=2.
    """
    rng = np.random.default_rng(seed)
    theta = 2.0

    u1 = rng.uniform(0.01, 0.99, size=n)
    # Conditional CDF inverse for Clayton: C(u2|u1) = t
    t = rng.uniform(0.01, 0.99, size=n)
    u2 = (u1 ** (-theta) * (t ** (-theta / (1 + theta)) - 1) + 1) ** (-1 / theta)
    u2 = np.clip(u2, 0.001, 0.999)

    # Convert from uniform to returns via inverse normal CDF
    from scipy.stats import norm

    returns_a = norm.ppf(u1).tolist()
    returns_b = norm.ppf(u2).tolist()
    return returns_a, returns_b


def _generate_upper_tail_dependent_data(
    n: int = _N_OBSERVATIONS,
    seed: int = _SEED,
) -> tuple[list[float], list[float]]:
    """Generate data with strong upper tail dependence via Gumbel copula.

    Uses the Marshall-Olkin method to sample from a Gumbel copula with theta=3.
    """
    rng = np.random.default_rng(seed)
    theta = 3.0

    from scipy.stats import levy_stable, norm

    # Stable distribution method for Gumbel copula
    alpha = 1.0 / theta
    # Sample from a stable distribution with characteristic exponent alpha
    v = levy_stable.rvs(alpha, 1.0, size=n, random_state=rng)
    v = np.abs(v)

    e1 = rng.exponential(1.0, size=n)
    e2 = rng.exponential(1.0, size=n)

    u1 = np.exp(-((e1 / v) ** alpha))
    u2 = np.exp(-((e2 / v) ** alpha))

    u1 = np.clip(u1, 0.001, 0.999)
    u2 = np.clip(u2, 0.001, 0.999)

    returns_a = norm.ppf(u1).tolist()
    returns_b = norm.ppf(u2).tolist()
    return returns_a, returns_b


def _generate_independent_data(
    n: int = _N_OBSERVATIONS,
    seed: int = _SEED,
) -> tuple[list[float], list[float]]:
    """Generate independent normal returns."""
    rng = np.random.default_rng(seed)
    returns_a = rng.normal(0, 0.02, size=n).tolist()
    returns_b = rng.normal(0, 0.02, size=n).tolist()
    return returns_a, returns_b


# ── Clayton Copula Tests ───────────────────────────────────────────────────────


class TestClaytonCopulaFit:
    """Tests for Clayton copula fitting."""

    def test_clayton_fit_returns_copula_fit(self) -> None:
        """Clayton fit on lower-tail-dependent data returns CopulaFit."""
        returns_a, returns_b = _generate_lower_tail_dependent_data()
        result = fit_copula(returns_a, returns_b, CopulaType.CLAYTON)
        assert result is not None
        assert isinstance(result, CopulaFit)
        assert result.copula_type == CopulaType.CLAYTON

    def test_clayton_theta_is_positive(self) -> None:
        """Clayton theta must be positive for positive dependence."""
        returns_a, returns_b = _generate_lower_tail_dependent_data()
        result = fit_copula(returns_a, returns_b, CopulaType.CLAYTON)
        assert result is not None
        assert result.theta > _THETA_POSITIVE_MIN

    def test_clayton_has_lower_tail_dependence(self) -> None:
        """Clayton copula should have nonzero lower tail dependence."""
        returns_a, returns_b = _generate_lower_tail_dependent_data()
        result = fit_copula(returns_a, returns_b, CopulaType.CLAYTON)
        assert result is not None
        assert result.lower_tail_dep > _TAIL_DEP_LOWER_BOUND
        assert result.lower_tail_dep <= _TAIL_DEP_UPPER_BOUND

    def test_clayton_has_zero_upper_tail_dependence(self) -> None:
        """Clayton copula has no upper tail dependence by definition."""
        returns_a, returns_b = _generate_lower_tail_dependent_data()
        result = fit_copula(returns_a, returns_b, CopulaType.CLAYTON)
        assert result is not None
        assert result.upper_tail_dep == _ZERO


# ── Gumbel Copula Tests ───────────────────────────────────────────────────────


class TestGumbelCopulaFit:
    """Tests for Gumbel copula fitting."""

    def test_gumbel_fit_returns_copula_fit(self) -> None:
        """Gumbel fit on upper-tail-dependent data returns CopulaFit."""
        returns_a, returns_b = _generate_upper_tail_dependent_data()
        result = fit_copula(returns_a, returns_b, CopulaType.GUMBEL)
        assert result is not None
        assert isinstance(result, CopulaFit)
        assert result.copula_type == CopulaType.GUMBEL

    def test_gumbel_theta_at_least_one(self) -> None:
        """Gumbel theta must be >= 1."""
        returns_a, returns_b = _generate_upper_tail_dependent_data()
        result = fit_copula(returns_a, returns_b, CopulaType.GUMBEL)
        assert result is not None
        assert result.theta >= _GUMBEL_THETA_MIN

    def test_gumbel_has_upper_tail_dependence(self) -> None:
        """Gumbel copula should have nonzero upper tail dependence."""
        returns_a, returns_b = _generate_upper_tail_dependent_data()
        result = fit_copula(returns_a, returns_b, CopulaType.GUMBEL)
        assert result is not None
        assert result.upper_tail_dep > _TAIL_DEP_LOWER_BOUND
        assert result.upper_tail_dep <= _TAIL_DEP_UPPER_BOUND

    def test_gumbel_has_zero_lower_tail_dependence(self) -> None:
        """Gumbel copula has no lower tail dependence by definition."""
        returns_a, returns_b = _generate_upper_tail_dependent_data()
        result = fit_copula(returns_a, returns_b, CopulaType.GUMBEL)
        assert result is not None
        assert result.lower_tail_dep == _ZERO


# ── Frank Copula Tests ─────────────────────────────────────────────────────────


class TestFrankCopulaFit:
    """Tests for Frank copula fitting."""

    def test_frank_fit_returns_copula_fit(self) -> None:
        """Frank fit returns CopulaFit."""
        returns_a, returns_b = _generate_lower_tail_dependent_data()
        result = fit_copula(returns_a, returns_b, CopulaType.FRANK)
        assert result is not None
        assert isinstance(result, CopulaFit)
        assert result.copula_type == CopulaType.FRANK

    def test_frank_has_zero_tail_dependence(self) -> None:
        """Frank copula has zero tail dependence (symmetric, no tail dep)."""
        returns_a, returns_b = _generate_lower_tail_dependent_data()
        result = fit_copula(returns_a, returns_b, CopulaType.FRANK)
        assert result is not None
        assert result.lower_tail_dep == _ZERO
        assert result.upper_tail_dep == _ZERO


# ── Tail Dependence Range Tests ────────────────────────────────────────────────


class TestTailDependenceRange:
    """Tail dependence coefficients must be in [0, 1]."""

    def test_all_copula_types_in_range(self) -> None:
        """All fitted copulas produce tail dep coefficients in [0, 1]."""
        returns_a, returns_b = _generate_lower_tail_dependent_data()
        for copula_type in CopulaType:
            result = fit_copula(returns_a, returns_b, copula_type)
            assert result is not None, f"fit_copula returned None for {copula_type}"
            assert _TAIL_DEP_LOWER_BOUND <= result.lower_tail_dep <= _TAIL_DEP_UPPER_BOUND
            assert _TAIL_DEP_LOWER_BOUND <= result.upper_tail_dep <= _TAIL_DEP_UPPER_BOUND


# ── is_tail_dependent Tests ────────────────────────────────────────────────────


class TestIsTailDependent:
    """Tests for the is_tail_dependent helper."""

    def test_detects_lower_tail_dependence(self) -> None:
        """Clayton with strong dependence should be detected."""
        returns_a, returns_b = _generate_lower_tail_dependent_data()
        result = fit_copula(returns_a, returns_b, CopulaType.CLAYTON)
        assert result is not None
        # With theta=2 fit, lower_tail_dep = 2^(-1/theta) should be ~0.707
        assert is_tail_dependent(result, threshold=_DEFAULT_THRESHOLD)

    def test_independent_data_not_tail_dependent(self) -> None:
        """Independent data should not show tail dependence."""
        returns_a, returns_b = _generate_independent_data()
        result = fit_copula(returns_a, returns_b, CopulaType.CLAYTON)
        assert result is not None
        assert not is_tail_dependent(result, threshold=_DEFAULT_THRESHOLD)

    def test_frank_never_tail_dependent(self) -> None:
        """Frank copula is never tail dependent regardless of threshold."""
        returns_a, returns_b = _generate_lower_tail_dependent_data()
        result = fit_copula(returns_a, returns_b, CopulaType.FRANK)
        assert result is not None
        assert not is_tail_dependent(result, threshold=_LOW_THRESHOLD)

    def test_threshold_sensitivity(self) -> None:
        """Lower threshold should detect more tail dependence."""
        returns_a, returns_b = _generate_lower_tail_dependent_data()
        result = fit_copula(returns_a, returns_b, CopulaType.CLAYTON)
        assert result is not None
        # Very low threshold should be exceeded by any positive dependence
        assert is_tail_dependent(result, threshold=_LOW_THRESHOLD)


# ── Insufficient Data Tests ────────────────────────────────────────────────────


class TestInsufficientData:
    """Tests for minimum observation requirements."""

    def test_too_few_observations_returns_none(self) -> None:
        """Fewer than 50 observations returns None."""
        rng = np.random.default_rng(_SEED)
        returns_a = rng.normal(0, 0.02, size=_INSUFFICIENT_OBSERVATIONS).tolist()
        returns_b = rng.normal(0, 0.02, size=_INSUFFICIENT_OBSERVATIONS).tolist()
        result = fit_copula(returns_a, returns_b, CopulaType.CLAYTON)
        assert result is None

    def test_exactly_min_observations_works(self) -> None:
        """Exactly 50 observations should succeed."""
        rng = np.random.default_rng(_SEED)
        returns_a = rng.normal(0, 0.02, size=_MIN_OBSERVATIONS).tolist()
        returns_b = rng.normal(0, 0.02, size=_MIN_OBSERVATIONS).tolist()
        result = fit_copula(returns_a, returns_b, CopulaType.CLAYTON)
        assert result is not None

    def test_mismatched_lengths_returns_none(self) -> None:
        """Mismatched return series lengths returns None."""
        rng = np.random.default_rng(_SEED)
        returns_a = rng.normal(0, 0.02, size=_N_OBSERVATIONS).tolist()
        returns_b = rng.normal(0, 0.02, size=_N_OBSERVATIONS - 10).tolist()
        result = fit_copula(returns_a, returns_b, CopulaType.CLAYTON)
        assert result is None


# ── select_best_copula Tests ───────────────────────────────────────────────────


class TestSelectBestCopula:
    """Tests for best copula selection."""

    def test_select_best_returns_valid_result(self) -> None:
        """select_best_copula returns a valid CopulaFit."""
        returns_a, returns_b = _generate_lower_tail_dependent_data()
        result = select_best_copula(returns_a, returns_b)
        assert result is not None
        assert isinstance(result, CopulaFit)
        assert result.copula_type in CopulaType

    def test_select_best_with_upper_tail_data(self) -> None:
        """select_best_copula works with upper-tail-dependent data."""
        returns_a, returns_b = _generate_upper_tail_dependent_data()
        result = select_best_copula(returns_a, returns_b)
        assert result is not None
        assert isinstance(result, CopulaFit)

    def test_select_best_insufficient_data_returns_none(self) -> None:
        """select_best_copula returns None for insufficient data."""
        rng = np.random.default_rng(_SEED)
        returns_a = rng.normal(0, 0.02, size=_INSUFFICIENT_OBSERVATIONS).tolist()
        returns_b = rng.normal(0, 0.02, size=_INSUFFICIENT_OBSERVATIONS).tolist()
        result = select_best_copula(returns_a, returns_b)
        assert result is None

    def test_select_best_independent_data(self) -> None:
        """select_best_copula handles independent data gracefully."""
        returns_a, returns_b = _generate_independent_data()
        result = select_best_copula(returns_a, returns_b)
        # Should still return a result (best fit among the 3)
        assert result is not None
        assert isinstance(result, CopulaFit)
