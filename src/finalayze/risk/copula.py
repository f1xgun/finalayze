"""Copula tail dependence estimation (Layer 4).

Fits Clayton, Gumbel, and Frank copulas to asset return pairs to detect
asymmetric tail dependence that Pearson correlation misses. Critical for
MOEX markets where stocks crash together during sanctions events.

Clayton copula captures lower tail dependence (joint crashes).
Gumbel copula captures upper tail dependence (joint booms).
Frank copula is symmetric with zero tail dependence.

Copula parameters are estimated via Kendall's tau inversion, which is
computationally efficient and statistically robust.

Reference:
    Nelsen (2006), "An Introduction to Copulas", Springer.

See docs/architecture/DEPENDENCY_LAYERS.md for layering rules.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

import numpy as np
from scipy import integrate, optimize, stats

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Minimum observations required for a reliable copula fit.
_MIN_OBSERVATIONS = 50

# Numerical stability constants.
_CLIP_LO = 0.001
_CLIP_HI = 0.999
_TAU_ZERO_TOL = 1e-10
_THETA_FLOOR_CLAYTON = 1e-6
_THETA_FLOOR_GUMBEL = 1.0
_FRANK_INTEGRAL_LIMIT = 100
_FRANK_THETA_BOUND = 50.0
_FRANK_ZERO_TOL = 1e-6


class CopulaType(StrEnum):
    """Supported copula families."""

    CLAYTON = "clayton"
    GUMBEL = "gumbel"
    FRANK = "frank"


@dataclass(frozen=True, slots=True)
class CopulaFit:
    """Result of fitting a copula to bivariate returns.

    Attributes:
        copula_type: The copula family used.
        theta: Copula dependence parameter.
        lower_tail_dep: Lower tail dependence coefficient lambda_L in [0, 1].
        upper_tail_dep: Upper tail dependence coefficient lambda_U in [0, 1].
        log_likelihood: Log-likelihood of the fit (for model comparison).
    """

    copula_type: CopulaType
    theta: float
    lower_tail_dep: float
    upper_tail_dep: float
    log_likelihood: float = 0.0


def _returns_to_pseudo_observations(
    returns_a: list[float],
    returns_b: list[float],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Convert returns to pseudo-observations (empirical CDF / uniform margins).

    Uses rank-based transformation: u_i = rank(x_i) / (n + 1) to avoid
    boundary values of 0 and 1.
    """
    n = len(returns_a)
    arr_a = np.array(returns_a)
    arr_b = np.array(returns_b)

    # Rank-based empirical CDF (Weibull plotting positions)
    ranks_a = stats.rankdata(arr_a) / (n + 1)
    ranks_b = stats.rankdata(arr_b) / (n + 1)

    # Clip for numerical safety
    u = np.clip(ranks_a, _CLIP_LO, _CLIP_HI)
    v = np.clip(ranks_b, _CLIP_LO, _CLIP_HI)

    return u, v


def _kendall_tau(returns_a: list[float], returns_b: list[float]) -> float:
    """Compute Kendall's rank correlation tau."""
    tau, _ = stats.kendalltau(returns_a, returns_b)
    if np.isnan(tau):
        return 0.0
    return float(tau)


# ── Clayton copula ─────────────────────────────────────────────────────────────


def _clayton_theta_from_tau(tau: float) -> float:
    """Invert Kendall's tau for Clayton: tau = theta / (theta + 2)."""
    if tau <= _TAU_ZERO_TOL:
        return _THETA_FLOOR_CLAYTON
    theta = 2.0 * tau / (1.0 - tau)
    return max(theta, _THETA_FLOOR_CLAYTON)


def _clayton_pdf(
    u: NDArray[np.float64], v: NDArray[np.float64], theta: float
) -> NDArray[np.float64]:
    """Clayton copula density."""
    # c(u,v) = (1+theta) * (u*v)^(-theta-1) * (u^(-theta) + v^(-theta) - 1)^(-2-1/theta)
    t1 = 1.0 + theta
    t2 = (u * v) ** (-theta - 1.0)
    inner = u ** (-theta) + v ** (-theta) - 1.0
    inner = np.maximum(inner, 1e-300)  # avoid negative from numerical errors
    t3 = inner ** (-2.0 - 1.0 / theta)
    result: NDArray[np.float64] = t1 * t2 * t3
    return result


def _clayton_log_likelihood(
    u: NDArray[np.float64],
    v: NDArray[np.float64],
    theta: float,
) -> float:
    """Compute Clayton copula log-likelihood."""
    if theta <= 0:
        return float(-np.inf)
    pdf_vals = _clayton_pdf(u, v, theta)
    pdf_vals = np.maximum(pdf_vals, 1e-300)
    return float(np.sum(np.log(pdf_vals)))


def _clayton_tail_dep(theta: float) -> tuple[float, float]:
    """Clayton tail dependence: lambda_L = 2^(-1/theta), lambda_U = 0."""
    if theta <= _TAU_ZERO_TOL:
        return 0.0, 0.0
    lower = 2.0 ** (-1.0 / theta)
    return lower, 0.0


# ── Gumbel copula ─────────────────────────────────────────────────────────────


def _gumbel_theta_from_tau(tau: float) -> float:
    """Invert Kendall's tau for Gumbel: tau = 1 - 1/theta."""
    if tau <= _TAU_ZERO_TOL:
        return _THETA_FLOOR_GUMBEL
    theta = 1.0 / (1.0 - tau)
    return max(theta, _THETA_FLOOR_GUMBEL)


def _gumbel_pdf(
    u: NDArray[np.float64], v: NDArray[np.float64], theta: float
) -> NDArray[np.float64]:
    """Gumbel copula density (numerically stable)."""
    log_u = -np.log(u)
    log_v = -np.log(v)

    a = log_u**theta + log_v**theta
    a_inv_theta = a ** (1.0 / theta)

    # C(u,v) = exp(-a^(1/theta))
    c_uv = np.exp(-a_inv_theta)

    # Derivative terms
    t1 = c_uv / (u * v)
    t2 = (log_u * log_v) ** (theta - 1.0)
    t3 = a ** (2.0 / theta - 2.0)
    t4 = a_inv_theta + theta - 1.0

    result: NDArray[np.float64] = t1 * t2 * t3 * t4
    return result


def _gumbel_log_likelihood(
    u: NDArray[np.float64],
    v: NDArray[np.float64],
    theta: float,
) -> float:
    """Compute Gumbel copula log-likelihood."""
    if theta < 1.0:
        return float(-np.inf)
    pdf_vals = _gumbel_pdf(u, v, theta)
    pdf_vals = np.maximum(pdf_vals, 1e-300)
    return float(np.sum(np.log(pdf_vals)))


def _gumbel_tail_dep(theta: float) -> tuple[float, float]:
    """Gumbel tail dependence: lambda_L = 0, lambda_U = 2 - 2^(1/theta)."""
    if theta <= _THETA_FLOOR_GUMBEL:
        return 0.0, 0.0
    upper = 2.0 - 2.0 ** (1.0 / theta)
    return 0.0, upper


# ── Frank copula ───────────────────────────────────────────────────────────────


def _frank_theta_from_tau(tau: float) -> float:
    """Invert Kendall's tau for Frank copula numerically.

    tau = 1 - 4/theta * (1 - D_1(theta)) where D_1 is the first Debye function.
    """
    if abs(tau) < _TAU_ZERO_TOL:
        return _FRANK_ZERO_TOL

    def _debye1(t: float) -> float:
        """First Debye function: D_1(x) = (1/x) * int_0^x t/(exp(t)-1) dt."""
        if abs(t) < _FRANK_ZERO_TOL:
            return 1.0
        quad_result: float
        quad_result, _ = integrate.quad(
            lambda s: s / (np.exp(s) - 1.0) if s > 0 else 1.0,
            0,
            abs(t),
            limit=_FRANK_INTEGRAL_LIMIT,
        )
        return quad_result / abs(t)

    # Solve: tau = 1 - 4/theta * (1 - D_1(theta))
    def _equation(theta_val: float) -> float:
        if abs(theta_val) < _FRANK_ZERO_TOL:
            return -tau
        d1 = _debye1(theta_val)
        return 1.0 - 4.0 / theta_val * (1.0 - d1) - tau

    # Bracket search
    try:
        result = optimize.brentq(_equation, -_FRANK_THETA_BOUND, _FRANK_THETA_BOUND)
        return float(result)
    except ValueError:
        # If bracketing fails, fall back to sign-based estimate
        return 4.0 * tau  # rough approximation


def _frank_pdf(u: NDArray[np.float64], v: NDArray[np.float64], theta: float) -> NDArray[np.float64]:
    """Frank copula density."""
    if abs(theta) < _FRANK_ZERO_TOL:
        # Independence copula: density = 1
        return np.ones_like(u)

    e_t = math.exp(-theta)
    e_tu = np.exp(-theta * u)
    e_tv = np.exp(-theta * v)
    e_tuv = np.exp(-theta * (u + v))

    numer = -theta * (1.0 - e_t) * e_tuv
    denom = ((1.0 - e_t) - (1.0 - e_tu) * (1.0 - e_tv)) ** 2
    denom = np.maximum(denom, 1e-300)
    result: NDArray[np.float64] = numer / denom
    return result


def _frank_log_likelihood(
    u: NDArray[np.float64],
    v: NDArray[np.float64],
    theta: float,
) -> float:
    """Compute Frank copula log-likelihood."""
    pdf_vals = _frank_pdf(u, v, theta)
    pdf_vals = np.maximum(np.abs(pdf_vals), 1e-300)
    return float(np.sum(np.log(pdf_vals)))


def _frank_tail_dep() -> tuple[float, float]:
    """Frank tail dependence: both coefficients are zero."""
    return 0.0, 0.0


# ── Public API ─────────────────────────────────────────────────────────────────


def fit_copula(
    returns_a: list[float],
    returns_b: list[float],
    copula_type: CopulaType = CopulaType.CLAYTON,
) -> CopulaFit | None:
    """Fit a copula to bivariate return series.

    Converts returns to pseudo-observations (uniform margins via empirical CDF),
    then estimates the copula parameter via Kendall's tau inversion.

    Args:
        returns_a: Return series for asset A.
        returns_b: Return series for asset B.
        copula_type: Copula family to fit.

    Returns:
        A :class:`CopulaFit` if successful, ``None`` if data is insufficient
        (fewer than 50 observations or mismatched lengths).
    """
    n_a = len(returns_a)
    n_b = len(returns_b)

    if n_a != n_b or n_a < _MIN_OBSERVATIONS:
        return None

    u, v = _returns_to_pseudo_observations(returns_a, returns_b)
    tau = _kendall_tau(returns_a, returns_b)

    if copula_type == CopulaType.CLAYTON:
        theta = _clayton_theta_from_tau(tau)
        ll = _clayton_log_likelihood(u, v, theta)
        lower, upper = _clayton_tail_dep(theta)
    elif copula_type == CopulaType.GUMBEL:
        theta = _gumbel_theta_from_tau(tau)
        ll = _gumbel_log_likelihood(u, v, theta)
        lower, upper = _gumbel_tail_dep(theta)
    else:  # FRANK
        theta = _frank_theta_from_tau(tau)
        ll = _frank_log_likelihood(u, v, theta)
        lower, upper = _frank_tail_dep()

    return CopulaFit(
        copula_type=copula_type,
        theta=theta,
        lower_tail_dep=lower,
        upper_tail_dep=upper,
        log_likelihood=ll,
    )


def is_tail_dependent(fit: CopulaFit, threshold: float = 0.3) -> bool:
    """Check if tail dependence exceeds the given threshold.

    Returns ``True`` if either lower or upper tail dependence coefficient
    exceeds *threshold*.

    Args:
        fit: A fitted copula result.
        threshold: Tail dependence threshold (default 0.3).

    Returns:
        ``True`` if max(lower_tail_dep, upper_tail_dep) > threshold.
    """
    return max(fit.lower_tail_dep, fit.upper_tail_dep) > threshold


def select_best_copula(
    returns_a: list[float],
    returns_b: list[float],
) -> CopulaFit | None:
    """Fit all three copulas and return the one with the highest log-likelihood.

    Args:
        returns_a: Return series for asset A.
        returns_b: Return series for asset B.

    Returns:
        The best :class:`CopulaFit` by log-likelihood, or ``None`` if data
        is insufficient.
    """
    fits: list[CopulaFit] = []
    for copula_type in CopulaType:
        result = fit_copula(returns_a, returns_b, copula_type)
        if result is not None:
            fits.append(result)

    if not fits:
        return None

    return max(fits, key=lambda f: f.log_likelihood)
