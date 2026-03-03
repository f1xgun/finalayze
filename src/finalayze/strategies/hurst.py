"""Hurst exponent via Rescaled Range (R/S) analysis (Layer 4)."""

from __future__ import annotations

import math

_MIN_DATA_POINTS = 20
_MIN_REGRESSION_POINTS = 2
_DEFAULT_HURST = 0.5
_EPSILON = 1e-12


def _compute_log_returns(data: list[float]) -> list[float] | None:
    """Compute log returns from price data. Returns None if any price <= 0."""
    log_returns: list[float] = []
    for i in range(1, len(data)):
        if data[i] <= 0 or data[i - 1] <= 0:
            return None
        log_returns.append(math.log(data[i] / data[i - 1]))
    return log_returns or None


def _subseries_rs(log_returns: list[float], size: int) -> float | None:
    """Compute average R/S for all non-overlapping sub-series of given size."""
    n_subseries = len(log_returns) // size
    rs_values: list[float] = []

    for j in range(n_subseries):
        start = j * size
        subseries = log_returns[start : start + size]

        mean = sum(subseries) / size
        variance = sum((x - mean) ** 2 for x in subseries) / size
        std = math.sqrt(variance)

        if std < _EPSILON:
            continue

        # Cumulative deviation from mean (profile)
        running = 0.0
        cumdev: list[float] = []
        for x in subseries:
            running += x - mean
            cumdev.append(running)

        r = max(cumdev) - min(cumdev)
        rs_values.append(r / std)

    if not rs_values:
        return None
    avg_rs = sum(rs_values) / len(rs_values)
    return avg_rs if avg_rs > 0 else None


def _linear_slope(xs: list[float], ys: list[float]) -> float | None:
    """Compute slope of simple linear regression. Returns None if degenerate."""
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    denom = sum((xs[i] - mean_x) ** 2 for i in range(n))
    if abs(denom) < _EPSILON:
        return None
    numer = sum((xs[i] - mean_x) * (ys[i] - mean_y) for i in range(n))
    return numer / denom


def compute_hurst_exponent(closes: list[float], window: int = 252) -> float:
    """Compute the Hurst exponent using R/S (Rescaled Range) analysis.

    Args:
        closes: List of closing prices.
        window: Number of most recent closes to use (default 252, ~1 trading year).

    Returns:
        Hurst exponent in [0, 1]. Returns 0.5 if insufficient data or on failure.
        H > 0.5 indicates trending (persistent) behavior.
        H < 0.5 indicates mean-reverting (anti-persistent) behavior.
        H = 0.5 indicates a random walk.
    """
    data = closes[-window:] if len(closes) > window else closes

    if len(data) < _MIN_DATA_POINTS:
        return _DEFAULT_HURST

    log_returns = _compute_log_returns(data)
    if log_returns is None:
        return _DEFAULT_HURST

    # Generate sub-series sizes: powers of 2 that fit within the data
    n_returns = len(log_returns)
    sizes: list[int] = []
    power = 2
    while power <= n_returns // 2:
        sizes.append(power)
        power *= 2

    if len(sizes) < _MIN_REGRESSION_POINTS:
        return _DEFAULT_HURST

    log_n: list[float] = []
    log_rs: list[float] = []

    for size in sizes:
        avg_rs = _subseries_rs(log_returns, size)
        if avg_rs is not None:
            log_n.append(math.log(size))
            log_rs.append(math.log(avg_rs))

    if len(log_n) < _MIN_REGRESSION_POINTS:
        return _DEFAULT_HURST

    slope = _linear_slope(log_n, log_rs)
    if slope is None:
        return _DEFAULT_HURST

    return max(0.0, min(1.0, slope))
