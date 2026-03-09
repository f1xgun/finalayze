"""Trend-scanning labels per López de Prado (2020).

For each observation t, fits OLS from t to t+L for L=min_horizon..max_horizon.
Selects L* with maximum |t-statistic| for slope coefficient.
Label = 1 if slope > 0, else 0. Weight = |t-value|.

Advantages over triple-barrier:
- No barrier width parameter (adapts to local volatility)
- No label bias from market drift
- t-value provides natural sample weight
"""

from __future__ import annotations

import math

import numpy as np

# Minimum number of points for a meaningful OLS regression.
# With fewer than 3 points, the t-statistic is degenerate.
_MIN_REGRESSION_PTS = 3

# Default look-ahead horizons (bars)
_DEFAULT_MAX_HORIZON = 20
_DEFAULT_MIN_HORIZON = 3


def trend_scan_labels(
    prices: np.ndarray,  # type: ignore[type-arg]
    max_horizon: int = _DEFAULT_MAX_HORIZON,
    min_horizon: int = _DEFAULT_MIN_HORIZON,
) -> tuple[np.ndarray, np.ndarray]:  # type: ignore[type-arg]
    """Compute trend-scanning labels and t-values.

    Args:
        prices: 1D array of close prices (positive values).
        max_horizon: Maximum look-ahead bars for regression.
        min_horizon: Minimum look-ahead bars (avoid noise at L=1,2).

    Returns:
        labels: Binary array (1=uptrend, 0=downtrend). NaN for last bars
                where insufficient lookahead remains.
        t_values: Absolute t-statistic of best regression slope. NaN for tail.

    Raises:
        ValueError: If any price is non-positive.
    """
    prices = np.asarray(prices, dtype=np.float64)

    if np.any(prices <= 0):
        msg = "All prices must be positive (log-prices require this)."
        raise ValueError(msg)

    # Enforce minimum regression point constraint
    min_horizon = max(min_horizon, _MIN_REGRESSION_PTS)

    n = len(prices)
    labels = np.full(n, np.nan, dtype=np.float64)
    t_values = np.full(n, np.nan, dtype=np.float64)

    # Pre-compute log-prices once
    log_prices = np.log(prices)

    for t in range(n):
        # Maximum L we can fit: need points t..t+L inclusive, so L <= n-t-1
        max_l = min(max_horizon, n - t - 1)
        if max_l < min_horizon:
            # Not enough lookahead for even the shortest regression
            continue

        best_abs_t = -1.0
        best_slope = 0.0

        for horizon in range(min_horizon, max_l + 1):
            # Points: t, t+1, ..., t+horizon  (horizon+1 points)
            num_pts = horizon + 1

            # x = 0, 1, ..., horizon
            # Closed-form OLS: slope = Σ(x - x̄)(y - ȳ) / Σ(x - x̄)²
            # For x = 0..horizon:
            #   x̄ = horizon / 2
            #   SS_xx = Σ(x - x̄)² = horizon * (horizon + 1) * (horizon + 2) / 12
            #          when x = 0..horizon (derived from sum of squares formula)
            # Actually: Σ_{i=0}^{h} (i - h/2)^2 = h*(h+1)*(2h+1)/6 - h*(h+1)/2 * h/1 + ...
            # Simpler: use numpy for the small window

            y = log_prices[t : t + num_pts]
            x = np.arange(num_pts, dtype=np.float64)

            x_mean = x.mean()
            y_mean = y.mean()

            x_centered = x - x_mean
            y_centered = y - y_mean

            ss_xx = np.dot(x_centered, x_centered)
            ss_xy = np.dot(x_centered, y_centered)

            if ss_xx == 0:
                continue

            slope = ss_xy / ss_xx

            # Residuals
            y_hat = y_mean + slope * x_centered
            residuals = y - y_hat
            sse = np.dot(residuals, residuals)

            # Degrees of freedom: n - 2 (intercept + slope)
            dof = num_pts - 2
            if dof <= 0:
                continue

            mse = sse / dof
            se_slope = math.sqrt(mse / ss_xx) if mse > 0 and ss_xx > 0 else 0.0

            # Perfect fit (residuals = 0) → large finite t-stat; else normal ratio
            abs_t = 1e6 if se_slope <= 0 else abs(slope / se_slope)

            if abs_t > best_abs_t:
                best_abs_t = abs_t
                best_slope = slope

        if best_abs_t >= 0:
            labels[t] = 1.0 if best_slope > 0 else 0.0
            t_values[t] = best_abs_t

    return labels, t_values
