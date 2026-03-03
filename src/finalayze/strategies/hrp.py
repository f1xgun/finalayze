"""Hierarchical Risk Parity (HRP) weight allocation (Layer 4).

Implements the Lopez de Prado HRP algorithm for dynamic strategy weight allocation:
1. Compute correlation matrix from strategy return series
2. Distance matrix: d = sqrt(0.5 * (1 - corr))
3. Hierarchical clustering via single-linkage
4. Quasi-diagonalization: reorder by dendrogram leaf order
5. Recursive bisection: allocate inversely proportional to cluster variance
"""

from __future__ import annotations

import math

from scipy.cluster.hierarchy import leaves_list, linkage

_MIN_HISTORY = 20
_MIN_STRATEGIES = 2


def compute_hrp_weights(
    returns_matrix: list[list[float]],
    strategy_names: list[str],
) -> dict[str, float]:
    """Compute HRP-based portfolio weights from strategy return series.

    Args:
        returns_matrix: Each inner list is the time series of returns for one strategy.
        strategy_names: Names corresponding to each row of returns_matrix.

    Returns:
        Dict mapping strategy_name to weight (weights sum to 1.0).
        Returns equal weights if fewer than 2 strategies or fewer than 20 time steps.
    """
    n_strategies = len(returns_matrix)

    # Edge cases: return equal weights
    if n_strategies == 0:
        return {}
    if n_strategies == 1:
        return {strategy_names[0]: 1.0}

    # H-1: if rows have unequal length, truncate all to the minimum length
    min_len = min(len(row) for row in returns_matrix)
    if any(len(row) != min_len for row in returns_matrix):
        returns_matrix = [row[:min_len] for row in returns_matrix]

    if len(returns_matrix[0]) < _MIN_HISTORY:
        equal_w = 1.0 / n_strategies
        return dict(zip(strategy_names, [equal_w] * n_strategies, strict=True))

    # Step 1: Compute correlation matrix
    corr = _correlation_matrix(returns_matrix)

    # Step 2: Distance matrix d = sqrt(0.5 * (1 - corr))
    n = len(corr)
    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            dist[i][j] = math.sqrt(0.5 * (1.0 - corr[i][j]))

    # Step 3: Condensed distance for scipy linkage
    condensed = _to_condensed(dist)
    link = linkage(condensed, method="single")

    # Step 4: Quasi-diagonalization — leaf order from dendrogram
    sort_ix = list(leaves_list(link).astype(int))

    # Step 5: Recursive bisection using full covariance matrix
    cov = _covariance_matrix(returns_matrix)
    weights_arr = [1.0] * n_strategies
    _recursive_bisect(weights_arr, sort_ix, cov)

    # Normalize to sum to 1
    total = sum(weights_arr)
    if total > 0:
        weights_arr = [w / total for w in weights_arr]

    return dict(zip(strategy_names, weights_arr, strict=True))


def _correlation_matrix(returns_matrix: list[list[float]]) -> list[list[float]]:
    """Compute Pearson correlation matrix from return series."""
    n = len(returns_matrix)
    means = [sum(r) / len(r) for r in returns_matrix]
    stds = [_std(r, means[i]) for i, r in enumerate(returns_matrix)]
    corr = [[0.0] * n for _ in range(n)]
    for i in range(n):
        corr[i][i] = 1.0
        for j in range(i + 1, n):
            if stds[i] == 0 or stds[j] == 0:
                corr[i][j] = 0.0
                corr[j][i] = 0.0
            else:
                cov = sum(
                    (returns_matrix[i][k] - means[i]) * (returns_matrix[j][k] - means[j])
                    for k in range(len(returns_matrix[i]))
                ) / (len(returns_matrix[i]) - 1)
                c = cov / (stds[i] * stds[j])
                # Clamp to [-1, 1] for numerical stability
                c = max(-1.0, min(1.0, c))
                corr[i][j] = c
                corr[j][i] = c
    return corr


def _std(series: list[float], mean: float) -> float:
    """Compute sample standard deviation (ddof=1)."""
    n = len(series)
    if n < _MIN_STRATEGIES:
        return 0.0
    var = sum((x - mean) ** 2 for x in series) / (n - 1)
    return math.sqrt(var)


def _variance(series: list[float]) -> float:
    """Compute sample variance (ddof=1)."""
    n = len(series)
    if n < _MIN_STRATEGIES:
        return 0.0
    mean = sum(series) / n
    return sum((x - mean) ** 2 for x in series) / (n - 1)


def _covariance_matrix(returns_matrix: list[list[float]]) -> list[list[float]]:
    """Compute sample covariance matrix (ddof=1) from return series."""
    n = len(returns_matrix)
    t = len(returns_matrix[0])
    means = [sum(r) / t for r in returns_matrix]
    cov = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            val = sum(
                (returns_matrix[i][k] - means[i]) * (returns_matrix[j][k] - means[j])
                for k in range(t)
            ) / (t - 1)
            cov[i][j] = val
            cov[j][i] = val
    return cov


def _cluster_variance(cluster: list[int], cov: list[list[float]]) -> float:
    """Compute Lopez de Prado cluster variance: w^T @ Cov_cluster @ w.

    Weights within the cluster are inverse-variance (diagonal of cov), normalised
    so they sum to 1.  This matches the original HRP paper's recursive bisection.
    """
    if len(cluster) == 1:
        return cov[cluster[0]][cluster[0]]

    # Diagonal variances for the cluster members
    diag_vars = [cov[i][i] for i in cluster]

    # Inverse-variance weights, guard against zero variance
    inv_vars = [1.0 / v if v > 0 else 0.0 for v in diag_vars]
    total_inv = sum(inv_vars)
    if total_inv == 0:
        w = [1.0 / len(cluster)] * len(cluster)
    else:
        w = [iv / total_inv for iv in inv_vars]

    # w^T @ Cov_cluster @ w
    result = 0.0
    for a, i in enumerate(cluster):
        for b, j in enumerate(cluster):
            result += w[a] * w[b] * cov[i][j]
    return result


def _to_condensed(dist: list[list[float]]) -> list[float]:
    """Convert a symmetric distance matrix to condensed form for scipy."""
    n = len(dist)
    condensed: list[float] = []
    for i in range(n):
        condensed.extend(dist[i][j] for j in range(i + 1, n))
    return condensed


def _recursive_bisect(
    weights: list[float],
    sort_ix: list[int],
    cov: list[list[float]],
) -> None:
    """Recursive bisection per Lopez de Prado HRP.

    Cluster variance is computed as w^T @ Cov_cluster @ w with inverse-variance
    weights inside each cluster, matching the original algorithm.
    """
    if len(sort_ix) <= 1:
        return

    mid = len(sort_ix) // 2
    left = sort_ix[:mid]
    right = sort_ix[mid:]

    # Proper cluster variance using covariance sub-matrix
    var_left = _cluster_variance(left, cov)
    var_right = _cluster_variance(right, cov)
    total_var = var_left + var_right

    # Allocate inversely proportional to cluster variance
    alpha = 0.5 if total_var == 0 else 1.0 - var_left / total_var

    for i in left:
        weights[i] *= alpha
    for i in right:
        weights[i] *= 1.0 - alpha

    _recursive_bisect(weights, left, cov)
    _recursive_bisect(weights, right, cov)
