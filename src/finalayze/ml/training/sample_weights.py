"""Sample uniqueness weighting (Lopez de Prado, AFML Ch. 4).

Reduces overfitting to clustered events (earnings seasons, crises) by
weighting each training sample by its average uniqueness — the inverse of
how many other labels overlap it in time.

Layer 3 — no upward imports.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from datetime import datetime


def compute_sample_uniqueness(
    timestamps: list[datetime],
    label_spans: list[tuple[datetime, datetime]],
) -> np.ndarray:  # type: ignore[type-arg]
    """Compute average uniqueness for each sample.

    For each sample *i*, count how many other samples' label spans overlap
    with sample *i*'s span.  Uniqueness is ``1 / avg_concurrency`` where
    concurrency is the average number of labels active at any point in
    sample *i*'s span.

    Args:
        timestamps: Observation timestamps (unused for overlap logic but kept
            for API symmetry with ``build_windows``).
        label_spans: ``(start, end)`` datetime pairs defining each label's
            temporal extent.

    Returns:
        1-D float64 array of uniqueness values in ``(0, 1]``.  Length equals
        ``len(timestamps)``.

    Raises:
        ValueError: If *timestamps* and *label_spans* have different lengths.
    """
    n = len(timestamps)
    if n != len(label_spans):
        msg = "timestamps and label_spans must have the same length"
        raise ValueError(msg)
    if n == 0:
        return np.array([], dtype=np.float64)

    # Build concurrency matrix: concurrency[i] = number of spans overlapping span i
    # Two spans (s1, e1) and (s2, e2) overlap iff s1 < e2 and s2 < e1
    uniqueness = np.empty(n, dtype=np.float64)
    for i in range(n):
        s_i, e_i = label_spans[i]
        overlap_count = 0
        for j in range(n):
            s_j, e_j = label_spans[j]
            if s_j < e_i and s_i < e_j:
                overlap_count += 1
        # overlap_count includes self, so avg concurrency = overlap_count
        # uniqueness = 1 / concurrency
        uniqueness[i] = 1.0 / overlap_count

    return uniqueness


def compute_sample_weights(
    uniqueness: np.ndarray,  # type: ignore[type-arg]
) -> np.ndarray:  # type: ignore[type-arg]
    """Normalize uniqueness so weights sum to ``len(uniqueness)``.

    This keeps the effective sample count unchanged while redistributing
    importance toward more unique (less overlapping) observations.

    Args:
        uniqueness: 1-D array of per-sample uniqueness values.

    Returns:
        1-D float64 array of weights summing to ``len(uniqueness)``.
    """
    n = len(uniqueness)
    if n == 0:
        return np.array([], dtype=np.float64)
    total = float(np.sum(uniqueness))
    if total == 0.0:
        return np.ones(n, dtype=np.float64)
    return uniqueness * (n / total)


def compute_decay_weights(n_samples: int, decay: float = 0.5) -> np.ndarray:  # type: ignore[type-arg]
    """Exponential decay weights giving more importance to recent samples.

    Weight for sample at index *i* (0 = oldest, n-1 = newest) is
    ``exp(decay * i / (n - 1))`` normalised so the sum equals *n_samples*.

    Args:
        n_samples: Number of samples.
        decay: Decay strength.  Higher values widen the gap between old and
            new sample weights.

    Returns:
        1-D float64 array of length *n_samples* summing to *n_samples*.

    Raises:
        ValueError: If *n_samples* is not positive.
    """
    if n_samples <= 0:
        msg = "n_samples must be positive"
        raise ValueError(msg)
    if n_samples == 1:
        return np.array([1.0], dtype=np.float64)
    indices = np.arange(n_samples, dtype=np.float64)
    raw = np.exp(decay * indices / (n_samples - 1))
    total = float(np.sum(raw))
    return raw * (n_samples / total)
