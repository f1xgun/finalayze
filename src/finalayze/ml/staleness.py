"""KL divergence staleness detection for ML models (Layer 3).

Detects when the distribution of recent market data has shifted away from the
training data distribution, signalling that a model may be stale and should be
retrained.  Uses KL(train || recent) — the Kullback-Leibler divergence from
the training distribution to the recent data distribution.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.stats import entropy

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Small constant added to histogram bins to avoid log(0)
_EPSILON = 1e-10


def compute_kl_divergence(
    train_dist: NDArray[Any],
    recent_dist: NDArray[Any],
    n_bins: int = 50,
) -> float:
    """Compute KL divergence between two sample arrays via histograms.

    Both arrays are discretised into the same ``n_bins`` bins (range
    determined by the union of both arrays).  A small epsilon is added
    to every bin count to avoid division-by-zero in the log term.

    Parameters
    ----------
    train_dist:
        1-D array of training-time values.
    recent_dist:
        1-D array of recent (production-time) values.
    n_bins:
        Number of histogram bins.

    Returns
    -------
    float
        KL(train || recent) — non-negative scalar.
    """
    # Compute shared bin edges from the union of both datasets
    combined = np.concatenate([train_dist, recent_dist])
    bin_edges = np.histogram_bin_edges(combined, bins=n_bins)

    train_counts, _ = np.histogram(train_dist, bins=bin_edges)
    recent_counts, _ = np.histogram(recent_dist, bins=bin_edges)

    # Normalise to probability distributions with epsilon smoothing
    train_prob = (train_counts + _EPSILON) / (train_counts + _EPSILON).sum()
    recent_prob = (recent_counts + _EPSILON) / (recent_counts + _EPSILON).sum()

    # scipy.stats.entropy(pk, qk) computes KL(pk || qk)
    kl: float = float(entropy(train_prob, recent_prob))
    return kl


class StalenessDetector:
    """Detect model staleness via KL divergence between train and recent data.

    Parameters
    ----------
    threshold:
        KL divergence value above which the model is considered stale.
    window_size:
        Maximum number of recent data points to keep.
    min_samples:
        Minimum number of recent data points required before computing
        KL divergence.  ``get_kl_score()`` returns ``None`` until this
        many points have been collected.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        window_size: int = 252,
        min_samples: int = 50,
    ) -> None:
        self._threshold = threshold
        self._window_size = window_size
        self._min_samples = min_samples
        self._recent_values: deque[float] = deque(maxlen=window_size)
        self._training_values: NDArray[Any] | None = None

    def set_training_distribution(self, values: list[float]) -> None:
        """Record the training data distribution."""
        self._training_values = np.array(values, dtype=np.float64)

    def update(self, value: float) -> None:
        """Add a new data point to the recent window."""
        self._recent_values.append(value)

    def get_kl_score(self) -> float | None:
        """Return current KL divergence, or ``None`` if insufficient data.

        Returns ``None`` when:
        - Training distribution has not been set, OR
        - Fewer than ``min_samples`` recent values have been collected.
        """
        if self._training_values is None:
            return None
        if len(self._recent_values) < self._min_samples:
            return None
        recent_array = np.array(self._recent_values, dtype=np.float64)
        return compute_kl_divergence(self._training_values, recent_array)

    def is_stale(self) -> bool:
        """Return ``True`` if KL divergence exceeds the threshold."""
        score = self.get_kl_score()
        if score is None:
            return False
        return score > self._threshold
