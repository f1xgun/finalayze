"""KL divergence staleness detection for ML models (Layer 3).

Detects when the distribution of recent market data has shifted away from the
training data distribution, signalling that a model may be stale and should be
retrained.  Uses KL(train || recent) — the Kullback-Leibler divergence from
the training distribution to the recent data distribution.

Enhanced (E3): output-distribution KL, per-feature drift, rolling Brier score.
"""

from __future__ import annotations

import threading
from collections import deque
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.stats import entropy

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Small constant added to histogram bins to avoid log(0)
_EPSILON = 1e-10
_DEFAULT_THRESHOLD = 0.3  # lowered from 0.5 (E3)


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
        threshold: float = _DEFAULT_THRESHOLD,
        window_size: int = 252,
        min_samples: int = 50,
    ) -> None:
        self._threshold = threshold
        self._window_size = window_size
        self._min_samples = min_samples
        self._recent_values: deque[float] = deque(maxlen=window_size)
        self._training_values: NDArray[Any] | None = None
        self._lock = threading.Lock()

        # E3: Per-feature drift tracking
        self._feature_training: dict[str, NDArray[Any]] = {}
        self._feature_recent: dict[str, deque[float]] = {}

        # E3: Output-distribution KL tracking
        self._train_output_values: NDArray[Any] | None = None
        self._recent_output_values: deque[float] = deque(maxlen=window_size)

        # E3: Rolling Brier score (60-day window)
        self._brier_window = 60
        self._recent_probas: deque[float] = deque(maxlen=self._brier_window)
        self._recent_actuals: deque[int] = deque(maxlen=self._brier_window)

    def set_training_distribution(self, values: list[float]) -> None:
        """Record the training data distribution."""
        self._training_values = np.array(values, dtype=np.float64)

    def update(self, value: float) -> None:
        """Add a new data point to the recent window."""
        with self._lock:
            self._recent_values.append(value)

    def get_kl_score(self) -> float | None:
        """Return current KL divergence, or ``None`` if insufficient data.

        Returns ``None`` when:
        - Training distribution has not been set, OR
        - Fewer than ``min_samples`` recent values have been collected.
        """
        with self._lock:
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

    # --- E3: Per-feature drift ---

    def set_feature_training(self, features: dict[str, list[float]]) -> None:
        """Record per-feature training distributions."""
        self._feature_training = {k: np.array(v, dtype=np.float64) for k, v in features.items()}
        self._feature_recent = {k: deque(maxlen=self._window_size) for k in features}

    def update_features(self, features: dict[str, float]) -> None:
        """Add a new feature observation to per-feature recent windows."""
        for k, v in features.items():
            if k in self._feature_recent:
                self._feature_recent[k].append(v)

    def get_top_drifting_features(self, n: int = 3) -> list[tuple[str, float]]:
        """Return top-n features by KL divergence (descending).

        Only features with >= min_samples recent values are considered.
        """
        drifts: list[tuple[str, float]] = []
        for name, train_vals in self._feature_training.items():
            recent_deque = self._feature_recent.get(name)
            if recent_deque is None or len(recent_deque) < self._min_samples:
                continue
            recent_arr = np.array(recent_deque, dtype=np.float64)
            kl = compute_kl_divergence(train_vals, recent_arr)
            drifts.append((name, kl))
        drifts.sort(key=lambda x: x[1], reverse=True)
        return drifts[:n]

    # --- E3: Output-distribution KL ---

    def set_output_training(self, values: list[float]) -> None:
        """Record the training output (prediction) distribution."""
        self._train_output_values = np.array(values, dtype=np.float64)

    def update_output(self, value: float) -> None:
        """Add a new model prediction to the recent output window."""
        self._recent_output_values.append(value)

    def get_output_kl_score(self) -> float | None:
        """Return KL divergence of output distributions, or None."""
        if self._train_output_values is None:
            return None
        if len(self._recent_output_values) < self._min_samples:
            return None
        recent = np.array(self._recent_output_values, dtype=np.float64)
        return compute_kl_divergence(self._train_output_values, recent)

    # --- E3: Rolling Brier score ---

    def update_brier(self, predicted_prob: float, actual: int) -> None:
        """Add a prediction-outcome pair for rolling Brier score."""
        self._recent_probas.append(predicted_prob)
        self._recent_actuals.append(actual)

    def get_rolling_brier(self) -> float | None:
        """Return rolling Brier score over the last 60 observations.

        Returns None if fewer than min_samples observations.
        """
        if len(self._recent_probas) < self._min_samples:
            return None
        probas = np.array(self._recent_probas, dtype=np.float64)
        actuals = np.array(self._recent_actuals, dtype=np.float64)
        return float(np.mean((probas - actuals) ** 2))
