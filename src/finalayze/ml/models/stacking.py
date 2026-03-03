"""Stacking ensemble meta-learner using LogisticRegression (Layer 3)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from sklearn.linear_model import LogisticRegression

if TYPE_CHECKING:
    from numpy.typing import NDArray

_MIN_SAMPLES = 10


class StackingEnsemble:
    """Meta-learner that combines sub-model probabilities via LogisticRegression.

    When not fitted, ``predict_proba`` falls back to simple mean averaging
    (matching the original EnsembleModel behavior).
    """

    def __init__(self) -> None:
        self._meta: LogisticRegression | None = None

    @property
    def is_fitted(self) -> bool:
        """Return True if the meta-learner has been trained."""
        return self._meta is not None

    def fit(self, holdout_predictions: list[list[float]], labels: list[int]) -> None:
        """Train the meta-learner on holdout sub-model predictions.

        Parameters
        ----------
        holdout_predictions:
            Each element is a list of probabilities from each sub-model for one sample.
        labels:
            Binary outcome labels (1 = BUY correct, 0 = not).

        Raises
        ------
        ValueError
            If fewer than ``_MIN_SAMPLES`` samples are provided.
        """
        if len(holdout_predictions) < _MIN_SAMPLES:
            n = len(holdout_predictions)
            msg = f"Stacking requires a minimum of {_MIN_SAMPLES} samples, got {n}"
            raise ValueError(msg)

        x: NDArray[np.float64] = np.array(holdout_predictions, dtype=np.float64)
        y: NDArray[np.int64] = np.array(labels, dtype=np.int64)

        meta = LogisticRegression(solver="lbfgs", max_iter=1000)
        meta.fit(x, y)
        self._meta = meta

    def predict_proba(self, model_probs: list[float]) -> float:
        """Return calibrated probability from the meta-learner.

        Falls back to ``mean(model_probs)`` if the meta-learner has not been fitted.
        """
        if self._meta is None:
            return float(np.mean(model_probs))

        x: NDArray[np.float64] = np.array([model_probs], dtype=np.float64)
        # [:, 1] is the probability of the positive class
        return float(self._meta.predict_proba(x)[0, 1])
