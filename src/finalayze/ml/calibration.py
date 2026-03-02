"""Ensemble-level probability calibration (Layer 3).

Provides a single Platt scaler for calibrating ensemble output probabilities.
This replaces per-model calibrators to avoid double-calibration.

NOTE: For now, per-model calibrators in xgboost_model.py and lightgbm_model.py
remain as-is. This module is created for future consolidation.

See docs/plans/2026-03-02-enhanced-improvement-plan.md, task B.5.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

_MIN_SAMPLES = 50
_MIN_CLASSES = 2


class EnsembleCalibrator:
    """Single Platt scaler for ensemble output probabilities.

    Fits a logistic regression on raw ensemble probabilities vs true labels.
    When insufficient data is available (< 50 samples or single class),
    calibration is skipped and raw probabilities are returned unchanged.
    """

    def __init__(self) -> None:
        self._calibrator: LogisticRegression | None = None
        self._fitted: bool = False

    @property
    def is_fitted(self) -> bool:
        """Whether the calibrator has been fitted successfully."""
        return self._fitted

    def fit(self, raw_probas: np.ndarray, labels: np.ndarray) -> None:
        """Fit Platt scaler on ensemble raw probabilities.

        Args:
            raw_probas: 1-D array of raw ensemble probability outputs.
            labels: 1-D array of true binary labels (0/1).

        Skips fitting silently when:
        - fewer than _MIN_SAMPLES samples are provided
        - only one class is present in labels
        """
        if len(labels) < _MIN_SAMPLES:
            return
        if len(set(labels.tolist())) < _MIN_CLASSES:
            return

        self._calibrator = LogisticRegression()
        self._calibrator.fit(raw_probas.reshape(-1, 1), labels)
        self._fitted = True

    def calibrate(self, raw_proba: float) -> float:
        """Calibrate a single raw probability.

        Args:
            raw_proba: Raw ensemble output probability.

        Returns:
            Calibrated probability if fitted, otherwise raw_proba unchanged.
        """
        if not self._fitted or self._calibrator is None:
            return raw_proba
        calibrated = float(self._calibrator.predict_proba(np.array([[raw_proba]]))[0, 1])
        return max(0.0, min(1.0, calibrated))
