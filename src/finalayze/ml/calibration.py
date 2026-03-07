"""Ensemble-level probability calibration (Layer 3).

Provides a single Platt scaler for calibrating ensemble output probabilities.
Per-model calibrators have been removed; calibration is now applied only at the
ensemble level to avoid double-calibration.

See docs/plans/2026-03-02-enhanced-improvement-plan.md, task B.5.
"""

from __future__ import annotations

import numpy as np
import structlog
from sklearn.linear_model import LogisticRegression

_MIN_SAMPLES = 50
_MIN_CLASSES = 2
_MIN_OUTPUT_RANGE = 0.30

_log = structlog.get_logger()


class EnsembleCalibrator:
    """Single Platt scaler for ensemble output probabilities.

    Fits a logistic regression on raw ensemble probabilities vs true labels.
    When insufficient data is available (< 50 samples or single class),
    calibration is skipped and raw probabilities are returned unchanged.

    After fitting, the output range is measured on calibration data. If the
    range is below ``_MIN_OUTPUT_RANGE`` (0.30), the calibrator is flagged as
    bypassed to prevent over-compression of ensemble probabilities.
    """

    def __init__(self) -> None:
        self._calibrator: LogisticRegression | None = None
        self._fitted: bool = False
        self.fit_output_range: float = 0.0
        self.calibrator_bypassed: bool = False

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

        After fitting, measures the calibrated output range on the training
        data. If the range is below _MIN_OUTPUT_RANGE, sets
        ``calibrator_bypassed = True`` and logs a warning.
        """
        if len(labels) < _MIN_SAMPLES:
            return
        if len(set(labels.tolist())) < _MIN_CLASSES:
            return

        self._calibrator = LogisticRegression()
        self._calibrator.fit(raw_probas.reshape(-1, 1), labels)
        self._fitted = True

        # Measure output range on calibration data
        calibrated = self._calibrator.predict_proba(raw_probas.reshape(-1, 1))[:, 1]
        self.fit_output_range = float(np.max(calibrated) - np.min(calibrated))

        if self.fit_output_range < _MIN_OUTPUT_RANGE:
            self.calibrator_bypassed = True
            _log.warning(
                "calibrator_over_compression_detected",
                output_range=round(self.fit_output_range, 4),
                min_required=_MIN_OUTPUT_RANGE,
                action="bypassing_calibrator",
            )
        else:
            self.calibrator_bypassed = False

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
