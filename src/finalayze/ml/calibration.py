"""Ensemble-level probability calibration (Layer 3).

Provides a single Platt scaler for calibrating ensemble output probabilities,
with isotonic regression as a fallback when Platt over-compresses.
Per-model calibrators have been removed; calibration is now applied only at the
ensemble level to avoid double-calibration.

See docs/plans/2026-03-02-enhanced-improvement-plan.md, task B.5.
"""

from __future__ import annotations

import numpy as np
import structlog
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

_MIN_SAMPLES = 50
_MIN_CLASSES = 2
_MIN_OUTPUT_RANGE = 0.30
_CONFORMAL_THRESHOLD = 0.5

_log = structlog.get_logger()


class ConformalCalibrator:
    """Split conformal prediction for binary classification.

    Uses nonconformity scores to construct prediction sets with
    guaranteed coverage at level (1 - alpha).

    For trading: singleton set {1} -> confident BUY, singleton {0} -> confident SELL,
    multi-set {0,1} -> abstain (deadzone).

    Reference: Kaya et al. (2025) "Conformal Prediction for Reliable Stock Selections"
    """

    def __init__(self, alpha: float = 0.10) -> None:
        self._alpha = alpha
        self._quantile_0: float = 1.0  # nonconformity quantile for class 0
        self._quantile_1: float = 1.0  # nonconformity quantile for class 1
        self._fitted: bool = False

    def fit(self, probs: np.ndarray, labels: np.ndarray) -> None:  # type: ignore[type-arg]
        """Compute nonconformity score quantiles from calibration data.

        Nonconformity score for class k: 1 - P(k|x)
        - For class 0: score = probs[i] (higher prob = less conforming to class 0)
        - For class 1: score = 1 - probs[i] (lower prob = less conforming to class 1)

        Quantile at level (1-alpha)(1+1/n) for finite-sample correction (Vovk).
        """
        mask_0 = labels == 0
        mask_1 = labels == 1

        scores_0 = probs[mask_0]  # nonconformity for class 0
        scores_1 = 1.0 - probs[mask_1]  # nonconformity for class 1

        if len(scores_0) > 0:
            n_0 = len(scores_0)
            level_0 = min(1.0, (1.0 - self._alpha) * (1.0 + 1.0 / n_0))
            self._quantile_0 = float(np.quantile(scores_0, level_0))
        else:
            self._quantile_0 = 1.0  # no class-0 data: include class 0 for any input

        if len(scores_1) > 0:
            n_1 = len(scores_1)
            level_1 = min(1.0, (1.0 - self._alpha) * (1.0 + 1.0 / n_1))
            self._quantile_1 = float(np.quantile(scores_1, level_1))
        else:
            self._quantile_1 = 1.0  # no class-1 data: include class 1 for any input

        self._fitted = True

    def predict_set(self, prob: float) -> set[int]:
        """Return prediction set: classes whose conformity exceeds threshold.

        Returns:
            {0}: confident SELL prediction
            {1}: confident BUY prediction
            {0, 1}: uncertain (abstain / deadzone)
        """
        result: set[int] = set()

        # Class 0: nonconformity score = prob; include if score <= quantile
        if prob <= self._quantile_0:
            result.add(0)

        # Class 1: nonconformity score = 1 - prob; include if score <= quantile
        if (1.0 - prob) <= self._quantile_1:
            result.add(1)

        # Empty set means neither class conforms -- treat as uncertain
        if not result:
            return {0, 1}

        return result

    @property
    def is_fitted(self) -> bool:
        """Whether the calibrator has been fitted."""
        return self._fitted


class EnsembleCalibrator:
    """Probability calibrator for ensemble outputs.

    Primary method: Platt scaling (logistic regression).
    Fallback: isotonic regression when Platt over-compresses output range.

    Fits on raw ensemble probabilities vs true labels.
    When insufficient data is available (< 50 samples or single class),
    calibration is skipped and raw probabilities are returned unchanged.

    After fitting, the output range is measured on calibration data. If the
    Platt scaler range is below ``_MIN_OUTPUT_RANGE`` (0.30), isotonic
    regression is tried as a fallback. If isotonic also compresses, the
    calibrator is flagged as bypassed.
    """

    def __init__(self) -> None:
        self._calibrator: LogisticRegression | None = None
        self._isotonic: IsotonicRegression | None = None
        self._use_isotonic: bool = False
        self._fitted: bool = False
        self.fit_output_range: float = 0.0
        self.calibrator_bypassed: bool = False
        self._conformal: ConformalCalibrator | None = None

    @property
    def is_fitted(self) -> bool:
        """Whether the calibrator has been fitted successfully."""
        return self._fitted

    def fit(self, raw_probas: np.ndarray, labels: np.ndarray) -> None:  # type: ignore[type-arg]
        """Fit calibrator on ensemble raw probabilities.

        First tries Platt scaling (logistic regression). If the calibrated
        output range is below ``_MIN_OUTPUT_RANGE``, falls back to isotonic
        regression. If isotonic also produces a narrow range, the calibrator
        is bypassed entirely.

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

        # Measure Platt output range on calibration data
        calibrated = self._calibrator.predict_proba(raw_probas.reshape(-1, 1))[:, 1]
        platt_range = float(np.max(calibrated) - np.min(calibrated))
        self.fit_output_range = platt_range

        if platt_range >= _MIN_OUTPUT_RANGE:
            self.calibrator_bypassed = False
            self._use_isotonic = False
            return

        # Platt over-compressed -- try isotonic regression as fallback
        _log.info(
            "platt_over_compression_trying_isotonic",
            platt_range=round(platt_range, 4),
            min_required=_MIN_OUTPUT_RANGE,
        )

        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        iso.fit(raw_probas, labels)
        iso_calibrated = iso.predict(raw_probas)
        iso_range = float(np.max(iso_calibrated) - np.min(iso_calibrated))

        if iso_range >= _MIN_OUTPUT_RANGE:
            # Isotonic produces wider range -- use it
            self._isotonic = iso
            self._use_isotonic = True
            self.calibrator_bypassed = False
            self.fit_output_range = iso_range
            _log.info(
                "isotonic_fallback_activated",
                isotonic_range=round(iso_range, 4),
                platt_range=round(platt_range, 4),
            )
        else:
            # Both methods compress -- bypass calibration entirely
            self._use_isotonic = False
            self.calibrator_bypassed = True
            _log.warning(
                "calibrator_over_compression_detected",
                platt_range=round(platt_range, 4),
                isotonic_range=round(iso_range, 4),
                min_required=_MIN_OUTPUT_RANGE,
                action="bypassing_calibrator",
            )

    def calibrate(self, raw_proba: float) -> float:
        """Calibrate a single raw probability.

        Uses isotonic regression when active, otherwise Platt scaling.

        Args:
            raw_proba: Raw ensemble output probability.

        Returns:
            Calibrated probability if fitted, otherwise raw_proba unchanged.
        """
        if not self._fitted:
            return raw_proba

        if self._use_isotonic and self._isotonic is not None:
            calibrated = float(self._isotonic.predict(np.array([raw_proba]))[0])
            return max(0.0, min(1.0, calibrated))

        if self._calibrator is None:
            return raw_proba
        calibrated = float(self._calibrator.predict_proba(np.array([[raw_proba]]))[0, 1])
        return max(0.0, min(1.0, calibrated))

    def predict_proba(self, raw_probas: np.ndarray) -> np.ndarray:  # type: ignore[type-arg]
        """Calibrate an array of raw probabilities.

        Vectorized version of calibrate() for batch evaluation.

        Args:
            raw_probas: 1-D array of raw ensemble probability outputs.

        Returns:
            1-D array of calibrated probabilities. If not fitted, returns input unchanged.
        """
        if not self._fitted:
            return raw_probas.copy()

        if self._use_isotonic and self._isotonic is not None:
            calibrated = self._isotonic.predict(raw_probas)
            return np.clip(calibrated, 0.0, 1.0)  # type: ignore[no-any-return]

        if self._calibrator is None:
            return raw_probas.copy()

        calibrated = self._calibrator.predict_proba(raw_probas.reshape(-1, 1))[:, 1]
        return np.clip(calibrated, 0.0, 1.0)  # type: ignore[no-any-return]

    def get_prediction_set(self, raw_prob: float) -> set[int]:
        """Get conformal prediction set for a raw probability.

        When a ConformalCalibrator is attached, delegates to it.
        Otherwise falls back to simple threshold-based classification.

        Args:
            raw_prob: Raw ensemble output probability.

        Returns:
            Set of predicted classes ({0}, {1}, or {0, 1}).
        """
        if self._conformal is None:
            return {1} if raw_prob > _CONFORMAL_THRESHOLD else {0}
        return self._conformal.predict_set(raw_prob)
