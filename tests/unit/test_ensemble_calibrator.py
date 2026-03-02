"""Tests for EnsembleCalibrator."""

from __future__ import annotations

import numpy as np

from finalayze.ml.calibration import EnsembleCalibrator

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_MIN_SAMPLES = 50
_LARGE_SAMPLE_SIZE = 200


class TestEnsembleCalibrator:
    """Tests for the ensemble-level Platt scaler."""

    def test_calibrator_fit_and_calibrate(self) -> None:
        """Fitted calibrator transforms probabilities."""
        rng = np.random.default_rng(42)
        # Generate well-separated probabilities
        raw_probas = np.concatenate(
            [
                rng.uniform(0.1, 0.4, _LARGE_SAMPLE_SIZE // 2),
                rng.uniform(0.6, 0.9, _LARGE_SAMPLE_SIZE // 2),
            ]
        )
        labels = np.array([0] * (_LARGE_SAMPLE_SIZE // 2) + [1] * (_LARGE_SAMPLE_SIZE // 2))

        cal = EnsembleCalibrator()
        cal.fit(raw_probas, labels)

        assert cal.is_fitted
        # Low raw proba should produce low calibrated proba
        low_cal = cal.calibrate(0.2)
        high_cal = cal.calibrate(0.8)
        assert low_cal < high_cal
        # Output must be in [0, 1]
        assert 0.0 <= low_cal <= 1.0
        assert 0.0 <= high_cal <= 1.0

    def test_calibrator_unfitted_passthrough(self) -> None:
        """Unfitted calibrator returns raw probability unchanged."""
        cal = EnsembleCalibrator()
        assert not cal.is_fitted
        raw = 0.73
        assert cal.calibrate(raw) == raw

    def test_calibrator_min_samples(self) -> None:
        """Fewer than 50 samples skips fitting."""
        small_size = _MIN_SAMPLES - 1
        raw_probas = np.random.default_rng(42).uniform(0.0, 1.0, small_size)
        labels = np.array([0] * (small_size // 2) + [1] * (small_size - small_size // 2))

        cal = EnsembleCalibrator()
        cal.fit(raw_probas, labels)

        assert not cal.is_fitted
        assert cal.calibrate(0.6) == 0.6  # noqa: PLR2004

    def test_calibrator_single_class(self) -> None:
        """Single class in labels skips fitting."""
        n = 100
        raw_probas = np.random.default_rng(42).uniform(0.0, 1.0, n)
        labels = np.ones(n, dtype=int)  # all class 1

        cal = EnsembleCalibrator()
        cal.fit(raw_probas, labels)

        assert not cal.is_fitted
        assert cal.calibrate(0.5) == 0.5  # noqa: PLR2004

    def test_calibrator_boundary_values(self) -> None:
        """Calibrated output is clamped to [0, 1]."""
        rng = np.random.default_rng(42)
        raw_probas = np.concatenate(
            [
                rng.uniform(0.0, 0.3, _LARGE_SAMPLE_SIZE // 2),
                rng.uniform(0.7, 1.0, _LARGE_SAMPLE_SIZE // 2),
            ]
        )
        labels = np.array([0] * (_LARGE_SAMPLE_SIZE // 2) + [1] * (_LARGE_SAMPLE_SIZE // 2))

        cal = EnsembleCalibrator()
        cal.fit(raw_probas, labels)

        # Test extreme values
        assert 0.0 <= cal.calibrate(0.0) <= 1.0
        assert 0.0 <= cal.calibrate(1.0) <= 1.0
