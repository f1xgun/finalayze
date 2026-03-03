"""Tests for Bayesian Online Changepoint Detection (BOCPD).

Covers: stationary series, mean shift, variance shift, reset, convenience
function, and hazard rate sensitivity.
"""

from __future__ import annotations

import numpy as np
import pytest

from finalayze.risk.bocpd import BOCPDDetector, detect_changepoints

# Reproducibility
RNG = np.random.default_rng(42)

# --- constants to avoid magic numbers (ruff PLR2004) ---
SERIES_LEN = 200
SHIFT_POINT = 100
MEAN_BEFORE = 0.0
MEAN_AFTER = 3.0
LOW_VAR = 0.1
HIGH_VAR = 2.0
CP_THRESHOLD = 0.5
STATIONARY_MAX_CP_PROB = 0.3  # allow some noise but no strong signal
DETECTION_WINDOW = 20  # changepoint should be detected within this many steps


class TestBOCPDStationary:
    """No changepoint should be flagged in a stationary series."""

    def test_no_changepoint_in_stationary(self) -> None:
        data = RNG.normal(loc=0.0, scale=1.0, size=SERIES_LEN).tolist()
        detector = BOCPDDetector()

        cp_probs: list[float] = []
        for x in data:
            p = detector.update(x)
            cp_probs.append(p)

        # After the initial warmup (first few points), probabilities should stay low
        assert max(cp_probs[10:]) < STATIONARY_MAX_CP_PROB, (
            f"Stationary series should not trigger high P(cp), got max={max(cp_probs[10:]):.4f}"
        )


class TestBOCPDMeanShift:
    """A clear mean shift should be detected near the shift point."""

    def test_detects_mean_shift(self) -> None:
        before = RNG.normal(loc=MEAN_BEFORE, scale=LOW_VAR, size=SHIFT_POINT)
        after = RNG.normal(loc=MEAN_AFTER, scale=LOW_VAR, size=SERIES_LEN - SHIFT_POINT)
        data = np.concatenate([before, after]).tolist()

        detector = BOCPDDetector()
        cp_probs: list[float] = []
        for x in data:
            p = detector.update(x)
            cp_probs.append(p)

        # There should be a high P(cp) near the shift point
        window_probs = cp_probs[SHIFT_POINT : SHIFT_POINT + DETECTION_WINDOW]
        assert max(window_probs) > CP_THRESHOLD, (
            f"Mean shift at {SHIFT_POINT} not detected, "
            f"max P(cp) in window = {max(window_probs):.4f}"
        )


class TestBOCPDVarianceShift:
    """A variance change should also be detectable."""

    def test_detects_variance_shift(self) -> None:
        before = RNG.normal(loc=0.0, scale=LOW_VAR, size=SHIFT_POINT)
        after = RNG.normal(loc=0.0, scale=HIGH_VAR, size=SERIES_LEN - SHIFT_POINT)
        data = np.concatenate([before, after]).tolist()

        detector = BOCPDDetector()
        cp_probs: list[float] = []
        for x in data:
            p = detector.update(x)
            cp_probs.append(p)

        # Variance shift should trigger detection within a reasonable window
        window_probs = cp_probs[SHIFT_POINT : SHIFT_POINT + DETECTION_WINDOW]
        assert max(window_probs) > CP_THRESHOLD, (
            f"Variance shift at {SHIFT_POINT} not detected, "
            f"max P(cp) in window = {max(window_probs):.4f}"
        )


class TestBOCPDReset:
    """Reset should clear all internal state."""

    def test_reset_clears_state(self) -> None:
        detector = BOCPDDetector()

        # Feed some data
        for x in RNG.normal(size=50).tolist():
            detector.update(x)

        detector.reset()

        # After reset, internal run-length array should be back to initial
        # Feed one observation -- should behave like fresh detector
        p = detector.update(0.0)
        assert isinstance(p, float)
        assert 0.0 <= p <= 1.0


class TestDetectChangepointsConvenience:
    """The convenience function should return indices of detected changepoints."""

    def test_detect_changepoints_convenience(self) -> None:
        before = RNG.normal(loc=0.0, scale=LOW_VAR, size=SHIFT_POINT)
        after = RNG.normal(loc=MEAN_AFTER, scale=LOW_VAR, size=SERIES_LEN - SHIFT_POINT)
        data = np.concatenate([before, after]).tolist()

        cps = detect_changepoints(data, threshold=CP_THRESHOLD)

        assert isinstance(cps, list)
        assert len(cps) >= 1, "Should detect at least one changepoint"

        # At least one detected changepoint should be near the actual shift
        near_shift = [i for i in cps if abs(i - SHIFT_POINT) <= DETECTION_WINDOW]
        assert len(near_shift) >= 1, f"No changepoint detected near index {SHIFT_POINT}, got {cps}"


class TestHazardRateSensitivity:
    """Higher hazard rate should produce more (or equal) changepoint detections."""

    def test_hazard_rate_sensitivity(self) -> None:
        data = RNG.normal(loc=0.0, scale=1.0, size=SERIES_LEN).tolist()

        low_hz = 1 / 500
        high_hz = 1 / 50

        cps_low = detect_changepoints(data, hazard_rate=low_hz, threshold=CP_THRESHOLD)
        cps_high = detect_changepoints(data, hazard_rate=high_hz, threshold=CP_THRESHOLD)

        # Higher hazard rate means more prior belief in changepoints,
        # so at minimum it should not detect fewer
        assert len(cps_high) >= len(cps_low), (
            f"Higher hazard should detect >= changepoints: low={len(cps_low)}, high={len(cps_high)}"
        )
