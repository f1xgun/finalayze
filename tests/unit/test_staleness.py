"""Tests for KL divergence staleness detection (ml/staleness.py)."""

from __future__ import annotations

import numpy as np

from finalayze.ml.staleness import StalenessDetector, compute_kl_divergence

# ---------------------------------------------------------------------------
# Constants (no magic numbers — ruff PLR2004)
# ---------------------------------------------------------------------------
N_BINS = 50
N_SAMPLES = 500
N_RECENT = 500
MIN_SAMPLES = 100
WINDOW_SIZE = 1000
DEFAULT_THRESHOLD = 2.0
LOW_THRESHOLD = 0.01
HIGH_THRESHOLD = 50.0
SHIFTED_MEAN = 5.0
KL_ZERO_TOLERANCE = 1e-6
SEED = 42


class TestComputeKlDivergence:
    """Unit tests for the standalone compute_kl_divergence function."""

    def test_identical_distributions_return_zero(self) -> None:
        """KL divergence of a distribution with itself should be ~0."""
        rng = np.random.default_rng(SEED)
        data = rng.standard_normal(N_SAMPLES)
        kl = compute_kl_divergence(data, data, n_bins=N_BINS)
        assert kl < KL_ZERO_TOLERANCE

    def test_different_distributions_return_positive(self) -> None:
        """KL divergence between shifted distributions should be > 0."""
        rng = np.random.default_rng(SEED)
        train = rng.standard_normal(N_SAMPLES)
        recent = rng.standard_normal(N_SAMPLES) + SHIFTED_MEAN
        kl = compute_kl_divergence(train, recent, n_bins=N_BINS)
        assert kl > 0.0

    def test_kl_is_non_negative(self) -> None:
        """KL divergence should always be >= 0."""
        rng = np.random.default_rng(SEED)
        a = rng.standard_normal(N_SAMPLES)
        b = rng.uniform(-1, 1, N_SAMPLES)
        kl = compute_kl_divergence(a, b, n_bins=N_BINS)
        assert kl >= 0.0


class TestStalenessDetector:
    """Unit tests for StalenessDetector."""

    def test_insufficient_data_returns_none(self) -> None:
        """get_kl_score returns None when fewer than min_samples collected."""
        detector = StalenessDetector(
            threshold=DEFAULT_THRESHOLD,
            window_size=WINDOW_SIZE,
            min_samples=MIN_SAMPLES,
        )
        rng = np.random.default_rng(SEED)
        detector.set_training_distribution(rng.standard_normal(N_SAMPLES).tolist())
        # Add fewer points than min_samples
        few_points = 10
        for val in rng.standard_normal(few_points):
            detector.update(float(val))
        assert detector.get_kl_score() is None

    def test_not_stale_when_distributions_match(self) -> None:
        """Detector should not flag staleness for same-distribution data."""
        rng = np.random.default_rng(SEED)
        detector = StalenessDetector(
            threshold=DEFAULT_THRESHOLD,
            window_size=WINDOW_SIZE,
            min_samples=MIN_SAMPLES,
        )
        train_data = rng.standard_normal(N_SAMPLES).tolist()
        detector.set_training_distribution(train_data)
        # Feed data from same distribution (enough samples for stable histogram)
        for val in rng.standard_normal(N_RECENT):
            detector.update(float(val))
        assert not detector.is_stale()

    def test_detects_regime_shift(self) -> None:
        """Detector should flag staleness after a large distribution shift."""
        rng = np.random.default_rng(SEED)
        detector = StalenessDetector(
            threshold=DEFAULT_THRESHOLD,
            window_size=WINDOW_SIZE,
            min_samples=MIN_SAMPLES,
        )
        train_data = rng.standard_normal(N_SAMPLES).tolist()
        detector.set_training_distribution(train_data)
        # Feed data from a very different distribution
        for val in rng.standard_normal(N_RECENT) + SHIFTED_MEAN:
            detector.update(float(val))
        assert detector.is_stale()

    def test_threshold_controls_sensitivity(self) -> None:
        """A very high threshold should prevent staleness detection."""
        rng = np.random.default_rng(SEED)
        detector = StalenessDetector(
            threshold=HIGH_THRESHOLD,
            window_size=WINDOW_SIZE,
            min_samples=MIN_SAMPLES,
        )
        train_data = rng.standard_normal(N_SAMPLES).tolist()
        detector.set_training_distribution(train_data)
        for val in rng.standard_normal(N_RECENT) + SHIFTED_MEAN:
            detector.update(float(val))
        assert not detector.is_stale()

    def test_low_threshold_triggers_easily(self) -> None:
        """A very low threshold should trigger even for small shifts."""
        rng = np.random.default_rng(SEED)
        detector = StalenessDetector(
            threshold=LOW_THRESHOLD,
            window_size=WINDOW_SIZE,
            min_samples=MIN_SAMPLES,
        )
        train_data = rng.standard_normal(N_SAMPLES).tolist()
        detector.set_training_distribution(train_data)
        # Even a mild shift (mean=1) should trigger with low threshold
        mild_shift = 1.0
        for val in rng.standard_normal(N_RECENT) + mild_shift:
            detector.update(float(val))
        assert detector.is_stale()

    def test_get_kl_score_returns_float(self) -> None:
        """get_kl_score returns a float when sufficient data exists."""
        rng = np.random.default_rng(SEED)
        detector = StalenessDetector(
            threshold=DEFAULT_THRESHOLD,
            window_size=WINDOW_SIZE,
            min_samples=MIN_SAMPLES,
        )
        detector.set_training_distribution(rng.standard_normal(N_SAMPLES).tolist())
        for val in rng.standard_normal(N_RECENT):
            detector.update(float(val))
        score = detector.get_kl_score()
        assert score is not None
        assert isinstance(score, float)

    def test_is_stale_returns_false_without_training_data(self) -> None:
        """is_stale returns False if training distribution was never set."""
        detector = StalenessDetector()
        rng = np.random.default_rng(SEED)
        for val in rng.standard_normal(MIN_SAMPLES + 1):
            detector.update(float(val))
        assert not detector.is_stale()

    def test_window_size_limits_recent_data(self) -> None:
        """Only the most recent window_size points should be kept."""
        small_window = 10
        detector = StalenessDetector(
            threshold=DEFAULT_THRESHOLD,
            window_size=small_window,
            min_samples=small_window,
        )
        rng = np.random.default_rng(SEED)
        detector.set_training_distribution(rng.standard_normal(N_SAMPLES).tolist())
        # Add more points than window_size
        total_points = small_window * 3
        for val in rng.standard_normal(total_points):
            detector.update(float(val))
        assert len(detector._recent_values) == small_window
