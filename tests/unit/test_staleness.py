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

    def test_default_threshold_is_030(self) -> None:
        """E3: Default threshold lowered from 0.5 to 0.3."""
        detector = StalenessDetector()
        assert detector._threshold == 0.3  # noqa: PLR2004


class TestPerFeatureDrift:
    """E3: Per-feature drift tracking."""

    def test_top_drifting_features_identifies_shifted(self) -> None:
        """Feature that shifts should appear in top drifting list."""
        rng = np.random.default_rng(SEED)
        detector = StalenessDetector(min_samples=MIN_SAMPLES)

        # Train: feature_a stable, feature_b stable
        detector.set_feature_training({
            "feature_a": rng.standard_normal(N_SAMPLES).tolist(),
            "feature_b": rng.standard_normal(N_SAMPLES).tolist(),
        })

        # Recent: feature_a shifts, feature_b stays
        for _ in range(N_RECENT):
            detector.update_features({
                "feature_a": float(rng.standard_normal() + SHIFTED_MEAN),
                "feature_b": float(rng.standard_normal()),
            })

        top = detector.get_top_drifting_features(n=2)
        assert len(top) > 0
        # feature_a should have higher drift than feature_b
        assert top[0][0] == "feature_a"
        assert top[0][1] > top[1][1]

    def test_top_drifting_insufficient_data(self) -> None:
        """With too few recent values, returns empty."""
        detector = StalenessDetector(min_samples=MIN_SAMPLES)
        detector.set_feature_training({"f1": [1.0, 2.0, 3.0]})
        detector.update_features({"f1": 1.0})
        assert detector.get_top_drifting_features() == []

    def test_top_drifting_respects_n(self) -> None:
        """Returns at most n features."""
        rng = np.random.default_rng(SEED)
        detector = StalenessDetector(min_samples=10)
        features = {f"f{i}": rng.standard_normal(N_SAMPLES).tolist() for i in range(5)}
        detector.set_feature_training(features)
        for _ in range(100):
            detector.update_features({f"f{i}": float(rng.standard_normal()) for i in range(5)})
        top = detector.get_top_drifting_features(n=2)
        assert len(top) <= 2  # noqa: PLR2004


class TestOutputDistributionKL:
    """E3: Output-distribution KL tracking."""

    def test_output_kl_none_without_training(self) -> None:
        """Returns None when training output not set."""
        detector = StalenessDetector()
        assert detector.get_output_kl_score() is None

    def test_output_kl_none_insufficient_recent(self) -> None:
        """Returns None when fewer than min_samples predictions."""
        detector = StalenessDetector(min_samples=MIN_SAMPLES)
        detector.set_output_training([0.5] * N_SAMPLES)
        detector.update_output(0.7)
        assert detector.get_output_kl_score() is None

    def test_output_kl_detects_shift(self) -> None:
        """KL > 0 when output distribution shifts."""
        rng = np.random.default_rng(SEED)
        detector = StalenessDetector(min_samples=MIN_SAMPLES)
        train_outputs = rng.uniform(0.3, 0.7, N_SAMPLES).tolist()
        detector.set_output_training(train_outputs)

        # Feed biased predictions
        for _ in range(N_RECENT):
            detector.update_output(float(rng.uniform(0.7, 0.9)))

        score = detector.get_output_kl_score()
        assert score is not None
        assert score > 0.0


class TestRollingBrierScore:
    """E3: Rolling 60-day Brier score tracker."""

    def test_rolling_brier_none_insufficient(self) -> None:
        """Returns None with too few observations."""
        detector = StalenessDetector(min_samples=50)
        detector.update_brier(0.7, 1)
        assert detector.get_rolling_brier() is None

    def test_rolling_brier_perfect_predictions(self) -> None:
        """Perfect predictions give Brier score ~ 0."""
        detector = StalenessDetector(min_samples=10)
        for _ in range(60):
            detector.update_brier(1.0, 1)
            detector.update_brier(0.0, 0)
        brier = detector.get_rolling_brier()
        assert brier is not None
        assert brier < 0.01  # noqa: PLR2004

    def test_rolling_brier_random_predictions(self) -> None:
        """Random 0.5 predictions give Brier score = 0.25."""
        detector = StalenessDetector(min_samples=10)
        for _ in range(60):
            detector.update_brier(0.5, 1)
            detector.update_brier(0.5, 0)
        brier = detector.get_rolling_brier()
        assert brier is not None
        assert abs(brier - 0.25) < 0.01  # noqa: PLR2004

    def test_rolling_brier_bad_predictions(self) -> None:
        """Inverted predictions give Brier score close to 1.0."""
        detector = StalenessDetector(min_samples=10)
        for _ in range(60):
            detector.update_brier(0.0, 1)
            detector.update_brier(1.0, 0)
        brier = detector.get_rolling_brier()
        assert brier is not None
        assert brier > 0.90  # noqa: PLR2004
