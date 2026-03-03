"""Tests for sample uniqueness weighting (Lopez de Prado)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from finalayze.ml.training.sample_weights import (
    compute_decay_weights,
    compute_sample_uniqueness,
    compute_sample_weights,
)

# ---------------------------------------------------------------------------
# Constants (no magic numbers — ruff PLR2004)
# ---------------------------------------------------------------------------
_N_SAMPLES = 10
_N_LARGE = 50
_EXPECTED_UNIQUENESS_NO_OVERLAP = 1.0
_DECAY_DEFAULT = 0.5
_DECAY_CUSTOM = 0.8
_SINGLE_SAMPLE = 1
_TOLERANCE = 1e-9


# ---------------------------------------------------------------------------
# compute_sample_uniqueness
# ---------------------------------------------------------------------------
class TestComputeSampleUniqueness:
    """Tests for compute_sample_uniqueness."""

    def test_non_overlapping_samples_all_unique(self) -> None:
        """Non-overlapping label spans should all have uniqueness = 1.0."""
        base = datetime(2024, 1, 1, tzinfo=UTC)
        timestamps = [base + timedelta(days=i) for i in range(_N_SAMPLES)]
        # Each label span covers exactly one day, no overlap
        label_spans = [
            (base + timedelta(days=i), base + timedelta(days=i, hours=12))
            for i in range(_N_SAMPLES)
        ]
        uniqueness = compute_sample_uniqueness(timestamps, label_spans)
        assert len(uniqueness) == _N_SAMPLES
        np.testing.assert_allclose(uniqueness, _EXPECTED_UNIQUENESS_NO_OVERLAP, atol=_TOLERANCE)

    def test_fully_overlapping_samples_lower_uniqueness(self) -> None:
        """Fully overlapping label spans should have uniqueness < 1.0."""
        base = datetime(2024, 1, 1, tzinfo=UTC)
        timestamps = [base + timedelta(hours=i) for i in range(_N_SAMPLES)]
        # All labels span the same full range — maximal overlap
        span_start = base
        span_end = base + timedelta(days=1)
        label_spans = [(span_start, span_end)] * _N_SAMPLES
        uniqueness = compute_sample_uniqueness(timestamps, label_spans)
        assert len(uniqueness) == _N_SAMPLES
        # All should have the same (low) uniqueness
        expected = 1.0 / _N_SAMPLES
        np.testing.assert_allclose(uniqueness, expected, atol=_TOLERANCE)

    def test_partial_overlap(self) -> None:
        """Partially overlapping spans: edge samples more unique than center."""
        n_samples = 5
        base = datetime(2024, 1, 1, tzinfo=UTC)
        timestamps = [base + timedelta(days=i) for i in range(n_samples)]
        # Each span is 2 days wide, so adjacent samples overlap
        label_spans = [
            (base + timedelta(days=i), base + timedelta(days=i + 2)) for i in range(n_samples)
        ]
        uniqueness = compute_sample_uniqueness(timestamps, label_spans)
        assert len(uniqueness) == n_samples
        # Edge samples (first, last) overlap fewer neighbours than center
        assert uniqueness[0] > uniqueness[2]  # edge > center
        assert uniqueness[-1] > uniqueness[2]  # edge > center

    def test_single_sample(self) -> None:
        """Single sample should have uniqueness = 1.0."""
        base = datetime(2024, 1, 1, tzinfo=UTC)
        uniqueness = compute_sample_uniqueness(
            [base],
            [(base, base + timedelta(days=1))],
        )
        assert len(uniqueness) == _SINGLE_SAMPLE
        np.testing.assert_allclose(uniqueness[0], _EXPECTED_UNIQUENESS_NO_OVERLAP)

    def test_empty_input(self) -> None:
        """Empty input should return empty array."""
        uniqueness = compute_sample_uniqueness([], [])
        assert len(uniqueness) == 0

    def test_mismatched_lengths_raises(self) -> None:
        """timestamps and label_spans with different lengths should raise."""
        base = datetime(2024, 1, 1, tzinfo=UTC)
        with pytest.raises(ValueError, match="same length"):
            compute_sample_uniqueness(
                [base, base + timedelta(days=1)],
                [(base, base + timedelta(days=1))],
            )


# ---------------------------------------------------------------------------
# compute_sample_weights
# ---------------------------------------------------------------------------
class TestComputeSampleWeights:
    """Tests for compute_sample_weights (normalization)."""

    def test_weights_sum_equals_n_samples(self) -> None:
        """Normalized weights should sum to n_samples."""
        uniqueness = np.array([1.0, 0.5, 0.25, 0.75, 1.0])
        n = len(uniqueness)
        weights = compute_sample_weights(uniqueness)
        np.testing.assert_allclose(np.sum(weights), n, atol=_TOLERANCE)

    def test_uniform_uniqueness_gives_uniform_weights(self) -> None:
        """All-equal uniqueness should produce all-1.0 weights."""
        uniqueness = np.ones(_N_SAMPLES)
        weights = compute_sample_weights(uniqueness)
        np.testing.assert_allclose(weights, 1.0, atol=_TOLERANCE)

    def test_higher_uniqueness_gets_higher_weight(self) -> None:
        """More unique samples should receive higher weight."""
        uniqueness = np.array([1.0, 0.1])
        weights = compute_sample_weights(uniqueness)
        assert weights[0] > weights[1]

    def test_empty_uniqueness(self) -> None:
        """Empty array should return empty weights."""
        weights = compute_sample_weights(np.array([]))
        assert len(weights) == 0

    def test_single_sample_weight(self) -> None:
        """Single sample: weight should be 1.0."""
        weights = compute_sample_weights(np.array([0.42]))
        np.testing.assert_allclose(weights[0], 1.0, atol=_TOLERANCE)


# ---------------------------------------------------------------------------
# compute_decay_weights
# ---------------------------------------------------------------------------
class TestComputeDecayWeights:
    """Tests for compute_decay_weights (exponential decay)."""

    def test_shape(self) -> None:
        """Output shape should match n_samples."""
        weights = compute_decay_weights(_N_LARGE)
        assert weights.shape == (_N_LARGE,)

    def test_recent_samples_weighted_more(self) -> None:
        """Last element (most recent) should have highest weight."""
        weights = compute_decay_weights(_N_SAMPLES)
        assert weights[-1] > weights[0]

    def test_monotonically_increasing(self) -> None:
        """Weights should be monotonically non-decreasing (older -> newer)."""
        weights = compute_decay_weights(_N_SAMPLES)
        assert np.all(np.diff(weights) >= 0)

    def test_sum_equals_n_samples(self) -> None:
        """Decay weights should also be normalized to sum = n_samples."""
        weights = compute_decay_weights(_N_SAMPLES)
        np.testing.assert_allclose(np.sum(weights), _N_SAMPLES, atol=_TOLERANCE)

    def test_custom_decay(self) -> None:
        """Custom decay rate should still produce valid normalized weights."""
        weights = compute_decay_weights(_N_SAMPLES, decay=_DECAY_CUSTOM)
        np.testing.assert_allclose(np.sum(weights), _N_SAMPLES, atol=_TOLERANCE)
        assert weights[-1] > weights[0]

    def test_single_sample_decay(self) -> None:
        """Single sample should get weight = 1.0."""
        weights = compute_decay_weights(_SINGLE_SAMPLE)
        np.testing.assert_allclose(weights[0], 1.0, atol=_TOLERANCE)

    def test_zero_samples_raises(self) -> None:
        """Zero samples should raise ValueError."""
        with pytest.raises(ValueError, match="positive"):
            compute_decay_weights(0)
