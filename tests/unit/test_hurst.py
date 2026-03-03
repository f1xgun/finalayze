"""Unit tests for Hurst exponent computation."""

from __future__ import annotations

import math

import pytest

from finalayze.strategies.hurst import compute_hurst_exponent

# Constants (no magic numbers — ruff PLR2004)
HURST_TRENDING_THRESHOLD = 0.55
HURST_MEAN_REVERTING_THRESHOLD = 0.45
HURST_RANDOM_WALK = 0.5
HURST_MIN = 0.0
HURST_MAX = 1.0
MIN_DATA_POINTS = 20
TRENDING_SERIES_LENGTH = 300
MR_SERIES_LENGTH = 300
TRENDING_STEP = 1.0
MR_AMPLITUDE = 10.0


class TestComputeHurstExponent:
    def test_trending_series_hurst_above_055(self) -> None:
        """Monotonically increasing prices should yield H > 0.55 (trending)."""
        # Steady uptrend: each price = previous + constant step
        closes = [100.0 + TRENDING_STEP * i for i in range(TRENDING_SERIES_LENGTH)]
        h = compute_hurst_exponent(closes)
        assert h > HURST_TRENDING_THRESHOLD, (
            f"Expected H > {HURST_TRENDING_THRESHOLD} for trending series, got {h:.4f}"
        )

    def test_mean_reverting_series_hurst_below_045(self) -> None:
        """Oscillating prices should yield H < 0.45 (mean-reverting)."""
        # Alternating up/down pattern around a mean
        closes = [100.0 + MR_AMPLITUDE * ((-1) ** i) for i in range(MR_SERIES_LENGTH)]
        h = compute_hurst_exponent(closes)
        assert h < HURST_MEAN_REVERTING_THRESHOLD, (
            f"Expected H < {HURST_MEAN_REVERTING_THRESHOLD} for mean-reverting series, got {h:.4f}"
        )

    def test_insufficient_data_returns_05(self) -> None:
        """Fewer than 20 data points should return 0.5 (random walk assumption)."""
        closes = [100.0 + i for i in range(MIN_DATA_POINTS - 1)]
        h = compute_hurst_exponent(closes)
        assert h == HURST_RANDOM_WALK, (
            f"Expected H = {HURST_RANDOM_WALK} for insufficient data, got {h:.4f}"
        )

    def test_hurst_clamped_to_01(self) -> None:
        """Result must always be in [0.0, 1.0] regardless of input."""
        # Test with various series types
        test_cases: list[list[float]] = [
            # Trending
            [100.0 + i for i in range(TRENDING_SERIES_LENGTH)],
            # Mean-reverting
            [100.0 + MR_AMPLITUDE * ((-1) ** i) for i in range(MR_SERIES_LENGTH)],
            # Constant (degenerate)
            [100.0] * 50,
            # Very short (above min threshold)
            [100.0 + i * 0.1 for i in range(MIN_DATA_POINTS)],
        ]
        for closes in test_cases:
            h = compute_hurst_exponent(closes)
            assert HURST_MIN <= h <= HURST_MAX, (
                f"H must be in [{HURST_MIN}, {HURST_MAX}], got {h:.4f}"
            )
