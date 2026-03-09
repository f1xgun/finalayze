"""Tests for trend-scanning label generation (López de Prado 2020)."""

from __future__ import annotations

import numpy as np
import pytest


class TestTrendScanning:
    """Trend scanning should detect the most significant local trend."""

    def test_uptrend_labeled_positive(self) -> None:
        """Clear uptrend should produce label=1 with high t-value."""
        from finalayze.ml.training.trend_scanning import trend_scan_labels

        prices = np.array([100 + i * 0.5 for i in range(50)], dtype=float)
        labels, t_values = trend_scan_labels(prices, max_horizon=20)

        valid = ~np.isnan(labels)
        assert valid.sum() > 0, "Expected some valid labels"
        assert (labels[valid] == 1).mean() > 0.8
        assert np.nanmean(np.abs(t_values[valid])) > 2.0

    def test_downtrend_labeled_negative(self) -> None:
        """Clear downtrend should produce label=0."""
        from finalayze.ml.training.trend_scanning import trend_scan_labels

        prices = np.array([200 - i * 0.5 for i in range(50)], dtype=float)
        labels, _t_values = trend_scan_labels(prices, max_horizon=20)

        valid = ~np.isnan(labels)
        assert valid.sum() > 0, "Expected some valid labels"
        assert (labels[valid] == 0).mean() > 0.8

    def test_output_shapes(self) -> None:
        """Labels and t-values should match input length."""
        from finalayze.ml.training.trend_scanning import trend_scan_labels

        prices = np.linspace(100, 120, 100)
        labels, t_values = trend_scan_labels(prices, max_horizon=20)

        assert len(labels) == 100
        assert len(t_values) == 100

    def test_tail_is_nan(self) -> None:
        """Last min_horizon-1 bars should have NaN (insufficient lookahead)."""
        from finalayze.ml.training.trend_scanning import trend_scan_labels

        prices = np.linspace(100, 120, 50)
        labels, _t_values = trend_scan_labels(prices, max_horizon=20, min_horizon=3)

        # Last 2 bars (min_horizon-1) should be NaN
        assert np.isnan(labels[-2])
        assert np.isnan(labels[-1])

    def test_noise_has_lower_t_values_than_trend(self) -> None:
        """Random walk t-values should be lower than strong trend t-values."""
        from finalayze.ml.training.trend_scanning import trend_scan_labels

        rng = np.random.default_rng(42)
        noise_prices = 100 + rng.standard_normal(200).cumsum() * 0.01
        trend_prices = np.array([100 + i * 1.0 for i in range(200)], dtype=float)

        _, t_noise = trend_scan_labels(noise_prices, max_horizon=20)
        _, t_trend = trend_scan_labels(trend_prices, max_horizon=20)

        assert np.nanmean(t_noise) < np.nanmean(t_trend)

    def test_reproducible(self) -> None:
        """Same input should produce same output (deterministic OLS)."""
        from finalayze.ml.training.trend_scanning import trend_scan_labels

        prices = np.linspace(100, 150, 80)
        l1, t1 = trend_scan_labels(prices)
        l2, t2 = trend_scan_labels(prices)
        np.testing.assert_array_equal(l1, l2)
        np.testing.assert_array_equal(t1, t2)

    def test_min_horizon_respected(self) -> None:
        """min_horizon < _MIN_REGRESSION_PTS should be capped at _MIN_REGRESSION_PTS."""
        from finalayze.ml.training.trend_scanning import trend_scan_labels

        prices = np.linspace(100, 120, 30)
        labels, _ = trend_scan_labels(prices, max_horizon=10, min_horizon=1)
        # Should still work (min capped internally to _MIN_REGRESSION_PTS=3)
        assert len(labels) == 30

    def test_non_positive_prices_raises(self) -> None:
        """Prices with zeros or negatives should raise ValueError."""
        from finalayze.ml.training.trend_scanning import trend_scan_labels

        prices = np.array([100, 50, 0, 10, 20], dtype=float)
        with pytest.raises(ValueError, match="positive"):
            trend_scan_labels(prices)

    def test_short_array_all_nan(self) -> None:
        """Array shorter than min_horizon should produce all NaN."""
        from finalayze.ml.training.trend_scanning import trend_scan_labels

        prices = np.array([100.0, 101.0])
        labels, t_values = trend_scan_labels(prices, min_horizon=3)
        assert np.all(np.isnan(labels))
        assert np.all(np.isnan(t_values))

    def test_v_shaped_prices(self) -> None:
        """V-shaped prices should produce mixed labels — down then up."""
        from finalayze.ml.training.trend_scanning import trend_scan_labels

        down = np.linspace(200, 100, 30)
        up = np.linspace(100, 200, 30)
        prices = np.concatenate([down, up])

        labels, _ = trend_scan_labels(prices, max_horizon=10, min_horizon=3)
        valid = ~np.isnan(labels)
        # First half should be mostly 0 (downtrend)
        first_half_valid = valid[:30]
        if first_half_valid.sum() > 0:
            assert (labels[:30][first_half_valid] == 0).mean() > 0.5
        # Second half should be mostly 1 (uptrend)
        second_half_valid = valid[30:]
        if second_half_valid.sum() > 0:
            assert (labels[30:][second_half_valid] == 1).mean() > 0.5

    def test_returns_best_horizon(self) -> None:
        """For a strong linear trend, the function should find a significant t-stat."""
        from finalayze.ml.training.trend_scanning import trend_scan_labels

        # Perfect linear trend — t-stat should be very high
        prices = np.array([100 + i * 2.0 for i in range(40)], dtype=float)
        _labels, t_values = trend_scan_labels(prices, max_horizon=15, min_horizon=3)

        valid = ~np.isnan(t_values)
        # Perfect trend should give very high t-values
        assert np.nanmin(t_values[valid]) > 5.0
