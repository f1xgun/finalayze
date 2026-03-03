"""Unit tests for wavelet energy features."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import numpy as np
import pytest

from finalayze.core.schemas import Candle
from finalayze.ml.features.technical import _compute_wavelet_features, compute_features

_WAVELET_KEYS = {
    "wavelet_approx_energy",
    "wavelet_detail1_energy",
    "wavelet_detail2_energy",
    "wavelet_detail3_energy",
}


class TestWaveletFeaturesBasic:
    """test_wavelet_features_basic -- returns 4 keys with values in [0, 1]."""

    def test_returns_four_keys(self) -> None:
        rng = np.random.default_rng(42)
        log_returns = list(rng.standard_normal(50))
        result = _compute_wavelet_features(log_returns)
        assert set(result.keys()) == _WAVELET_KEYS

    def test_values_in_zero_one_range(self) -> None:
        rng = np.random.default_rng(42)
        log_returns = list(rng.standard_normal(50))
        result = _compute_wavelet_features(log_returns)
        for key, val in result.items():
            assert 0.0 <= val <= 1.0, f"{key}={val} not in [0, 1]"


class TestWaveletEnergiesSumToOne:
    """test_wavelet_energies_sum_to_one -- normalized energies sum to ~1.0."""

    def test_energies_sum_to_one(self) -> None:
        rng = np.random.default_rng(99)
        log_returns = list(rng.standard_normal(50))
        result = _compute_wavelet_features(log_returns)
        total = sum(result.values())
        assert total == pytest.approx(1.0, abs=1e-6)


class TestWaveletInsufficientData:
    """test_wavelet_insufficient_data -- returns 0.0s for very short series."""

    def test_short_series_returns_zeros(self) -> None:
        result = _compute_wavelet_features([0.01, -0.02])
        assert all(v == 0.0 for v in result.values())

    def test_empty_series_returns_zeros(self) -> None:
        result = _compute_wavelet_features([])
        assert all(v == 0.0 for v in result.values())


class TestWaveletFeaturesInComputeFeatures:
    """test_wavelet_features_in_compute_features -- integrated into main pipeline."""

    def test_compute_features_includes_wavelet_keys(self) -> None:
        rng = np.random.default_rng(42)
        prices = 100.0 + rng.standard_normal(40).cumsum()
        base_date = datetime(2024, 1, 1, tzinfo=UTC)
        candles = [
            Candle(
                symbol="AAPL",
                market_id="us",
                timeframe="1d",
                timestamp=base_date + timedelta(days=i),
                open=Decimal(str(round(float(prices[i]) * 0.999, 2))),
                high=Decimal(str(round(float(prices[i]) * 1.005, 2))),
                low=Decimal(str(round(float(prices[i]) * 0.995, 2))),
                close=Decimal(str(round(float(prices[i]), 2))),
                volume=int(1000 + rng.integers(0, 500)),
            )
            for i in range(40)
        ]
        features = compute_features(candles)
        assert _WAVELET_KEYS.issubset(set(features.keys()))

    def test_wavelet_values_are_finite_floats(self) -> None:
        import math

        rng = np.random.default_rng(42)
        prices = 100.0 + rng.standard_normal(40).cumsum()
        base_date = datetime(2024, 1, 1, tzinfo=UTC)
        candles = [
            Candle(
                symbol="AAPL",
                market_id="us",
                timeframe="1d",
                timestamp=base_date + timedelta(days=i),
                open=Decimal(str(round(float(prices[i]) * 0.999, 2))),
                high=Decimal(str(round(float(prices[i]) * 1.005, 2))),
                low=Decimal(str(round(float(prices[i]) * 0.995, 2))),
                close=Decimal(str(round(float(prices[i]), 2))),
                volume=int(1000 + rng.integers(0, 500)),
            )
            for i in range(40)
        ]
        features = compute_features(candles)
        for key in _WAVELET_KEYS:
            assert isinstance(features[key], float)
            assert math.isfinite(features[key])
