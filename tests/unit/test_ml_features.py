"""Tests for ML feature engineering (6C.1 + 6C.2)."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from finalayze.core.schemas import Candle
from finalayze.ml.features.technical import compute_features

_EXPECTED_MIN_FEATURES = 16


def _make_candles(
    n: int = 50,
    base_price: float = 100.0,
    start_weekday: int = 0,
) -> list[Candle]:
    """Create synthetic candles with controlled price/volume."""
    candles: list[Candle] = []
    # Start on a Monday (weekday=0)
    base_ts = datetime(2025, 1, 6, 10, 0, tzinfo=UTC)  # Monday
    for i in range(n):
        price = base_price + i * 0.5
        ts = base_ts + timedelta(days=i)
        candles.append(
            Candle(
                symbol="TEST",
                market_id="us",
                timeframe="1d",
                timestamp=ts,
                open=Decimal(str(price - 0.5)),
                high=Decimal(str(price + 1.0)),
                low=Decimal(str(price - 1.0)),
                close=Decimal(str(price)),
                volume=1000 + i * 10,
            )
        )
    return candles


class TestFeatureDiversity:
    """6C.1: Verify new features are present and valid."""

    def test_compute_features_returns_all_expected_keys(self) -> None:
        candles = _make_candles(50)
        features = compute_features(candles)
        assert len(features) >= _EXPECTED_MIN_FEATURES

    def test_compute_features_no_nans(self) -> None:
        candles = _make_candles(50)
        features = compute_features(candles)
        for key, val in features.items():
            assert math.isfinite(val), f"Feature {key} is not finite: {val}"

    def test_compute_features_day_of_week_cyclical(self) -> None:
        candles = _make_candles(50)
        features = compute_features(candles)
        assert "dow_sin" in features
        assert "dow_cos" in features
        assert -1.0 <= features["dow_sin"] <= 1.0
        assert -1.0 <= features["dow_cos"] <= 1.0

    def test_compute_features_garman_klass_non_negative(self) -> None:
        candles = _make_candles(50)
        features = compute_features(candles)
        assert "gk_vol_20" in features
        assert features["gk_vol_20"] >= 0.0

    def test_compute_features_minimum_candles_unchanged(self) -> None:
        """30 candles should still work."""
        candles = _make_candles(30)
        features = compute_features(candles)
        assert len(features) >= _EXPECTED_MIN_FEATURES

    def test_compute_features_has_expected_new_keys(self) -> None:
        candles = _make_candles(50)
        features = compute_features(candles)
        expected_keys = [
            "roc_10",
            "willr_14",
            "adx_14",
            "ma_slope_20",
            "hist_vol_20",
            "gk_vol_20",
            "dow_sin",
            "dow_cos",
            "obv_slope_10",
            "rsi_divergence",
        ]
        for key in expected_keys:
            assert key in features, f"Missing feature: {key}"


class TestATRMACDNormalization:
    """6C.2: Verify ATR and MACD are normalized by price."""

    def test_atr_pct_scales_with_price(self) -> None:
        """Two candle sets at different price levels produce similar atr_14_pct."""
        candles_low = _make_candles(50, base_price=20.0)
        candles_high = _make_candles(50, base_price=200.0)
        feat_low = compute_features(candles_low)
        feat_high = compute_features(candles_high)
        # The percentage ATR should be in the same order of magnitude
        assert abs(feat_low["atr_14_pct"] - feat_high["atr_14_pct"]) < 0.1

    def test_macd_hist_pct_scales_with_price(self) -> None:
        """Two candle sets at different price levels produce similar macd_hist_pct."""
        candles_low = _make_candles(50, base_price=20.0)
        candles_high = _make_candles(50, base_price=200.0)
        feat_low = compute_features(candles_low)
        feat_high = compute_features(candles_high)
        # The percentage MACD hist should be in the same order of magnitude
        assert abs(feat_low["macd_hist_pct"] - feat_high["macd_hist_pct"]) < 0.1

    def test_old_feature_names_absent(self) -> None:
        """Renamed features should not have old names."""
        candles = _make_candles(50)
        features = compute_features(candles)
        assert "atr_14" not in features
        assert "macd_hist" not in features
        assert "atr_14_pct" in features
        assert "macd_hist_pct" in features
