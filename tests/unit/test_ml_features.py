"""Tests for ML feature engineering (6C.1 + 6C.2 + Phase B improvements)."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from finalayze.core.schemas import Candle
from finalayze.ml.features.technical import compute_features

_EXPECTED_MIN_FEATURES = 25
_EXPECTED_TOTAL_FEATURES = 28


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

    def test_compute_features_has_expected_keys(self) -> None:
        candles = _make_candles(50)
        features = compute_features(candles)
        expected_keys = [
            "roc_10",
            "willr_14",
            "adx_14",
            "hist_vol_20",
            "gk_vol_20",
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


class TestMicrostructureFeatures:
    """Tests for proximity_rolling_high, amihud_20d, and corwin_schultz_spread."""

    def test_proximity_rolling_high_present(self) -> None:
        candles = _make_candles(50)
        features = compute_features(candles)
        assert "proximity_rolling_high" in features

    def test_proximity_rolling_high_at_high_is_one(self) -> None:
        """When close is monotonically increasing, last close == rolling max -> 1.0."""
        candles = _make_candles(50)
        features = compute_features(candles)
        # _make_candles produces monotonically increasing close prices,
        # so the last close IS the rolling max
        assert features["proximity_rolling_high"] == pytest.approx(1.0)

    def test_amihud_20d_present_in_features(self) -> None:
        candles = _make_candles(50)
        features = compute_features(candles)
        assert "amihud_20d" in features

    def test_amihud_log_transformed_positive(self) -> None:
        """Amihud should be log-transformed: positive and reasonable for any stock."""
        candles = _make_candles(50)
        features = compute_features(candles)
        # log1p(x * 1e6) for a small illiquidity value should be positive
        assert features["amihud_20d"] >= 0.0
        # Should be a reasonable log-scale value (not raw tiny float)
        # For liquid stocks with volume ~1000, the raw amihud is very small,
        # but log1p(val * 1e6) should produce a value > 0
        assert features["amihud_20d"] > 0.0

    def test_corwin_schultz_present_in_features(self) -> None:
        candles = _make_candles(50)
        features = compute_features(candles)
        assert "corwin_schultz_spread" in features

    def test_corwin_schultz_non_negative(self) -> None:
        candles = _make_candles(50)
        features = compute_features(candles)
        assert features["corwin_schultz_spread"] >= 0.0


class TestNewLaggedReturnFeatures:
    """Phase B: lagged returns, return distribution, short RSI."""

    def test_lagged_returns_present(self) -> None:
        candles = _make_candles(50)
        features = compute_features(candles)
        assert "ret_1d" in features
        assert "ret_5d" in features
        assert "ret_21d" in features

    def test_lagged_returns_positive_for_uptrend(self) -> None:
        """_make_candles produces monotonically increasing prices."""
        candles = _make_candles(50)
        features = compute_features(candles)
        assert features["ret_1d"] > 0.0
        assert features["ret_5d"] > 0.0
        assert features["ret_21d"] > 0.0

    def test_return_distribution_features_present(self) -> None:
        candles = _make_candles(50)
        features = compute_features(candles)
        assert "skew_20d" in features
        assert "kurt_20d" in features
        assert "max_ret_20d" in features
        assert "min_ret_20d" in features

    def test_return_distribution_features_finite(self) -> None:
        candles = _make_candles(50)
        features = compute_features(candles)
        for key in ("skew_20d", "kurt_20d", "max_ret_20d", "min_ret_20d"):
            assert math.isfinite(features[key]), f"{key} is not finite"

    def test_short_rsi_present(self) -> None:
        candles = _make_candles(50)
        features = compute_features(candles)
        assert "rsi_2" in features
        assert "rsi_5" in features

    def test_short_rsi_in_valid_range(self) -> None:
        candles = _make_candles(50)
        features = compute_features(candles)
        rsi_2_lower = 0.0
        rsi_upper = 100.0
        assert rsi_2_lower <= features["rsi_2"] <= rsi_upper
        assert rsi_2_lower <= features["rsi_5"] <= rsi_upper


class TestRemovedFeatures:
    """Verify dead/noise features are removed from output."""

    def test_sentiment_not_in_output(self) -> None:
        candles = _make_candles(50)
        features = compute_features(candles)
        assert "sentiment" not in features

    def test_dow_sin_cos_not_in_output(self) -> None:
        candles = _make_candles(50)
        features = compute_features(candles)
        assert "dow_sin" not in features
        assert "dow_cos" not in features

    def test_ma_slope_20_not_in_output(self) -> None:
        candles = _make_candles(50)
        features = compute_features(candles)
        assert "ma_slope_20" not in features

    def test_proximity_52wk_not_in_output(self) -> None:
        """Old name should be replaced by proximity_rolling_high."""
        candles = _make_candles(50)
        features = compute_features(candles)
        assert "proximity_52wk" not in features
        assert "proximity_rolling_high" in features

    def test_backward_compat_sentiment_param(self) -> None:
        """compute_features(candles, sentiment_score=0.5) still works without error."""
        candles = _make_candles(50)
        features = compute_features(candles, sentiment_score=0.5)
        # sentiment_score param accepted but not included in output
        assert "sentiment" not in features
        assert len(features) >= _EXPECTED_MIN_FEATURES


class TestTotalFeatureCount:
    """Verify total feature count after Phase B changes."""

    def test_total_feature_count(self) -> None:
        candles = _make_candles(50)
        features = compute_features(candles)
        assert len(features) == _EXPECTED_TOTAL_FEATURES
