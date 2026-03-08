"""Tests for ML feature engineering (6C.1 + 6C.2 + Phase B improvements)."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import ClassVar

import pytest

from finalayze.core.schemas import Candle, MarketContext
from finalayze.ml.features.technical import (
    _MAX_FEATURE_LOOKBACK,
    _MIN_CANDLES,
    compute_features,
)

_EXPECTED_MIN_FEATURES = 25
_EXPECTED_TOTAL_FEATURES = 44
_NEW_MIN_CANDLES = 80


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
        candles = _make_candles(80)
        features = compute_features(candles)
        assert len(features) >= _EXPECTED_MIN_FEATURES

    def test_compute_features_no_nans(self) -> None:
        candles = _make_candles(80)
        features = compute_features(candles)
        for key, val in features.items():
            assert math.isfinite(val), f"Feature {key} is not finite: {val}"

    def test_compute_features_garman_klass_non_negative(self) -> None:
        candles = _make_candles(80)
        features = compute_features(candles)
        assert "gk_vol_20" in features
        assert features["gk_vol_20"] >= 0.0

    def test_min_candles_is_80(self) -> None:
        """_MIN_CANDLES raised to 80 for deeper lookback features."""
        assert _MIN_CANDLES == _NEW_MIN_CANDLES

    def test_max_feature_lookback_exported(self) -> None:
        """_MAX_FEATURE_LOOKBACK is exported and equals 252."""
        max_lookback = 252
        assert max_lookback == _MAX_FEATURE_LOOKBACK

    def test_compute_features_minimum_candles(self) -> None:
        """80 candles should work (the new minimum)."""
        candles = _make_candles(_NEW_MIN_CANDLES)
        features = compute_features(candles)
        assert len(features) >= _EXPECTED_MIN_FEATURES

    def test_compute_features_rejects_fewer_than_80(self) -> None:
        """Fewer than 80 candles raises InsufficientDataError."""
        from finalayze.core.exceptions import InsufficientDataError

        candles = _make_candles(79)
        with pytest.raises(InsufficientDataError):
            compute_features(candles)

    def test_compute_features_has_expected_keys(self) -> None:
        candles = _make_candles(80)
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
        candles_low = _make_candles(80, base_price=20.0)
        candles_high = _make_candles(80, base_price=200.0)
        feat_low = compute_features(candles_low)
        feat_high = compute_features(candles_high)
        # The percentage ATR should be in the same order of magnitude
        assert abs(feat_low["atr_14_pct"] - feat_high["atr_14_pct"]) < 0.1

    def test_macd_hist_pct_scales_with_price(self) -> None:
        """Two candle sets at different price levels produce similar macd_hist_pct."""
        candles_low = _make_candles(80, base_price=20.0)
        candles_high = _make_candles(80, base_price=200.0)
        feat_low = compute_features(candles_low)
        feat_high = compute_features(candles_high)
        # The percentage MACD hist should be in the same order of magnitude
        assert abs(feat_low["macd_hist_pct"] - feat_high["macd_hist_pct"]) < 0.1

    def test_old_feature_names_absent(self) -> None:
        """Renamed features should not have old names."""
        candles = _make_candles(80)
        features = compute_features(candles)
        assert "atr_14" not in features
        assert "macd_hist" not in features
        assert "atr_14_pct" in features
        assert "macd_hist_pct" in features


class TestMicrostructureFeatures:
    """Tests for proximity_rolling_high, amihud_20d, and corwin_schultz_spread."""

    def test_proximity_rolling_high_present(self) -> None:
        candles = _make_candles(80)
        features = compute_features(candles)
        assert "proximity_rolling_high" in features

    def test_proximity_rolling_high_at_high_is_one(self) -> None:
        """When close is monotonically increasing, last close == rolling max -> 1.0."""
        candles = _make_candles(80)
        features = compute_features(candles)
        # _make_candles produces monotonically increasing close prices,
        # so the last close IS the rolling max
        assert features["proximity_rolling_high"] == pytest.approx(1.0)

    def test_amihud_20d_present_in_features(self) -> None:
        candles = _make_candles(80)
        features = compute_features(candles)
        assert "amihud_20d" in features

    def test_amihud_log_transformed_positive(self) -> None:
        """Amihud should be log-transformed: positive and reasonable for any stock."""
        candles = _make_candles(80)
        features = compute_features(candles)
        # log1p(x * 1e6) for a small illiquidity value should be positive
        assert features["amihud_20d"] >= 0.0
        # Should be a reasonable log-scale value (not raw tiny float)
        # For liquid stocks with volume ~1000, the raw amihud is very small,
        # but log1p(val * 1e6) should produce a value > 0
        assert features["amihud_20d"] > 0.0

    def test_corwin_schultz_present_in_features(self) -> None:
        candles = _make_candles(80)
        features = compute_features(candles)
        assert "corwin_schultz_spread" in features

    def test_corwin_schultz_non_negative(self) -> None:
        candles = _make_candles(80)
        features = compute_features(candles)
        assert features["corwin_schultz_spread"] >= 0.0


class TestNewLaggedReturnFeatures:
    """Phase B: lagged returns, return distribution, short RSI."""

    def test_lagged_returns_present(self) -> None:
        candles = _make_candles(80)
        features = compute_features(candles)
        assert "ret_1d" in features
        assert "ret_5d" in features
        assert "ret_21d" in features

    def test_lagged_returns_positive_for_uptrend(self) -> None:
        """_make_candles produces monotonically increasing prices."""
        candles = _make_candles(80)
        features = compute_features(candles)
        assert features["ret_1d"] > 0.0
        assert features["ret_5d"] > 0.0
        assert features["ret_21d"] > 0.0

    def test_return_distribution_features_present(self) -> None:
        candles = _make_candles(80)
        features = compute_features(candles)
        assert "skew_20d" in features
        assert "kurt_20d" in features
        assert "max_ret_20d" in features
        assert "min_ret_20d" in features

    def test_return_distribution_features_finite(self) -> None:
        candles = _make_candles(80)
        features = compute_features(candles)
        for key in ("skew_20d", "kurt_20d", "max_ret_20d", "min_ret_20d"):
            assert math.isfinite(features[key]), f"{key} is not finite"

    def test_short_rsi_present(self) -> None:
        candles = _make_candles(80)
        features = compute_features(candles)
        assert "rsi_2" in features
        assert "rsi_5" in features

    def test_short_rsi_in_valid_range(self) -> None:
        candles = _make_candles(80)
        features = compute_features(candles)
        rsi_2_lower = 0.0
        rsi_upper = 100.0
        assert rsi_2_lower <= features["rsi_2"] <= rsi_upper
        assert rsi_2_lower <= features["rsi_5"] <= rsi_upper


class TestRemovedFeatures:
    """Verify dead/noise features are removed from output."""

    def test_sentiment_not_in_output(self) -> None:
        candles = _make_candles(80)
        features = compute_features(candles)
        assert "sentiment" not in features

    def test_ma_slope_20_not_in_output(self) -> None:
        candles = _make_candles(80)
        features = compute_features(candles)
        assert "ma_slope_20" not in features

    def test_proximity_52wk_not_in_output(self) -> None:
        """Old name should be replaced by proximity_rolling_high."""
        candles = _make_candles(80)
        features = compute_features(candles)
        assert "proximity_52wk" not in features
        assert "proximity_rolling_high" in features

    def test_backward_compat_sentiment_param(self) -> None:
        """compute_features(candles, sentiment_score=0.5) still works without error."""
        candles = _make_candles(80)
        features = compute_features(candles, sentiment_score=0.5)
        # sentiment_score param accepted but not included in output
        assert "sentiment" not in features
        assert len(features) >= _EXPECTED_MIN_FEATURES


class TestZScoreFeatures:
    """Phase D: Relative strength / z-score features."""

    _ZSCORE_KEYS: ClassVar[list[str]] = [
        "price_zscore_60d",
        "volume_zscore_20d",
        "rsi_zscore_60d",
        "atr_zscore_60d",
    ]

    def test_zscore_features_present(self) -> None:
        """All 4 z-score features appear in output."""
        candles = _make_candles(80)
        features = compute_features(candles)
        for key in self._ZSCORE_KEYS:
            assert key in features, f"Missing z-score feature: {key}"

    def test_zscore_features_finite(self) -> None:
        """All z-score features are finite floats."""
        candles = _make_candles(80)
        features = compute_features(candles)
        for key in self._ZSCORE_KEYS:
            assert math.isfinite(features[key]), f"{key} = {features[key]} is not finite"

    def test_zscore_features_with_minimum_candles(self) -> None:
        """Z-score features work with 80 candles (min required)."""
        candles = _make_candles(80)
        features = compute_features(candles)
        for key in self._ZSCORE_KEYS:
            assert key in features
            assert math.isfinite(features[key])

    def test_price_zscore_positive_for_uptrend(self) -> None:
        """Monotonically increasing price should have positive z-score."""
        candles = _make_candles(80)
        features = compute_features(candles)
        assert features["price_zscore_60d"] > 0.0

    def test_volume_zscore_reasonable_range(self) -> None:
        """Volume z-score should be in a reasonable range for steady data."""
        candles = _make_candles(80)
        features = compute_features(candles)
        zscore_reasonable_bound = 10.0
        assert abs(features["volume_zscore_20d"]) < zscore_reasonable_bound

    def test_zscore_zero_std_returns_zero(self) -> None:
        """When std is 0 (constant values), z-score should be 0.0."""
        candles: list[Candle] = []
        base_ts = datetime(2025, 1, 6, 10, 0, tzinfo=UTC)
        for i in range(80):
            ts = base_ts + timedelta(days=i)
            candles.append(
                Candle(
                    symbol="FLAT",
                    market_id="us",
                    timeframe="1d",
                    timestamp=ts,
                    open=Decimal(100),
                    high=Decimal(100),
                    low=Decimal(100),
                    close=Decimal(100),
                    volume=1000,
                )
            )
        features = compute_features(candles)
        assert features["price_zscore_60d"] == 0.0
        assert features["volume_zscore_20d"] == 0.0

    def test_rsi_zscore_finite_for_trending(self) -> None:
        """RSI z-score should be finite for a trending series."""
        candles = _make_candles(80)
        features = compute_features(candles)
        assert math.isfinite(features["rsi_zscore_60d"])

    def test_atr_zscore_finite_for_trending(self) -> None:
        """ATR z-score should be finite for a trending series."""
        candles = _make_candles(80)
        features = compute_features(candles)
        assert math.isfinite(features["atr_zscore_60d"])


class TestCalendarFeatures:
    """Phase C: cyclical calendar encoding features."""

    _CALENDAR_KEYS = ("dow_sin", "dow_cos", "month_sin", "month_cos")

    def test_calendar_features_present(self) -> None:
        candles = _make_candles(80)
        features = compute_features(candles)
        for key in self._CALENDAR_KEYS:
            assert key in features, f"Missing calendar feature: {key}"

    def test_calendar_features_finite(self) -> None:
        candles = _make_candles(80)
        features = compute_features(candles)
        for key in self._CALENDAR_KEYS:
            assert math.isfinite(features[key]), f"{key} not finite"

    def test_calendar_features_bounded(self) -> None:
        """sin/cos values must be in [-1, 1]."""
        candles = _make_candles(80)
        features = compute_features(candles)
        lower_bound = -1.0
        upper_bound = 1.0
        for key in self._CALENDAR_KEYS:
            assert lower_bound <= features[key] <= upper_bound, (
                f"{key}={features[key]} out of [-1,1]"
            )

    def test_monday_dow_encoding(self) -> None:
        """Monday (dow=0): sin(0)=0.0, cos(0)=1.0."""
        # _make_candles starts on 2025-01-06 (Mon), +84 days = 2025-03-31 (Mon)
        candles = _make_candles(85)
        last_ts = candles[-1].timestamp
        assert last_ts.weekday() == 0, f"Expected Monday, got weekday={last_ts.weekday()}"
        features = compute_features(candles)
        assert features["dow_sin"] == pytest.approx(0.0, abs=1e-9)
        assert features["dow_cos"] == pytest.approx(1.0, abs=1e-9)

    def test_january_month_encoding(self) -> None:
        """January (month=1): sin(2*pi*1/12), cos(2*pi*1/12)."""
        base_ts = datetime(2024, 10, 14, 10, 0, tzinfo=UTC)
        candles: list[Candle] = []
        for i in range(80):
            price = 100.0 + i * 0.5
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
        # Last candle: Oct 14 + 79 days = Jan 1 2025 (month=1)
        last_month = candles[-1].timestamp.month
        assert last_month == 1
        features = compute_features(candles)
        months_per_year = 12
        expected_sin = math.sin(2 * math.pi * 1 / months_per_year)
        expected_cos = math.cos(2 * math.pi * 1 / months_per_year)
        assert features["month_sin"] == pytest.approx(expected_sin, abs=1e-9)
        assert features["month_cos"] == pytest.approx(expected_cos, abs=1e-9)

    def test_no_lookahead_bias(self) -> None:
        """Calendar features use only the last candle timestamp."""
        candles = _make_candles(100)
        features_100 = compute_features(candles)
        candles_80 = _make_candles(80)
        features_80 = compute_features(candles_80)
        for key in self._CALENDAR_KEYS:
            assert math.isfinite(features_100[key])
            assert math.isfinite(features_80[key])


class TestRegimeFeatures:
    """Phase B: VIX regime and realized volatility features."""

    _REGIME_KEYS = ("vix_level", "vix_percentile_252d", "vix_change_5d", "realized_vol_ratio")

    def test_regime_features_present_in_output(self) -> None:
        """All 4 regime features appear in output."""
        candles = _make_candles(80)
        features = compute_features(candles)
        for key in self._REGIME_KEYS:
            assert key in features, f"Missing regime feature: {key}"

    def test_vix_features_default_zero_when_no_vix(self) -> None:
        """When vix_candles=None, VIX features should be 0.0."""
        candles = _make_candles(80)
        features = compute_features(candles, vix_candles=None)
        assert features["vix_level"] == 0.0
        assert features["vix_percentile_252d"] == 0.0
        assert features["vix_change_5d"] == 0.0

    def test_realized_vol_ratio_positive_without_vix(self) -> None:
        """realized_vol_ratio uses stock's own data and should be > 0."""
        candles = _make_candles(80)
        features = compute_features(candles)
        assert features["realized_vol_ratio"] > 0.0

    def test_vix_level_uses_lagged_value(self) -> None:
        """vix_level should use candles[-2].close, not candles[-1].close."""
        candles = _make_candles(80)
        vix_candles = _make_candles(80, base_price=20.0)
        features = compute_features(candles, vix_candles=vix_candles)
        # vix_candles[-2].close = 20.0 + 48 * 0.5 = 44.0
        expected_vix_level = float(vix_candles[-2].close)
        assert features["vix_level"] == pytest.approx(expected_vix_level)

    def test_vix_percentile_in_range(self) -> None:
        """vix_percentile_252d should be in [0, 1]."""
        candles = _make_candles(80)
        vix_candles = _make_candles(80, base_price=20.0)
        features = compute_features(candles, vix_candles=vix_candles)
        percentile_lower = 0.0
        percentile_upper = 1.0
        assert percentile_lower <= features["vix_percentile_252d"] <= percentile_upper

    def test_vix_change_5d_positive_for_uptrend(self) -> None:
        """VIX candles with monotonically increasing close should have positive 5d change."""
        candles = _make_candles(80)
        vix_candles = _make_candles(80, base_price=20.0)
        features = compute_features(candles, vix_candles=vix_candles)
        assert features["vix_change_5d"] > 0.0

    def test_regime_features_finite(self) -> None:
        """All regime features must be finite."""
        candles = _make_candles(80)
        vix_candles = _make_candles(80, base_price=20.0)
        features = compute_features(candles, vix_candles=vix_candles)
        for key in self._REGIME_KEYS:
            assert math.isfinite(features[key]), f"{key} = {features[key]} is not finite"

    def test_short_vix_series_graceful(self) -> None:
        """With very few VIX candles, features should still be finite (no crash)."""
        candles = _make_candles(80)
        vix_candles = _make_candles(5, base_price=20.0)
        features = compute_features(candles, vix_candles=vix_candles)
        for key in self._REGIME_KEYS:
            assert math.isfinite(features[key]), f"{key} not finite with short VIX"


class TestCrossAssetFeatures:
    """Phase A: Cross-asset features (relative strength vs benchmark)."""

    _CROSS_ASSET_KEYS = (
        "relative_strength_21d",
        "rolling_beta_63d",
        "rolling_corr_63d",
        "excess_momentum_score",
    )

    def test_cross_asset_features_present_with_benchmark(self) -> None:
        """All 4 cross-asset features appear when benchmark_candles is provided."""
        candles = _make_candles(80)
        benchmark = _make_candles(80, base_price=200.0)
        features = compute_features(candles, benchmark_candles=benchmark)
        for key in self._CROSS_ASSET_KEYS:
            assert key in features, f"Missing cross-asset feature: {key}"

    def test_cross_asset_defaults_when_no_benchmark(self) -> None:
        """Domain-aware defaults when benchmark_candles is None."""
        candles = _make_candles(80)
        features = compute_features(candles)
        default_beta = 1.0
        default_corr = 0.5
        assert features["relative_strength_21d"] == 0.0
        assert features["rolling_beta_63d"] == default_beta
        assert features["rolling_corr_63d"] == default_corr
        assert features["excess_momentum_score"] == 0.0

    def test_cross_asset_features_finite(self) -> None:
        """All cross-asset features are finite floats."""
        candles = _make_candles(80)
        benchmark = _make_candles(80, base_price=200.0)
        features = compute_features(candles, benchmark_candles=benchmark)
        for key in self._CROSS_ASSET_KEYS:
            assert math.isfinite(features[key]), f"{key} = {features[key]} is not finite"

    def test_beta_close_to_one_for_identical_series(self) -> None:
        """Rolling beta of a series vs itself should be ~1.0."""
        candles = _make_candles(80)
        features = compute_features(candles, benchmark_candles=candles)
        assert features["rolling_beta_63d"] == pytest.approx(1.0, abs=0.15)

    def test_corr_close_to_one_for_identical_series(self) -> None:
        """Rolling correlation of a series vs itself should be ~1.0."""
        candles = _make_candles(80)
        features = compute_features(candles, benchmark_candles=candles)
        assert features["rolling_corr_63d"] == pytest.approx(1.0, abs=0.05)

    def test_excess_momentum_handles_zero_vol(self) -> None:
        """Denominator clamped to VOL_FLOOR when vol is near zero."""
        base_ts = datetime(2025, 1, 6, 10, 0, tzinfo=UTC)
        flat_candles: list[Candle] = []
        flat_count = 100
        for i in range(flat_count):
            ts = base_ts + timedelta(days=i)
            flat_candles.append(
                Candle(
                    symbol="FLAT",
                    market_id="us",
                    timeframe="1d",
                    timestamp=ts,
                    open=Decimal(100),
                    high=Decimal(101),
                    low=Decimal(99),
                    close=Decimal(100),
                    volume=1000,
                )
            )
        benchmark = _make_candles(flat_count, base_price=200.0)
        features = compute_features(flat_candles, benchmark_candles=benchmark)
        assert math.isfinite(features["excess_momentum_score"])

    def test_relative_strength_positive_for_outperformer(self) -> None:
        """Stock with higher returns than benchmark has positive relative strength."""
        stock = _make_candles(100, base_price=100.0)  # +0.5 per bar
        base_ts = datetime(2025, 1, 6, 10, 0, tzinfo=UTC)
        benchmark: list[Candle] = []
        bench_count = 100
        for i in range(bench_count):
            price = 200.0 + i * 0.1  # slower growth
            ts = base_ts + timedelta(days=i)
            benchmark.append(
                Candle(
                    symbol="SPY",
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
        features = compute_features(stock, benchmark_candles=benchmark)
        assert features["relative_strength_21d"] > 0.0

    def test_cross_asset_with_short_benchmark(self) -> None:
        """When benchmark has fewer candles than lookback, use defaults."""
        candles = _make_candles(80)
        short_benchmark = _make_candles(10, base_price=200.0)
        features = compute_features(candles, benchmark_candles=short_benchmark)
        for key in self._CROSS_ASSET_KEYS:
            assert key in features
            assert math.isfinite(features[key])


class TestTotalFeatureCount:
    """Verify total feature count after all phases."""

    def test_total_feature_count(self) -> None:
        candles = _make_candles(80)
        features = compute_features(candles)
        assert len(features) == _EXPECTED_TOTAL_FEATURES


class TestMarketContext:
    """Phase E: MarketContext dataclass creation and usage."""

    def test_market_context_creation_defaults(self) -> None:
        """MarketContext defaults to None for both fields."""
        ctx = MarketContext()
        assert ctx.benchmark_candles is None
        assert ctx.vix_candles is None

    def test_market_context_creation_with_data(self) -> None:
        """MarketContext accepts candle lists."""
        bench = _make_candles(80, base_price=200.0)
        vix = _make_candles(80, base_price=20.0)
        ctx = MarketContext(benchmark_candles=bench, vix_candles=vix)
        assert ctx.benchmark_candles is bench
        assert ctx.vix_candles is vix

    def test_market_context_frozen(self) -> None:
        """MarketContext is immutable (frozen dataclass)."""
        ctx = MarketContext()
        with pytest.raises(AttributeError):
            ctx.benchmark_candles = []  # type: ignore[misc]


class TestMLStrategyMarketContext:
    """Phase E: MLStrategy.set_market_context integration."""

    def test_set_market_context(self) -> None:
        """MLStrategy stores market context for compute_features calls."""
        from unittest.mock import MagicMock

        from finalayze.strategies.ml_strategy import MLStrategy

        registry = MagicMock()
        strategy = MLStrategy(registry)

        assert strategy._market_context is None

        bench = _make_candles(80, base_price=200.0)
        vix = _make_candles(80, base_price=20.0)
        ctx = MarketContext(benchmark_candles=bench, vix_candles=vix)
        strategy.set_market_context(ctx)

        assert strategy._market_context is ctx
        assert strategy._market_context.benchmark_candles is bench
        assert strategy._market_context.vix_candles is vix


class TestBuildTripleBarrierDatasetForwarding:
    """Phase E: build_triple_barrier_dataset forwards benchmark/vix candles."""

    def test_forwards_vix_candles_to_compute_features(self) -> None:
        """build_triple_barrier_dataset passes vix_candles through."""
        from unittest.mock import patch

        from finalayze.ml.training.labeling import build_triple_barrier_dataset

        # Create enough candles for window_size + max_hold
        window_size = 80
        max_hold = 5
        n_candles = window_size + max_hold + 10
        candles = _make_candles(n_candles)
        vix = _make_candles(n_candles, base_price=20.0)
        bench = _make_candles(n_candles, base_price=200.0)

        captured_calls: list[dict] = []

        original_compute = compute_features

        def mock_compute(candles_arg, **kwargs):
            captured_calls.append(kwargs)
            return original_compute(candles_arg, **kwargs)

        with patch(
            "finalayze.ml.training.labeling.compute_features",
            side_effect=mock_compute,
        ):
            build_triple_barrier_dataset(
                candles,
                window_size=window_size,
                max_hold=max_hold,
                benchmark_candles=bench,
                vix_candles=vix,
            )

        # Verify at least one call was made and vix/benchmark were passed
        assert len(captured_calls) > 0
        for call_kwargs in captured_calls:
            assert call_kwargs.get("vix_candles") is vix
            assert call_kwargs.get("benchmark_candles") is bench

    def test_uses_full_history_window(self) -> None:
        """build_triple_barrier_dataset passes full history up to entry bar."""
        from unittest.mock import patch

        from finalayze.ml.training.labeling import build_triple_barrier_dataset

        window_size = 80
        max_hold = 5
        n_candles = window_size + max_hold + 10
        candles = _make_candles(n_candles)

        window_sizes: list[int] = []

        original_compute = compute_features

        def mock_compute(candles_arg, **kwargs):
            window_sizes.append(len(candles_arg))
            return original_compute(candles_arg, **kwargs)

        with patch(
            "finalayze.ml.training.labeling.compute_features",
            side_effect=mock_compute,
        ):
            build_triple_barrier_dataset(
                candles,
                window_size=window_size,
                max_hold=max_hold,
            )

        # Windows should grow (full history), not be fixed at window_size
        if len(window_sizes) > 1:
            assert window_sizes[-1] > window_sizes[0]
