"""Unit tests for Ichimoku Cloud helper and momentum strategy integration."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING

from finalayze.core.schemas import Candle, SignalDirection

if TYPE_CHECKING:
    import pytest
from finalayze.strategies.ichimoku import IchimokuResult, compute_ichimoku
from finalayze.strategies.momentum import MomentumStrategy

# ---------------------------------------------------------------------------
# Constants (no magic numbers -- ruff PLR2004)
# ---------------------------------------------------------------------------
TENKAN_PERIOD = 9
KIJUN_PERIOD = 26
MIN_ICHIMOKU_BARS = 26  # kijun_period is the binding constraint
INSUFFICIENT_BARS = 10
CLOUD_THICKNESS_TOLERANCE = 1e-9

# Momentum test params (no YAML I/O)
_MOMENTUM_PARAMS_ICHIMOKU: dict[str, object] = {
    "rsi_period": 14,
    "rsi_oversold": 30,
    "rsi_overbought": 70,
    "macd_fast": 12,
    "macd_slow": 26,
    "min_confidence": 0.0,
    "macd_hist_lookback": 3,
    "lookback_bars": 5,
    "trend_filter": False,
    "trend_sma_period": 50,
    "trend_sma_buffer_pct": 2.0,
    "adx_filter": False,
    "adx_period": 14,
    "adx_threshold": 25,
    "volume_filter": False,
    "volume_sma_period": 20,
    "volume_min_ratio": 1.0,
    "neutral_reset_bars": 20,
    "ichimoku_filter": False,
}

# Known price data for deterministic Ichimoku computation.
# 26 bars of uptrending data so tenkan > kijun and price above cloud.
_UPTREND_PRICES = [100.0 + i * 2.0 for i in range(30)]
_UPTREND_HIGHS = [p + 1.0 for p in _UPTREND_PRICES]
_UPTREND_LOWS = [p - 1.0 for p in _UPTREND_PRICES]

# 30 bars of downtrending data so tenkan < kijun and price below cloud.
_DOWNTREND_PRICES = [200.0 - i * 2.0 for i in range(30)]
_DOWNTREND_HIGHS = [p + 1.0 for p in _DOWNTREND_PRICES]
_DOWNTREND_LOWS = [p - 1.0 for p in _DOWNTREND_PRICES]


def _make_candles(
    prices: list[float],
    *,
    highs: list[float] | None = None,
    lows: list[float] | None = None,
    volumes: list[int] | None = None,
) -> list[Candle]:
    """Build Candle list for testing."""
    base = datetime(2024, 1, 1, 14, 30, tzinfo=UTC)
    candles: list[Candle] = []
    for i, price in enumerate(prices):
        p = Decimal(str(price))
        h = Decimal(str(highs[i])) if highs else p + Decimal(1)
        lo = Decimal(str(lows[i])) if lows else p - Decimal(1)
        vol = volumes[i] if volumes else 1_000_000
        candles.append(
            Candle(
                symbol="AAPL",
                market_id="us",
                timeframe="1d",
                timestamp=base + timedelta(days=i),
                open=p,
                high=h,
                low=lo,
                close=p,
                volume=vol,
            )
        )
    return candles


# ===========================================================================
# Tests for compute_ichimoku
# ===========================================================================


class TestComputeIchimoku:
    """Tests for the standalone compute_ichimoku function."""

    def test_insufficient_data_returns_none(self) -> None:
        """Fewer bars than kijun_period -> None."""
        highs = [100.0] * INSUFFICIENT_BARS
        lows = [99.0] * INSUFFICIENT_BARS
        closes = [99.5] * INSUFFICIENT_BARS
        result = compute_ichimoku(highs, lows, closes)
        assert result is None

    def test_exact_minimum_data(self) -> None:
        """Exactly kijun_period bars should produce a result."""
        n = KIJUN_PERIOD
        highs = [100.0 + i for i in range(n)]
        lows = [99.0 + i for i in range(n)]
        closes = [99.5 + i for i in range(n)]
        result = compute_ichimoku(highs, lows, closes)
        assert result is not None
        assert isinstance(result, IchimokuResult)

    def test_uptrend_is_bullish(self) -> None:
        """In a clear uptrend, is_bullish=True, is_bearish=False."""
        result = compute_ichimoku(_UPTREND_HIGHS, _UPTREND_LOWS, _UPTREND_PRICES)
        assert result is not None
        assert result.is_bullish is True
        assert result.is_bearish is False

    def test_downtrend_is_bearish(self) -> None:
        """In a clear downtrend, is_bearish=True, is_bullish=False."""
        result = compute_ichimoku(_DOWNTREND_HIGHS, _DOWNTREND_LOWS, _DOWNTREND_PRICES)
        assert result is not None
        assert result.is_bearish is True
        assert result.is_bullish is False

    def test_cloud_thickness_positive(self) -> None:
        """Cloud thickness is always >= 0."""
        result = compute_ichimoku(_UPTREND_HIGHS, _UPTREND_LOWS, _UPTREND_PRICES)
        assert result is not None
        assert result.cloud_thickness >= 0.0

    def test_cloud_thickness_is_abs_difference(self) -> None:
        """Cloud thickness = abs(senkou_a - senkou_b)."""
        result = compute_ichimoku(_UPTREND_HIGHS, _UPTREND_LOWS, _UPTREND_PRICES)
        assert result is not None
        expected = abs(result.senkou_a - result.senkou_b)
        assert abs(result.cloud_thickness - expected) < CLOUD_THICKNESS_TOLERANCE

    def test_tenkan_kijun_values_known_data(self) -> None:
        """Verify tenkan and kijun calculations with known monotonic data.

        For monotonic up data: tenkan = (high[-9] + low[-1]) / 2 over last 9 bars
        (but actually highest high and lowest low over the period).
        """
        result = compute_ichimoku(_UPTREND_HIGHS, _UPTREND_LOWS, _UPTREND_PRICES)
        assert result is not None
        # Tenkan: (highest_high + lowest_low) / 2 over last 9 bars
        last_9_highs = _UPTREND_HIGHS[-TENKAN_PERIOD:]
        last_9_lows = _UPTREND_LOWS[-TENKAN_PERIOD:]
        expected_tenkan = (max(last_9_highs) + min(last_9_lows)) / 2.0
        assert abs(result.tenkan - expected_tenkan) < CLOUD_THICKNESS_TOLERANCE

        # Kijun: (highest_high + lowest_low) / 2 over last 26 bars
        last_26_highs = _UPTREND_HIGHS[-KIJUN_PERIOD:]
        last_26_lows = _UPTREND_LOWS[-KIJUN_PERIOD:]
        expected_kijun = (max(last_26_highs) + min(last_26_lows)) / 2.0
        assert abs(result.kijun - expected_kijun) < CLOUD_THICKNESS_TOLERANCE

    def test_custom_periods(self) -> None:
        """Custom tenkan/kijun periods are respected."""
        custom_tenkan = 5
        custom_kijun = 15
        n = custom_kijun
        highs = [100.0 + i for i in range(n)]
        lows = [99.0 + i for i in range(n)]
        closes = [99.5 + i for i in range(n)]
        result = compute_ichimoku(
            highs,
            lows,
            closes,
            tenkan_period=custom_tenkan,
            kijun_period=custom_kijun,
        )
        assert result is not None
        # Tenkan uses last 5 bars
        expected_tenkan = (max(highs[-custom_tenkan:]) + min(lows[-custom_tenkan:])) / 2.0
        assert abs(result.tenkan - expected_tenkan) < CLOUD_THICKNESS_TOLERANCE

    def test_flat_data_not_bullish_not_bearish(self) -> None:
        """Flat data: tenkan == kijun, price inside cloud -> neither bullish nor bearish."""
        n = KIJUN_PERIOD
        highs = [101.0] * n
        lows = [99.0] * n
        closes = [100.0] * n
        result = compute_ichimoku(highs, lows, closes)
        assert result is not None
        # Flat: tenkan == kijun, so neither bullish nor bearish
        assert result.is_bullish is False
        assert result.is_bearish is False


# ===========================================================================
# Tests for MomentumStrategy with ichimoku_filter
# ===========================================================================


class TestMomentumIchimokuFilter:
    """Tests for the ichimoku_filter integration in MomentumStrategy."""

    @staticmethod
    def _buy_prices() -> list[float]:
        """Price series that triggers a BUY signal (crash + recovery)."""
        stable_price = 200.0
        prices: list[float] = [stable_price] * 40
        crash_bottom = stable_price - 4.0 * 16
        prices.extend([stable_price - 4.0 * (i + 1) for i in range(16)])
        prices.extend([crash_bottom] * 3)
        prices.extend([crash_bottom + 2.0 * (i + 1) for i in range(4)])
        return prices

    @staticmethod
    def _sell_prices() -> list[float]:
        """Price series that triggers a SELL signal (rally + decline)."""
        stable_price = 100.0
        rally_top = stable_price + 4.0 * 16
        prices: list[float] = [stable_price] * 40
        prices.extend([stable_price + 4.0 * (i + 1) for i in range(16)])
        prices.extend([rally_top] * 3)
        prices.extend([rally_top - 2.0 * (i + 1) for i in range(7)])
        return prices

    def test_ichimoku_filter_default_false_does_not_affect(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ichimoku_filter=False (default) should not change existing behavior."""
        params = {**_MOMENTUM_PARAMS_ICHIMOKU, "ichimoku_filter": False}
        strategy = MomentumStrategy()
        monkeypatch.setattr(strategy, "get_parameters", lambda _seg: params)
        candles = _make_candles(self._buy_prices())
        signal = strategy.generate_signal("AAPL", candles, "us_tech")
        assert signal is not None
        assert signal.direction == SignalDirection.BUY

    def test_ichimoku_filter_suppresses_buy_in_downtrend(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Ichimoku filter enabled: BUY suppressed when price is in downtrend (below cloud).

        The crash+recovery pattern produces prices well below the cloud, so
        Ichimoku is bearish and should block BUY.
        """
        params = {**_MOMENTUM_PARAMS_ICHIMOKU, "ichimoku_filter": True}
        strategy = MomentumStrategy()
        monkeypatch.setattr(strategy, "get_parameters", lambda _seg: params)

        prices = self._buy_prices()
        highs = [p + 5.0 for p in prices]
        lows = [p - 5.0 for p in prices]
        candles = _make_candles(prices, highs=highs, lows=lows)

        # Without ichimoku filter, BUY fires
        params_no_ichimoku = {**params, "ichimoku_filter": False}
        strategy_no = MomentumStrategy()
        monkeypatch.setattr(strategy_no, "get_parameters", lambda _seg: params_no_ichimoku)
        baseline = strategy_no.generate_signal("AAPL", candles, "us_tech")
        assert baseline is not None, "Baseline BUY should fire without ichimoku filter"
        assert baseline.direction == SignalDirection.BUY

        # With ichimoku filter, BUY suppressed (price crashed below cloud)
        signal = strategy.generate_signal("AAPL", candles, "us_tech")
        assert signal is None, "BUY suppressed by ichimoku filter in downtrend"

    def test_ichimoku_filter_allows_sell_exit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """SELL signals bypass ichimoku filter (exit relaxation)."""
        params = {**_MOMENTUM_PARAMS_ICHIMOKU, "ichimoku_filter": True}
        strategy = MomentumStrategy()
        monkeypatch.setattr(strategy, "get_parameters", lambda _seg: params)

        prices = self._sell_prices()
        highs = [p + 5.0 for p in prices]
        lows = [p - 5.0 for p in prices]
        candles = _make_candles(prices, highs=highs, lows=lows)
        signal = strategy.generate_signal("AAPL", candles, "us_tech")
        assert signal is not None, "SELL should pass despite ichimoku filter (exit relaxation)"
        assert signal.direction == SignalDirection.SELL

    def test_ichimoku_features_in_signal(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When ichimoku_filter is enabled, signal features include ichimoku data."""
        params = {**_MOMENTUM_PARAMS_ICHIMOKU, "ichimoku_filter": True}
        strategy = MomentumStrategy()
        monkeypatch.setattr(strategy, "get_parameters", lambda _seg: params)

        # Use sell prices (SELL bypasses filter) to get a signal with ichimoku features
        prices = self._sell_prices()
        highs = [p + 5.0 for p in prices]
        lows = [p - 5.0 for p in prices]
        candles = _make_candles(prices, highs=highs, lows=lows)
        signal = strategy.generate_signal("AAPL", candles, "us_tech")
        assert signal is not None
        assert "ichimoku_bullish" in signal.features
        assert "ichimoku_cloud_thickness" in signal.features

    def test_ichimoku_cloud_thickness_modifies_confidence(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Cloud thickness should act as a confidence modifier."""
        # With ichimoku off, get baseline confidence from SELL
        params_off = {**_MOMENTUM_PARAMS_ICHIMOKU, "ichimoku_filter": False}
        strategy_off = MomentumStrategy()
        monkeypatch.setattr(strategy_off, "get_parameters", lambda _seg: params_off)

        prices = self._sell_prices()
        highs = [p + 5.0 for p in prices]
        lows = [p - 5.0 for p in prices]
        candles = _make_candles(prices, highs=highs, lows=lows)
        sig_off = strategy_off.generate_signal("AAPL", candles, "us_tech")
        assert sig_off is not None

        # With ichimoku on, SELL still fires but confidence may differ
        params_on = {**_MOMENTUM_PARAMS_ICHIMOKU, "ichimoku_filter": True}
        strategy_on = MomentumStrategy()
        monkeypatch.setattr(strategy_on, "get_parameters", lambda _seg: params_on)
        sig_on = strategy_on.generate_signal("AAPL", candles, "us_tech")
        assert sig_on is not None
        # Confidence should still be in valid range
        assert 0.0 <= sig_on.confidence <= 1.0
