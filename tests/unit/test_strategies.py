"""Unit tests for trading strategies."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest
import yaml

from finalayze.core.schemas import Candle, SignalDirection
from finalayze.strategies.base import BaseStrategy
from finalayze.strategies.mean_reversion import MeanReversionStrategy
from finalayze.strategies.momentum import MomentumStrategy, _SignalState

MIN_CANDLES_FOR_INDICATORS = 35
RSI_PERIOD = 14
MIN_SUPPORTED_SEGMENTS = 2
LOOKBACK_BARS = 5
CONFIDENCE_TOLERANCE = 0.01
NEUTRAL_RESET_BARS = 20

_MOMENTUM_PARAMS: dict[str, object] = {
    "rsi_period": 14,
    "rsi_oversold": 30,
    "rsi_overbought": 70,
    "macd_fast": 12,
    "macd_slow": 26,
    "min_confidence": 0.6,
    "macd_hist_lookback": 3,
}

_MOMENTUM_PARAMS_V2: dict[str, object] = {
    **_MOMENTUM_PARAMS,
    "lookback_bars": 5,
}

_MOMENTUM_PARAMS_V3: dict[str, object] = {
    **_MOMENTUM_PARAMS_V2,
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
}


def _make_candles(
    prices: list[float],
    start_year: int = 2024,
    *,
    volumes: list[int] | None = None,
    highs: list[float] | None = None,
    lows: list[float] | None = None,
) -> list[Candle]:
    candles = []
    base = datetime(start_year, 1, 1, 14, 30, tzinfo=UTC)
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


def _write_momentum_preset(preset_dir: Path, segment_id: str) -> None:
    """Write a minimal momentum-enabled YAML preset to a temp directory."""
    data = {
        "segment_id": segment_id,
        "strategies": {
            "momentum": {
                "enabled": True,
                "weight": 0.4,
                "params": dict(_MOMENTUM_PARAMS_V2),
            }
        },
    }
    with (preset_dir / f"{segment_id}.yaml").open("w") as f:
        yaml.dump(data, f)


class TestBaseStrategy:
    def test_is_abstract(self) -> None:
        with pytest.raises(TypeError):
            BaseStrategy()  # type: ignore[abstract]


class TestMomentumStrategy:
    def test_name(self) -> None:
        assert MomentumStrategy().name == "momentum"

    def test_supported_segments(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _write_momentum_preset(tmp_path, "us_tech")
        _write_momentum_preset(tmp_path, "us_broad")

        import finalayze.strategies.momentum as momentum_module

        monkeypatch.setattr(momentum_module, "_PRESETS_DIR", tmp_path)

        supported = MomentumStrategy().supported_segments()
        assert "us_tech" in supported
        assert "us_broad" in supported
        assert "nonexistent_segment" not in supported
        assert len(supported) >= MIN_SUPPORTED_SEGMENTS

    def test_get_parameters_us_tech(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _write_momentum_preset(tmp_path, "us_tech")

        import finalayze.strategies.momentum as momentum_module

        monkeypatch.setattr(momentum_module, "_PRESETS_DIR", tmp_path)

        params = MomentumStrategy().get_parameters("us_tech")
        assert params["rsi_period"] == RSI_PERIOD

    def test_insufficient_data_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        strategy = MomentumStrategy()
        monkeypatch.setattr(strategy, "get_parameters", lambda _seg: _MOMENTUM_PARAMS_V2)
        short = _make_candles([100.0] * 5)
        assert strategy.generate_signal("AAPL", short, "us_tech") is None

    def test_hold_when_no_signal(self, monkeypatch: pytest.MonkeyPatch) -> None:
        strategy = MomentumStrategy()
        monkeypatch.setattr(strategy, "get_parameters", lambda _seg: _MOMENTUM_PARAMS_V2)
        flat = _make_candles([150.0] * (MIN_CANDLES_FOR_INDICATORS + 5))
        assert strategy.generate_signal("AAPL", flat, "us_tech") is None

    def test_buy_signal_on_oversold_rsi(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Build a deterministic sequence that forces RSI < 30 and MACD histogram cross above zero.
        # Phase 1: 40 stable candles at 200 -- seeds EMA(12) and EMA(26) at the same level.
        # Phase 2: 16 crash candles dropping 4 points each -> RSI near 0.
        # Phase 3: 3 level candles (no change) -> allows MACD to recover slightly.
        # Phase 4: 4 recovery candles at +2 -> MACD histogram crossover at RSI=23.7.
        stable_price = 200.0
        stable_count = 40
        crash_drop = 4.0
        crash_count = 16
        level_count = 3
        recovery_step = 2.0
        recovery_count = 4
        prices: list[float] = [stable_price] * stable_count
        crash_bottom = stable_price - crash_drop * crash_count
        prices.extend([stable_price - crash_drop * (i + 1) for i in range(crash_count)])
        prices.extend([crash_bottom] * level_count)
        prices.extend([crash_bottom + recovery_step * (i + 1) for i in range(recovery_count)])

        strategy = MomentumStrategy()
        monkeypatch.setattr(strategy, "get_parameters", lambda _seg: _MOMENTUM_PARAMS_V2)
        candles = _make_candles(prices)
        signal = strategy.generate_signal("AAPL", candles, "us_tech")
        assert signal is not None, (
            "Expected a BUY signal after a crash+level+recovery pattern with RSI < 30 "
            "and MACD histogram cross above zero"
        )
        assert signal.direction == SignalDirection.BUY
        assert signal.strategy_name == "momentum"
        assert 0.0 <= signal.confidence <= 1.0

    def test_buy_signal_with_lookback_regime(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """RSI dipped below 30 within last 5 bars but current RSI ~35-40.

        Old logic would NOT fire (current RSI > oversold). New regime lookback SHOULD fire BUY.
        """
        # Phase 1: 40 stable candles at 200
        # Phase 2: 16 crash candles -> RSI near 0
        # Phase 3: 3 level candles
        # Phase 4: 7 recovery candles at +2 each -> RSI recovers above 30 but within lookback
        stable_price = 200.0
        stable_count = 40
        crash_drop = 4.0
        crash_count = 16
        level_count = 3
        recovery_step = 2.0
        recovery_count = 7  # more recovery than basic test -> RSI > 30 currently
        prices: list[float] = [stable_price] * stable_count
        crash_bottom = stable_price - crash_drop * crash_count
        prices.extend([stable_price - crash_drop * (i + 1) for i in range(crash_count)])
        prices.extend([crash_bottom] * level_count)
        prices.extend([crash_bottom + recovery_step * (i + 1) for i in range(recovery_count)])

        strategy = MomentumStrategy()
        monkeypatch.setattr(strategy, "get_parameters", lambda _seg: _MOMENTUM_PARAMS_V2)
        candles = _make_candles(prices)
        signal = strategy.generate_signal("AAPL", candles, "us_tech")
        assert signal is not None, (
            "Expected BUY: RSI was oversold within lookback window and histogram is rising"
        )
        assert signal.direction == SignalDirection.BUY

    def test_sell_signal_on_overbought_rsi_regime(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """RSI peaked above 70 within last 5 bars, histogram falling. Should fire SELL."""
        # Phase 1: 40 stable candles at 100
        # Phase 2: 16 rally candles rising 4 points each -> RSI near 100
        # Phase 3: 3 level candles
        # Phase 4: 7 decline candles -> RSI drops from overbought but within lookback
        stable_price = 100.0
        stable_count = 40
        rally_step = 4.0
        rally_count = 16
        level_count = 3
        decline_step = 2.0
        decline_count = 7
        prices: list[float] = [stable_price] * stable_count
        rally_top = stable_price + rally_step * rally_count
        prices.extend([stable_price + rally_step * (i + 1) for i in range(rally_count)])
        prices.extend([rally_top] * level_count)
        prices.extend([rally_top - decline_step * (i + 1) for i in range(decline_count)])

        strategy = MomentumStrategy()
        monkeypatch.setattr(strategy, "get_parameters", lambda _seg: _MOMENTUM_PARAMS_V2)
        candles = _make_candles(prices)
        signal = strategy.generate_signal("AAPL", candles, "us_tech")
        assert signal is not None, (
            "Expected SELL: RSI was overbought within lookback window and histogram is falling"
        )
        assert signal.direction == SignalDirection.SELL

    def test_no_buy_when_lookback_expired(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """RSI was oversold 6+ bars ago (outside lookback window). No signal."""
        # Phase 1: 40 stable candles at 200
        # Phase 2: 16 crash candles
        # Phase 3: 3 level candles
        # Phase 4: 12 recovery candles -> RSI recovers well past lookback window
        stable_price = 200.0
        stable_count = 40
        crash_drop = 4.0
        crash_count = 16
        level_count = 3
        recovery_step = 1.5
        recovery_count = 12  # far beyond lookback_bars=5
        prices: list[float] = [stable_price] * stable_count
        crash_bottom = stable_price - crash_drop * crash_count
        prices.extend([stable_price - crash_drop * (i + 1) for i in range(crash_count)])
        prices.extend([crash_bottom] * level_count)
        prices.extend([crash_bottom + recovery_step * (i + 1) for i in range(recovery_count)])

        strategy = MomentumStrategy()
        monkeypatch.setattr(strategy, "get_parameters", lambda _seg: _MOMENTUM_PARAMS_V2)
        candles = _make_candles(prices)
        signal = strategy.generate_signal("AAPL", candles, "us_tech")
        assert signal is None, "Expected no signal: oversold condition expired outside lookback"

    def test_no_sell_when_lookback_expired(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """RSI was overbought 6+ bars ago (outside lookback window). No signal."""
        # Phase 1: 40 stable candles at 100
        # Phase 2: 16 rally candles
        # Phase 3: 3 level candles
        # Phase 4: 12 decline candles -> RSI drops well past lookback window
        stable_price = 100.0
        stable_count = 40
        rally_step = 4.0
        rally_count = 16
        level_count = 3
        decline_step = 1.5
        decline_count = 12
        prices: list[float] = [stable_price] * stable_count
        rally_top = stable_price + rally_step * rally_count
        prices.extend([stable_price + rally_step * (i + 1) for i in range(rally_count)])
        prices.extend([rally_top] * level_count)
        prices.extend([rally_top - decline_step * (i + 1) for i in range(decline_count)])

        strategy = MomentumStrategy()
        monkeypatch.setattr(strategy, "get_parameters", lambda _seg: _MOMENTUM_PARAMS_V2)
        candles = _make_candles(prices)
        signal = strategy.generate_signal("AAPL", candles, "us_tech")
        assert signal is None, "Expected no signal: overbought condition expired outside lookback"

    def test_no_buy_when_hist_not_rising_and_no_crossover(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """RSI window has oversold, histogram declining, no MACD crossover. No BUY signal."""
        # Phase 1: 40 stable candles at 200
        # Phase 2: 16 continuous crash candles at -4 each -> RSI oversold
        # Phase 3: 2 more accelerating crash candles -> histogram keeps falling, no crossover
        stable_price = 200.0
        stable_count = 40
        crash_drop = 4.0
        crash_count = 16
        prices: list[float] = [stable_price] * stable_count
        crash_bottom = stable_price - crash_drop * crash_count
        prices.extend([stable_price - crash_drop * (i + 1) for i in range(crash_count)])
        # Accelerating crash -> histogram keeps getting more negative, no crossover
        prices.append(crash_bottom - 6.0)
        prices.append(crash_bottom - 14.0)

        strategy = MomentumStrategy()
        monkeypatch.setattr(strategy, "get_parameters", lambda _seg: _MOMENTUM_PARAMS_V2)
        candles = _make_candles(prices)
        signal = strategy.generate_signal("AAPL", candles, "us_tech")
        assert signal is None, "Expected no BUY: histogram not rising and no MACD crossover"

    def test_no_sell_when_hist_not_falling_and_no_crossover(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """RSI window has overbought, histogram rising, no MACD crossover. No SELL signal."""
        # Phase 1: 40 stable candles at 100
        # Phase 2: 16 continuous rally candles -> RSI overbought
        # Phase 3: 2 more accelerating rally candles -> histogram keeps rising, no crossover
        stable_price = 100.0
        stable_count = 40
        rally_step = 4.0
        rally_count = 16
        prices: list[float] = [stable_price] * stable_count
        rally_top = stable_price + rally_step * rally_count
        prices.extend([stable_price + rally_step * (i + 1) for i in range(rally_count)])
        # Accelerating rally -> histogram keeps getting more positive, no bearish crossover
        prices.append(rally_top + 6.0)
        prices.append(rally_top + 14.0)

        strategy = MomentumStrategy()
        monkeypatch.setattr(strategy, "get_parameters", lambda _seg: _MOMENTUM_PARAMS_V2)
        candles = _make_candles(prices)
        signal = strategy.generate_signal("AAPL", candles, "us_tech")
        assert signal is None, "Expected no SELL: histogram not falling and no MACD crossover"

    def test_confidence_normalized_by_price(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Same price pattern at 2 different scales should produce similar confidence."""
        # Low-price sequence
        stable_price_low = 50.0
        stable_count = 40
        crash_drop_low = 1.0
        crash_count = 16
        level_count = 3
        recovery_step_low = 0.5
        recovery_count = 4

        prices_low: list[float] = [stable_price_low] * stable_count
        crash_bottom_low = stable_price_low - crash_drop_low * crash_count
        prices_low.extend([stable_price_low - crash_drop_low * (i + 1) for i in range(crash_count)])
        prices_low.extend([crash_bottom_low] * level_count)
        prices_low.extend(
            [crash_bottom_low + recovery_step_low * (i + 1) for i in range(recovery_count)]
        )

        # High-price sequence (10x scale)
        scale = 10.0
        stable_price_high = stable_price_low * scale
        crash_drop_high = crash_drop_low * scale
        recovery_step_high = recovery_step_low * scale

        prices_high: list[float] = [stable_price_high] * stable_count
        crash_bottom_high = stable_price_high - crash_drop_high * crash_count
        prices_high.extend(
            [stable_price_high - crash_drop_high * (i + 1) for i in range(crash_count)]
        )
        prices_high.extend([crash_bottom_high] * level_count)
        prices_high.extend(
            [crash_bottom_high + recovery_step_high * (i + 1) for i in range(recovery_count)]
        )

        strategy_low = MomentumStrategy()
        strategy_high = MomentumStrategy()
        monkeypatch.setattr(strategy_low, "get_parameters", lambda _seg: _MOMENTUM_PARAMS_V2)
        monkeypatch.setattr(strategy_high, "get_parameters", lambda _seg: _MOMENTUM_PARAMS_V2)

        candles_low = _make_candles(prices_low)
        candles_high = _make_candles(prices_high)

        signal_low = strategy_low.generate_signal("AAPL", candles_low, "us_tech")
        signal_high = strategy_high.generate_signal("AAPL", candles_high, "us_tech")

        assert signal_low is not None, "Expected BUY signal for low-price sequence"
        assert signal_high is not None, "Expected BUY signal for high-price sequence"

        assert abs(signal_low.confidence - signal_high.confidence) < CONFIDENCE_TOLERANCE, (
            f"Confidence should be similar regardless of price scale: "
            f"low={signal_low.confidence:.4f}, high={signal_high.confidence:.4f}"
        )


class TestMomentumSignalFilters:
    """Tests for trend filter, signal state machine, ADX filter, and volume filter."""

    # ---------- helper: BUY-producing price series ----------
    @staticmethod
    def _buy_prices() -> list[float]:
        """Return price series that triggers a BUY signal (crash + recovery)."""
        stable_price = 200.0
        prices: list[float] = [stable_price] * 40
        crash_bottom = stable_price - 4.0 * 16
        prices.extend([stable_price - 4.0 * (i + 1) for i in range(16)])
        prices.extend([crash_bottom] * 3)
        prices.extend([crash_bottom + 2.0 * (i + 1) for i in range(4)])
        return prices

    # ---------- helper: SELL-producing price series ----------
    @staticmethod
    def _sell_prices() -> list[float]:
        """Return price series that triggers a SELL signal (rally + decline)."""
        stable_price = 100.0
        rally_top = stable_price + 4.0 * 16
        prices: list[float] = [stable_price] * 40
        prices.extend([stable_price + 4.0 * (i + 1) for i in range(16)])
        prices.extend([rally_top] * 3)
        prices.extend([rally_top - 2.0 * (i + 1) for i in range(7)])
        return prices

    # ---------- 1. Trend filter: suppress sell in uptrend ----------
    def test_trend_filter_suppresses_sell_in_uptrend(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Price well above 50 SMA, overbought RSI -> SELL suppressed."""
        params = {
            **_MOMENTUM_PARAMS_V3,
            "trend_filter": True,
            "trend_sma_period": 50,
            "trend_sma_buffer_pct": 2.0,
        }
        # Use the sell-signal price series. The last close should be well above
        # the SMA of the whole series because the rally pushed it high.
        prices = self._sell_prices()
        strategy = MomentumStrategy()
        monkeypatch.setattr(strategy, "get_parameters", lambda _seg: params)
        candles = _make_candles(prices)

        # Without trend filter, this produces a SELL
        params_no_filter = {**params, "trend_filter": False}
        strategy_no_filter = MomentumStrategy()
        monkeypatch.setattr(strategy_no_filter, "get_parameters", lambda _seg: params_no_filter)
        sell_signal = strategy_no_filter.generate_signal("AAPL", candles, "us_tech")
        assert sell_signal is not None, "Baseline: expected SELL without trend filter"
        assert sell_signal.direction == SignalDirection.SELL

        # With trend filter enabled, SELL should be suppressed (price above SMA)
        signal = strategy.generate_signal("AAPL", candles, "us_tech")
        assert signal is None, "Expected SELL suppressed: price above SMA in uptrend"

    # ---------- 2. Trend filter: suppress buy in downtrend ----------
    def test_trend_filter_suppresses_buy_in_downtrend(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Price well below 50 SMA, oversold RSI -> BUY suppressed."""
        params = {
            **_MOMENTUM_PARAMS_V3,
            "trend_filter": True,
            "trend_sma_period": 50,
            "trend_sma_buffer_pct": 2.0,
        }
        prices = self._buy_prices()
        strategy = MomentumStrategy()
        monkeypatch.setattr(strategy, "get_parameters", lambda _seg: params)
        candles = _make_candles(prices)

        # Baseline: confirm BUY without filter
        params_no_filter = {**params, "trend_filter": False}
        strategy_no_filter = MomentumStrategy()
        monkeypatch.setattr(strategy_no_filter, "get_parameters", lambda _seg: params_no_filter)
        buy_signal = strategy_no_filter.generate_signal("AAPL", candles, "us_tech")
        assert buy_signal is not None, "Baseline: expected BUY without trend filter"
        assert buy_signal.direction == SignalDirection.BUY

        # With trend filter, BUY suppressed (price well below SMA)
        signal = strategy.generate_signal("AAPL", candles, "us_tech")
        assert signal is None, "Expected BUY suppressed: price below SMA in downtrend"

    # ---------- 3. Trend filter: allows signal in buffer zone ----------
    def test_trend_filter_allows_signal_in_buffer_zone(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Price near SMA (within buffer) -> signal passes through."""
        # Use a very large buffer (100%) so close is always within buffer
        params = {
            **_MOMENTUM_PARAMS_V3,
            "trend_filter": True,
            "trend_sma_period": 50,
            "trend_sma_buffer_pct": 100.0,
        }
        prices = self._buy_prices()
        strategy = MomentumStrategy()
        monkeypatch.setattr(strategy, "get_parameters", lambda _seg: params)
        candles = _make_candles(prices)
        signal = strategy.generate_signal("AAPL", candles, "us_tech")
        assert signal is not None, "Expected signal to pass through with large buffer"

    # ---------- 4. Signal state machine: suppress duplicate ----------
    def test_signal_state_machine_suppresses_duplicate(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two consecutive BUY signals -> only first emitted."""
        params = {**_MOMENTUM_PARAMS_V3, "neutral_reset_bars": 20}
        prices = self._buy_prices()
        strategy = MomentumStrategy()
        monkeypatch.setattr(strategy, "get_parameters", lambda _seg: params)
        candles = _make_candles(prices)

        sig1 = strategy.generate_signal("AAPL", candles, "us_tech")
        assert sig1 is not None, "First BUY should be emitted"
        assert sig1.direction == SignalDirection.BUY

        sig2 = strategy.generate_signal("AAPL", candles, "us_tech")
        assert sig2 is None, "Duplicate BUY should be suppressed"

    # ---------- 5. Signal state machine: allows direction change ----------
    def test_signal_state_machine_allows_direction_change(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """BUY then SELL -> both emitted."""
        params = {**_MOMENTUM_PARAMS_V3, "neutral_reset_bars": 20}
        buy_prices = self._buy_prices()
        sell_prices = self._sell_prices()
        strategy = MomentumStrategy()
        monkeypatch.setattr(strategy, "get_parameters", lambda _seg: params)

        buy_candles = _make_candles(buy_prices)
        sig_buy = strategy.generate_signal("AAPL", buy_candles, "us_tech")
        assert sig_buy is not None and sig_buy.direction == SignalDirection.BUY

        sell_candles = _make_candles(sell_prices)
        sig_sell = strategy.generate_signal("AAPL", sell_candles, "us_tech")
        assert sig_sell is not None, "SELL after BUY should be emitted (direction change)"
        assert sig_sell.direction == SignalDirection.SELL

    # ---------- 6. Signal state machine: resets after neutral bars ----------
    def test_signal_state_machine_resets_after_neutral_bars(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """After 20 neutral bars, same direction re-emitted."""
        params = {**_MOMENTUM_PARAMS_V3, "neutral_reset_bars": 20}
        prices = self._buy_prices()
        strategy = MomentumStrategy()
        monkeypatch.setattr(strategy, "get_parameters", lambda _seg: params)
        candles = _make_candles(prices)

        sig1 = strategy.generate_signal("AAPL", candles, "us_tech")
        assert sig1 is not None and sig1.direction == SignalDirection.BUY

        # Simulate 20 ticks with flat prices (no signal) -> state resets
        flat_prices: list[float] = [150.0] * (MIN_CANDLES_FOR_INDICATORS + 5)
        flat_candles = _make_candles(flat_prices)
        for _ in range(NEUTRAL_RESET_BARS):
            strategy.generate_signal("AAPL", flat_candles, "us_tech")

        # Now same direction should be re-emitted
        sig2 = strategy.generate_signal("AAPL", candles, "us_tech")
        assert sig2 is not None, "BUY should re-emit after neutral_reset_bars ticks"
        assert sig2.direction == SignalDirection.BUY

    # ---------- 7. ADX filter: suppresses low ADX ----------
    def test_adx_filter_suppresses_low_adx(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ADX < 25 -> signal suppressed."""
        from finalayze.strategies.momentum import _Indicators

        params = {
            **_MOMENTUM_PARAMS_V3,
            "adx_filter": True,
            "adx_period": 14,
            "adx_threshold": 25,
        }
        prices = self._buy_prices()
        strategy = MomentumStrategy()
        monkeypatch.setattr(strategy, "get_parameters", lambda _seg: params)
        candles = _make_candles(prices)

        # First verify a BUY signal would fire without ADX filter
        params_no_adx = {**params, "adx_filter": False}
        strategy_no_adx = MomentumStrategy()
        monkeypatch.setattr(strategy_no_adx, "get_parameters", lambda _seg: params_no_adx)
        baseline = strategy_no_adx.generate_signal("AAPL", candles, "us_tech")
        assert baseline is not None, "Baseline: expected BUY without ADX filter"

        # Now monkeypatch _compute_indicators to force low ADX
        original_compute = strategy._compute_indicators

        def _compute_with_low_adx(candles: list[Candle], segment_id: str) -> _Indicators | None:
            ind = original_compute(candles, segment_id)
            if ind is None:
                return None
            return _Indicators(
                current_rsi=ind.current_rsi,
                rsi_window=ind.rsi_window,
                current_hist=ind.current_hist,
                prev_hist=ind.prev_hist,
                hist_window=ind.hist_window,
                macd_line=ind.macd_line,
                signal_line=ind.signal_line,
                macd_crossover_buy=ind.macd_crossover_buy,
                macd_crossover_sell=ind.macd_crossover_sell,
                avg_hist_range=ind.avg_hist_range,
                current_close=ind.current_close,
                min_confidence=ind.min_confidence,
                current_sma=ind.current_sma,
                current_adx=10.0,  # Force low ADX
                volume_ratio=ind.volume_ratio,
            )

        monkeypatch.setattr(strategy, "_compute_indicators", _compute_with_low_adx)
        signal = strategy.generate_signal("AAPL", candles, "us_tech")
        assert signal is None, "Expected signal suppressed: ADX too low (range-bound)"

    # ---------- 8. ADX filter: allows high ADX ----------
    def test_adx_filter_allows_high_adx(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ADX > 25 -> signal passes."""
        params = {
            **_MOMENTUM_PARAMS_V3,
            "adx_filter": True,
            "adx_period": 14,
            "adx_threshold": 25,
        }
        # Use buy prices with wide high-low spread -> high ADX
        prices = self._buy_prices()
        highs = [p + 10.0 for p in prices]
        lows = [p - 10.0 for p in prices]
        strategy = MomentumStrategy()
        monkeypatch.setattr(strategy, "get_parameters", lambda _seg: params)
        candles = _make_candles(prices, highs=highs, lows=lows)
        signal = strategy.generate_signal("AAPL", candles, "us_tech")
        assert signal is not None, "Expected signal to pass: ADX should be high"
        assert signal.direction == SignalDirection.BUY

    # ---------- 9. Volume filter: suppresses low volume ----------
    def test_volume_filter_suppresses_low_volume(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Volume < 1x SMA -> suppressed."""
        params = {
            **_MOMENTUM_PARAMS_V3,
            "volume_filter": True,
            "volume_sma_period": 20,
            "volume_min_ratio": 1.0,
        }
        prices = self._buy_prices()
        n = len(prices)
        # High volume for first candles, then low volume at the end
        volumes = [1_000_000] * (n - 1) + [100]
        strategy = MomentumStrategy()
        monkeypatch.setattr(strategy, "get_parameters", lambda _seg: params)
        candles = _make_candles(prices, volumes=volumes)
        signal = strategy.generate_signal("AAPL", candles, "us_tech")
        assert signal is None, "Expected signal suppressed: volume too low"

    # ---------- 10. Volume filter: allows high volume ----------
    def test_volume_filter_allows_high_volume(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Volume >= 1x SMA -> passes."""
        params = {
            **_MOMENTUM_PARAMS_V3,
            "volume_filter": True,
            "volume_sma_period": 20,
            "volume_min_ratio": 1.0,
        }
        prices = self._buy_prices()
        n = len(prices)
        # All candles have same volume -> ratio = 1.0 exactly
        volumes = [1_000_000] * n
        strategy = MomentumStrategy()
        monkeypatch.setattr(strategy, "get_parameters", lambda _seg: params)
        candles = _make_candles(prices, volumes=volumes)
        signal = strategy.generate_signal("AAPL", candles, "us_tech")
        assert signal is not None, "Expected signal to pass: volume ratio >= 1.0"
        assert signal.direction == SignalDirection.BUY

    # ---------- 11. Multi-bar MACD histogram check ----------
    def test_multibar_hist_rising_triggers_buy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Histogram improved over last 3 bars even if prev bar was higher."""
        params = {**_MOMENTUM_PARAMS_V3, "macd_hist_lookback": 3}
        prices = self._buy_prices()
        strategy = MomentumStrategy()
        monkeypatch.setattr(strategy, "get_parameters", lambda _seg: params)
        candles = _make_candles(prices)
        signal = strategy.generate_signal("AAPL", candles, "us_tech")
        assert signal is not None, "Expected BUY with multi-bar histogram check"
        assert signal.direction == SignalDirection.BUY

    # ---------- 12. MACD crossover triggers buy ----------
    def test_macd_crossover_triggers_buy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """MACD line crossing above signal line triggers BUY even if histogram isn't rising."""
        from finalayze.strategies.momentum import _Indicators

        params = {**_MOMENTUM_PARAMS_V3}
        prices = self._buy_prices()
        strategy = MomentumStrategy()
        monkeypatch.setattr(strategy, "get_parameters", lambda _seg: params)
        candles = _make_candles(prices)

        # Force crossover buy flag via monkeypatching _compute_indicators
        original_compute = strategy._compute_indicators

        def _compute_with_crossover(candles: list[Candle], segment_id: str) -> _Indicators | None:
            ind = original_compute(candles, segment_id)
            if ind is None:
                return None
            return _Indicators(
                current_rsi=ind.current_rsi,
                rsi_window=ind.rsi_window,
                current_hist=ind.current_hist,
                prev_hist=ind.current_hist + 1.0,  # hist NOT rising (prev > current)
                hist_window=[ind.current_hist + 1.0, ind.current_hist + 0.5, ind.current_hist],
                macd_line=ind.macd_line,
                signal_line=ind.signal_line,
                macd_crossover_buy=True,  # Force crossover
                macd_crossover_sell=False,
                avg_hist_range=ind.avg_hist_range,
                current_close=ind.current_close,
                min_confidence=ind.min_confidence,
                current_sma=ind.current_sma,
                current_adx=ind.current_adx,
                volume_ratio=ind.volume_ratio,
            )

        monkeypatch.setattr(strategy, "_compute_indicators", _compute_with_crossover)
        signal = strategy.generate_signal("AAPL", candles, "us_tech")
        assert signal is not None, "Expected BUY via MACD crossover"
        assert signal.direction == SignalDirection.BUY

    # ---------- 13. Improved confidence uses histogram strength ----------
    def test_confidence_includes_hist_strength(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Confidence should factor in histogram strength (0.5 + rsi*0.3 + hist*0.2)."""
        params = {**_MOMENTUM_PARAMS_V3, "min_confidence": 0.0}
        prices = self._buy_prices()
        strategy = MomentumStrategy()
        monkeypatch.setattr(strategy, "get_parameters", lambda _seg: params)
        candles = _make_candles(prices)
        signal = strategy.generate_signal("AAPL", candles, "us_tech")
        assert signal is not None
        # With the new formula, confidence should be between 0.5 and 1.0
        assert signal.confidence >= 0.5
        assert signal.confidence <= 1.0


class TestMomentumPerSegmentState:
    """Tests for per-segment _SignalState isolation (6B.1)."""

    @staticmethod
    def _buy_prices() -> list[float]:
        stable_price = 200.0
        prices: list[float] = [stable_price] * 40
        crash_bottom = stable_price - 4.0 * 16
        prices.extend([stable_price - 4.0 * (i + 1) for i in range(16)])
        prices.extend([crash_bottom] * 3)
        prices.extend([crash_bottom + 2.0 * (i + 1) for i in range(4)])
        return prices

    def test_signal_state_isolated_per_segment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """BUY for AAPL in us_tech should NOT suppress BUY for AAPL in us_broad."""
        params = {**_MOMENTUM_PARAMS_V3, "neutral_reset_bars": 20}
        prices = self._buy_prices()
        strategy = MomentumStrategy()
        monkeypatch.setattr(strategy, "get_parameters", lambda _seg: params)
        candles = _make_candles(prices)

        sig1 = strategy.generate_signal("AAPL", candles, "us_tech")
        assert sig1 is not None, "Expected BUY in us_tech"
        assert sig1.direction == SignalDirection.BUY

        sig2 = strategy.generate_signal("AAPL", candles, "us_broad")
        assert sig2 is not None, "Expected BUY in us_broad (state isolated per segment)"
        assert sig2.direction == SignalDirection.BUY

    def test_signal_state_duplicate_suppressed_within_segment(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Same (symbol, segment_id) still suppresses duplicates."""
        params = {**_MOMENTUM_PARAMS_V3, "neutral_reset_bars": 20}
        prices = self._buy_prices()
        strategy = MomentumStrategy()
        monkeypatch.setattr(strategy, "get_parameters", lambda _seg: params)
        candles = _make_candles(prices)

        sig1 = strategy.generate_signal("AAPL", candles, "us_tech")
        assert sig1 is not None and sig1.direction == SignalDirection.BUY

        sig2 = strategy.generate_signal("AAPL", candles, "us_tech")
        assert sig2 is None, "Duplicate BUY in same segment should be suppressed"

    def test_neutral_reset_bars_per_segment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Each segment respects its own neutral_reset_bars independently."""
        prices = self._buy_prices()
        strategy = MomentumStrategy()
        params_a = {**_MOMENTUM_PARAMS_V3, "neutral_reset_bars": 2}
        params_b = {**_MOMENTUM_PARAMS_V3, "neutral_reset_bars": 100}

        def get_params(seg: str) -> dict[str, object]:
            return params_a if seg == "seg_a" else params_b

        monkeypatch.setattr(strategy, "get_parameters", get_params)
        candles = _make_candles(prices)

        # Emit BUY in both segments
        strategy.generate_signal("AAPL", candles, "seg_a")
        strategy.generate_signal("AAPL", candles, "seg_b")

        # Tick both segments with flat candles (no signal) -> only seg_a resets
        flat_candles = _make_candles([150.0] * (MIN_CANDLES_FOR_INDICATORS + 5))
        for _ in range(2):
            strategy.generate_signal("AAPL", flat_candles, "seg_a")
            strategy.generate_signal("AAPL", flat_candles, "seg_b")

        # seg_a should have reset (2 bars), so re-emit BUY
        sig_a = strategy.generate_signal("AAPL", candles, "seg_a")
        assert sig_a is not None, "seg_a should have reset after 2 neutral bars"

        # seg_b should NOT have reset (needs 100 bars)
        sig_b = strategy.generate_signal("AAPL", candles, "seg_b")
        assert sig_b is None, "seg_b should NOT have reset (100 bar threshold not reached)"

    def test_signal_state_no_mutation_of_neutral_reset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Calling generate_signal for segment A does not change segment B's reset bars."""
        prices = self._buy_prices()
        strategy = MomentumStrategy()
        params_a = {**_MOMENTUM_PARAMS_V3, "neutral_reset_bars": 5}
        params_b = {**_MOMENTUM_PARAMS_V3, "neutral_reset_bars": 50}

        def get_params(seg: str) -> dict[str, object]:
            return params_a if seg == "seg_a" else params_b

        monkeypatch.setattr(strategy, "get_parameters", get_params)
        candles = _make_candles(prices)

        # Initialize both segment states
        strategy.generate_signal("AAPL", candles, "seg_b")
        strategy.generate_signal("AAPL", candles, "seg_a")

        # seg_b state should still have neutral_reset_bars=50
        state_b = strategy._signal_states["seg_b"]
        assert state_b._neutral_reset_bars == 50, (
            "seg_b neutral_reset_bars should not be mutated by seg_a call"
        )


_MR_PARAMS: dict[str, object] = {
    "bb_period": 20,
    "bb_std_dev": 2.0,
    "min_confidence": 0.0,
    "squeeze_threshold": 0.0,
    "min_band_distance_pct": 0.0,
    "rsi_oversold_mr": 100,
    "rsi_overbought_mr": 0,
    "rsi_period": 14,
    "exit_at_mean": False,
}


def _make_mr_candles(
    prices: list[float],
    symbol: str = "AAPL",
) -> list[Candle]:
    """Build candles for mean reversion tests."""
    candles = []
    base = datetime(2024, 1, 1, 14, 30, tzinfo=UTC)
    for i, price in enumerate(prices):
        p = Decimal(str(price))
        candles.append(
            Candle(
                symbol=symbol,
                market_id="us",
                timeframe="1d",
                timestamp=base + timedelta(days=i),
                open=p,
                high=p + Decimal(1),
                low=p - Decimal(1),
                close=p,
                volume=1_000_000,
            )
        )
    return candles


class TestMeanReversionExitAtMean:
    """Tests for exit-at-mean signal (6B.3)."""

    @staticmethod
    def _prices_below_lower_then_mean() -> list[float]:
        """Prices: stable at 100, drop below lower BB, then return to mean."""
        # 21 stable candles for BB(20) computation, then crash, then recover
        prices: list[float] = [100.0] * 21
        # Drop below lower band (std dev of flat data is 0, so add some noise first)
        # Use small oscillation to create non-zero std dev
        prices = [100.0 + (i % 3 - 1) * 0.5 for i in range(21)]
        # Sharp drop below lower band
        prices.append(90.0)
        # Return to mean
        prices.append(100.0)
        return prices

    def test_exit_at_mean_generates_sell_after_buy(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Price below lower BB (BUY), then returns to mean -> exit SELL."""
        params = {**_MR_PARAMS, "exit_at_mean": True}
        strategy = MeanReversionStrategy()
        monkeypatch.setattr(strategy, "get_parameters", lambda _seg: params)

        prices = self._prices_below_lower_then_mean()
        # First call: price at 90 (below lower) -> BUY
        candles_drop = _make_mr_candles(prices[:-1])
        sig1 = strategy.generate_signal("AAPL", candles_drop, "us_tech")
        assert sig1 is not None, "Expected BUY signal when price below lower BB"
        assert sig1.direction == SignalDirection.BUY

        # Second call: price at 100 (back to mean) -> exit SELL
        candles_mean = _make_mr_candles(prices)
        sig2 = strategy.generate_signal("AAPL", candles_mean, "us_tech")
        assert sig2 is not None, "Expected SELL exit signal when price returns to mean"
        assert sig2.direction == SignalDirection.SELL
        assert sig2.features.get("exit_at_mean") == 1.0

    def test_exit_at_mean_generates_buy_after_sell(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Price above upper BB (SELL), then returns to mean -> exit BUY."""
        params = {**_MR_PARAMS, "exit_at_mean": True}
        strategy = MeanReversionStrategy()
        monkeypatch.setattr(strategy, "get_parameters", lambda _seg: params)

        # Stable prices with small oscillation, then spike up, then return
        prices = [100.0 + (i % 3 - 1) * 0.5 for i in range(21)]
        prices.append(110.0)  # above upper BB -> SELL
        prices.append(100.0)  # back to mean -> exit BUY

        candles_spike = _make_mr_candles(prices[:-1])
        sig1 = strategy.generate_signal("AAPL", candles_spike, "us_tech")
        assert sig1 is not None, "Expected SELL signal when price above upper BB"
        assert sig1.direction == SignalDirection.SELL

        candles_mean = _make_mr_candles(prices)
        sig2 = strategy.generate_signal("AAPL", candles_mean, "us_tech")
        assert sig2 is not None, "Expected BUY exit signal when price returns to mean"
        assert sig2.direction == SignalDirection.BUY

    def test_exit_at_mean_disabled_by_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without exit_at_mean, no exit signal on mean return."""
        params = {**_MR_PARAMS, "exit_at_mean": False}
        strategy = MeanReversionStrategy()
        monkeypatch.setattr(strategy, "get_parameters", lambda _seg: params)

        prices = self._prices_below_lower_then_mean()
        candles_drop = _make_mr_candles(prices[:-1])
        sig1 = strategy.generate_signal("AAPL", candles_drop, "us_tech")
        assert sig1 is not None and sig1.direction == SignalDirection.BUY

        candles_mean = _make_mr_candles(prices)
        sig2 = strategy.generate_signal("AAPL", candles_mean, "us_tech")
        assert sig2 is None, "No exit signal when exit_at_mean is False"

    def test_exit_at_mean_no_signal_without_active_position(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Price inside bands with no prior active signal -> no exit signal."""
        params = {**_MR_PARAMS, "exit_at_mean": True}
        strategy = MeanReversionStrategy()
        monkeypatch.setattr(strategy, "get_parameters", lambda _seg: params)

        prices = [100.0 + (i % 3 - 1) * 0.5 for i in range(22)]
        candles = _make_mr_candles(prices)
        sig = strategy.generate_signal("AAPL", candles, "us_tech")
        assert sig is None, "No signal when price is inside bands and no prior active"
