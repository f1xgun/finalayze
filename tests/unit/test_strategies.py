"""Unit tests for trading strategies."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest
import yaml

from finalayze.core.schemas import Candle, SignalDirection
from finalayze.strategies.base import BaseStrategy
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

    def test_no_buy_when_hist_not_rising(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """RSI window has oversold but current_hist <= prev_hist. No BUY signal."""
        # Phase 1: 40 stable candles at 200
        # Phase 2: 14 crash candles at -4 each -> RSI oversold
        # Phase 3: 2 candles slight recovery then steeper crash -> histogram drops
        stable_price = 200.0
        stable_count = 40
        crash_drop = 4.0
        crash_count = 14
        prices: list[float] = [stable_price] * stable_count
        crash_bottom = stable_price - crash_drop * crash_count
        prices.extend([stable_price - crash_drop * (i + 1) for i in range(crash_count)])
        # Small bounce then sharp drop -> histogram falls on last bar
        prices.append(crash_bottom + 2.0)
        prices.append(crash_bottom - 8.0)

        strategy = MomentumStrategy()
        monkeypatch.setattr(strategy, "get_parameters", lambda _seg: _MOMENTUM_PARAMS_V2)
        candles = _make_candles(prices)
        signal = strategy.generate_signal("AAPL", candles, "us_tech")
        assert signal is None, "Expected no BUY: histogram is not rising"

    def test_no_sell_when_hist_not_falling(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """RSI window has overbought but current_hist >= prev_hist. No SELL signal."""
        # Phase 1: 40 stable candles at 100
        # Phase 2: 14 rally candles -> RSI overbought
        # Phase 3: small dip then sharp rally -> histogram rises on last bar
        stable_price = 100.0
        stable_count = 40
        rally_step = 4.0
        rally_count = 14
        prices: list[float] = [stable_price] * stable_count
        rally_top = stable_price + rally_step * rally_count
        prices.extend([stable_price + rally_step * (i + 1) for i in range(rally_count)])
        # Small dip then sharp rise -> histogram rises on last bar
        prices.append(rally_top - 2.0)
        prices.append(rally_top + 8.0)

        strategy = MomentumStrategy()
        monkeypatch.setattr(strategy, "get_parameters", lambda _seg: _MOMENTUM_PARAMS_V2)
        candles = _make_candles(prices)
        signal = strategy.generate_signal("AAPL", candles, "us_tech")
        assert signal is None, "Expected no SELL: histogram is not falling"

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
