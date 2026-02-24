"""Unit tests for trading strategies."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest
import yaml

from finalayze.core.schemas import Candle, SignalDirection
from finalayze.strategies.base import BaseStrategy
from finalayze.strategies.momentum import MomentumStrategy

MIN_CANDLES_FOR_INDICATORS = 35
RSI_PERIOD = 14
MIN_SUPPORTED_SEGMENTS = 2

_MOMENTUM_PARAMS: dict[str, object] = {
    "rsi_period": 14,
    "rsi_oversold": 30,
    "rsi_overbought": 70,
    "macd_fast": 12,
    "macd_slow": 26,
    "min_confidence": 0.6,
}


def _make_candles(prices: list[float], start_year: int = 2024) -> list[Candle]:
    candles = []
    base = datetime(start_year, 1, 1, 14, 30, tzinfo=UTC)
    for i, price in enumerate(prices):
        p = Decimal(str(price))
        candles.append(
            Candle(
                symbol="AAPL",
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


def _write_momentum_preset(preset_dir: Path, segment_id: str) -> None:
    """Write a minimal momentum-enabled YAML preset to a temp directory."""
    data = {
        "segment_id": segment_id,
        "strategies": {
            "momentum": {
                "enabled": True,
                "weight": 0.4,
                "params": dict(_MOMENTUM_PARAMS),
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
        monkeypatch.setattr(strategy, "get_parameters", lambda _seg: _MOMENTUM_PARAMS)
        short = _make_candles([100.0] * 5)
        assert strategy.generate_signal("AAPL", short, "us_tech") is None

    def test_hold_when_no_signal(self, monkeypatch: pytest.MonkeyPatch) -> None:
        strategy = MomentumStrategy()
        monkeypatch.setattr(strategy, "get_parameters", lambda _seg: _MOMENTUM_PARAMS)
        flat = _make_candles([150.0] * (MIN_CANDLES_FOR_INDICATORS + 5))
        assert strategy.generate_signal("AAPL", flat, "us_tech") is None

    def test_buy_signal_on_oversold_rsi(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Build a deterministic sequence that forces RSI < 30 and MACD histogram cross above zero.
        # Verified manually: this pattern produces RSI=23.7, hist crosses 0 at the last candle.
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
        monkeypatch.setattr(strategy, "get_parameters", lambda _seg: _MOMENTUM_PARAMS)
        candles = _make_candles(prices)
        signal = strategy.generate_signal("AAPL", candles, "us_tech")
        assert signal is not None, (
            "Expected a BUY signal after a crash+level+recovery pattern with RSI < 30 "
            "and MACD histogram cross above zero"
        )
        assert signal.direction == SignalDirection.BUY
        assert signal.strategy_name == "momentum"
        assert 0.0 <= signal.confidence <= 1.0
