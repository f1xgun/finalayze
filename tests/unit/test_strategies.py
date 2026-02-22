"""Unit tests for trading strategies."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from finalayze.core.schemas import Candle, SignalDirection
from finalayze.strategies.base import BaseStrategy
from finalayze.strategies.momentum import MomentumStrategy

MIN_CANDLES_FOR_INDICATORS = 35
RSI_PERIOD = 14
MIN_SUPPORTED_SEGMENTS = 2


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


class TestBaseStrategy:
    def test_is_abstract(self) -> None:
        with pytest.raises(TypeError):
            BaseStrategy()  # type: ignore[abstract]


class TestMomentumStrategy:
    def test_name(self) -> None:
        assert MomentumStrategy().name == "momentum"

    def test_supported_segments(self) -> None:
        supported = MomentumStrategy().supported_segments()
        assert "us_tech" in supported
        assert "us_broad" in supported
        assert "nonexistent_segment" not in supported  # should not be there without preset
        assert len(supported) >= MIN_SUPPORTED_SEGMENTS

    def test_get_parameters_us_tech(self) -> None:
        params = MomentumStrategy().get_parameters("us_tech")
        assert params["rsi_period"] == RSI_PERIOD

    def test_insufficient_data_returns_none(self) -> None:
        short = _make_candles([100.0] * 5)
        assert MomentumStrategy().generate_signal("AAPL", short, "us_tech") is None

    def test_hold_when_no_signal(self) -> None:
        flat = _make_candles([150.0] * (MIN_CANDLES_FOR_INDICATORS + 5))
        assert MomentumStrategy().generate_signal("AAPL", flat, "us_tech") is None

    def test_buy_signal_on_oversold_rsi(self) -> None:
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
        candles = _make_candles(prices)
        signal = MomentumStrategy().generate_signal("AAPL", candles, "us_tech")
        assert signal is not None, (
            "Expected a BUY signal after a crash+level+recovery pattern with RSI < 30 "
            "and MACD histogram cross above zero"
        )
        assert signal.direction == SignalDirection.BUY
        assert signal.strategy_name == "momentum"
        assert 0.0 <= signal.confidence <= 1.0
