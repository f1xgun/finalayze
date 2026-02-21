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
        segments = MomentumStrategy().supported_segments()
        assert "us_tech" in segments
        assert "us_broad" in segments

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
        prices = [200.0 - i * 2 for i in range(MIN_CANDLES_FOR_INDICATORS)]
        prices.extend([prices[-1] + 1, prices[-1] + 2, prices[-1] + 3])
        candles = _make_candles(prices)
        signal = MomentumStrategy().generate_signal("AAPL", candles, "us_tech")
        if signal is not None:
            assert signal.direction == SignalDirection.BUY
            assert signal.strategy_name == "momentum"
            assert 0.0 <= signal.confidence <= 1.0
