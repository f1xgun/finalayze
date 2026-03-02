"""Unit tests for RSI(2) Connors strategy."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING

from finalayze.core.schemas import Candle, SignalDirection

if TYPE_CHECKING:
    import pytest
from finalayze.strategies.rsi2_connors import RSI2ConnorsStrategy

# Strategy parameters passed directly (no YAML I/O in unit tests)
_RSI2_PARAMS: dict[str, object] = {
    "rsi_period": 2,
    "rsi_buy_threshold": 10.0,
    "rsi_sell_threshold": 90.0,
    "sma_trend_period": 200,
    "sma_exit_period": 5,
    "min_confidence": 0.35,
    "enabled": True,
}

# Number of candles needed for SMA(200) to be valid
_SMA_WARMUP = 201


def _make_candles(
    prices: list[float],
    symbol: str = "AAPL",
    market_id: str = "us",
) -> list[Candle]:
    """Build candles from a list of close prices."""
    candles: list[Candle] = []
    base = datetime(2024, 1, 1, 14, 30, tzinfo=UTC)
    for i, price in enumerate(prices):
        p = Decimal(str(price))
        candles.append(
            Candle(
                symbol=symbol,
                market_id=market_id,
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


def _prices_with_rsi2_buy(base_price: float = 150.0) -> list[float]:
    """Generate prices where RSI(2) drops below 10 and price > SMA(200).

    Strategy: start low, trend up so SMA(200) is well below current price,
    then 2 small drops to push RSI(2) near 0 while price stays above SMA(200).
    """
    # Trend upward: SMA(200) will lag behind, keeping it well below current price
    start = base_price - 40.0
    step = 40.0 / _SMA_WARMUP
    prices = [start + step * i for i in range(_SMA_WARMUP)]
    # Two consecutive drops (small enough to stay above SMA(200))
    last = prices[-1]
    prices.append(last - 1.0)
    prices.append(last - 2.0)
    return prices


def _prices_with_rsi2_sell(base_price: float = 150.0) -> list[float]:
    """Generate prices where RSI(2) rises above 90 and price < SMA(200).

    Strategy: 200 candles trending down so SMA(200) is above current price,
    then 2 sharp rises to push RSI(2) near 100.
    """
    # Start high, trend down so SMA(200) remains above the last price
    start_price = base_price + 50.0
    end_price = base_price - 20.0
    step = (start_price - end_price) / _SMA_WARMUP
    prices = [start_price - step * i for i in range(_SMA_WARMUP)]
    # Two consecutive rises -> RSI(2) near 100
    last = prices[-1]
    prices.append(last + 5.0)
    prices.append(last + 10.0)
    return prices


class TestRSI2ConnorsStrategy:
    """Tests for RSI2ConnorsStrategy."""

    def test_name(self) -> None:
        strategy = RSI2ConnorsStrategy()
        assert strategy.name == "rsi2_connors"

    def test_buy_signal_when_rsi2_below_threshold_and_above_sma200(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """RSI(2) < 10 AND price > SMA(200) -> BUY signal."""
        strategy = RSI2ConnorsStrategy()
        monkeypatch.setattr(strategy, "get_parameters", lambda _seg: _RSI2_PARAMS)

        prices = _prices_with_rsi2_buy()
        candles = _make_candles(prices)
        signal = strategy.generate_signal("AAPL", candles, "us_tech")

        assert signal is not None, "Expected BUY when RSI(2) < 10 and price > SMA(200)"
        assert signal.direction == SignalDirection.BUY
        assert signal.strategy_name == "rsi2_connors"
        assert 0.0 <= signal.confidence <= 1.0

    def test_sell_signal_when_rsi2_above_threshold_and_below_sma200(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """RSI(2) > 90 AND price < SMA(200) -> SELL signal."""
        strategy = RSI2ConnorsStrategy()
        monkeypatch.setattr(strategy, "get_parameters", lambda _seg: _RSI2_PARAMS)

        prices = _prices_with_rsi2_sell()
        candles = _make_candles(prices)
        signal = strategy.generate_signal("AAPL", candles, "us_tech")

        assert signal is not None, "Expected SELL when RSI(2) > 90 and price < SMA(200)"
        assert signal.direction == SignalDirection.SELL
        assert signal.strategy_name == "rsi2_connors"
        assert 0.0 <= signal.confidence <= 1.0

    def test_sma200_blocks_buy_when_price_below(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """RSI(2) < 10 but price < SMA(200) -> no BUY signal."""
        strategy = RSI2ConnorsStrategy()
        monkeypatch.setattr(strategy, "get_parameters", lambda _seg: _RSI2_PARAMS)

        # Trend down so SMA(200) is above price, then crash further for low RSI(2)
        start_price = 200.0
        end_price = 100.0
        step = (start_price - end_price) / _SMA_WARMUP
        prices = [start_price - step * i for i in range(_SMA_WARMUP)]
        # Two more drops -> RSI(2) near 0, but price well below SMA(200)
        last = prices[-1]
        prices.append(last - 5.0)
        prices.append(last - 10.0)

        candles = _make_candles(prices)
        signal = strategy.generate_signal("AAPL", candles, "us_tech")
        assert signal is None, "BUY should be blocked when price < SMA(200)"

    def test_sma200_blocks_sell_when_price_above(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """RSI(2) > 90 but price > SMA(200) -> no SELL signal."""
        strategy = RSI2ConnorsStrategy()
        monkeypatch.setattr(strategy, "get_parameters", lambda _seg: _RSI2_PARAMS)

        # Stable prices so SMA(200) ~ base_price, then 2 sharp rises -> RSI(2) near 100
        # but price remains above SMA(200)
        base_price = 150.0
        prices = [base_price] * _SMA_WARMUP
        prices.append(base_price + 5.0)
        prices.append(base_price + 10.0)

        candles = _make_candles(prices)
        signal = strategy.generate_signal("AAPL", candles, "us_tech")
        assert signal is None, "SELL should be blocked when price > SMA(200)"

    def test_confidence_scaling_rsi2_zero_gives_max(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """RSI(2) = 0 should give confidence near 1.0."""
        strategy = RSI2ConnorsStrategy()
        params = {**_RSI2_PARAMS, "min_confidence": 0.0}
        monkeypatch.setattr(strategy, "get_parameters", lambda _seg: params)

        # Trend upward so SMA(200) is well below, then consecutive large drops
        # that still stay above SMA(200) but push RSI(2) very low
        start = 110.0
        end = 250.0
        step = (end - start) / _SMA_WARMUP
        prices = [start + step * i for i in range(_SMA_WARMUP)]
        last = prices[-1]
        # Steep consecutive drops -> RSI(2) near 0
        prices.append(last - 20.0)
        prices.append(last - 50.0)

        candles = _make_candles(prices)
        signal = strategy.generate_signal("AAPL", candles, "us_tech")
        assert signal is not None, "Expected BUY with very low RSI(2)"
        # RSI(2) near 0 -> confidence near 1.0 (formula: (10 - rsi2)/10 * 0.8 + 0.2)
        # With steep drops, RSI(2) should be very low -> high confidence
        assert signal.confidence >= 0.85, (
            f"Expected confidence >= 0.85 for very low RSI(2), got {signal.confidence}"
        )

    def test_confidence_scaling_rsi2_nine_gives_low(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """RSI(2) = 9 should give confidence near 0.28."""
        strategy = RSI2ConnorsStrategy()
        params = {**_RSI2_PARAMS, "min_confidence": 0.0}
        monkeypatch.setattr(strategy, "get_parameters", lambda _seg: params)

        # We need RSI(2) just barely below 10. This is harder to construct precisely,
        # so we test that the confidence formula works correctly by directly testing
        # the _compute_confidence method.
        # confidence = (10 - 9) / 10 * 0.8 + 0.2 = 0.1 * 0.8 + 0.2 = 0.28
        confidence = strategy._compute_buy_confidence(9.0)
        assert abs(confidence - 0.28) < 0.01, f"Expected ~0.28, got {confidence}"

    def test_min_confidence_filter(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Signal below min_confidence should be filtered out."""
        strategy = RSI2ConnorsStrategy()
        # Set min_confidence very high so signals get filtered
        params = {**_RSI2_PARAMS, "min_confidence": 0.99}
        monkeypatch.setattr(strategy, "get_parameters", lambda _seg: params)

        prices = _prices_with_rsi2_buy()
        candles = _make_candles(prices)
        signal = strategy.generate_signal("AAPL", candles, "us_tech")
        assert signal is None, "Signal should be filtered when confidence < min_confidence"

    def test_neutral_rsi_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """RSI(2) in neutral zone (10-90) should produce no signal."""
        strategy = RSI2ConnorsStrategy()
        monkeypatch.setattr(strategy, "get_parameters", lambda _seg: _RSI2_PARAMS)

        # Flat prices -> RSI(2) around 50 (neutral)
        prices = [150.0] * (_SMA_WARMUP + 5)
        candles = _make_candles(prices)
        signal = strategy.generate_signal("AAPL", candles, "us_tech")
        assert signal is None, "No signal when RSI(2) is in neutral zone (10-90)"

    def test_insufficient_data_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Not enough candles for SMA(200) should return None."""
        strategy = RSI2ConnorsStrategy()
        monkeypatch.setattr(strategy, "get_parameters", lambda _seg: _RSI2_PARAMS)

        prices = [150.0] * 50  # Far less than 200+
        candles = _make_candles(prices)
        signal = strategy.generate_signal("AAPL", candles, "us_tech")
        assert signal is None, "Should return None with insufficient data for SMA(200)"
