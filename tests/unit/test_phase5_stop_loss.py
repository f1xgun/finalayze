"""Tests for stop-loss wiring after BUY/SELL fills (5.1)."""

from __future__ import annotations

import datetime
from decimal import Decimal
from unittest.mock import MagicMock, patch

from finalayze.core.schemas import Candle


def _make_candles(n: int, base_price: float = 100.0, symbol: str = "AAPL") -> list[Candle]:
    """Create n synthetic candles for stop-loss testing."""
    candles: list[Candle] = []
    for i in range(n):
        price = Decimal(str(base_price + i * 0.5))
        candles.append(
            Candle(
                symbol=symbol,
                market_id="us",
                timeframe="1d",
                timestamp=datetime.datetime(2025, 1, 1, tzinfo=datetime.UTC)
                + datetime.timedelta(days=i),
                open=price - Decimal("0.1"),
                high=price + Decimal(1),
                low=price - Decimal(1),
                close=price,
                volume=1000,
            )
        )
    return candles


def _make_trading_loop() -> MagicMock:
    """Create a minimal TradingLoop-like object for testing _submit_order."""
    from finalayze.core.trading_loop import TradingLoop

    settings = MagicMock()
    settings.news_cycle_minutes = 30
    settings.strategy_cycle_minutes = 60
    settings.daily_reset_hour_utc = 0
    settings.mode = "test"
    settings.max_position_pct = 0.20
    settings.max_positions_per_market = 10
    settings.daily_loss_limit_pct = 0.05
    settings.kelly_fraction = 0.5

    loop = MagicMock(spec=TradingLoop)
    loop._stop_loss_prices = {}
    loop._broker_router = MagicMock()
    loop._alerter = MagicMock()
    loop._submit_order = TradingLoop._submit_order.__get__(loop)

    # Store OrderRequest class for the method
    from finalayze.execution.broker_base import OrderRequest

    loop._OrderRequest = OrderRequest
    return loop


class TestStopLossWiring:
    def test_buy_fill_sets_stop_loss(self) -> None:
        """A filled BUY should compute and store a stop-loss price."""
        loop = _make_trading_loop()
        from finalayze.execution.broker_base import OrderRequest, OrderResult

        order = OrderRequest(symbol="AAPL", side="BUY", quantity=Decimal(10))
        result = OrderResult(
            filled=True,
            fill_price=Decimal("110.0"),
            symbol="AAPL",
            side="BUY",
            quantity=Decimal(10),
        )
        loop._broker_router.submit.return_value = result
        candles = _make_candles(20)

        loop._submit_order(order, "us", candles=candles)

        assert "AAPL" in loop._stop_loss_prices
        assert loop._stop_loss_prices["AAPL"] > Decimal(0)
        assert loop._stop_loss_prices["AAPL"] < Decimal("110.0")

    def test_sell_fill_clears_stop_loss(self) -> None:
        """A filled SELL should remove the stop-loss for the symbol."""
        loop = _make_trading_loop()
        loop._stop_loss_prices["AAPL"] = Decimal("95.0")

        from finalayze.execution.broker_base import OrderRequest, OrderResult

        order = OrderRequest(symbol="AAPL", side="SELL", quantity=Decimal(10))
        result = OrderResult(
            filled=True,
            fill_price=Decimal("112.0"),
            symbol="AAPL",
            side="SELL",
            quantity=Decimal(10),
        )
        loop._broker_router.submit.return_value = result

        loop._submit_order(order, "us", candles=_make_candles(20))

        assert "AAPL" not in loop._stop_loss_prices

    def test_rejected_order_no_stop(self) -> None:
        """A rejected order should not set any stop-loss."""
        loop = _make_trading_loop()

        from finalayze.execution.broker_base import OrderRequest, OrderResult

        order = OrderRequest(symbol="AAPL", side="BUY", quantity=Decimal(10))
        result = OrderResult(
            filled=False,
            symbol="AAPL",
            side="BUY",
            quantity=Decimal(0),
            reason="Insufficient funds",
        )
        loop._broker_router.submit.return_value = result

        loop._submit_order(order, "us", candles=_make_candles(20))

        assert "AAPL" not in loop._stop_loss_prices

    def test_moex_uses_higher_multiplier(self) -> None:
        """MOEX BUY fills should use the 2.5x ATR multiplier (wider stop)."""
        loop = _make_trading_loop()

        from finalayze.execution.broker_base import OrderRequest, OrderResult

        order = OrderRequest(symbol="SBER", side="BUY", quantity=Decimal(10))
        result = OrderResult(
            filled=True,
            fill_price=Decimal("110.0"),
            symbol="SBER",
            side="BUY",
            quantity=Decimal(10),
        )
        loop._broker_router.submit.return_value = result
        candles = _make_candles(20, symbol="SBER")

        with patch("finalayze.risk.stop_loss.compute_atr_stop_loss") as mock_stop:
            mock_stop.return_value = Decimal("100.0")
            loop._submit_order(order, "moex", candles=candles)

            # Verify 2.5 multiplier was used for MOEX
            call_kwargs = mock_stop.call_args
            assert call_kwargs[1]["atr_multiplier"] == Decimal("2.5")

    def test_buy_without_candles_no_crash(self) -> None:
        """BUY fill with no candles should not crash or set stop-loss."""
        loop = _make_trading_loop()

        from finalayze.execution.broker_base import OrderRequest, OrderResult

        order = OrderRequest(symbol="AAPL", side="BUY", quantity=Decimal(10))
        result = OrderResult(
            filled=True,
            fill_price=Decimal("110.0"),
            symbol="AAPL",
            side="BUY",
            quantity=Decimal(10),
        )
        loop._broker_router.submit.return_value = result

        loop._submit_order(order, "us", candles=None)

        assert "AAPL" not in loop._stop_loss_prices
