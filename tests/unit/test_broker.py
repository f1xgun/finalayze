"""Unit tests for BrokerBase ABC and SimulatedBroker."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from finalayze.core.schemas import Candle
from finalayze.execution.broker_base import BrokerBase, OrderRequest, OrderResult
from finalayze.execution.simulated_broker import SimulatedBroker

INITIAL_CASH = Decimal(100000)
SHARE_PRICE = Decimal(150)
ORDER_QTY = Decimal(10)
STOP_PRICE = Decimal(140)
LOW_PRICE = Decimal(135)
VOLUME = 1_000_000


def _candle(
    price: Decimal,
    day: int = 0,
    *,
    symbol: str = "AAPL",
    low: Decimal | None = None,
) -> Candle:
    """Create a candle at the given price."""
    return Candle(
        symbol=symbol,
        market_id="us",
        timeframe="1d",
        timestamp=datetime(2024, 1, 1, 14, 30, tzinfo=UTC) + timedelta(days=day),
        open=price,
        high=price + Decimal(5),
        low=low if low is not None else price - Decimal(5),
        close=price,
        volume=VOLUME,
    )


class TestBrokerBase:
    """BrokerBase is an abstract class and cannot be instantiated."""

    def test_is_abstract(self) -> None:
        with pytest.raises(TypeError):
            BrokerBase()  # type: ignore[abstract]

    def test_order_request_creation(self) -> None:
        req = OrderRequest(symbol="AAPL", side="BUY", quantity=ORDER_QTY)
        assert req.symbol == "AAPL"
        assert req.side == "BUY"
        assert req.quantity == ORDER_QTY

    def test_order_result_defaults(self) -> None:
        result = OrderResult(filled=False)
        assert result.filled is False
        assert result.fill_price is None
        assert result.symbol == ""
        assert result.side == ""
        assert result.quantity == Decimal(0)
        assert result.reason == ""


class TestSimulatedBrokerInitialState:
    """Initial portfolio should reflect starting cash."""

    def test_initial_portfolio_cash(self) -> None:
        broker = SimulatedBroker(initial_cash=INITIAL_CASH)
        portfolio = broker.get_portfolio()
        assert portfolio.cash == INITIAL_CASH

    def test_initial_portfolio_equity(self) -> None:
        broker = SimulatedBroker(initial_cash=INITIAL_CASH)
        portfolio = broker.get_portfolio()
        assert portfolio.equity == INITIAL_CASH

    def test_initial_portfolio_no_positions(self) -> None:
        broker = SimulatedBroker(initial_cash=INITIAL_CASH)
        portfolio = broker.get_portfolio()
        assert portfolio.positions == {}


class TestSimulatedBrokerBuy:
    """Buy orders should fill at candle open, deduct cash, create position."""

    def test_buy_fills_at_open(self) -> None:
        broker = SimulatedBroker(initial_cash=INITIAL_CASH)
        candle = _candle(SHARE_PRICE)
        order = OrderRequest(symbol="AAPL", side="BUY", quantity=ORDER_QTY)

        result = broker.submit_order(order, candle)

        assert result.filled is True
        assert result.fill_price == SHARE_PRICE
        assert result.symbol == "AAPL"
        assert result.side == "BUY"
        assert result.quantity == ORDER_QTY

    def test_buy_deducts_cash(self) -> None:
        broker = SimulatedBroker(initial_cash=INITIAL_CASH)
        candle = _candle(SHARE_PRICE)
        order = OrderRequest(symbol="AAPL", side="BUY", quantity=ORDER_QTY)
        broker.submit_order(order, candle)

        expected_cash = INITIAL_CASH - SHARE_PRICE * ORDER_QTY
        assert broker.get_portfolio().cash == expected_cash

    def test_buy_creates_position(self) -> None:
        broker = SimulatedBroker(initial_cash=INITIAL_CASH)
        candle = _candle(SHARE_PRICE)
        order = OrderRequest(symbol="AAPL", side="BUY", quantity=ORDER_QTY)
        broker.submit_order(order, candle)

        portfolio = broker.get_portfolio()
        assert portfolio.positions["AAPL"] == ORDER_QTY


class TestSimulatedBrokerSell:
    """Sell orders should fill at candle open, increase cash, remove position."""

    def test_sell_fills_at_open(self) -> None:
        broker = SimulatedBroker(initial_cash=INITIAL_CASH)
        buy_candle = _candle(SHARE_PRICE)
        broker.submit_order(OrderRequest("AAPL", "BUY", ORDER_QTY), buy_candle)

        sell_price = Decimal(160)
        sell_candle = _candle(sell_price, day=1)
        result = broker.submit_order(OrderRequest("AAPL", "SELL", ORDER_QTY), sell_candle)

        assert result.filled is True
        assert result.fill_price == sell_price
        assert result.side == "SELL"

    def test_sell_adds_cash(self) -> None:
        broker = SimulatedBroker(initial_cash=INITIAL_CASH)
        buy_candle = _candle(SHARE_PRICE)
        broker.submit_order(OrderRequest("AAPL", "BUY", ORDER_QTY), buy_candle)

        sell_price = Decimal(160)
        sell_candle = _candle(sell_price, day=1)
        broker.submit_order(OrderRequest("AAPL", "SELL", ORDER_QTY), sell_candle)

        expected = INITIAL_CASH - SHARE_PRICE * ORDER_QTY + sell_price * ORDER_QTY
        assert broker.get_portfolio().cash == expected

    def test_sell_removes_position(self) -> None:
        broker = SimulatedBroker(initial_cash=INITIAL_CASH)
        buy_candle = _candle(SHARE_PRICE)
        broker.submit_order(OrderRequest("AAPL", "BUY", ORDER_QTY), buy_candle)

        sell_candle = _candle(Decimal(160), day=1)
        broker.submit_order(OrderRequest("AAPL", "SELL", ORDER_QTY), sell_candle)

        assert "AAPL" not in broker.get_portfolio().positions

    def test_sell_partial_position(self) -> None:
        """Selling more than held should sell only what is held."""
        broker = SimulatedBroker(initial_cash=INITIAL_CASH)
        buy_candle = _candle(SHARE_PRICE)
        broker.submit_order(OrderRequest("AAPL", "BUY", Decimal(5)), buy_candle)

        sell_candle = _candle(Decimal(160), day=1)
        result = broker.submit_order(OrderRequest("AAPL", "SELL", Decimal(20)), sell_candle)

        assert result.filled is True
        assert result.quantity == Decimal(5)
        assert "AAPL" not in broker.get_portfolio().positions


class TestSimulatedBrokerInsufficientCash:
    """Orders that exceed available cash should be rejected."""

    def test_insufficient_cash_rejected(self) -> None:
        broker = SimulatedBroker(initial_cash=Decimal(100))
        candle = _candle(SHARE_PRICE)
        order = OrderRequest(symbol="AAPL", side="BUY", quantity=ORDER_QTY)

        result = broker.submit_order(order, candle)

        assert result.filled is False
        assert result.reason != ""

    def test_insufficient_cash_no_position(self) -> None:
        broker = SimulatedBroker(initial_cash=Decimal(100))
        candle = _candle(SHARE_PRICE)
        order = OrderRequest(symbol="AAPL", side="BUY", quantity=ORDER_QTY)
        broker.submit_order(order, candle)

        assert broker.get_portfolio().positions == {}


class TestSimulatedBrokerStopLoss:
    """Stop losses should trigger when candle low drops to or below stop price."""

    def test_stop_loss_triggers(self) -> None:
        broker = SimulatedBroker(initial_cash=INITIAL_CASH)
        buy_candle = _candle(SHARE_PRICE)
        broker.submit_order(OrderRequest("AAPL", "BUY", ORDER_QTY), buy_candle)
        broker.set_stop_loss("AAPL", STOP_PRICE)

        trigger_candle = _candle(SHARE_PRICE, day=1, low=LOW_PRICE)
        results = broker.check_stop_losses(trigger_candle)

        assert len(results) == 1
        assert results[0].filled is True
        assert results[0].fill_price == STOP_PRICE
        assert results[0].side == "SELL"

    def test_stop_loss_closes_position(self) -> None:
        broker = SimulatedBroker(initial_cash=INITIAL_CASH)
        buy_candle = _candle(SHARE_PRICE)
        broker.submit_order(OrderRequest("AAPL", "BUY", ORDER_QTY), buy_candle)
        broker.set_stop_loss("AAPL", STOP_PRICE)

        trigger_candle = _candle(SHARE_PRICE, day=1, low=LOW_PRICE)
        broker.check_stop_losses(trigger_candle)

        assert "AAPL" not in broker.get_portfolio().positions

    def test_stop_loss_no_trigger(self) -> None:
        broker = SimulatedBroker(initial_cash=INITIAL_CASH)
        buy_candle = _candle(SHARE_PRICE)
        broker.submit_order(OrderRequest("AAPL", "BUY", ORDER_QTY), buy_candle)
        broker.set_stop_loss("AAPL", STOP_PRICE)

        safe_candle = _candle(SHARE_PRICE, day=1, low=Decimal(145))
        results = broker.check_stop_losses(safe_candle)

        assert len(results) == 0
        assert "AAPL" in broker.get_portfolio().positions


class TestSimulatedBrokerEquity:
    """Equity should reflect cash + market value of positions."""

    def test_equity_with_position(self) -> None:
        broker = SimulatedBroker(initial_cash=INITIAL_CASH)
        candle = _candle(SHARE_PRICE)
        broker.submit_order(OrderRequest("AAPL", "BUY", ORDER_QTY), candle)

        portfolio = broker.get_portfolio()
        expected_cash = INITIAL_CASH - SHARE_PRICE * ORDER_QTY
        expected_equity = expected_cash + candle.close * ORDER_QTY
        assert portfolio.equity == expected_equity

    def test_equity_updates_with_price(self) -> None:
        broker = SimulatedBroker(initial_cash=INITIAL_CASH)
        candle = _candle(SHARE_PRICE)
        broker.submit_order(OrderRequest("AAPL", "BUY", ORDER_QTY), candle)

        new_price = Decimal(200)
        new_candle = _candle(new_price, day=1)
        broker.update_prices(new_candle)

        portfolio = broker.get_portfolio()
        expected_cash = INITIAL_CASH - SHARE_PRICE * ORDER_QTY
        expected_equity = expected_cash + new_price * ORDER_QTY
        assert portfolio.equity == expected_equity

    def test_sell_clears_stop_loss_entry(self) -> None:
        """Selling a position must remove its stop-loss to avoid stale entries."""
        broker = SimulatedBroker(initial_cash=INITIAL_CASH)
        buy_candle = _candle(SHARE_PRICE)
        broker.submit_order(OrderRequest("AAPL", "BUY", ORDER_QTY), buy_candle)
        broker.set_stop_loss("AAPL", STOP_PRICE)

        sell_candle = _candle(Decimal(160), day=1)
        broker.submit_order(OrderRequest("AAPL", "SELL", ORDER_QTY), sell_candle)

        # Stop-loss entry must be cleared after position is fully closed
        assert "AAPL" not in broker._stop_states


class TestSimulatedBrokerFillCandleOptional:
    """SimulatedBroker must raise ValueError when fill_candle is None."""

    def test_submit_order_raises_if_no_candle(self) -> None:
        """SimulatedBroker must reject orders when no candle is provided."""
        broker = SimulatedBroker(initial_cash=INITIAL_CASH)
        order = OrderRequest(symbol="AAPL", side="BUY", quantity=Decimal(1))
        with pytest.raises(ValueError, match="fill_candle"):
            broker.submit_order(order, fill_candle=None)
