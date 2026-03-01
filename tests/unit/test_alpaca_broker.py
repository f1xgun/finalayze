"""Unit tests for AlpacaBroker."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from finalayze.core.exceptions import BrokerError, InsufficientFundsError
from finalayze.execution.alpaca_broker import AlpacaBroker
from finalayze.execution.broker_base import OrderRequest

FILL_PRICE_BUY = "150.00"
FILL_PRICE_SELL = "155.00"
FILL_QTY_BUY = "10"
FILL_QTY_SELL = "5"
BUYING_POWER = "50000.00"
CASH_AMOUNT = "50000.00"
PORTFOLIO_VALUE = "50000.00"
AAPL_MARKET_VALUE = "1500.00"
AAPL_QTY = "10"


# ---------- helpers ----------


def _make_broker(paper: bool = True) -> AlpacaBroker:
    return AlpacaBroker(api_key="fake_key", secret_key="fake_secret", paper=paper)  # noqa: S106


def _mock_trading_client() -> MagicMock:
    client = MagicMock()
    account = MagicMock()
    account.buying_power = BUYING_POWER
    account.cash = CASH_AMOUNT
    account.portfolio_value = PORTFOLIO_VALUE
    account.status = "ACTIVE"
    client.get_account.return_value = account
    return client


def _mock_position(
    symbol: str, qty: str = AAPL_QTY, market_value: str = AAPL_MARKET_VALUE
) -> MagicMock:
    pos = MagicMock()
    pos.symbol = symbol
    pos.qty = qty
    pos.market_value = market_value
    return pos


# ---------- tests ----------


class TestAlpacaBrokerSubmitOrder:
    def test_buy_order_success(self) -> None:
        mock_client = _mock_trading_client()
        mock_order = MagicMock()
        mock_order.filled_avg_price = FILL_PRICE_BUY
        mock_order.filled_qty = FILL_QTY_BUY
        mock_client.submit_order.return_value = mock_order

        with patch("finalayze.execution.alpaca_broker.TradingClient", return_value=mock_client):
            broker = _make_broker()
            order = OrderRequest(symbol="AAPL", side="BUY", quantity=Decimal(10))
            result = broker.submit_order(order)

        assert result.filled is True
        assert result.symbol == "AAPL"
        assert result.side == "BUY"
        assert result.fill_price == Decimal(FILL_PRICE_BUY)

    def test_sell_order_success(self) -> None:
        mock_client = _mock_trading_client()
        mock_order = MagicMock()
        mock_order.filled_avg_price = FILL_PRICE_SELL
        mock_order.filled_qty = FILL_QTY_SELL
        mock_client.submit_order.return_value = mock_order

        with patch("finalayze.execution.alpaca_broker.TradingClient", return_value=mock_client):
            broker = _make_broker()
            order = OrderRequest(symbol="AAPL", side="SELL", quantity=Decimal(5))
            result = broker.submit_order(order)

        assert result.filled is True
        assert result.side == "SELL"
        assert result.fill_price == Decimal(FILL_PRICE_SELL)

    def test_insufficient_funds_raises(self) -> None:
        mock_client = _mock_trading_client()
        mock_client.submit_order.side_effect = Exception("insufficient buying power")

        with patch("finalayze.execution.alpaca_broker.TradingClient", return_value=mock_client):
            broker = _make_broker()
            order = OrderRequest(symbol="AAPL", side="BUY", quantity=Decimal(1000))
            with pytest.raises(InsufficientFundsError):
                broker.submit_order(order)

    def test_api_error_raises_broker_error(self) -> None:
        mock_client = _mock_trading_client()
        mock_client.submit_order.side_effect = Exception("connection timeout")

        with patch("finalayze.execution.alpaca_broker.TradingClient", return_value=mock_client):
            broker = _make_broker()
            order = OrderRequest(symbol="AAPL", side="BUY", quantity=Decimal(1))
            with pytest.raises(BrokerError):
                broker.submit_order(order)

    def test_fill_candle_ignored(self) -> None:
        """AlpacaBroker must accept fill_candle=None (live broker doesn't need it)."""
        mock_client = _mock_trading_client()
        mock_order = MagicMock()
        mock_order.filled_avg_price = FILL_PRICE_BUY
        mock_order.filled_qty = "1"
        mock_client.submit_order.return_value = mock_order

        with patch("finalayze.execution.alpaca_broker.TradingClient", return_value=mock_client):
            broker = _make_broker()
            order = OrderRequest(symbol="AAPL", side="BUY", quantity=Decimal(1))
            result = broker.submit_order(order, fill_candle=None)
        assert result.filled is True


class TestAlpacaBrokerFillReporting:
    """6A.9: Partial fill and None fill_qty reporting."""

    def test_partial_fill_reports_actual_quantity(self) -> None:
        mock_client = _mock_trading_client()
        mock_order = MagicMock()
        mock_order.filled_avg_price = FILL_PRICE_BUY
        mock_order.filled_qty = "5"
        mock_client.submit_order.return_value = mock_order

        with patch("finalayze.execution.alpaca_broker.TradingClient", return_value=mock_client):
            broker = _make_broker()
            order = OrderRequest(symbol="AAPL", side="BUY", quantity=Decimal(10))
            result = broker.submit_order(order)

        expected_qty = Decimal(5)
        assert result.quantity == expected_qty

    def test_none_fill_qty_reports_zero(self) -> None:
        mock_client = _mock_trading_client()
        mock_order = MagicMock()
        mock_order.filled_avg_price = None
        mock_order.filled_qty = None
        mock_client.submit_order.return_value = mock_order

        with patch("finalayze.execution.alpaca_broker.TradingClient", return_value=mock_client):
            broker = _make_broker()
            order = OrderRequest(symbol="AAPL", side="BUY", quantity=Decimal(10))
            result = broker.submit_order(order)

        assert result.quantity == Decimal(0)

    def test_zero_fill_qty_reports_zero(self) -> None:
        mock_client = _mock_trading_client()
        mock_order = MagicMock()
        mock_order.filled_avg_price = None
        mock_order.filled_qty = 0
        mock_client.submit_order.return_value = mock_order

        with patch("finalayze.execution.alpaca_broker.TradingClient", return_value=mock_client):
            broker = _make_broker()
            order = OrderRequest(symbol="AAPL", side="BUY", quantity=Decimal(10))
            result = broker.submit_order(order)

        assert result.quantity == Decimal(0)


class TestAlpacaBrokerGetPortfolio:
    def test_portfolio_returns_state(self) -> None:
        mock_client = _mock_trading_client()
        mock_client.get_all_positions.return_value = [
            _mock_position("AAPL", qty=AAPL_QTY, market_value=AAPL_MARKET_VALUE),
        ]

        with patch("finalayze.execution.alpaca_broker.TradingClient", return_value=mock_client):
            broker = _make_broker()
            portfolio = broker.get_portfolio()

        assert portfolio.cash == Decimal(CASH_AMOUNT)
        assert "AAPL" in portfolio.positions

    def test_portfolio_api_error_raises(self) -> None:
        mock_client = _mock_trading_client()
        mock_client.get_account.side_effect = Exception("API unavailable")

        with patch("finalayze.execution.alpaca_broker.TradingClient", return_value=mock_client):
            broker = _make_broker()
            with pytest.raises(BrokerError):
                broker.get_portfolio()


class TestAlpacaBrokerHasPosition:
    def test_has_position_true(self) -> None:
        mock_client = _mock_trading_client()
        mock_client.get_all_positions.return_value = [
            _mock_position("AAPL", qty=AAPL_QTY),
        ]

        with patch("finalayze.execution.alpaca_broker.TradingClient", return_value=mock_client):
            broker = _make_broker()
            assert broker.has_position("AAPL") is True

    def test_has_position_false(self) -> None:
        mock_client = _mock_trading_client()
        mock_client.get_all_positions.return_value = []

        with patch("finalayze.execution.alpaca_broker.TradingClient", return_value=mock_client):
            broker = _make_broker()
            assert broker.has_position("MSFT") is False


class TestAlpacaBrokerCancelOrder:
    def test_cancel_order_success(self) -> None:
        mock_client = _mock_trading_client()
        mock_client.cancel_order_by_id.return_value = None

        with patch("finalayze.execution.alpaca_broker.TradingClient", return_value=mock_client):
            broker = _make_broker()
            broker.cancel_order("order-123")

        mock_client.cancel_order_by_id.assert_called_once_with("order-123")

    def test_cancel_order_api_error_raises(self) -> None:
        mock_client = _mock_trading_client()
        mock_client.cancel_order_by_id.side_effect = Exception("order not found")

        with patch("finalayze.execution.alpaca_broker.TradingClient", return_value=mock_client):
            broker = _make_broker()
            with pytest.raises(BrokerError, match="order not found"):
                broker.cancel_order("order-999")
