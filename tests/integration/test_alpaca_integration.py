"""Integration tests for AlpacaBroker — SDK-boundary mocked order lifecycle."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from finalayze.core.exceptions import BrokerError, InsufficientFundsError
from finalayze.execution.alpaca_broker import AlpacaBroker
from finalayze.execution.broker_base import OrderRequest
from finalayze.execution.retry import RetryPolicy


@pytest.fixture
def mock_client() -> MagicMock:
    """Create a mock TradingClient."""
    return MagicMock()


@pytest.fixture
def broker(mock_client: MagicMock) -> AlpacaBroker:
    """Create AlpacaBroker with mocked TradingClient."""
    with patch("finalayze.execution.alpaca_broker.TradingClient", return_value=mock_client):
        return AlpacaBroker(api_key="test", secret_key="test", paper=True)  # noqa: S106


@pytest.fixture
def broker_with_retry(mock_client: MagicMock) -> AlpacaBroker:
    """Create AlpacaBroker with retry policy and mocked TradingClient."""
    with patch("finalayze.execution.alpaca_broker.TradingClient", return_value=mock_client):
        policy = RetryPolicy(max_retries=2, base_delay=0.001)
        return AlpacaBroker(
            api_key="test",
            secret_key="test",  # noqa: S106
            paper=True,
            retry_policy=policy,
        )


class TestOrderLifecycle:
    """Full order submission lifecycle."""

    def test_buy_order_filled(self, broker: AlpacaBroker, mock_client: MagicMock) -> None:
        mock_result = MagicMock()
        mock_result.filled_avg_price = "150.50"
        mock_result.filled_qty = "10"
        mock_client.submit_order.return_value = mock_result

        order = OrderRequest(symbol="AAPL", side="BUY", quantity=Decimal(10))
        result = broker.submit_order(order)

        assert result.filled is True
        assert result.fill_price == Decimal("150.50")
        assert result.quantity == Decimal(10)

    def test_sell_order_filled(self, broker: AlpacaBroker, mock_client: MagicMock) -> None:
        mock_result = MagicMock()
        mock_result.filled_avg_price = "155.00"
        mock_result.filled_qty = "5"
        mock_client.submit_order.return_value = mock_result

        order = OrderRequest(symbol="AAPL", side="SELL", quantity=Decimal(5))
        result = broker.submit_order(order)

        assert result.filled is True
        assert result.side == "SELL"

    def test_insufficient_funds_detected(
        self, broker: AlpacaBroker, mock_client: MagicMock
    ) -> None:
        mock_client.submit_order.side_effect = Exception("insufficient buying power")

        order = OrderRequest(symbol="AAPL", side="BUY", quantity=Decimal(1000))
        with pytest.raises(InsufficientFundsError, match="Insufficient funds"):
            broker.submit_order(order)

    def test_generic_broker_error(self, broker: AlpacaBroker, mock_client: MagicMock) -> None:
        mock_client.submit_order.side_effect = Exception("server error 500")

        order = OrderRequest(symbol="AAPL", side="BUY", quantity=Decimal(1))
        with pytest.raises(BrokerError, match="Alpaca order failed"):
            broker.submit_order(order)


class TestReconnection:
    """Test retry on transient failures."""

    def test_retry_on_connection_error(
        self, broker_with_retry: AlpacaBroker, mock_client: MagicMock
    ) -> None:
        mock_result = MagicMock()
        mock_result.filled_avg_price = "100.00"
        mock_result.filled_qty = "1"
        mock_client.submit_order.side_effect = [
            ConnectionError("connection reset"),
            mock_result,
        ]

        order = OrderRequest(symbol="AAPL", side="BUY", quantity=Decimal(1))
        with patch("finalayze.execution.retry.time.sleep"):
            result = broker_with_retry.submit_order(order)

        assert result.filled is True
        assert mock_client.submit_order.call_count == 2

    def test_retry_exhausted_raises(
        self, broker_with_retry: AlpacaBroker, mock_client: MagicMock
    ) -> None:
        mock_client.submit_order.side_effect = ConnectionError("always down")

        order = OrderRequest(symbol="AAPL", side="BUY", quantity=Decimal(1))
        with (
            patch("finalayze.execution.retry.time.sleep"),
            pytest.raises(BrokerError, match="Alpaca order failed"),
        ):
            broker_with_retry.submit_order(order)

    def test_insufficient_funds_not_retried(
        self, broker_with_retry: AlpacaBroker, mock_client: MagicMock
    ) -> None:
        mock_client.submit_order.side_effect = InsufficientFundsError("no funds")

        order = OrderRequest(symbol="AAPL", side="BUY", quantity=Decimal(1))
        with pytest.raises(InsufficientFundsError):
            broker_with_retry.submit_order(order)
        mock_client.submit_order.assert_called_once()


class TestPortfolio:
    """Test portfolio retrieval with retry."""

    def test_get_portfolio_success(self, broker: AlpacaBroker, mock_client: MagicMock) -> None:
        mock_account = MagicMock()
        mock_account.cash = "10000.00"
        mock_client.get_account.return_value = mock_account

        mock_pos = MagicMock()
        mock_pos.symbol = "AAPL"
        mock_pos.qty = "10"
        mock_pos.market_value = "1500.00"
        mock_client.get_all_positions.return_value = [mock_pos]

        portfolio = broker.get_portfolio()
        assert portfolio.cash == Decimal("10000.00")
        assert portfolio.positions["AAPL"] == Decimal(10)

    def test_get_portfolio_retry(
        self, broker_with_retry: AlpacaBroker, mock_client: MagicMock
    ) -> None:
        mock_account = MagicMock()
        mock_account.cash = "5000.00"
        mock_client.get_account.side_effect = [
            ConnectionError("timeout"),
            mock_account,
        ]
        mock_client.get_all_positions.return_value = []

        with patch("finalayze.execution.retry.time.sleep"):
            portfolio = broker_with_retry.get_portfolio()
        assert portfolio.cash == Decimal("5000.00")
