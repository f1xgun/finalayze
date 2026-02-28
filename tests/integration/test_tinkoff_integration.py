"""Integration tests for TinkoffBroker — SDK-boundary mocked lifecycle."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from finalayze.core.exceptions import BrokerError, InstrumentNotFoundError
from finalayze.execution.broker_base import OrderRequest
from finalayze.execution.retry import RetryPolicy
from finalayze.execution.tinkoff_broker import TinkoffBroker


def _make_quotation(units: int, nano: int) -> MagicMock:
    q = MagicMock()
    q.units = units
    q.nano = nano
    return q


@pytest.fixture
def registry() -> MagicMock:
    """Mock instrument registry."""
    reg = MagicMock()
    instrument = MagicMock()
    instrument.figi = "BBG000B9XRY4"
    instrument.lot_size = 10
    reg.get.return_value = instrument
    return reg


@pytest.fixture
def broker(registry: MagicMock) -> TinkoffBroker:
    """Create TinkoffBroker with mock registry."""
    return TinkoffBroker(token="test-token", registry=registry, sandbox=True)  # noqa: S106


@pytest.fixture
def broker_with_retry(registry: MagicMock) -> TinkoffBroker:
    """Create TinkoffBroker with retry policy."""
    policy = RetryPolicy(max_retries=2, base_delay=0.001)
    return TinkoffBroker(
        token="test-token",  # noqa: S106
        registry=registry,
        sandbox=True,
        retry_policy=policy,
    )


class TestOrderLifecycle:
    """Full order submission lifecycle."""

    def test_buy_order_lot_rounding(self, broker: TinkoffBroker) -> None:
        mock_result = MagicMock()
        mock_result.executed_order_price = _make_quotation(250, 500_000_000)

        with patch("finalayze.execution.tinkoff_broker.asyncio.run", return_value=mock_result):
            order = OrderRequest(symbol="SBER", side="BUY", quantity=Decimal(15))
            result = broker.submit_order(order)

        assert result.filled is True
        assert result.quantity == Decimal(10)  # rounded down to lot_size

    def test_quantity_below_lot_size_returns_not_filled(self, broker: TinkoffBroker) -> None:
        order = OrderRequest(symbol="SBER", side="BUY", quantity=Decimal(5))
        result = broker.submit_order(order)

        assert result.filled is False
        assert result.quantity == Decimal(0)
        assert "lot size" in (result.reason or "")

    def test_sell_order_filled(self, broker: TinkoffBroker) -> None:
        mock_result = MagicMock()
        mock_result.executed_order_price = _make_quotation(260, 0)

        with patch("finalayze.execution.tinkoff_broker.asyncio.run", return_value=mock_result):
            order = OrderRequest(symbol="SBER", side="SELL", quantity=Decimal(20))
            result = broker.submit_order(order)

        assert result.filled is True
        assert result.side == "SELL"
        assert result.quantity == Decimal(20)

    def test_no_figi_raises(self, registry: MagicMock) -> None:
        instrument = MagicMock()
        instrument.figi = None
        registry.get.return_value = instrument
        broker = TinkoffBroker(token="test", registry=registry)  # noqa: S106

        order = OrderRequest(symbol="UNKNOWN", side="BUY", quantity=Decimal(10))
        with pytest.raises(InstrumentNotFoundError, match="no FIGI"):
            broker.submit_order(order)


class TestReconnection:
    """Test retry on transient gRPC failures."""

    def test_retry_on_connection_error(self, broker_with_retry: TinkoffBroker) -> None:
        mock_result = MagicMock()
        mock_result.executed_order_price = _make_quotation(100, 0)

        with (
            patch(
                "finalayze.execution.tinkoff_broker.asyncio.run",
                side_effect=[ConnectionError("gRPC down"), mock_result],
            ),
            patch("finalayze.execution.retry.time.sleep"),
        ):
            order = OrderRequest(symbol="SBER", side="BUY", quantity=Decimal(10))
            result = broker_with_retry.submit_order(order)

        assert result.filled is True

    def test_instrument_not_found_not_retried(
        self, broker_with_retry: TinkoffBroker, registry: MagicMock
    ) -> None:
        instrument = MagicMock()
        instrument.figi = None
        registry.get.return_value = instrument

        order = OrderRequest(symbol="UNKNOWN", side="BUY", quantity=Decimal(10))
        with pytest.raises(InstrumentNotFoundError):
            broker_with_retry.submit_order(order)


class TestPortfolio:
    """Test portfolio retrieval."""

    def test_get_portfolio(self, broker: TinkoffBroker) -> None:
        mock_portfolio = MagicMock()
        mock_portfolio.total_amount_portfolio = _make_quotation(100000, 0)
        mock_pos = MagicMock()
        mock_pos.figi = "BBG000B9XRY4"
        mock_pos.quantity = _make_quotation(50, 0)
        mock_portfolio.positions = [mock_pos]

        with patch("finalayze.execution.tinkoff_broker.asyncio.run", return_value=mock_portfolio):
            portfolio = broker.get_portfolio()

        assert portfolio.equity == Decimal(100000)
        assert portfolio.positions["BBG000B9XRY4"] == Decimal(50)

    def test_get_portfolio_retry(self, broker_with_retry: TinkoffBroker) -> None:
        mock_portfolio = MagicMock()
        mock_portfolio.total_amount_portfolio = _make_quotation(50000, 0)
        mock_portfolio.positions = []

        with (
            patch(
                "finalayze.execution.tinkoff_broker.asyncio.run",
                side_effect=[ConnectionError("timeout"), mock_portfolio],
            ),
            patch("finalayze.execution.retry.time.sleep"),
        ):
            portfolio = broker_with_retry.get_portfolio()

        assert portfolio.equity == Decimal(50000)
