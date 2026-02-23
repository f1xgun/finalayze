"""Unit tests for TinkoffBroker."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from finalayze.core.exceptions import BrokerError, InstrumentNotFoundError
from finalayze.execution.broker_base import OrderRequest
from finalayze.execution.tinkoff_broker import TinkoffBroker
from finalayze.markets.instruments import DEFAULT_MOEX_INSTRUMENTS, InstrumentRegistry


def _make_registry() -> InstrumentRegistry:
    registry = InstrumentRegistry()
    for inst in DEFAULT_MOEX_INSTRUMENTS:
        registry.register(inst)
    return registry


def _make_broker(sandbox: bool = True) -> TinkoffBroker:
    return TinkoffBroker(token="fake_token", registry=_make_registry(), sandbox=sandbox)  # noqa: S106


class TestTinkoffBrokerSubmitOrder:
    def test_buy_order_success(self) -> None:
        mock_result = MagicMock()
        mock_result.order_id = "ord-123"
        mock_result.executed_order_price.units = 270
        mock_result.executed_order_price.nano = 0
        mock_result.lots_executed = 1

        with patch(
            "finalayze.execution.tinkoff_broker.asyncio.run",
            return_value=mock_result,
        ):
            broker = _make_broker()
            order = OrderRequest(symbol="SBER", side="BUY", quantity=Decimal(10))
            result = broker.submit_order(order)

        assert result.filled is True
        assert result.symbol == "SBER"
        assert result.side == "BUY"
        assert result.fill_price == Decimal(270)

    def test_unknown_symbol_raises(self) -> None:
        broker = _make_broker()
        order = OrderRequest(symbol="UNKNOWN", side="BUY", quantity=Decimal(10))
        with pytest.raises(InstrumentNotFoundError):
            broker.submit_order(order)

    def test_api_error_raises_broker_error(self) -> None:
        with patch(
            "finalayze.execution.tinkoff_broker.asyncio.run",
            side_effect=RuntimeError("gRPC unavailable"),
        ):
            broker = _make_broker()
            order = OrderRequest(symbol="SBER", side="BUY", quantity=Decimal(10))
            with pytest.raises(BrokerError, match="gRPC unavailable"):
                broker.submit_order(order)

    def test_fill_candle_ignored(self) -> None:
        """TinkoffBroker ignores fill_candle (live broker)."""
        mock_result = MagicMock()
        mock_result.order_id = "ord-456"
        mock_result.executed_order_price.units = 270
        mock_result.executed_order_price.nano = 0
        mock_result.lots_executed = 1

        with patch(
            "finalayze.execution.tinkoff_broker.asyncio.run",
            return_value=mock_result,
        ):
            broker = _make_broker()
            order = OrderRequest(symbol="SBER", side="BUY", quantity=Decimal(10))
            result = broker.submit_order(order, fill_candle=None)
        assert result.filled is True

    def test_lot_size_rounding(self) -> None:
        """Quantity must be rounded down to nearest lot_size multiple.

        SBER has lot_size=10. Requesting qty=15 -> actual qty=10.
        """

        def capture_run(coro: object) -> MagicMock:
            mock_result = MagicMock()
            mock_result.order_id = "ord-789"
            mock_result.executed_order_price.units = 270
            mock_result.executed_order_price.nano = 0
            mock_result.lots_executed = 1
            return mock_result

        with patch("finalayze.execution.tinkoff_broker.asyncio.run", side_effect=capture_run):
            broker = _make_broker()
            order = OrderRequest(symbol="SBER", side="BUY", quantity=Decimal(15))
            result = broker.submit_order(order)

        # filled=True but actual quantity rounded to 10
        assert result.filled is True
        assert result.quantity == Decimal(10)


class TestTinkoffBrokerGetPortfolio:
    def test_portfolio_returned(self) -> None:
        mock_portfolio = MagicMock()
        mock_portfolio.total_amount_portfolio.units = 1_000_000
        mock_portfolio.total_amount_portfolio.nano = 0
        mock_portfolio.positions = []

        with patch(
            "finalayze.execution.tinkoff_broker.asyncio.run",
            return_value=mock_portfolio,
        ):
            broker = _make_broker()
            portfolio = broker.get_portfolio()

        assert portfolio.equity == Decimal(1000000)

    def test_portfolio_api_error_raises(self) -> None:
        with patch(
            "finalayze.execution.tinkoff_broker.asyncio.run",
            side_effect=RuntimeError("gRPC timeout"),
        ):
            broker = _make_broker()
            with pytest.raises(BrokerError):
                broker.get_portfolio()
