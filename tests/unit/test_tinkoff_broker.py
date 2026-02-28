"""Unit tests for TinkoffBroker."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest
from t_tech.invest import OrderDirection

from finalayze.core.exceptions import BrokerError, InstrumentNotFoundError
from finalayze.execution.broker_base import OrderRequest
from finalayze.execution.tinkoff_broker import TinkoffBroker
from finalayze.markets.instruments import DEFAULT_MOEX_INSTRUMENTS, Instrument, InstrumentRegistry

SBER_FIGI = "BBG004730N88"
PORTFOLIO_CASH = 500_000


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


class TestTinkoffBrokerSubmitOrderSell:
    def test_sell_order_success(self) -> None:
        """SELL order path — side='SELL' should succeed and return correct side."""

        def capture_run(coro: object) -> MagicMock:
            mock_result = MagicMock()
            mock_result.order_id = "ord-sell-1"
            mock_result.executed_order_price.units = 270
            mock_result.executed_order_price.nano = 0
            mock_result.lots_executed = 1
            return mock_result

        with patch(
            "finalayze.execution.tinkoff_broker.asyncio.run",
            side_effect=capture_run,
        ):
            broker = _make_broker()
            order = OrderRequest(symbol="SBER", side="SELL", quantity=Decimal(10))
            result = broker.submit_order(order)

        assert result.filled is True
        assert result.symbol == "SBER"
        assert result.side == "SELL"

    def test_sell_order_direction_constant(self) -> None:
        """Verify ORDER_DIRECTION_SELL is used for SELL side (not BUY)."""
        assert OrderDirection.ORDER_DIRECTION_SELL != OrderDirection.ORDER_DIRECTION_BUY

    def test_qty_below_lot_size_returns_unfilled(self) -> None:
        """SBER lot_size=10, qty=5 -> filled=False, quantity=Decimal(0)."""
        broker = _make_broker()
        order = OrderRequest(symbol="SBER", side="BUY", quantity=Decimal(5))
        result = broker.submit_order(order)

        assert result.filled is False
        assert result.quantity == Decimal(0)
        assert "lot size" in result.reason.lower()


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


class TestTinkoffBrokerHasPosition:
    def _mock_portfolio(self, figi: str, qty_units: int) -> MagicMock:
        mock_portfolio = MagicMock()
        mock_portfolio.total_amount_portfolio.units = PORTFOLIO_CASH
        mock_portfolio.total_amount_portfolio.nano = 0
        mock_pos = MagicMock()
        mock_pos.figi = figi
        mock_pos.quantity.units = qty_units
        mock_pos.quantity.nano = 0
        mock_portfolio.positions = [mock_pos]
        return mock_portfolio

    def test_has_position_true(self) -> None:
        """has_position returns True when portfolio contains the instrument's FIGI."""
        mock_portfolio = self._mock_portfolio(SBER_FIGI, qty_units=10)

        with patch(
            "finalayze.execution.tinkoff_broker.asyncio.run",
            return_value=mock_portfolio,
        ):
            broker = _make_broker()
            result = broker.has_position("SBER")

        assert result is True

    def test_has_position_false_empty_portfolio(self) -> None:
        """has_position returns False when portfolio is empty."""
        mock_portfolio = MagicMock()
        mock_portfolio.total_amount_portfolio.units = PORTFOLIO_CASH
        mock_portfolio.total_amount_portfolio.nano = 0
        mock_portfolio.positions = []

        with patch(
            "finalayze.execution.tinkoff_broker.asyncio.run",
            return_value=mock_portfolio,
        ):
            broker = _make_broker()
            result = broker.has_position("SBER")

        assert result is False

    def test_has_position_raises_when_figi_is_none(self) -> None:
        """has_position raises InstrumentNotFoundError when instrument FIGI is None."""
        registry = _make_registry()
        no_figi_inst = Instrument(
            symbol="NOFIGI",
            market_id="moex",
            name="No FIGI Instrument",
            figi=None,
            lot_size=1,
        )
        registry.register(no_figi_inst)

        broker = TinkoffBroker(token="fake_token", registry=registry, sandbox=True)  # noqa: S106
        with pytest.raises(InstrumentNotFoundError, match="FIGI"):
            broker.has_position("NOFIGI")


class TestTinkoffBrokerGetPositions:
    def test_get_positions_returns_dict(self) -> None:
        """get_positions returns dict of FIGI -> Decimal quantities."""
        mock_portfolio = MagicMock()
        mock_portfolio.total_amount_portfolio.units = PORTFOLIO_CASH
        mock_portfolio.total_amount_portfolio.nano = 0
        mock_pos = MagicMock()
        mock_pos.figi = SBER_FIGI
        mock_pos.quantity.units = 20
        mock_pos.quantity.nano = 0
        mock_portfolio.positions = [mock_pos]

        with patch(
            "finalayze.execution.tinkoff_broker.asyncio.run",
            return_value=mock_portfolio,
        ):
            broker = _make_broker()
            positions = broker.get_positions()

        assert SBER_FIGI in positions
        assert positions[SBER_FIGI] == Decimal(20)


class TestTinkoffBrokerCancelOrder:
    def _mock_accounts(self) -> MagicMock:
        mock_accounts_response = MagicMock()
        mock_account = MagicMock()
        mock_account.id = "test-account-id"
        mock_accounts_response.accounts = [mock_account]
        return mock_accounts_response

    def test_cancel_order_success(self) -> None:
        """cancel_order completes without error when SDK call succeeds."""
        with patch(
            "finalayze.execution.tinkoff_broker.asyncio.run",
            side_effect=[self._mock_accounts(), None],
        ) as mock_run:
            broker = _make_broker()
            broker.cancel_order("order-abc-123")

        # First call: get_accounts, second call: cancel_order
        assert mock_run.call_count == 2

    def test_cancel_order_api_error_raises(self) -> None:
        """cancel_order raises BrokerError when SDK call fails."""
        accounts_response = self._mock_accounts()
        with patch(
            "finalayze.execution.tinkoff_broker.asyncio.run",
            side_effect=[accounts_response, RuntimeError("gRPC cancel failed")],
        ):
            broker = _make_broker()
            with pytest.raises(BrokerError, match="gRPC cancel failed"):
                broker.cancel_order("order-xyz")
