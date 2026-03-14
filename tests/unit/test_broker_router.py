"""Unit tests for BrokerRouter."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from finalayze.core.exceptions import BrokerError
from finalayze.execution.broker_base import OrderRequest, OrderResult
from finalayze.execution.broker_router import BrokerRouter


def _make_mock_broker(market_id: str) -> MagicMock:
    broker = MagicMock()
    broker.market_id = market_id
    return broker


def _make_router() -> tuple[BrokerRouter, MagicMock, MagicMock]:
    us_broker = _make_mock_broker("us")
    moex_broker = _make_mock_broker("moex")
    router = BrokerRouter({"us": us_broker, "moex": moex_broker})
    return router, us_broker, moex_broker


# ---------- tests ----------


class TestBrokerRouterRoute:
    def test_routes_us_order_to_alpaca(self) -> None:
        router, us_broker, _ = _make_router()
        routed = router.route("us")
        assert routed is us_broker

    def test_routes_moex_order_to_tinkoff(self) -> None:
        router, _, moex_broker = _make_router()
        routed = router.route("moex")
        assert routed is moex_broker

    def test_unknown_market_raises_broker_error(self) -> None:
        router, _, _ = _make_router()
        with pytest.raises(BrokerError, match="No broker registered for market"):
            router.route("london")


class TestBrokerRouterSubmit:
    def test_submit_delegates_to_correct_broker(self) -> None:
        router, us_broker, _ = _make_router()
        expected_result = OrderResult(filled=True, symbol="AAPL", side="BUY")
        us_broker.submit_order.return_value = expected_result

        order = OrderRequest(symbol="AAPL", side="BUY", quantity=Decimal(5))
        result = router.submit(order, market_id="us")

        us_broker.submit_order.assert_called_once_with(order, fill_candle=None)
        assert result is expected_result

    def test_submit_moex_delegates_to_tinkoff(self) -> None:
        router, _, moex_broker = _make_router()
        expected_result = OrderResult(filled=True, symbol="SBER", side="SELL")
        moex_broker.submit_order.return_value = expected_result

        order = OrderRequest(symbol="SBER", side="SELL", quantity=Decimal(10))
        result = router.submit(order, market_id="moex")

        moex_broker.submit_order.assert_called_once_with(order, fill_candle=None)
        assert result is expected_result


class TestBrokerRouterRegistration:
    def test_empty_router_raises_on_route(self) -> None:
        router = BrokerRouter({})
        with pytest.raises(BrokerError, match="No broker registered"):
            router.route("us")

    def test_registered_markets(self) -> None:
        router, _, _ = _make_router()
        assert set(router.registered_markets) == {"us", "moex"}


# ── moex_bonds routing ────────────────────────────────────────────────────


class TestBrokerRouterMoexBonds:
    def test_routes_moex_bonds_to_bond_broker(self) -> None:
        """BrokerRouter with 'moex_bonds' key routes to bond broker."""
        mock_broker = _make_mock_broker("moex")
        mock_bond_broker = _make_mock_broker("moex_bonds")
        router = BrokerRouter({"moex": mock_broker, "moex_bonds": mock_bond_broker})
        routed = router.route("moex_bonds")
        assert routed is mock_bond_broker

    def test_submit_through_moex_bonds(self) -> None:
        """submit() works through moex_bonds route."""
        mock_bond_broker = _make_mock_broker("moex_bonds")
        expected = OrderResult(filled=True, symbol="SU26244RMFS2", side="BUY")
        mock_bond_broker.submit_order.return_value = expected

        router = BrokerRouter({"moex_bonds": mock_bond_broker})
        order = OrderRequest(symbol="SU26244RMFS2", side="BUY", quantity=Decimal(5))
        result = router.submit(order, market_id="moex_bonds")
        assert result is expected
        mock_bond_broker.submit_order.assert_called_once()


# ── make_bond_broker factory ──────────────────────────────────────────────


class TestMakeBondBroker:
    def test_creates_broker_sharing_client(self) -> None:
        """make_bond_broker creates TinkoffBroker sharing equity broker's AsyncClient."""
        from finalayze.execution.tinkoff_broker import TinkoffBroker, make_bond_broker

        mock_registry = MagicMock()
        equity_broker = TinkoffBroker(
            token="test-token",
            registry=mock_registry,
            sandbox=True,
        )
        # Manually set a mock client
        mock_client = MagicMock()
        equity_broker._client = mock_client

        bond_broker = make_bond_broker(equity_broker)
        assert isinstance(bond_broker, TinkoffBroker)
        assert bond_broker._client is mock_client  # same client
        assert bond_broker is not equity_broker  # different instance
