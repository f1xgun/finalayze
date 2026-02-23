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
