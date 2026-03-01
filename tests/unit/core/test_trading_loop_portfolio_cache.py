"""Tests for 6D.14: Per-cycle portfolio caching in TradingLoop."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import MagicMock, patch

from finalayze.core.schemas import PortfolioState


def _make_portfolio(equity: Decimal = Decimal(10000)) -> PortfolioState:
    return PortfolioState(
        equity=equity,
        cash=equity,
        positions={},
        timestamp=datetime.now(UTC),
    )


def _make_loop() -> MagicMock:
    """Create a minimal TradingLoop-like mock with _get_cached_portfolio wired."""
    from finalayze.core.trading_loop import TradingLoop

    # We need to test _get_cached_portfolio in isolation, so patch __init__
    with patch.object(TradingLoop, "__init__", lambda self: None):
        loop = TradingLoop()

    loop._cycle_portfolio_cache = {}
    loop._broker_router = MagicMock()
    return loop


class TestPortfolioCache:
    """Verify per-cycle portfolio caching reduces broker calls."""

    def test_get_cached_portfolio_calls_broker_once(self) -> None:
        """Second call for same market should return cached value."""
        loop = _make_loop()
        portfolio = _make_portfolio()
        loop._broker_router.route.return_value.get_portfolio.return_value = portfolio

        result1 = loop._get_cached_portfolio("us")
        result2 = loop._get_cached_portfolio("us")

        assert result1 is result2
        assert result1 is portfolio
        # Broker should only be called once
        loop._broker_router.route.return_value.get_portfolio.assert_called_once()

    def test_different_markets_get_separate_cache_entries(self) -> None:
        """Each market should get its own cached portfolio."""
        loop = _make_loop()
        us_portfolio = _make_portfolio(Decimal(10000))
        moex_portfolio = _make_portfolio(Decimal(500000))

        broker_us = MagicMock()
        broker_us.get_portfolio.return_value = us_portfolio
        broker_moex = MagicMock()
        broker_moex.get_portfolio.return_value = moex_portfolio

        loop._broker_router.route.side_effect = lambda m: broker_us if m == "us" else broker_moex

        result_us = loop._get_cached_portfolio("us")
        result_moex = loop._get_cached_portfolio("moex")

        assert result_us is us_portfolio
        assert result_moex is moex_portfolio

    def test_cache_returns_none_on_broker_error(self) -> None:
        """If broker raises, should return None and not cache."""
        loop = _make_loop()
        loop._broker_router.route.return_value.get_portfolio.side_effect = RuntimeError("fail")

        result = loop._get_cached_portfolio("us")
        assert result is None
        assert "us" not in loop._cycle_portfolio_cache

    def test_get_market_equity_uses_cache(self) -> None:
        """_get_market_equity should delegate to _get_cached_portfolio."""
        loop = _make_loop()
        portfolio = _make_portfolio(Decimal(12345))
        loop._broker_router.route.return_value.get_portfolio.return_value = portfolio

        equity = loop._get_market_equity("us")
        assert equity == Decimal(12345)

        # Second call should not hit broker again
        equity2 = loop._get_market_equity("us")
        assert equity2 == Decimal(12345)
        loop._broker_router.route.return_value.get_portfolio.assert_called_once()
