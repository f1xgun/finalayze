"""Tests for MetricsCollector wiring in TradingLoop (6D.9)."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest


class TestSubmitOrderMetrics:
    """Verify MetricsCollector is called on trade fill and rejection."""

    def test_record_trade_on_fill(self) -> None:
        """MetricsCollector.record_trade called when order is filled."""
        from finalayze.core.trading_loop import TradingLoop

        loop = _make_loop_stub()
        order = MagicMock()
        order.side = "BUY"
        order.symbol = "AAPL"

        result = MagicMock()
        result.filled = True
        result.fill_price = Decimal("150.00")
        loop._broker_router.submit.return_value = result

        with patch("finalayze.api.metrics.MetricsCollector") as mc:
            TradingLoop._submit_order(loop, order, "us", candles=[_fake_candle()])

        mc.record_trade.assert_called_once()
        call_kwargs = mc.record_trade.call_args
        assert call_kwargs[1]["market"] == "us" or call_kwargs[0][0] == "us"

    def test_record_rejection_on_unfilled(self) -> None:
        """MetricsCollector.record_rejection called when order is not filled."""
        from finalayze.core.trading_loop import TradingLoop

        loop = _make_loop_stub()
        order = MagicMock()
        order.side = "BUY"
        order.symbol = "AAPL"

        result = MagicMock()
        result.filled = False
        result.reason = "insufficient funds"
        loop._broker_router.submit.return_value = result

        with patch("finalayze.api.metrics.MetricsCollector") as mc:
            TradingLoop._submit_order(loop, order, "us")

        mc.record_rejection.assert_called_once()


class TestProcessInstrumentMetrics:
    """Verify MetricsCollector.record_signal is called after signal generation."""

    def test_record_signal_called(self) -> None:
        from finalayze.core.schemas import SignalDirection
        from finalayze.core.trading_loop import TradingLoop

        loop = _make_loop_stub()

        signal = MagicMock()
        signal.direction = SignalDirection.BUY
        signal.confidence = 0.8
        signal.strategy_name = "momentum"
        loop._strategy.generate_signal.return_value = signal

        candles = [_fake_candle()]
        fetcher = MagicMock()
        fetcher.fetch_candles.return_value = candles

        instrument = MagicMock()
        instrument.symbol = "AAPL"
        instrument.segment_id = "us_tech"

        from finalayze.risk.circuit_breaker import CircuitLevel

        # Mock pre-trade checker to pass
        pre_result = MagicMock()
        pre_result.passed = True
        loop._pre_trade_checker.check.return_value = pre_result

        # Mock broker
        portfolio = MagicMock()
        portfolio.cash = Decimal("10000")
        portfolio.equity = Decimal("10000")
        portfolio.positions = {}
        broker = MagicMock()
        broker.get_portfolio.return_value = portfolio
        loop._broker_router.route.return_value = broker

        # Mock _compute_total_equity_base
        loop._compute_total_equity_base = MagicMock(return_value=Decimal("10000"))

        # Mock submit result
        submit_result = MagicMock()
        submit_result.filled = True
        submit_result.fill_price = Decimal("150.00")
        loop._broker_router.submit.return_value = submit_result

        with patch("finalayze.api.metrics.MetricsCollector") as mc:
            TradingLoop._process_instrument(
                loop, instrument, "us", CircuitLevel.NORMAL, fetcher, MagicMock()
            )

        mc.record_signal.assert_called_once()


class TestMarketCycleMetrics:
    """Verify portfolio equity and circuit breaker level metrics are set."""

    def test_equity_and_cb_level_set(self) -> None:
        from finalayze.core.trading_loop import TradingLoop
        from finalayze.risk.circuit_breaker import CircuitLevel

        loop = _make_loop_stub()
        loop._registry.list_by_market.return_value = []  # no instruments

        market_equities = {"us": Decimal("50000")}

        with patch("finalayze.api.metrics.MetricsCollector") as mc:
            TradingLoop._process_market_cycle(
                loop, "us", CircuitLevel.NORMAL, market_equities, MagicMock()
            )

        mc.set_portfolio_equity.assert_called_once_with("us", 50000.0)
        mc.set_circuit_breaker_level.assert_called_once()


# ── Helpers ──────────────────────────────────────────────────────────────────


def _fake_candle() -> MagicMock:
    c = MagicMock()
    c.close = Decimal("150.00")
    c.high = Decimal("155.00")
    c.low = Decimal("145.00")
    c.open = Decimal("148.00")
    c.volume = 1000
    return c


def _make_loop_stub() -> MagicMock:
    """Create a MagicMock that acts as a TradingLoop instance with enough attributes."""
    import threading

    from finalayze.execution.broker_base import OrderRequest
    from finalayze.risk.circuit_breaker import CircuitLevel

    loop = MagicMock()
    loop._OrderRequest = OrderRequest
    loop._CircuitLevel = CircuitLevel
    loop._stop_loss_lock = threading.Lock()
    loop._stop_loss_prices = {}
    loop._sentiment_lock = threading.Lock()
    loop._sentiment_cache = {}
    loop._cache = None
    loop._event_bus = None
    loop._fx_service = None
    loop._alerter = MagicMock()
    loop._broker_router = MagicMock()
    loop._strategy = MagicMock()
    loop._registry = MagicMock()
    loop._settings = MagicMock()
    loop._settings.max_cross_market_exposure_pct = 0.80
    loop._pre_trade_checker = MagicMock()
    loop._kelly_sizer = MagicMock()
    loop._kelly_sizer.optimal_fraction.return_value = Decimal("0.1")
    loop._circuit_breakers = {}
    loop._fx = MagicMock()
    return loop
