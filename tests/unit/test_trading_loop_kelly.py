"""Unit tests for TradingLoop Kelly update wiring."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import MagicMock, patch

from config.settings import Settings

from finalayze.core.alerts import TelegramAlerter
from finalayze.core.modes import WorkMode
from finalayze.core.schemas import Candle, SignalDirection
from finalayze.core.trading_loop import TradingLoop
from finalayze.execution.broker_base import OrderResult
from finalayze.markets.instruments import Instrument, InstrumentRegistry
from finalayze.risk.circuit_breaker import CircuitBreaker, CircuitLevel, CrossMarketCircuitBreaker
from finalayze.risk.kelly import TradeRecord

# ── Constants ──────────────────────────────────────────────────────────
MARKET_US = "us"
SYMBOL = "AAPL"
FILL_PRICE_ENTRY = Decimal("150.00")
FILL_PRICE_EXIT = Decimal("160.00")
BASELINE_EQUITY = Decimal(100000)
NUM_CANDLES = 60
CANDLE_CLOSE = Decimal("150.00")


def _make_candle(idx: int = 0) -> Candle:
    base = datetime(2025, 1, 1, 14, 30, tzinfo=UTC)
    return Candle(
        symbol=SYMBOL,
        market_id=MARKET_US,
        timeframe="1d",
        timestamp=base + timedelta(days=idx),
        open=CANDLE_CLOSE,
        high=CANDLE_CLOSE,
        low=CANDLE_CLOSE,
        close=CANDLE_CLOSE,
        volume=1_000_000,
    )


def _make_candles(n: int = NUM_CANDLES) -> list[Candle]:
    return [_make_candle(idx=i) for i in range(n)]


def _make_settings() -> MagicMock:
    s = MagicMock(spec=Settings)
    s.news_cycle_minutes = 30
    s.strategy_cycle_minutes = 60
    s.daily_reset_hour_utc = 0
    s.max_position_pct = 0.20
    s.kelly_fraction = 0.5
    s.max_positions_per_market = 10
    s.daily_loss_limit_pct = 0.03
    s.max_cross_market_exposure_pct = 0.80
    s.mode = WorkMode.SANDBOX
    return s


def _make_loop() -> TradingLoop:
    settings = _make_settings()
    fetchers = {MARKET_US: MagicMock()}
    news_fetcher = MagicMock()
    news_analyzer = MagicMock()
    event_classifier = MagicMock()
    impact_estimator = MagicMock()
    strategy = MagicMock()
    broker_router = MagicMock()
    alerter = MagicMock(spec=TelegramAlerter)

    cb_us = CircuitBreaker(market_id=MARKET_US)
    circuit_breakers = {MARKET_US: cb_us}
    cross_market = CrossMarketCircuitBreaker()

    registry = InstrumentRegistry()

    return TradingLoop(
        settings=settings,
        fetchers=fetchers,
        news_fetcher=news_fetcher,
        news_analyzer=news_analyzer,
        event_classifier=event_classifier,
        impact_estimator=impact_estimator,
        strategy=strategy,
        broker_router=broker_router,
        circuit_breakers=circuit_breakers,
        cross_market_breaker=cross_market,
        alerter=alerter,
        instrument_registry=registry,
    )


class TestTradingLoopKellyUpdate:
    """Verify that Kelly sizer is updated on SELL fills."""

    def test_buy_fill_records_entry_price(self) -> None:
        """A BUY fill should store the entry price."""
        loop = _make_loop()
        candles = _make_candles()

        # Mock broker router to return a filled BUY order
        order = loop._OrderRequest(symbol=SYMBOL, side="BUY", quantity=Decimal(10))
        result = OrderResult(filled=True, fill_price=FILL_PRICE_ENTRY)
        loop._broker_router.submit.return_value = result

        loop._submit_order(order, MARKET_US, candles=candles)

        assert SYMBOL in loop._entry_prices
        assert loop._entry_prices[SYMBOL] == FILL_PRICE_ENTRY

    def test_sell_fill_updates_kelly(self) -> None:
        """A SELL fill should call kelly.update with PnL."""
        loop = _make_loop()

        # Simulate a prior BUY entry
        loop._entry_prices[SYMBOL] = FILL_PRICE_ENTRY

        order = loop._OrderRequest(symbol=SYMBOL, side="SELL", quantity=Decimal(10))
        result = OrderResult(filled=True, fill_price=FILL_PRICE_EXIT)
        loop._broker_router.submit.return_value = result

        initial_count = loop._kelly_sizer.trade_count
        loop._submit_order(order, MARKET_US)

        assert loop._kelly_sizer.trade_count == initial_count + 1
        assert SYMBOL not in loop._entry_prices

    def test_stop_loss_updates_kelly(self) -> None:
        """A stop-loss sell should also update Kelly."""
        loop = _make_loop()

        # Setup: position and entry price exist
        loop._entry_prices[SYMBOL] = FILL_PRICE_ENTRY
        loop._stop_loss_prices[SYMBOL] = Decimal("140.00")

        broker = MagicMock()
        broker.get_positions.return_value = {SYMBOL: Decimal(10)}
        loop._broker_router.route.return_value = broker

        initial_count = loop._kelly_sizer.trade_count
        stop_price = Decimal("139.00")
        loop._check_stop_losses(MARKET_US, SYMBOL, stop_price)

        assert loop._kelly_sizer.trade_count == initial_count + 1
        assert SYMBOL not in loop._entry_prices
