"""Integration test: full strategy cycle signal -> circuit breaker -> order -> alert."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from finalayze.analysis.event_classifier import EventClassifier, EventType
from finalayze.analysis.impact_estimator import ImpactEstimator
from finalayze.analysis.news_analyzer import NewsAnalyzer
from finalayze.core.alerts import TelegramAlerter
from finalayze.core.schemas import (
    Candle,
    SentimentResult,
    Signal,
    SignalDirection,
)
from finalayze.core.trading_loop import TradingLoop
from finalayze.data.fetchers.newsapi import NewsApiFetcher
from finalayze.execution.broker_base import OrderResult
from finalayze.execution.broker_router import BrokerRouter
from finalayze.markets.instruments import Instrument, InstrumentRegistry
from finalayze.risk.circuit_breaker import CircuitBreaker, CircuitLevel, CrossMarketCircuitBreaker

# ── Constants ──────────────────────────────────────────────────────────────
MARKET_US = "us"
SEGMENT_US_TECH = "us_tech"
SYMBOL_AAPL = "AAPL"
BASELINE_EQUITY = Decimal(100000)
AVAILABLE_CASH = Decimal(50000)
FILL_PRICE = Decimal("150.00")
ORDER_QTY = Decimal(10)
NUM_CANDLES = 60
CANDLE_CLOSE = Decimal("150.00")
NEWS_CYCLE_MINUTES = 30
STRATEGY_CYCLE_MINUTES = 60
DAILY_RESET_HOUR = 0


@pytest.mark.integration
class TestStrategyIntegration:
    """Full cycle: signal flows through circuit breaker check -> order submitted -> alert fired."""

    def _build_system(
        self,
        circuit_level: CircuitLevel = CircuitLevel.NORMAL,
        fill: bool = True,
    ) -> TradingLoop:
        settings = MagicMock()
        settings.news_cycle_minutes = NEWS_CYCLE_MINUTES
        settings.strategy_cycle_minutes = STRATEGY_CYCLE_MINUTES
        settings.daily_reset_hour_utc = DAILY_RESET_HOUR
        settings.max_position_pct = 0.20
        settings.kelly_fraction = 0.5
        settings.max_positions_per_market = 10

        # Real instrument registry
        registry = InstrumentRegistry()
        registry.register(
            Instrument(
                symbol=SYMBOL_AAPL,
                market_id=MARKET_US,
                name="Apple Inc.",
                segment_id=SEGMENT_US_TECH,
            )
        )

        # Mock fetcher
        base_ts = datetime(2025, 1, 1, 14, 30, tzinfo=UTC)
        candles = [
            Candle(
                symbol=SYMBOL_AAPL,
                market_id=MARKET_US,
                timeframe="1d",
                timestamp=base_ts + timedelta(days=i),
                open=CANDLE_CLOSE,
                high=CANDLE_CLOSE,
                low=CANDLE_CLOSE,
                close=CANDLE_CLOSE,
                volume=1_000_000,
            )
            for i in range(NUM_CANDLES)
        ]
        fetcher = MagicMock()
        fetcher.fetch_candles = MagicMock(return_value=candles)

        # Mock news stack
        news_fetcher = MagicMock(spec=NewsApiFetcher)
        news_fetcher.fetch_news = MagicMock(return_value=[])
        news_analyzer = MagicMock(spec=NewsAnalyzer)
        news_analyzer.analyze = AsyncMock(
            return_value=SentimentResult(sentiment=0.0, confidence=0.0, reasoning="")
        )
        event_classifier = MagicMock(spec=EventClassifier)
        event_classifier.classify = AsyncMock(return_value=EventType.OTHER)
        impact_estimator = MagicMock(spec=ImpactEstimator)
        impact_estimator.estimate = MagicMock(return_value=[])

        # Mock strategy -- returns a BUY signal
        strategy = MagicMock()
        strategy.generate_signal = MagicMock(
            return_value=Signal(
                strategy_name="combined",
                symbol=SYMBOL_AAPL,
                market_id=MARKET_US,
                segment_id=SEGMENT_US_TECH,
                direction=SignalDirection.BUY,
                confidence=0.80,
                features={},
                reasoning="integration test signal",
            )
        )

        # Mock broker
        portfolio = MagicMock()
        portfolio.equity = BASELINE_EQUITY
        portfolio.cash = AVAILABLE_CASH
        mock_broker = MagicMock()
        mock_broker.get_portfolio = MagicMock(return_value=portfolio)
        mock_broker.get_positions = MagicMock(return_value={})
        fill_result = OrderResult(
            filled=fill,
            fill_price=FILL_PRICE if fill else None,
            symbol=SYMBOL_AAPL,
            side="BUY",
            quantity=ORDER_QTY,
            reason="" if fill else "insufficient funds",
        )
        mock_broker.submit_order = MagicMock(return_value=fill_result)
        broker_router = MagicMock(spec=BrokerRouter)
        broker_router.route = MagicMock(return_value=mock_broker)
        broker_router.submit = MagicMock(return_value=fill_result)
        broker_router.registered_markets = [MARKET_US]

        # Real circuit breaker (with mocked check to return desired level)
        cb = MagicMock(spec=CircuitBreaker)
        cb.level = circuit_level
        cb.market_id = MARKET_US
        cb.check = MagicMock(return_value=circuit_level)
        cb.reset_daily = MagicMock()

        cmcb = MagicMock(spec=CrossMarketCircuitBreaker)
        cmcb.check = MagicMock(return_value=False)
        cmcb.reset_daily = MagicMock()

        alerter = MagicMock(spec=TelegramAlerter)

        return TradingLoop(
            settings=settings,
            fetchers={MARKET_US: fetcher},
            news_fetcher=news_fetcher,
            news_analyzer=news_analyzer,
            event_classifier=event_classifier,
            impact_estimator=impact_estimator,
            strategy=strategy,
            broker_router=broker_router,
            circuit_breakers={MARKET_US: cb},
            cross_market_breaker=cmcb,
            alerter=alerter,
            instrument_registry=registry,
        )

    def test_signal_flows_through_to_order_submit(self) -> None:
        loop = self._build_system(circuit_level=CircuitLevel.NORMAL, fill=True)
        loop._strategy_cycle()
        loop._broker_router.submit.assert_called_once()

    def test_filled_order_fires_alert(self) -> None:
        loop = self._build_system(circuit_level=CircuitLevel.NORMAL, fill=True)
        loop._strategy_cycle()
        loop._alerter.on_trade_filled.assert_called_once()

    def test_rejected_order_fires_reject_alert(self) -> None:
        loop = self._build_system(circuit_level=CircuitLevel.NORMAL, fill=False)
        loop._strategy_cycle()
        loop._alerter.on_trade_rejected.assert_called_once()

    def test_halted_circuit_blocks_all_orders(self) -> None:
        loop = self._build_system(circuit_level=CircuitLevel.HALTED)
        loop._strategy_cycle()
        loop._broker_router.submit.assert_not_called()

    def test_caution_circuit_allows_reduced_orders(self) -> None:
        loop = self._build_system(circuit_level=CircuitLevel.CAUTION, fill=True)
        loop._strategy_cycle()
        loop._broker_router.submit.assert_called()
