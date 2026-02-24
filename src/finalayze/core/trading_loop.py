"""APScheduler-based live trading loop (Layer -- sits above Layer 5).

Orchestrates three scheduled cycles:
  - _news_cycle: fetch news, analyze sentiment, update _sentiment_cache
  - _strategy_cycle: for each instrument, generate signal, apply circuit breakers,
    submit orders via BrokerRouter, fire alerts
  - _daily_reset: reset circuit breakers, send daily P&L summary

Thread safety: _sentiment_cache is protected by _sentiment_lock (threading.Lock).

See docs/architecture/DEPENDENCY_LAYERS.md for layering rules.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, Literal

from apscheduler.schedulers.background import BackgroundScheduler

from finalayze.core.schemas import NewsArticle, SignalDirection
from finalayze.execution.broker_base import OrderRequest
from finalayze.risk.circuit_breaker import CircuitLevel

if TYPE_CHECKING:
    from config.settings import Settings

    from finalayze.analysis.event_classifier import EventClassifier, EventType
    from finalayze.analysis.impact_estimator import ImpactEstimator
    from finalayze.analysis.news_analyzer import NewsAnalyzer
    from finalayze.core.alerts import TelegramAlerter
    from finalayze.core.schemas import Candle, SentimentResult, Signal
    from finalayze.data.fetchers.newsapi import NewsApiFetcher
    from finalayze.execution.broker_base import BrokerBase
    from finalayze.execution.broker_router import BrokerRouter
    from finalayze.markets.instruments import Instrument, InstrumentRegistry
    from finalayze.risk.circuit_breaker import CircuitBreaker, CrossMarketCircuitBreaker
    from finalayze.strategies.combiner import StrategyCombiner

# ── Constants ──────────────────────────────────────────────────────────────
_NEWS_QUERY = "stock market finance"
_NEWS_LOOKBACK_HOURS = 2
_CANDLE_LOOKBACK = 60  # number of bars to fetch per symbol
_CAUTION_SIZE_FACTOR = Decimal("0.5")  # halve position size at CAUTION
_MIN_CONFIDENCE_BOOST = 1.2  # raise required confidence 20% at CAUTION
_DEFAULT_SENTIMENT = 0.0
_ZERO = Decimal(0)

_log = logging.getLogger(__name__)


class TradingLoop:
    """Schedules and runs the news, strategy, and daily-reset cycles.

    Designed for TEST / SANDBOX modes. Will gate on WorkMode in real mode.
    """

    def __init__(
        self,
        settings: Settings,
        fetchers: dict[str, object],
        news_fetcher: NewsApiFetcher,
        news_analyzer: NewsAnalyzer,
        event_classifier: EventClassifier,
        impact_estimator: ImpactEstimator,
        strategy: StrategyCombiner,
        broker_router: BrokerRouter,
        circuit_breakers: dict[str, CircuitBreaker],
        cross_market_breaker: CrossMarketCircuitBreaker,
        alerter: TelegramAlerter,
        instrument_registry: InstrumentRegistry,
    ) -> None:
        self._settings = settings
        self._fetchers = fetchers
        self._news_fetcher = news_fetcher
        self._news_analyzer = news_analyzer
        self._event_classifier = event_classifier
        self._impact_estimator = impact_estimator
        self._strategy = strategy
        self._broker_router = broker_router
        self._circuit_breakers = circuit_breakers
        self._cross_market_breaker = cross_market_breaker
        self._alerter = alerter
        self._registry = instrument_registry

        # Thread-safe sentiment cache: segment_id -> weighted sentiment score
        self._sentiment_cache: dict[str, float] = {}
        self._sentiment_lock = threading.Lock()

        # Daily baseline equities: market_id -> equity at start of trading day
        self._baseline_equities: dict[str, Decimal] = {}

        self._scheduler: BackgroundScheduler | None = None
        self._stop_event = threading.Event()

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the APScheduler and block until stop() is called."""
        self._scheduler = BackgroundScheduler(timezone="UTC")
        self._scheduler.add_job(
            self._news_cycle,
            "interval",
            minutes=self._settings.news_cycle_minutes,
        )
        self._scheduler.add_job(
            self._strategy_cycle,
            "interval",
            minutes=self._settings.strategy_cycle_minutes,
        )
        self._scheduler.add_job(
            self._daily_reset,
            "cron",
            hour=self._settings.daily_reset_hour_utc,
            minute=0,
        )
        self._scheduler.start()
        _log.info(
            "TradingLoop started: news=%dm, strategy=%dm, daily_reset=UTC %02d:00",
            self._settings.news_cycle_minutes,
            self._settings.strategy_cycle_minutes,
            self._settings.daily_reset_hour_utc,
        )
        self._stop_event.wait()

    def stop(self) -> None:
        """Gracefully shut down the scheduler and unblock start()."""
        if self._scheduler is not None:
            self._scheduler.shutdown(wait=True)
        self._stop_event.set()

    # ── Cycles ───────────────────────────────────────────────────────────────

    def _news_cycle(self) -> None:
        """Fetch latest news, analyze sentiment, update _sentiment_cache."""
        now = datetime.now(UTC)
        from_date = now - timedelta(hours=_NEWS_LOOKBACK_HOURS)
        try:
            articles = self._news_fetcher.fetch_news(
                query=_NEWS_QUERY,
                from_date=from_date,
                to_date=now,
            )
        except Exception:
            _log.exception("_news_cycle: failed to fetch news")
            self._alerter.on_error("NewsApiFetcher", "fetch_news failed")
            return

        for article in articles:
            try:
                self._process_news_article(article)
            except Exception:
                _log.exception("_news_cycle: error processing article %s", article.id)

    async def _analyze_article(self, article: NewsArticle) -> tuple[SentimentResult, EventType]:
        """Run sentiment analysis and event classification concurrently."""
        sentiment, event = await asyncio.gather(
            self._news_analyzer.analyze(article),
            self._event_classifier.classify(article),
        )
        return sentiment, event

    def _process_news_article(self, article: NewsArticle) -> None:
        """Analyze a single article and update sentiment cache."""
        sentiment, event = asyncio.run(self._analyze_article(article))
        active_segments = self._collect_active_segments()
        impacts = self._impact_estimator.estimate(
            article,
            event,
            sentiment,
            active_segments,
        )
        with self._sentiment_lock:
            for impact in impacts:
                existing = self._sentiment_cache.get(impact.segment_id, _DEFAULT_SENTIMENT)
                self._sentiment_cache[impact.segment_id] = existing * 0.7 + impact.sentiment * 0.3

    def _collect_active_segments(self) -> list[str]:
        """Collect distinct segment IDs across all markets."""
        return list(
            {
                seg
                for market_id in self._fetchers
                for instr in self._registry.list_by_market(market_id)
                if hasattr(instr, "segment_id") and instr.segment_id
                for seg in [instr.segment_id]
            }
        )

    def _strategy_cycle(self) -> None:
        """For each market and instrument, generate a signal and submit orders."""
        market_equities: dict[str, Decimal] = {}
        baseline_equities: dict[str, Decimal] = {}

        for market_id, cb in self._circuit_breakers.items():
            equity = self._get_market_equity(market_id)
            if equity is None:
                continue

            market_equities[market_id] = equity
            baseline = self._baseline_equities.get(market_id, equity)
            baseline_equities[market_id] = baseline

            level = cb.check(current_equity=equity, baseline_equity=baseline)
            self._handle_market_level(market_id, level)

        if self._cross_market_breaker.check(market_equities, baseline_equities):
            _log.warning("CrossMarketCircuitBreaker tripped -- all markets halted")
            self._alerter.on_circuit_breaker_trip("all", CircuitLevel.HALTED, 0.0)

    def _get_market_equity(self, market_id: str) -> Decimal | None:
        """Return current portfolio equity for market, or None on failure."""
        try:
            broker = self._broker_router.route(market_id)
            portfolio = broker.get_portfolio()
            equity: Decimal = portfolio.equity
            return equity
        except Exception:
            _log.exception("_strategy_cycle: failed to get portfolio for %s", market_id)
            return None

    def _handle_market_level(self, market_id: str, level: CircuitLevel) -> None:
        """Process one market based on its current circuit breaker level."""
        if level == CircuitLevel.LIQUIDATE:
            _log.warning("Circuit breaker LIQUIDATE for %s -- liquidating", market_id)
            self._liquidate_market(market_id)
            return

        if level == CircuitLevel.HALTED:
            _log.warning("Circuit breaker HALTED for %s -- skipping cycle", market_id)
            return

        fetcher = self._fetchers.get(market_id)
        if fetcher is None:
            _log.warning("No fetcher for market %s", market_id)
            return

        instruments = self._registry.list_by_market(market_id)
        for instrument in instruments:
            self._process_instrument(instrument, market_id, level, fetcher)

    def _process_instrument(
        self,
        instrument: Instrument,
        market_id: str,
        level: CircuitLevel,
        fetcher: object,
    ) -> None:
        """Fetch candles, generate signal, and submit order for one instrument."""
        seg_id = getattr(instrument, "segment_id", "") or "us_tech"
        try:
            candles: list[Candle] = fetcher.fetch_candles(  # type: ignore[attr-defined]
                symbol=instrument.symbol,
                market_id=market_id,
                limit=_CANDLE_LOOKBACK,
            )
        except Exception:
            _log.exception("_strategy_cycle: failed to fetch candles for %s", instrument.symbol)
            return

        with self._sentiment_lock:
            sentiment_score = self._sentiment_cache.get(seg_id, _DEFAULT_SENTIMENT)

        signal = self._strategy.generate_signal(
            instrument.symbol, candles, seg_id, sentiment_score=sentiment_score
        )
        if signal is None:
            return

        _log.debug(
            "_process_instrument: signal=%s sentiment_score=%.3f symbol=%s",
            signal.direction,
            sentiment_score,
            instrument.symbol,
        )

        broker = self._broker_router.route(market_id)
        portfolio = broker.get_portfolio()
        order = self._build_order(signal, level, portfolio.cash, candles, instrument.symbol)
        if order is None:
            return

        self._submit_order(order, market_id)

    def _build_order(
        self,
        signal: Signal,
        level: CircuitLevel,
        available_cash: Decimal,
        candles: list[Candle],
        symbol: str,
    ) -> OrderRequest | None:
        """Build an order from signal, respecting CAUTION size reduction."""
        if level == CircuitLevel.CAUTION:
            min_conf = 0.5 * _MIN_CONFIDENCE_BOOST
            if signal.confidence < min_conf:
                return None

        order_value = Decimal(str(signal.confidence)) * available_cash
        if level == CircuitLevel.CAUTION:
            order_value = order_value * _CAUTION_SIZE_FACTOR

        qty = (order_value / Decimal(str(candles[-1].close))) if candles else _ZERO
        qty = qty.quantize(Decimal(1))
        if qty <= _ZERO:
            return None

        side: Literal["BUY", "SELL"] = "BUY" if signal.direction == SignalDirection.BUY else "SELL"
        return OrderRequest(symbol=symbol, side=side, quantity=qty)

    def _submit_order(self, order: OrderRequest, market_id: str) -> None:
        """Submit order and fire the appropriate alert."""
        try:
            result = self._broker_router.submit(order, market_id=market_id)
            if result.filled:
                self._alerter.on_trade_filled(result, market_id, broker=market_id)
            else:
                self._alerter.on_trade_rejected(order, result.reason)
        except Exception:
            _log.exception("_strategy_cycle: order submission failed for %s", order.symbol)

    def _daily_reset(self) -> None:
        """Reset circuit breakers and send daily P&L summary."""
        market_pnl: dict[str, Decimal] = {}
        new_baselines: dict[str, Decimal] = {}

        for market_id, cb in self._circuit_breakers.items():
            try:
                broker = self._broker_router.route(market_id)
                portfolio = broker.get_portfolio()
                equity = portfolio.equity
                new_baselines[market_id] = equity
                self._baseline_equities[market_id] = equity
                market_pnl[market_id] = _ZERO  # simplified: actual P&L tracked separately
                cb.reset_daily(new_baseline=equity)
            except Exception:
                _log.exception("_daily_reset: failed to reset for market %s", market_id)

        self._cross_market_breaker.reset_daily(new_baselines)
        total_equity = sum(new_baselines.values(), _ZERO)
        self._alerter.on_daily_summary(market_pnl, total_equity)
        _log.info("Daily reset complete. Total equity: %s", total_equity)

    def _liquidate_market(self, market_id: str) -> None:
        """Close all open positions in a market (L3 circuit breaker response)."""
        try:
            broker = self._broker_router.route(market_id)
            positions = broker.get_positions()
            portfolio = broker.get_portfolio()
            equity = portfolio.equity
            self._close_positions(broker, positions)
            drawdown = float(
                (equity - sum(positions.values(), _ZERO)) / equity if equity > _ZERO else 0
            )
            self._alerter.on_circuit_breaker_trip(market_id, CircuitLevel.LIQUIDATE, drawdown)
        except Exception:
            _log.exception("_liquidate_market: failed for market %s", market_id)
            self._alerter.on_error("TradingLoop", f"liquidation failed for {market_id}")

    def _close_positions(self, broker: BrokerBase, positions: dict[str, Decimal]) -> None:
        """Submit SELL orders for all non-zero positions."""
        for symbol, qty in positions.items():
            if qty <= _ZERO:
                continue
            order = OrderRequest(symbol=symbol, side="SELL", quantity=qty)
            try:
                broker.submit_order(order)
            except Exception as exc:
                _log.error("liquidation_order_failed", extra={"symbol": symbol, "error": str(exc)})
                self._alerter.on_error("TradingLoop", f"Liquidation failed for {symbol}: {exc}")
