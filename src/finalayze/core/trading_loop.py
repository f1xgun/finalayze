"""APScheduler-based live trading loop (Layer 6 -- top-level orchestrator).

Orchestrates three scheduled cycles:
  - _news_cycle: fetch news, analyze sentiment, update _sentiment_cache
  - _strategy_cycle: for each instrument, generate signal, apply circuit breakers,
    submit orders via BrokerRouter, fire alerts
  - _daily_reset: reset circuit breakers, send daily P&L summary

Thread safety: _sentiment_cache is protected by _sentiment_lock (threading.Lock).

Note: This module lives in ``core/`` for import convenience but it is
architecturally Layer 6 — it imports from L3 (analysis), L4 (risk/strategies),
and L5 (execution).  All higher-layer imports are deferred to avoid polluting
the ``core`` namespace at import time.

See docs/architecture/DEPENDENCY_LAYERS.md for layering rules.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from apscheduler.schedulers.background import BackgroundScheduler

from finalayze.core.schemas import NewsArticle, SignalDirection
from finalayze.markets.currency import CurrencyConverter

if TYPE_CHECKING:
    from config.settings import Settings

    from finalayze.analysis.event_classifier import EventClassifier, EventType
    from finalayze.analysis.impact_estimator import ImpactEstimator
    from finalayze.analysis.news_analyzer import NewsAnalyzer
    from finalayze.core.alerts import TelegramAlerter
    from finalayze.core.schemas import Candle, SentimentResult, Signal
    from finalayze.data.cache import RedisCache
    from finalayze.data.fetchers.newsapi import NewsApiFetcher
    from finalayze.execution.broker_base import BrokerBase, OrderRequest
    from finalayze.execution.broker_router import BrokerRouter
    from finalayze.markets.instruments import Instrument, InstrumentRegistry
    from finalayze.ml.registry import MLModelRegistry
    from finalayze.risk.circuit_breaker import (
        CircuitBreaker,
        CircuitLevel,
        CrossMarketCircuitBreaker,
    )
    from finalayze.strategies.combiner import StrategyCombiner

# ── Constants ──────────────────────────────────────────────────────────────
_NEWS_QUERY = "stock market finance"
_NEWS_LOOKBACK_HOURS = 2
_CANDLE_LOOKBACK = 60  # number of bars to fetch per symbol
_CAUTION_SIZE_FACTOR = Decimal("0.5")  # halve position size at CAUTION
_MIN_CONFIDENCE_BOOST = 1.2  # raise required confidence 20% at CAUTION
_DEFAULT_SENTIMENT = 0.0
_ZERO = Decimal(0)
_WEEKEND_WEEKDAY = 5  # Saturday=5, Sunday=6
_ATR_MULTIPLIER_US = Decimal("2.0")
_ATR_MULTIPLIER_MOEX = Decimal("2.5")
_MARKET_CURRENCY: dict[str, str] = {"us": "USD", "moex": "RUB"}

# US market hours in UTC: 9:30-16:00 ET = 14:30-21:00 UTC
_US_OPEN_UTC = (14, 30)
_US_CLOSE_UTC = (21, 0)
# MOEX market hours in UTC: 10:00-18:45 MSK = 07:00-15:45 UTC
_MOEX_OPEN_UTC = (7, 0)
_MOEX_CLOSE_UTC = (15, 45)

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
        cache: RedisCache | None = None,
        ml_registry: MLModelRegistry | None = None,
    ) -> None:
        from finalayze.execution.broker_base import OrderRequest  # noqa: PLC0415
        from finalayze.risk.circuit_breaker import CircuitLevel  # noqa: PLC0415
        from finalayze.risk.kelly import RollingKelly  # noqa: PLC0415
        from finalayze.risk.loss_limits import LossLimitTracker  # noqa: PLC0415
        from finalayze.risk.pre_trade_check import PreTradeChecker  # noqa: PLC0415

        # Store class references for runtime use without module-level imports
        self._OrderRequest = OrderRequest
        self._CircuitLevel = CircuitLevel

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
        self._cache = cache

        self._fx = CurrencyConverter(base_currency="USD")

        # Thread-safe sentiment cache: segment_id -> weighted sentiment score
        self._sentiment_cache: dict[str, float] = {}
        self._sentiment_lock = threading.Lock()

        # Daily baseline equities: market_id -> equity at start of trading day
        self._baseline_equities: dict[str, Decimal] = {}

        # Stop-loss tracking: symbol -> stop_loss_price
        self._stop_loss_prices: dict[str, Decimal] = {}

        # Risk management components
        self._pre_trade_checker = PreTradeChecker(
            max_position_pct=Decimal(str(settings.max_position_pct)),
            max_positions_per_market=settings.max_positions_per_market,
        )
        _raw_loss_limit = getattr(settings, "daily_loss_limit_pct", 0.05)
        self._loss_limit_tracker = LossLimitTracker(
            daily_loss_limit_pct=float(_raw_loss_limit) * 100,  # pct -> percent
        )
        self._kelly_sizer = RollingKelly(
            fraction=getattr(settings, "kelly_fraction", 0.5),
        )

        self._ml_registry = ml_registry
        self._scheduler: BackgroundScheduler | None = None
        self._stop_event = threading.Event()

        # Persistent background event loop for async calls (5.4)
        self._async_loop: asyncio.AbstractEventLoop | None = None
        self._async_thread: threading.Thread | None = None

    # ── Async helper ────────────────────────────────────────────────────────

    def _run_async(self, coro: Any) -> Any:
        """Run an async coroutine on a persistent background event loop.

        Lazily creates a daemon thread with its own event loop on first call.
        Uses ``run_coroutine_threadsafe`` with a 30-second timeout so the
        caller is never blocked indefinitely.
        """
        _async_timeout = 30
        if self._async_loop is None or self._async_loop.is_closed():
            loop = asyncio.new_event_loop()
            self._async_loop = loop
            thread = threading.Thread(target=loop.run_forever, daemon=True)
            thread.start()
            self._async_thread = thread
        future = asyncio.run_coroutine_threadsafe(coro, self._async_loop)
        return future.result(timeout=_async_timeout)

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the APScheduler and block until stop() is called."""
        from apscheduler.executors.pool import (  # noqa: PLC0415
            ThreadPoolExecutor as APSThreadPoolExecutor,
        )

        executors: dict[str, APSThreadPoolExecutor] = {
            "default": APSThreadPoolExecutor(max_workers=4),
            "retrain": APSThreadPoolExecutor(max_workers=1),
        }
        self._scheduler = BackgroundScheduler(timezone="UTC", executors=executors)
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
        if self._ml_registry is not None and getattr(self._settings, "ml_enabled", False):
            self._scheduler.add_job(
                self._retrain_cycle,
                "interval",
                hours=getattr(self._settings, "ml_retrain_interval_hours", 168),
                executor="retrain",
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
        """Gracefully shut down the scheduler, async loop, and unblock start()."""
        if self._scheduler is not None:
            self._scheduler.shutdown(wait=True)
        if self._async_loop is not None and not self._async_loop.is_closed():
            self._async_loop.call_soon_threadsafe(self._async_loop.stop)
            if self._async_thread is not None:
                self._async_thread.join(timeout=5)
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
        sentiment, event = self._run_async(self._analyze_article(article))
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
                new_score = existing * 0.7 + impact.sentiment * 0.3
                self._sentiment_cache[impact.segment_id] = new_score
                if self._cache is not None:
                    try:
                        self._run_async(self._cache.set_sentiment(impact.segment_id, new_score))
                    except Exception:
                        _log.debug("Failed to write sentiment to Redis cache")

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

    def _get_sentiment(self, seg_id: str) -> float:
        """Read sentiment from Redis cache (if available) or in-memory fallback."""
        if self._cache is not None:
            try:
                cached: float | None = self._run_async(self._cache.get_sentiment(seg_id))
                if cached is not None:
                    return cached
            except Exception:
                _log.debug("Failed to read sentiment from Redis cache")
        with self._sentiment_lock:
            return self._sentiment_cache.get(seg_id, _DEFAULT_SENTIMENT)

    def _now(self) -> datetime:
        """Return current UTC datetime. Extracted for testability."""
        return datetime.now(UTC)

    def _strategy_cycle(self) -> None:
        """For each market and instrument, generate a signal and submit orders."""
        # 6A.1: Mode gate -- DEBUG mode must not send real orders
        if not self._settings.mode.can_submit_orders():
            _log.info(
                "_strategy_cycle: mode=%s does not allow orders -- skipping",
                self._settings.mode,
            )
            return

        now = self._now()
        market_equities: dict[str, Decimal] = {}
        baseline_equities: dict[str, Decimal] = {}

        # Phase 1: Collect equities and evaluate circuit breaker levels.
        # Handle LIQUIDATE immediately (close positions), but defer instrument
        # processing until all safety gates have been checked.
        liquidate_markets: list[str] = []
        market_cb_levels: dict[str, CircuitLevel] = {}

        for market_id, cb in self._circuit_breakers.items():
            equity = self._get_market_equity(market_id)
            if equity is None:
                continue

            market_equities[market_id] = equity
            baseline = self._baseline_equities.get(market_id, equity)
            baseline_equities[market_id] = baseline

            level = cb.check(current_equity=equity, baseline_equity=baseline)
            market_cb_levels[market_id] = level

            if level == self._CircuitLevel.LIQUIDATE:
                liquidate_markets.append(market_id)

        # Always liquidate markets at L3 (regardless of other gate checks)
        for market_id in liquidate_markets:
            _log.warning("Circuit breaker LIQUIDATE for %s -- liquidating", market_id)
            self._liquidate_market(market_id)

        # Phase 2: Safety gates — check cross-market breaker and loss limits
        # BEFORE processing any instruments.

        # #144: CrossMarketCircuitBreaker trip halts ALL market processing.
        if self._cross_market_breaker.check(market_equities, baseline_equities):
            _log.warning("CrossMarketCircuitBreaker tripped -- all markets halted")
            self._alerter.on_circuit_breaker_trip("all", self._CircuitLevel.HALTED, 0.0)
            return  # halt all instrument processing

        # #146: Check daily loss limit before proceeding
        total_equity = sum(market_equities.values(), _ZERO)
        if self._loss_limit_tracker.is_halted(now, total_equity):
            _log.warning("LossLimitTracker halted trading -- daily loss limit exceeded")
            self._alerter.on_error("TradingLoop", "Daily loss limit exceeded -- trading halted")
            return

        # Phase 3: Process instruments for markets that are NORMAL or CAUTION
        for market_id, level in market_cb_levels.items():
            if level in (self._CircuitLevel.LIQUIDATE, self._CircuitLevel.HALTED):
                if level == self._CircuitLevel.HALTED:
                    _log.warning("Circuit breaker HALTED for %s -- skipping cycle", market_id)
                continue  # already liquidated or halted

            # #159: Market hours check before processing instruments
            if not self._is_market_open(market_id, now):
                _log.debug("Market %s is closed at %s -- skipping cycle", market_id, now)
                continue

            fetcher = self._fetchers.get(market_id)
            if fetcher is None:
                _log.warning("No fetcher for market %s", market_id)
                continue

            instruments = self._registry.list_by_market(market_id)
            for instrument in instruments:
                self._process_instrument(instrument, market_id, level, fetcher, now)

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

    def _compute_total_equity_base(self) -> Decimal:
        """Sum equities across all markets, converting to base currency (USD)."""
        total = _ZERO
        for m in self._circuit_breakers:
            equity = self._get_market_equity(m)
            if equity is None:
                continue
            currency = _MARKET_CURRENCY.get(m, "USD")
            total += self._fx.to_base(equity, currency)
        return total

    def _is_market_open(self, market_id: str, dt: datetime) -> bool:
        """Return True if the market is open at the given UTC datetime."""
        # Weekends: Saturday=5, Sunday=6
        if dt.weekday() >= _WEEKEND_WEEKDAY:
            return False

        if market_id == "us":
            open_h, open_m = _US_OPEN_UTC
            close_h, close_m = _US_CLOSE_UTC
        elif market_id == "moex":
            open_h, open_m = _MOEX_OPEN_UTC
            close_h, close_m = _MOEX_CLOSE_UTC
        else:
            # Unknown market: assume open (safe default — broker will reject if closed)
            return True

        open_minutes = open_h * 60 + open_m
        close_minutes = close_h * 60 + close_m
        current_minutes = dt.hour * 60 + dt.minute
        return open_minutes <= current_minutes < close_minutes

    def _process_instrument(
        self,
        instrument: Instrument,
        market_id: str,
        level: CircuitLevel,
        fetcher: object,
        now: datetime,
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

        # #157/#182: Check stop-losses against latest candle price
        if candles:
            current_price = candles[-1].close
            self._check_stop_losses(market_id, instrument.symbol, current_price)

        sentiment_score = self._get_sentiment(seg_id)

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

        # #162: Use RollingKelly for position sizing
        kelly_fraction = self._kelly_sizer.optimal_fraction()
        order = self._build_order(
            signal, level, portfolio.equity, portfolio.cash, candles,
            instrument.symbol, kelly_fraction,
        )
        if order is None:
            return

        # #141: Run PreTradeChecker before submitting
        order_value = order.quantity * (candles[-1].close if candles else _ZERO)
        open_position_count = len([q for q in portfolio.positions.values() if q > _ZERO])

        # #154: Compute cross-market exposure as invested value / total equity.
        # portfolio.positions maps symbol -> share quantity, so we must convert
        # to monetary values.  Use (equity - cash) as invested value since the
        # broker already tracks mark-to-market equity.
        total_equity: Decimal = self._compute_total_equity_base()
        invested_value = max(portfolio.equity - portfolio.cash, _ZERO)
        # Prospective exposure includes the proposed order value
        prospective_invested = invested_value + order_value
        cross_exposure: Decimal = (
            prospective_invested / total_equity if total_equity > _ZERO else _ZERO
        )
        try:
            _raw_max_exp = getattr(self._settings, "max_cross_market_exposure_pct", 0.80)
            max_exposure = Decimal(str(float(_raw_max_exp)))
        except (TypeError, ValueError):
            max_exposure = Decimal("0.80")

        pre_result = self._pre_trade_checker.check(
            order_value=order_value,
            portfolio_equity=portfolio.equity,
            available_cash=portfolio.cash,
            open_position_count=open_position_count,
            market_id=market_id,
            dt=now,
            circuit_breaker_level=self._circuit_breakers[market_id].level
            if market_id in self._circuit_breakers
            else None,
            cross_market_exposure_pct=cross_exposure,
            max_cross_market_exposure_pct=max_exposure,
        )

        if not pre_result.passed:
            _log.warning(
                "_process_instrument: pre-trade check failed for %s: %s",
                instrument.symbol,
                pre_result.violations,
            )
            return

        self._submit_order(order, market_id, candles=candles)

    def _build_order(
        self,
        signal: Signal,
        level: CircuitLevel,
        portfolio_equity: Decimal,
        available_cash: Decimal,
        candles: list[Candle],
        symbol: str,
        kelly_fraction: Decimal,
    ) -> OrderRequest | None:
        """Build an order from signal, using Kelly sizing and respecting CAUTION reduction."""
        if level == self._CircuitLevel.CAUTION:
            min_conf = 0.5 * _MIN_CONFIDENCE_BOOST
            if signal.confidence < min_conf:
                return None

        # 6A.11: Kelly sizes against portfolio equity, capped by available cash
        order_value = kelly_fraction * portfolio_equity
        order_value = min(order_value, available_cash)
        if level == self._CircuitLevel.CAUTION:
            order_value = order_value * _CAUTION_SIZE_FACTOR

        qty = (order_value / Decimal(str(candles[-1].close))) if candles else _ZERO
        qty = qty.quantize(Decimal(1))
        if qty <= _ZERO:
            return None

        side: Literal["BUY", "SELL"] = "BUY" if signal.direction == SignalDirection.BUY else "SELL"
        return self._OrderRequest(symbol=symbol, side=side, quantity=qty)

    def _submit_order(
        self,
        order: OrderRequest,
        market_id: str,
        candles: list[Candle] | None = None,
    ) -> None:
        """Submit order, set stop-loss on BUY fill, clear on SELL fill."""
        from finalayze.risk.stop_loss import compute_atr_stop_loss  # noqa: PLC0415

        try:
            result = self._broker_router.submit(order, market_id=market_id)
            if result.filled:
                self._alerter.on_trade_filled(result, market_id, broker=market_id)
                # Wire stop-loss on BUY fill
                if order.side == "BUY" and candles and result.fill_price is not None:
                    multiplier = _ATR_MULTIPLIER_MOEX if market_id == "moex" else _ATR_MULTIPLIER_US
                    stop = compute_atr_stop_loss(
                        result.fill_price, candles, atr_multiplier=multiplier
                    )
                    if stop is not None:
                        self._stop_loss_prices[order.symbol] = stop
                # Clear stop-loss on SELL fill
                elif order.side == "SELL":
                    self._stop_loss_prices.pop(order.symbol, None)
            else:
                self._alerter.on_trade_rejected(order, result.reason)
        except Exception:
            _log.exception("_strategy_cycle: order submission failed for %s", order.symbol)

    def _check_stop_losses(
        self,
        market_id: str,
        symbol: str,
        current_price: Decimal,
    ) -> None:
        """Check if current price has breached the stop-loss for a symbol.

        If price <= stop_loss_price, submit a SELL market order immediately.
        Clears the stop-loss entry after triggering to avoid duplicate orders.
        """
        stop_price = self._stop_loss_prices.get(symbol)
        if stop_price is None:
            return

        if current_price <= stop_price:
            _log.warning(
                "_check_stop_losses: stop triggered for %s @ %s (stop=%s)",
                symbol,
                current_price,
                stop_price,
            )
            broker = self._broker_router.route(market_id)
            positions = broker.get_positions()
            qty = positions.get(symbol, _ZERO)
            if qty > _ZERO:
                order = self._OrderRequest(symbol=symbol, side="SELL", quantity=qty)
                try:
                    broker.submit_order(order)
                except Exception:
                    _log.exception("_check_stop_losses: failed to submit stop-loss for %s", symbol)
                    return
            # Clear stop-loss after trigger
            del self._stop_loss_prices[symbol]

    def _retrain_cycle(self) -> None:
        """Periodically retrain ML ensemble models for all active segments.

        For each segment: fetch candles, build training windows, train an
        ensemble, validate accuracy > 52%, and hot-swap into the registry.
        Runs in a dedicated APScheduler executor to avoid starving other jobs.
        """
        from finalayze.ml.loader import save_ensemble  # noqa: PLC0415
        from finalayze.ml.training import DEFAULT_WINDOW_SIZE, build_windows  # noqa: PLC0415

        if self._ml_registry is None:
            return

        min_samples = getattr(self._settings, "ml_min_train_samples", 252)
        model_dir = Path(getattr(self._settings, "ml_model_dir", "models/"))
        segments = self._collect_active_segments()

        for segment_id in segments:
            try:
                self._retrain_segment(
                    segment_id,
                    model_dir,
                    min_samples,
                    DEFAULT_WINDOW_SIZE,
                    build_windows,
                    save_ensemble,
                )
            except Exception:
                _log.exception("_retrain_cycle: failed for segment %s", segment_id)
                self._alerter.on_error("MLRetrain", f"Retrain failed for {segment_id}")

    def _retrain_segment(
        self,
        segment_id: str,
        model_dir: Path,
        min_samples: int,
        window_size: int,
        build_windows_fn: object,
        save_ensemble_fn: object,
    ) -> None:
        """Retrain a single segment's ML ensemble with validation gating."""
        from sklearn.metrics import accuracy_score  # noqa: PLC0415

        # Fetch candles for each instrument in this segment
        market_id = segment_id.split("_", maxsplit=1)[0]
        instruments = [
            instr
            for instr in self._registry.list_by_market(market_id)
            if getattr(instr, "segment_id", "") == segment_id
        ]

        all_features: list[dict[str, float]] = []
        all_labels: list[int] = []
        fetcher = self._fetchers.get(market_id)
        if fetcher is None:
            return

        for instrument in instruments:
            try:
                candles = fetcher.fetch_candles(  # type: ignore[attr-defined]
                    symbol=instrument.symbol,
                    market_id=market_id,
                    limit=500,  # fetch more data for training
                )
            except Exception:
                _log.warning("_retrain: failed to fetch candles for %s", instrument.symbol)
                continue

            if len(candles) < window_size + 1:
                continue

            # Type-safe call to build_windows
            x_sym, y_sym, _ts = build_windows_fn(candles, window_size)  # type: ignore[operator]
            all_features.extend(x_sym)
            all_labels.extend(y_sym)

        if len(all_features) < min_samples:
            _log.info(
                "_retrain: only %d samples for %s (need %d) — skipping",
                len(all_features),
                segment_id,
                min_samples,
            )
            return

        # Temporal split: 70% train, gap of window_size, then validation
        n_train = int(len(all_features) * 0.7)
        gap_end = min(n_train + window_size, len(all_features))

        train_features = all_features[:n_train]
        train_labels = all_labels[:n_train]
        val_features = all_features[gap_end:]
        val_labels = all_labels[gap_end:]

        if not val_features:
            _log.info("_retrain: no validation data after gap for %s — skipping", segment_id)
            return

        # Train new ensemble
        assert self._ml_registry is not None
        ensemble = self._ml_registry.create_ensemble(segment_id)
        ensemble.fit(train_features, train_labels)

        # Validation gate: accuracy must exceed 52%
        val_preds = [round(ensemble.predict_proba(f)) for f in val_features]
        val_accuracy = float(accuracy_score(val_labels, val_preds))

        _min_accuracy = 0.52
        if val_accuracy < _min_accuracy:
            _log.warning(
                "_retrain: validation accuracy %.3f < %.2f for %s — rejecting",
                val_accuracy,
                _min_accuracy,
                segment_id,
            )
            return

        # Hot-swap into registry (thread-safe via lock)
        self._ml_registry.register(segment_id, ensemble)
        _log.info(
            "_retrain: registered new ensemble for %s (val_accuracy=%.3f)",
            segment_id,
            val_accuracy,
        )

        # Persist to disk
        try:
            save_ensemble_fn(model_dir, segment_id, ensemble)  # type: ignore[operator]
        except Exception:
            _log.exception("_retrain: failed to save ensemble for %s", segment_id)

    def _daily_reset(self) -> None:
        """Reset circuit breakers and send daily P&L summary."""
        market_pnl: dict[str, Decimal] = {}
        new_baselines: dict[str, Decimal] = {}

        now = self._now()
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

        # Reset loss limit tracker daily baseline
        self._loss_limit_tracker.reset_day(now, total_equity)

        self._alerter.on_daily_summary(market_pnl, total_equity)
        _log.info("Daily reset complete. Total equity: %s", total_equity)

    def _liquidate_market(self, market_id: str) -> None:
        """Close all open positions in a market (L3 circuit breaker response)."""
        try:
            broker = self._broker_router.route(market_id)
            positions = broker.get_positions()
            portfolio = broker.get_portfolio()
            equity = portfolio.equity

            # #174: Correct drawdown = (baseline - current) / baseline
            baseline = self._baseline_equities.get(market_id, equity)
            drawdown = float((baseline - equity) / baseline if baseline > _ZERO else _ZERO)

            # #129: No look-ahead bias — submit market orders without fill_candle
            self._close_positions(broker, positions)

            self._alerter.on_circuit_breaker_trip(market_id, self._CircuitLevel.LIQUIDATE, drawdown)
        except Exception:
            _log.exception("_liquidate_market: failed for market %s", market_id)
            self._alerter.on_error("TradingLoop", f"liquidation failed for {market_id}")

    def _close_positions(self, broker: BrokerBase, positions: dict[str, Decimal]) -> None:
        """Submit SELL orders for all non-zero positions.

        Uses market orders without fill_candle (#129: no look-ahead bias).
        """
        for symbol, qty in positions.items():
            if qty <= _ZERO:
                continue
            # #129: Do NOT pass fill_candle — live market orders have no look-ahead
            order = self._OrderRequest(symbol=symbol, side="SELL", quantity=qty)
            try:
                broker.submit_order(order)
            except Exception as exc:
                _log.error("liquidation_order_failed", extra={"symbol": symbol, "error": str(exc)})
                self._alerter.on_error("TradingLoop", f"Liquidation failed for {symbol}: {exc}")
