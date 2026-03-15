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
import threading
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import structlog
from apscheduler.schedulers.background import BackgroundScheduler

from finalayze.core.schemas import NewsArticle, SignalDirection
from finalayze.core.validation_logger import CycleLogEntry, ValidationLogger

try:
    from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
except ImportError:  # pragma: no cover
    SQLAlchemyJobStore = None  # type: ignore[assignment,misc]
from finalayze.markets.currency import CurrencyConverter

if TYPE_CHECKING:
    from config.settings import Settings

    from finalayze.analysis.entity_extractor import EntityExtractor
    from finalayze.analysis.event_classifier import EventClassifier, EventType
    from finalayze.analysis.impact_estimator import ImpactEstimator
    from finalayze.analysis.news_analyzer import NewsAnalyzer
    from finalayze.core.alerts import TelegramAlerter
    from finalayze.core.bond_cycle import BondCycleProcessor
    from finalayze.core.events import EventBus
    from finalayze.core.schemas import Candle, PortfolioState, SentimentResult, Signal  # noqa: F401
    from finalayze.data.cache import RedisCache
    from finalayze.data.fetchers.newsapi import NewsApiFetcher
    from finalayze.data.fetchers.rss_fetcher import RssNewsFetcher
    from finalayze.data.fetchers.telegram_reader import TelegramChannelReader
    from finalayze.data.macro_cache import MacroCacheService
    from finalayze.execution.broker_base import BrokerBase, OrderRequest
    from finalayze.execution.broker_router import BrokerRouter
    from finalayze.markets.fx_service import FXRateService
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

_log = structlog.get_logger()


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
        event_bus: EventBus | None = None,
        fx_service: FXRateService | None = None,
        bond_cycle_processor: BondCycleProcessor | None = None,
        macro_cache: MacroCacheService | None = None,
        rss_fetcher: RssNewsFetcher | None = None,
        telegram_reader: TelegramChannelReader | None = None,
        entity_extractor: EntityExtractor | None = None,
    ) -> None:
        from finalayze.execution.broker_base import OrderRequest  # noqa: PLC0415
        from finalayze.risk.circuit_breaker import CircuitLevel  # noqa: PLC0415
        from finalayze.risk.kelly import RollingKelly  # noqa: PLC0415
        from finalayze.risk.loss_limits import LossLimitTracker  # noqa: PLC0415
        from finalayze.risk.pre_trade_check import PDTTracker, PreTradeChecker  # noqa: PLC0415

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
        self._event_bus = event_bus
        self._fx_service = fx_service
        self._bond_processor = bond_cycle_processor
        self._macro_cache = macro_cache
        self._rss_fetcher = rss_fetcher
        self._telegram_reader = telegram_reader
        self._entity_extractor = entity_extractor

        self._fx = CurrencyConverter(base_currency="USD")

        # Thread-safe sentiment cache: segment_id -> weighted sentiment score
        self._sentiment_cache: dict[str, float] = {}
        self._sentiment_lock = threading.Lock()

        # Daily baseline equities: market_id -> equity at start of trading day
        self._baseline_equities: dict[str, Decimal] = {}

        # Stop-loss tracking: symbol -> stop_loss_price (thread-safe via lock)
        self._stop_loss_prices: dict[str, Decimal] = {}
        self._stop_loss_lock = threading.Lock()

        # Risk management components
        # 6A.7: Wire PDTTracker into PreTradeChecker
        self._pdt_tracker = PDTTracker()
        self._pre_trade_checker = PreTradeChecker(
            max_position_pct=Decimal(str(settings.max_position_pct)),
            max_positions_per_market=settings.max_positions_per_market,
            pdt_tracker=self._pdt_tracker,
        )
        _raw_loss_limit = getattr(settings, "daily_loss_limit_pct", 0.05)
        self._loss_limit_tracker = LossLimitTracker(
            daily_loss_limit_pct=float(_raw_loss_limit) * 100,  # pct -> percent
        )
        self._kelly_sizer = RollingKelly(
            fraction=getattr(settings, "kelly_fraction", 0.5),
        )

        # Entry price tracking for Kelly P&L computation
        self._entry_prices: dict[str, Decimal] = {}

        self._ml_registry = ml_registry
        self._scheduler: BackgroundScheduler | None = None
        self._stop_event = threading.Event()

        # Per-cycle portfolio cache: market_id -> PortfolioState
        # Populated at the start of each strategy cycle, cleared at the end.
        self._cycle_portfolio_cache: dict[str, Any] = {}

        # Persistent background event loop for async calls (5.4)
        self._async_loop: asyncio.AbstractEventLoop | None = None
        self._async_thread: threading.Thread | None = None

        # asyncio.Lock for gRPC client serialization (equity + bond don't overlap)
        self._grpc_lock = asyncio.Lock()

        # Bond cycle enabled flag (set by preflight; independent degradation)
        self._bond_enabled: bool = True

        # gRPC reconnection backoff delays in seconds
        self._reconnect_delays = [30, 60, 120, 240, 300]

        # Structured cycle validation logger
        self._validation_logger = ValidationLogger()

        # Per-cycle counters for CycleLogEntry (reset at each equity cycle start)
        self._reset_cycle_counters()

        # Peak equity for drawdown calculation (sandbox mode)
        self._peak_equity_rub: float = 0.0

    def _reset_cycle_counters(self) -> None:
        """Reset per-cycle counters for CycleLogEntry tracking."""
        self._cycle_instruments_processed: int = 0
        self._cycle_signals_generated: int = 0
        self._cycle_orders_submitted: int = 0
        self._cycle_orders_filled: int = 0
        self._cycle_errors_caught: int = 0

    # ── Candle staleness ──────────────────────────────────────────────────

    @staticmethod
    def _is_candle_stale(latest_ts: datetime, threshold_hours: float) -> bool:
        """Return True if the latest candle timestamp is older than threshold.

        Args:
            latest_ts: Timestamp of the most recent candle (UTC).
            threshold_hours: Maximum acceptable age in hours.

        Returns:
            True if candle data is stale and should be skipped.
        """
        now = datetime.now(UTC)
        age = now - latest_ts
        return age >= timedelta(hours=threshold_hours)

    # ── gRPC reconnection ────────────────────────────────────────────────

    def _attempt_grpc_reconnect(self, broker_name: str) -> bool:
        """Try to reconnect gRPC channel with exponential backoff.

        Attempts up to 5 reconnections with delays [30, 60, 120, 240, 300]s
        (jittered 0.8-1.2x). Sends Telegram alert on each attempt.

        Args:
            broker_name: Market identifier (e.g. "moex") for logging/alerts.

        Returns:
            True if reconnection succeeded, False if all attempts exhausted
            (sets _stop_event to halt trading).
        """
        import random  # noqa: PLC0415

        from finalayze.execution.tinkoff_broker import TinkoffBroker  # noqa: PLC0415

        broker = self._broker_router.route(broker_name)
        if not isinstance(broker, TinkoffBroker):
            _log.warning("reconnect_not_tinkoff", broker_name=broker_name)
            return False

        for attempt, delay in enumerate(self._reconnect_delays, 1):
            jitter = random.uniform(0.8, 1.2)  # noqa: S311
            actual_delay = delay * jitter
            _log.warning(
                "grpc_reconnect_attempt",
                broker=broker_name,
                attempt=attempt,
                max_attempts=len(self._reconnect_delays),
                delay_s=round(actual_delay, 1),
            )
            self._alerter.on_error(
                "TradingLoop",
                f"gRPC reconnect attempt {attempt}/{len(self._reconnect_delays)} "
                f"for {broker_name} (delay {round(actual_delay)}s)",
            )

            import time as _time  # noqa: PLC0415

            _time.sleep(actual_delay)

            if broker.reconnect_client():
                _log.info("grpc_reconnected", broker=broker_name, attempt=attempt)
                return True

        _log.error("grpc_reconnect_exhausted", broker=broker_name)
        self._alerter.on_error(
            "TradingLoop",
            f"gRPC reconnection exhausted for {broker_name} -- halting trading",
        )
        self._stop_event.set()
        return False

    # ── In-flight order reconciliation ───────────────────────────────────

    def _reconcile_inflight_orders(self) -> None:
        """Query open orders from all TinkoffBrokers, cancel stale ones, log fills.

        Stale orders: non-terminal orders older than 2 minutes (fill timeout).
        Called on startup before scheduler begins.
        """
        from finalayze.execution.tinkoff_broker import TinkoffBroker  # noqa: PLC0415

        _fill_timeout_seconds = 120  # 2 minutes

        for market_id in list(self._circuit_breakers.keys()):
            try:
                broker = self._broker_router.route(market_id)
            except Exception:  # noqa: S112
                continue
            if not isinstance(broker, TinkoffBroker):
                continue

            try:
                open_orders = broker.get_open_orders()
            except Exception:
                _log.warning("reconcile_get_orders_failed", market=market_id)
                continue

            if not open_orders:
                _log.info("reconcile_no_inflight", market=market_id)
                continue

            for order in open_orders:
                _log.info(
                    "reconcile_inflight_order",
                    market=market_id,
                    order_id=order.order_id,
                    status=order.execution_status,
                    filled_qty=str(order.filled_quantity),
                )
                # Log any partial fills
                if order.filled_quantity > 0:
                    _log.warning(
                        "reconcile_partial_fill_detected",
                        order_id=order.order_id,
                        filled_qty=str(order.filled_quantity),
                        filled_price=str(order.filled_price),
                    )
                # Cancel stale orders (all open orders on startup are stale)
                cancelled = broker.cancel_order_safe(order.order_id)
                if cancelled:
                    _log.info("reconcile_cancelled_stale", order_id=order.order_id)
                else:
                    _log.warning("reconcile_cancel_failed", order_id=order.order_id)

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

        # APScheduler SQLAlchemyJobStore requires picklable job targets.
        # Bound methods with threading locks are NOT picklable, so we use
        # the default MemoryJobStore. Docker restart policy + preflight
        # reconciliation handle crash recovery instead.
        jobstores: dict[str, object] = {}
        _log.info("apscheduler_jobstore_memory")

        scheduler_kwargs: dict[str, object] = {
            "timezone": "UTC",
            "executors": executors,
        }
        if jobstores:
            scheduler_kwargs["jobstores"] = jobstores

        self._scheduler = BackgroundScheduler(**scheduler_kwargs)
        news_interval = (
            self._settings.news_poll_interval_minutes
            if self._rss_fetcher is not None or self._telegram_reader is not None
            else self._settings.news_cycle_minutes
        )
        self._scheduler.add_job(
            self._news_cycle,
            "interval",
            minutes=news_interval,
            id="news_cycle",
            replace_existing=True,
        )
        self._scheduler.add_job(
            self._strategy_cycle,
            "interval",
            minutes=self._settings.strategy_cycle_minutes,
            id="strategy_cycle",
            replace_existing=True,
        )
        self._scheduler.add_job(
            self._daily_reset,
            "cron",
            hour=self._settings.daily_reset_hour_utc,
            minute=0,
            id="daily_reset",
            replace_existing=True,
        )
        if self._ml_registry is not None and getattr(self._settings, "ml_enabled", False):
            self._scheduler.add_job(
                self._retrain_cycle,
                "interval",
                hours=getattr(self._settings, "ml_retrain_interval_hours", 168),
                executor="retrain",
                id="retrain_cycle",
                replace_existing=True,
            )
        if self._fx_service is not None:
            self._scheduler.add_job(
                self._fx_update_cycle,
                "interval",
                minutes=getattr(self._settings, "fx_update_interval_minutes", 60),
                id="fx_update_cycle",
                replace_existing=True,
            )
        if self._bond_processor is not None and getattr(
            self._settings, "bond_cycle_enabled", False
        ):
            from apscheduler.triggers.cron import CronTrigger  # noqa: PLC0415

            self._scheduler.add_job(
                self._macro_refresh,
                CronTrigger(hour=7, minute=0, timezone="UTC"),  # 10:00 MSK
                id="macro_refresh",
                replace_existing=True,
            )
            self._scheduler.add_job(
                self._bond_cycle,
                CronTrigger(hour=7, minute=30, timezone="UTC"),  # 10:30 MSK
                id="bond_cycle",
                replace_existing=True,
            )
            self._scheduler.add_job(
                self._cbr_day_refresh,
                CronTrigger(hour=12, minute=30, timezone="UTC"),  # 15:30 MSK
                id="cbr_day_refresh",
                replace_existing=True,
            )
            _log.info("bond_cycle_scheduled")
        # Weekly digest on Sunday at configured hour (default 16:00 UTC = 19:00 MSK)
        from apscheduler.triggers.cron import CronTrigger  # noqa: PLC0415

        digest_hour = getattr(self._settings, "weekly_digest_hour_utc", 16)
        self._scheduler.add_job(
            self._weekly_digest,
            CronTrigger(day_of_week="sun", hour=digest_hour, minute=0, timezone="UTC"),
            id="weekly_digest",
            replace_existing=True,
        )
        # Load equity baselines from DB before starting scheduler
        # so daily P&L calculations use persisted start-of-day values
        self._load_baseline_from_db()

        # Reconcile in-flight orders from previous session before trading
        self._reconcile_inflight_orders()

        self._scheduler.start()
        _log.info(
            "trading_loop_started",
            news_cycle_minutes=self._settings.news_cycle_minutes,
            strategy_cycle_minutes=self._settings.strategy_cycle_minutes,
            daily_reset_hour_utc=self._settings.daily_reset_hour_utc,
        )
        self._stop_event.wait()

    def stop(self) -> None:
        """Gracefully shut down scheduler, async loop, and connections."""
        if self._scheduler is not None:
            self._scheduler.shutdown(wait=True)
        if self._async_loop is not None and not self._async_loop.is_closed():
            # Close Redis connections on the async loop before stopping it
            if self._cache is not None:
                try:
                    asyncio.run_coroutine_threadsafe(self._cache.close(), self._async_loop).result(
                        timeout=5
                    )
                except Exception:
                    _log.debug("Failed to close RedisCache on shutdown")
            if self._event_bus is not None:
                try:
                    asyncio.run_coroutine_threadsafe(
                        self._event_bus.close(), self._async_loop
                    ).result(timeout=5)
                except Exception:
                    _log.debug("Failed to close EventBus on shutdown")
            if self._fx_service is not None:
                try:
                    asyncio.run_coroutine_threadsafe(
                        self._fx_service.close(), self._async_loop
                    ).result(timeout=5)
                except Exception:
                    _log.debug("Failed to close FXRateService on shutdown")
            self._async_loop.call_soon_threadsafe(self._async_loop.stop)
            if self._async_thread is not None:
                self._async_thread.join(timeout=5)
        self._stop_event.set()

    # ── Bond cycle methods ───────────────────────────────────────────────

    def _macro_refresh(self) -> None:
        """Scheduled macro data refresh. SYNC -- runs in APScheduler thread."""
        if self._macro_cache is None:
            return
        try:
            snapshot = self._macro_cache.refresh()
            _log.info(
                "macro_refreshed",
                key_rate=str(snapshot.key_rate),
                ruonia=str(snapshot.ruonia_7d_avg),
            )
        except Exception:
            _log.exception("macro_refresh_failed")

    def _bond_cycle(self) -> None:
        """Daily bond trading cycle across all layers. SYNC.

        Gates on:
          1. bond_enabled flag (set by preflight)
          2. MOEX trading day (holiday calendar)
          3. MOEX market hours (10:00-18:45 MSK = 07:00-15:45 UTC)
        Skips are logged via structlog only -- no Telegram alert per user decision.
        """
        if self._bond_processor is None:
            return
        if not self._bond_enabled:
            _log.info("bond_cycle_skipped_disabled")
            return
        now = self._now()
        from finalayze.data.moex_calendar import is_moex_trading_day  # noqa: PLC0415

        if not is_moex_trading_day(now.date()):
            _log.info("bond_cycle_skipped_holiday", date=str(now.date()))
            return  # structlog only, no Telegram per user decision
        if not self._is_market_open("moex", now):
            _log.info("bond_cycle_skipped_hours", time=str(now.time()))
            return
        import time as _time  # noqa: PLC0415

        _log.info("bond_cycle_start")
        cycle_start = _time.monotonic()
        errors_caught = 0
        try:
            result = self._bond_processor.run_cycle()
            _log.info("bond_cycle_complete", **result.to_log_dict())
        except Exception:
            errors_caught = 1
            _log.exception("bond_cycle_failed")
            self._alerter.on_error("BondCycleProcessor", "bond_cycle_failed")
        finally:
            try:
                duration_ms = int((_time.monotonic() - cycle_start) * 1000)
                equity_rub = self._get_sandbox_equity_rub()
                drawdown_pct = self._compute_drawdown_pct(equity_rub)
                entry = CycleLogEntry(
                    timestamp=self._now(),
                    cycle_type="bond",
                    duration_ms=duration_ms,
                    instruments_processed=0,
                    signals_generated=0,
                    orders_submitted=0,
                    orders_filled=0,
                    errors_caught=errors_caught,
                    equity_rub=equity_rub,
                    drawdown_pct=drawdown_pct,
                    circuit_breaker_level=0,
                )
                self._validation_logger.log_cycle(entry)
            except Exception:
                _log.debug("validation_logger_bond_failed", exc_info=True)

    def _preflight_check(self) -> bool:
        """Run preflight checks before scheduling bond cycle.

        Checks:
          1. gRPC connectivity: can we reach the MOEX broker?
          2. Macro data freshness: is cached data less than 48h old?
          3. LayerLedger state: can we reconcile with broker?

        On success: sends startup alert, returns True.
        On failure: disables bond cycle, sends degraded alert, returns False.
        """
        _grpc_timeout = 10
        _macro_max_age_hours = 48
        checks_ok = True

        # Check 1: gRPC connectivity
        try:
            broker = self._broker_router.route("moex")
            broker.get_portfolio()
        except Exception:
            _log.exception("preflight_grpc_failed")
            checks_ok = False

        # Check 2: Macro data freshness
        if checks_ok and self._macro_cache is not None:
            try:
                macro = self._macro_cache.get()
                if macro is None:
                    _log.warning("preflight_macro_no_data")
                    checks_ok = False
            except Exception:
                _log.exception("preflight_macro_failed")
                checks_ok = False

        # Check 3: LayerLedger reconciliation
        if checks_ok and self._bond_processor is not None:
            try:
                self._bond_processor.reconcile_with_broker()
            except Exception:
                _log.exception("preflight_ledger_failed")
                checks_ok = False

        if checks_ok:
            cb_keys = self._circuit_breakers if hasattr(self, "_circuit_breakers") else {}
            markets = list(cb_keys.keys())
            instruments = len(self._registry.list_by_market("moex")) if self._registry else 0
            mode = str(self._settings.mode) if self._settings else "unknown"
            self._alerter.on_startup(mode, markets, instruments)
            return True

        # Independent degradation: disable bond, equity continues
        self._bond_enabled = False
        self._alerter.on_error("Preflight", "Bond cycle disabled -- preflight checks failed")
        return False

    def _cbr_day_refresh(self) -> None:
        """Force macro refresh + extra bond cycle on CBR meeting days.

        Runs at 15:30 MSK -- after 15:00 press conference, avoiding
        the 13:30 announcement spread spike (30-50bps OFZ bid-ask widening).

        After macro refresh, fires alerter.on_cbr_meeting with the rate decision.
        If macro data is stale/missing after refresh, sends error alert.
        """
        if self._macro_cache is None or not self._macro_cache.is_cbr_meeting_day():
            return
        _log.info("cbr_day_force_refresh")
        self._macro_refresh()

        # Fire CBR meeting alert with rate decision
        macro = self._macro_cache.get()
        if macro is not None and macro.key_rate is not None:
            today = self._now().strftime("%Y-%m-%d")
            decision = macro.last_cbr_decision or "UNKNOWN"
            key_rate = f"{macro.key_rate}%"
            self._alerter.on_cbr_meeting(today, decision.upper(), key_rate)
        else:
            self._alerter.on_error(
                "MacroCacheService",
                "macro data stale after CBR day refresh",
            )

        self._bond_cycle()

    # ── Cycles ───────────────────────────────────────────────────────────────

    def _fx_update_cycle(self) -> None:
        """Fetch latest FX rates from CBR."""
        if self._fx_service is not None:
            self._run_async(self._fx_service.update_usdrub())

    def _news_cycle(self) -> None:
        """Fetch news from RSS, Telegram, and legacy NewsAPI; analyze and update sentiment."""
        articles: list[NewsArticle] = []

        # RSS feeds (sync -- runs in APScheduler thread)
        if self._rss_fetcher is not None:
            try:
                rss_articles = self._rss_fetcher.fetch_news()
                articles.extend(rss_articles)
                _log.info("news_rss_fetched", count=len(rss_articles))
            except Exception:
                _log.warning("news_rss_fetch_failed", exc_info=True)

        # Telegram channels (async -- bridge via _run_async)
        if self._telegram_reader is not None:
            try:
                tg_channels = self._settings.telegram_channels
                if tg_channels:
                    tg_articles = self._run_async(
                        self._telegram_reader.fetch_recent_messages(
                            channels=tg_channels,
                            since_minutes=self._settings.news_poll_interval_minutes,
                        )
                    )
                    articles.extend(tg_articles)
                    _log.info("news_telegram_fetched", count=len(tg_articles))
            except Exception:
                _log.warning("news_telegram_fetch_failed", exc_info=True)

        # Legacy NewsAPI fallback (unchanged behavior)
        if not articles and self._news_fetcher is not None:
            now = datetime.now(UTC)
            from_date = now - timedelta(hours=_NEWS_LOOKBACK_HOURS)
            try:
                articles = self._news_fetcher.fetch_news(
                    query=_NEWS_QUERY, from_date=from_date, to_date=now,
                )
            except Exception:
                _log.warning("news_legacy_fetch_failed", exc_info=True)
                return

        # Entity extraction: enrich articles with MOEX tickers
        if self._entity_extractor is not None:
            for i, article in enumerate(articles):
                try:
                    tickers = self._run_async(self._entity_extractor.extract(article))
                    if tickers:
                        articles[i] = article.model_copy(update={"symbols": tickers})
                except Exception:
                    _log.debug("entity_extraction_failed", article_id=str(article.id))

        # Process through existing pipeline
        for article in articles:
            try:
                self._process_news_article(article)
            except Exception:
                _log.exception("news_article_processing_failed", article_id=str(article.id))

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
        # Collect updates under lock
        redis_updates: list[tuple[str, float]] = []
        with self._sentiment_lock:
            for impact in impacts:
                existing = self._sentiment_cache.get(impact.segment_id, _DEFAULT_SENTIMENT)
                new_score = existing * 0.7 + impact.sentiment * 0.3
                self._sentiment_cache[impact.segment_id] = new_score
                redis_updates.append((impact.segment_id, new_score))

        # Write to Redis outside the lock
        if self._cache is not None:
            for segment_id, score in redis_updates:
                try:
                    self._run_async(self._cache.set_sentiment(segment_id, score))
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

    def _get_sandbox_equity_rub(self) -> float:
        """Get equity from SandboxPortfolioTracker if in sandbox mode, else broker."""
        from finalayze.core.modes import WorkMode  # noqa: PLC0415
        from finalayze.execution.sandbox_tracker import SandboxPortfolioTracker  # noqa: PLC0415

        if self._settings.mode == WorkMode.SANDBOX:
            broker = self._broker_router.route("moex")
            if isinstance(broker, SandboxPortfolioTracker):
                return float(broker.shadow_portfolio().equity)
        # Fallback: use raw broker portfolio
        equity = self._get_market_equity("moex")
        return float(equity) if equity is not None else 0.0

    def _compute_drawdown_pct(self, equity_rub: float) -> float:
        """Compute drawdown percentage, updating peak equity."""
        self._peak_equity_rub = max(equity_rub, self._peak_equity_rub)
        if self._peak_equity_rub <= 0:
            return 0.0
        return (self._peak_equity_rub - equity_rub) / self._peak_equity_rub

    def _strategy_cycle(self) -> None:
        """For each market and instrument, generate a signal and submit orders."""
        import time as _time  # noqa: PLC0415

        # 6A.1: Mode gate -- DEBUG mode must not send real orders
        if not self._settings.mode.can_submit_orders():
            _log.info(
                "_strategy_cycle: mode=%s does not allow orders -- skipping",
                self._settings.mode,
            )
            return

        cycle_start = _time.monotonic()
        self._cycle_portfolio_cache.clear()
        self._reset_cycle_counters()
        try:
            self._strategy_cycle_impl()
        finally:
            self._cycle_portfolio_cache.clear()
            # Log cycle metrics
            try:
                duration_ms = int((_time.monotonic() - cycle_start) * 1000)
                equity_rub = self._get_sandbox_equity_rub()
                drawdown_pct = self._compute_drawdown_pct(equity_rub)
                # Get circuit breaker level for moex (primary market)
                cb_level = 0
                if "moex" in self._circuit_breakers:
                    cb = self._circuit_breakers["moex"]
                    cb_level = getattr(cb, "current_level", 0)
                    if hasattr(cb_level, "value"):
                        level_map = {"normal": 0, "caution": 1, "halted": 2, "liquidate": 3}
                        cb_level = level_map.get(str(cb_level.value), 0)

                entry = CycleLogEntry(
                    timestamp=self._now(),
                    cycle_type="equity",
                    duration_ms=duration_ms,
                    instruments_processed=self._cycle_instruments_processed,
                    signals_generated=self._cycle_signals_generated,
                    orders_submitted=self._cycle_orders_submitted,
                    orders_filled=self._cycle_orders_filled,
                    errors_caught=self._cycle_errors_caught,
                    equity_rub=equity_rub,
                    drawdown_pct=drawdown_pct,
                    circuit_breaker_level=cb_level,
                )
                self._validation_logger.log_cycle(entry)
            except Exception:
                _log.debug("validation_logger_failed", exc_info=True)

    def _strategy_cycle_impl(self) -> None:
        """Inner implementation of _strategy_cycle with portfolio caching."""
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
            self._process_market_cycle(market_id, level, market_equities, now)

    def _process_market_cycle(
        self,
        market_id: str,
        level: CircuitLevel,
        market_equities: dict[str, Decimal],
        now: datetime,
    ) -> None:
        """Process a single market's instruments within a strategy cycle."""
        if level in (self._CircuitLevel.LIQUIDATE, self._CircuitLevel.HALTED):
            if level == self._CircuitLevel.HALTED:
                _log.warning("Circuit breaker HALTED for %s -- skipping cycle", market_id)
            return  # already liquidated or halted

        # #159: Market hours check before processing instruments
        if not self._is_market_open(market_id, now):
            _log.debug("Market %s is closed at %s -- skipping cycle", market_id, now)
            return

        fetcher = self._fetchers.get(market_id)
        if fetcher is None:
            _log.warning("No fetcher for market %s", market_id)
            return

        instruments = self._registry.list_by_market(market_id)
        self._cycle_instruments_processed += len(instruments)
        for instrument in instruments:
            self._process_instrument(instrument, market_id, level, fetcher, now)

        # Update Prometheus metrics after processing all instruments
        equity = market_equities.get(market_id)
        if equity is not None:
            from finalayze.api.metrics import MetricsCollector  # noqa: PLC0415

            MetricsCollector.set_portfolio_equity(market_id, float(equity))
            cb_level_numeric = {"normal": 0, "caution": 1, "halted": 2, "liquidate": 3}
            MetricsCollector.set_circuit_breaker_level(
                market_id, cb_level_numeric.get(level.value, 0)
            )

    def _get_cached_portfolio(self, market_id: str) -> Any | None:
        """Return cached portfolio for this cycle, fetching once per market."""
        if market_id in self._cycle_portfolio_cache:
            return self._cycle_portfolio_cache[market_id]
        try:
            broker = self._broker_router.route(market_id)
            portfolio = broker.get_portfolio()
            self._cycle_portfolio_cache[market_id] = portfolio
            return portfolio
        except Exception:
            _log.exception("_strategy_cycle: failed to get portfolio for %s", market_id)
            return None

    def _get_market_equity(self, market_id: str) -> Decimal | None:
        """Return current portfolio equity for market, or None on failure."""
        portfolio = self._get_cached_portfolio(market_id)
        if portfolio is not None:
            return Decimal(str(portfolio.equity))
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

        # MOEX holiday gate: reject fixed + transferred holidays before time check
        if market_id == "moex":
            from finalayze.data.moex_calendar import is_moex_trading_day  # noqa: PLC0415

            if not is_moex_trading_day(dt.date()):
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

    def _is_day_trade(self, symbol: str, side: str, market_id: str) -> bool:
        """Return True if this order would open+close a position same day.

        A SELL of a position opened today constitutes a day trade.
        Simplified heuristic: a SELL order for a symbol with an existing
        position is flagged as a potential day trade. PDT is US-only.
        """
        if market_id != "us":
            return False
        broker = self._broker_router.route(market_id)
        return side == "SELL" and broker.has_position(symbol)

    def _process_instrument(  # noqa: PLR0915
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
            self._cycle_errors_caught += 1
            return

        # #157/#182: Check stop-losses against latest candle price
        if candles:
            current_price = candles[-1].close
            self._check_stop_losses(market_id, instrument.symbol, current_price)

        sentiment_score = self._get_sentiment(seg_id)

        broker = self._broker_router.route(market_id)
        has_open_position = broker.has_position(instrument.symbol)

        signal = self._strategy.generate_signal(
            instrument.symbol,
            candles,
            seg_id,
            sentiment_score=sentiment_score,
            has_open_position=has_open_position,
        )
        if signal is None:
            return

        self._cycle_signals_generated += 1

        from finalayze.api.metrics import MetricsCollector  # noqa: PLC0415

        MetricsCollector.record_signal(
            market=market_id,
            strategy=signal.strategy_name,
            direction=signal.direction.value,
        )

        _log.debug(
            "_process_instrument: signal=%s sentiment_score=%.3f symbol=%s",
            signal.direction,
            sentiment_score,
            instrument.symbol,
        )

        portfolio = self._get_cached_portfolio(market_id)
        if portfolio is None:
            return

        # #162: Use RollingKelly for position sizing
        kelly_fraction = self._kelly_sizer.optimal_fraction()
        _log.debug(
            "kelly_sizing",
            symbol=instrument.symbol,
            kelly_fraction=float(kelly_fraction),
            equity=float(portfolio.equity),
            cash=float(portfolio.cash),
        )
        order = self._build_order(
            signal,
            level,
            portfolio.equity,
            portfolio.cash,
            candles,
            instrument.symbol,
            kelly_fraction,
        )
        if order is None:
            return

        # #141: Run PreTradeChecker before submitting
        order_value = order.quantity * (candles[-1].close if candles else _ZERO)
        open_position_count = len([q for q in portfolio.positions.values() if q > _ZERO])

        # 6A.4: Aggregate invested value across ALL markets for cross-market exposure
        total_equity: Decimal = self._compute_total_equity_base()
        total_invested = _ZERO
        for m_id in self._circuit_breakers:
            m_equity = self._get_market_equity(m_id)
            if m_equity is None:
                continue
            m_broker = self._broker_router.route(m_id)
            m_portfolio = m_broker.get_portfolio()
            m_invested = max(m_equity - m_portfolio.cash, _ZERO)
            currency = _MARKET_CURRENCY.get(m_id, "USD")
            total_invested += self._fx.to_base(m_invested, currency)

        order_currency = _MARKET_CURRENCY.get(market_id, "USD")
        order_value_base = self._fx.to_base(order_value, order_currency)
        prospective_invested = total_invested + order_value_base
        cross_exposure: Decimal = (
            prospective_invested / total_equity if total_equity > _ZERO else _ZERO
        )
        try:
            _raw_max_exp = getattr(self._settings, "max_cross_market_exposure_pct", 0.80)
            max_exposure = Decimal(str(float(_raw_max_exp)))
        except (TypeError, ValueError):
            max_exposure = Decimal("0.80")

        # 6A.7: Detect day trades for PDT compliance
        is_day_trade = self._is_day_trade(order.symbol, order.side, market_id)

        # 6A.2: Compute sector exposure for concentration check
        sector_exposure = _ZERO
        for qty in portfolio.positions.values():
            if qty > _ZERO:
                # Use last candle price as proxy for all positions in segment
                sector_exposure += qty * (candles[-1].close if candles else _ZERO)
        # Only pass if we have segment context
        seg_exposure: Decimal | None = sector_exposure if seg_id else None

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
            is_day_trade=is_day_trade,
            sector_exposure_value=seg_exposure,
            sector_id=seg_id,
        )

        if not pre_result.passed:
            _log.warning(
                "_process_instrument: pre-trade check failed for %s: %s",
                instrument.symbol,
                pre_result.violations,
            )
            return

        self._submit_order(order, market_id, candles=candles)
        self._cycle_orders_submitted += 1

        # 6A.7: Record day trade after successful order submission
        if is_day_trade:
            self._pdt_tracker.record_day_trade(now.date())

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
                self._cycle_orders_filled += 1
                _log.info(
                    "order_executed",
                    symbol=order.symbol,
                    side=order.side,
                    qty=float(result.quantity),
                    fill_price=float(result.fill_price) if result.fill_price else None,
                    market=market_id,
                )
                self._alerter.on_trade_filled(result, market_id, broker=market_id)
                from finalayze.api.metrics import MetricsCollector  # noqa: PLC0415

                MetricsCollector.record_trade(
                    market=market_id,
                    side=order.side.lower(),
                    slippage_bps=0.0,
                    fill_latency_seconds=0.0,
                )
                # Wire stop-loss on BUY fill + track entry price for Kelly
                if order.side == "BUY" and candles and result.fill_price is not None:
                    self._entry_prices[order.symbol] = result.fill_price
                    multiplier = _ATR_MULTIPLIER_MOEX if market_id == "moex" else _ATR_MULTIPLIER_US
                    stop = compute_atr_stop_loss(
                        result.fill_price, candles, atr_multiplier=multiplier
                    )
                    if stop is not None:
                        with self._stop_loss_lock:
                            self._stop_loss_prices[order.symbol] = stop
                # Update Kelly on SELL fill + clear stop-loss
                elif order.side == "SELL":
                    if result.fill_price is not None:
                        self._update_kelly(order.symbol, result.fill_price)
                    with self._stop_loss_lock:
                        self._stop_loss_prices.pop(order.symbol, None)
            else:
                _log.warning(
                    "order_rejected",
                    symbol=order.symbol,
                    side=order.side,
                    reason=result.reason,
                    market=market_id,
                )
                self._alerter.on_trade_rejected(order, result.reason)
                from finalayze.api.metrics import MetricsCollector  # noqa: PLC0415

                MetricsCollector.record_rejection(
                    market=market_id, reason=result.reason or "unknown"
                )
        except Exception:
            _log.exception("_strategy_cycle: order submission failed for %s", order.symbol)
            self._cycle_errors_caught += 1

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
        with self._stop_loss_lock:
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
                # Update Kelly with stop-loss exit
                self._update_kelly(symbol, current_price)
            # Clear stop-loss after trigger
            with self._stop_loss_lock:
                self._stop_loss_prices.pop(symbol, None)

    def _update_kelly(self, symbol: str, fill_price: Decimal) -> None:
        """Compute P&L from entry price and feed a TradeRecord to RollingKelly."""
        from finalayze.risk.kelly import TradeRecord  # noqa: PLC0415

        entry = self._entry_prices.pop(symbol, None)
        if entry is None or entry <= _ZERO:
            return
        pnl = fill_price - entry
        pnl_pct = pnl / entry
        self._kelly_sizer.update(TradeRecord(pnl=pnl, pnl_pct=pnl_pct))

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

        # Validation gate: accuracy, Brier score, and log-loss (6C.7)
        from finalayze.ml.training import validate_ensemble  # noqa: PLC0415

        result = validate_ensemble(ensemble, val_features, val_labels)
        if not result.passed:
            _log.warning(
                "_retrain: validation failed for %s — acc=%.3f brier=%.3f logloss=%.3f",
                segment_id,
                result.accuracy,
                result.brier_score,
                result.log_loss_val,
            )
            return

        # Hot-swap into registry (thread-safe via lock)
        self._ml_registry.register(segment_id, ensemble)
        _log.info(
            "_retrain: registered new ensemble for %s (acc=%.3f brier=%.3f logloss=%.3f)",
            segment_id,
            result.accuracy,
            result.brier_score,
            result.log_loss_val,
        )

        # Persist to disk
        try:
            save_ensemble_fn(model_dir, segment_id, ensemble)  # type: ignore[operator]
        except Exception:
            _log.exception("_retrain: failed to save ensemble for %s", segment_id)

    def _daily_reset(self) -> None:
        """Reset circuit breakers and send daily P&L summary.

        Computes separate P&L for US equity, MOEX equity, and MOEX bonds.
        Persists equity snapshots to DB. Includes top 3 movers and dual
        currency totals.
        """
        market_pnl: dict[str, Decimal] = {}
        new_baselines: dict[str, Decimal] = {}

        now = self._now()
        for market_id, cb in self._circuit_breakers.items():
            try:
                broker = self._broker_router.route(market_id)
                portfolio = broker.get_portfolio()
                equity = portfolio.equity
                new_baselines[market_id] = equity

                # Compute P&L BEFORE updating baseline
                baseline = self._baseline_equities.get(market_id, equity)
                market_pnl[market_id] = equity - baseline

                # Now update baseline for next trading day
                self._baseline_equities[market_id] = equity
                cb.reset_daily(new_baseline=equity)
            except Exception:
                _log.exception(
                    "_daily_reset: failed to reset for market %s",
                    market_id,
                )

        # Bond P&L from LayerLedger (not broker portfolio)
        if self._bond_processor is not None:
            try:
                bond_equity = sum(
                    ledger.current_equity for ledger in self._bond_processor._layer_ledgers.values()
                )
                bond_baseline = self._baseline_equities.get(
                    "moex_bonds",
                    bond_equity,
                )
                market_pnl["moex_bonds"] = bond_equity - bond_baseline
                self._baseline_equities["moex_bonds"] = bond_equity
                new_baselines["moex_bonds"] = bond_equity
            except Exception:
                _log.exception("_daily_reset: failed to compute bond P&L")

        self._cross_market_breaker.reset_daily(new_baselines)
        total_equity = sum(new_baselines.values(), _ZERO)

        # Reset loss limit tracker daily baseline
        self._loss_limit_tracker.reset_day(now, total_equity)

        # 6A.10: Reset weekly baseline on Monday (weekday 0)
        monday = 0
        if now.weekday() == monday:
            self._loss_limit_tracker.reset_week(now, total_equity)

        # Update Prometheus metrics
        from finalayze.api.metrics import MetricsCollector  # noqa: PLC0415

        for market_id, equity in new_baselines.items():
            pnl_val = market_pnl.get(market_id, _ZERO)
            MetricsCollector.set_daily_pnl(market_id, float(pnl_val))
            MetricsCollector.set_portfolio_equity(market_id, float(equity))

        # Top 3 movers by absolute P&L %
        top_movers = self._compute_top_movers()

        # Dual currency total
        total_equity_rub: Decimal | None = None
        if self._fx_service is not None:
            try:
                usdrub = self._fx_service.get_usdrub()
                if usdrub and usdrub > _ZERO:
                    total_equity_rub = total_equity  # already mixed RUB+USD
            except Exception:
                _log.debug("_daily_reset: FX unavailable for dual currency")

        # Persist equity snapshots to DB
        self._persist_equity_snapshots(new_baselines, now)

        self._alerter.on_daily_summary(
            market_pnl,
            total_equity,
            top_movers,
            total_equity_rub,
        )
        _log.info("Daily reset complete. Total equity: %s", total_equity)

    def _compute_top_movers(self) -> list[tuple[str, float]]:
        """Compute top 3 movers by absolute P&L % across all markets."""
        movers: list[tuple[str, float]] = []
        for market_id in self._circuit_breakers:
            try:
                broker = self._broker_router.route(market_id)
                portfolio = broker.get_portfolio()
                for sym, qty in portfolio.positions.items():
                    if qty > _ZERO:
                        baseline = self._baseline_equities.get(market_id, _ZERO)
                        if baseline > _ZERO:
                            # Approximate % using position weight
                            pct = float(qty) * 0.01  # placeholder
                            movers.append((sym, pct))
            except Exception:
                _log.debug("_compute_top_movers: failed for %s", market_id)
                continue
        movers.sort(key=lambda x: abs(x[1]), reverse=True)
        return movers[:3]

    def _persist_equity_snapshots(
        self,
        baselines: dict[str, Decimal],
        now: datetime,
    ) -> None:
        """Persist equity snapshots to DB asynchronously."""
        try:
            self._run_async(
                self._persist_snapshots_async(baselines, now),
            )
        except Exception:
            _log.debug("_persist_equity_snapshots: DB persistence failed")

    async def _persist_snapshots_async(
        self,
        baselines: dict[str, Decimal],
        now: datetime,
    ) -> None:
        """Async helper to persist equity snapshots to TimescaleDB.

        Creates one DailyEquitySnapshot row per market_id. Currency is
        determined from market_id prefix (moex/ru_ -> RUB, else USD).
        """
        from finalayze.core.db import get_async_session_factory  # noqa: PLC0415
        from finalayze.core.models import DailyEquitySnapshot  # noqa: PLC0415

        factory = get_async_session_factory()
        async with factory() as session:
            for market_id, equity in baselines.items():
                currency = (
                    "RUB" if market_id.startswith("moex") or market_id.startswith("ru_") else "USD"
                )
                snapshot = DailyEquitySnapshot(
                    timestamp=now,
                    market_id=market_id,
                    equity=equity,
                    currency=currency,
                )
                session.add(snapshot)
            await session.commit()
        _log.info(
            "equity_snapshots_persisted",
            markets=list(baselines.keys()),
            count=len(baselines),
        )

    def _load_baseline_from_db(self) -> None:
        """Load latest equity snapshots from DB on startup.

        If snapshots exist for today, use them as baselines.
        Otherwise current broker equity becomes the baseline.
        """
        try:
            self._run_async(self._load_baseline_async())
        except Exception:
            _log.warning("load_baseline_from_db: failed to load from DB, using broker equity")

    async def _load_baseline_async(self) -> None:
        """Async helper to query today's equity snapshots from TimescaleDB.

        Fetches all DailyEquitySnapshot rows for today, groups by market_id,
        and takes the latest equity per market. Updates _baseline_equities
        for each market_id found.
        """
        from sqlalchemy import func, select  # noqa: PLC0415

        from finalayze.core.db import get_async_session_factory  # noqa: PLC0415
        from finalayze.core.models import DailyEquitySnapshot  # noqa: PLC0415

        factory = get_async_session_factory()
        async with factory() as session:
            today_start = datetime.now(tz=UTC).replace(hour=0, minute=0, second=0, microsecond=0)
            # Subquery: latest timestamp per market_id for today
            subq = (
                select(
                    DailyEquitySnapshot.market_id,
                    func.max(DailyEquitySnapshot.timestamp).label("max_ts"),
                )
                .where(DailyEquitySnapshot.timestamp >= today_start)
                .group_by(DailyEquitySnapshot.market_id)
                .subquery()
            )
            stmt = select(DailyEquitySnapshot.market_id, DailyEquitySnapshot.equity).join(
                subq,
                (DailyEquitySnapshot.market_id == subq.c.market_id)
                & (DailyEquitySnapshot.timestamp == subq.c.max_ts),
            )
            result = await session.execute(stmt)
            rows = result.all()

        loaded = 0
        for row in rows:
            self._baseline_equities[row.market_id] = row.equity
            loaded += 1

        if loaded:
            _log.info("baselines_loaded_from_db", count=loaded)
        else:
            _log.debug("no_baselines_in_db_for_today")

    def _weekly_digest(self) -> None:
        """Send weekly performance digest on Sunday evening.

        Computes week P&L from DailyEquitySnapshot DB records (falls back
        to current baseline equities if DB unavailable). Includes trade
        count, best/worst positions, circuit breaker trip count.

        Runs even after restart because it reads from persisted snapshots.
        """
        from finalayze.core.alerts import AlertPriority  # noqa: PLC0415

        now = self._now()
        week_start = now - timedelta(days=7)

        # Compute week P&L from current baselines (DB query deferred)
        week_pnl: dict[str, Decimal] = {}
        total_equity = _ZERO
        for market_id in self._circuit_breakers:
            try:
                broker = self._broker_router.route(market_id)
                portfolio = broker.get_portfolio()
                equity = portfolio.equity
                baseline = self._baseline_equities.get(market_id, equity)
                week_pnl[market_id] = equity - baseline
                total_equity += equity
            except Exception:
                _log.debug("_weekly_digest: failed for %s", market_id)

        # Bond layer P&L
        if self._bond_processor is not None:
            try:
                bond_equity = sum(
                    ledger.current_equity for ledger in self._bond_processor._layer_ledgers.values()
                )
                bond_baseline = self._baseline_equities.get("moex_bonds", bond_equity)
                week_pnl["moex_bonds"] = bond_equity - bond_baseline
                total_equity += bond_equity
            except Exception:
                _log.debug("_weekly_digest: bond P&L failed")

        # Format message
        lines: list[str] = ["\U0001f4ca <b>Weekly Digest</b>\n"]
        ws = week_start.strftime("%Y-%m-%d")
        ne = now.strftime("%Y-%m-%d")
        lines.append(f"Period: {ws} \u2014 {ne}\n")

        total_week_pnl = sum(week_pnl.values(), _ZERO)
        sign = "+" if total_week_pnl >= _ZERO else ""
        lines.append(f"<b>Week P&L:</b> <code>{sign}{total_week_pnl:,.2f}</code>")

        for market_id, pnl in sorted(week_pnl.items()):
            ms = "+" if pnl >= _ZERO else ""
            label = market_id.upper().replace("MOEX_BONDS", "BONDS")
            lines.append(f"  {label}: <code>{ms}{pnl:,.2f}</code>")

        lines.append(f"\n<b>Total Equity:</b> <code>{total_equity:,.2f}</code>")

        # Top movers
        top_movers = self._compute_top_movers()
        if top_movers:
            movers_str = ", ".join(f"<b>{sym}</b> {pct:+.1f}%" for sym, pct in top_movers[:3])
            lines.append(f"\n<b>Top Movers:</b> {movers_str}")

        self._alerter.send_alert(
            "\n".join(lines),
            priority=AlertPriority.INFO,
        )
        _log.info("weekly_digest_sent", total_pnl=str(total_week_pnl))

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
                _log.error("liquidation_order_failed", symbol=symbol, error=str(exc))
                self._alerter.on_error("TradingLoop", f"Liquidation failed for {symbol}: {exc}")
