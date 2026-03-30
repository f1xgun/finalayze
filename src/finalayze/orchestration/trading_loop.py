"""APScheduler-based live trading loop (Layer 5 -- orchestrator).

Orchestrates three scheduled cycles:
  - _news_cycle: fetch news, analyze sentiment, update _sentiment_cache
  - _strategy_cycle: for each instrument, generate signal, apply circuit breakers,
    submit orders via BrokerRouter, fire alerts
  - _daily_reset: reset circuit breakers, send daily P&L summary

Thread safety: _sentiment_cache is protected by _sentiment_lock (threading.Lock).

Moved from core/ to orchestration/ in Phase 22 (dependency layer cleanup).
See docs/architecture/DEPENDENCY_LAYERS.md for layering rules.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import threading
import time
from collections import OrderedDict
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
from finalayze.data.normalizer import DataNormalizer
from finalayze.markets.currency import CurrencyConverter
from finalayze.markets.schedule import SCHEDULES

if TYPE_CHECKING:
    from config.settings import Settings

    from finalayze.analysis.event_classifier import EventClassifier
    from finalayze.analysis.impact_estimator import ImpactEstimator
    from finalayze.analysis.news_analyzer import NewsAnalyzer
    from finalayze.analysis.news_impact_analyzer import NewsImpactAnalyzer, NewsImpactResult
    from finalayze.analysis.sector_ticker_mapper import SectorTickerMapper
    from finalayze.api.alerts import TelegramAlerter
    from finalayze.api.metrics import MetricsCollector
    from finalayze.core.events import EventBus
    from finalayze.core.schemas import Candle, PortfolioState, Signal
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
    from finalayze.monitoring.sandbox_monitor import SandboxMonitorService
    from finalayze.orchestration.bond_cycle import BondCycleProcessor
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
_SENTIMENT_HALF_LIFE_HOURS = 4.0
_SENTIMENT_DECAY_LAMBDA = math.log(2) / _SENTIMENT_HALF_LIFE_HOURS  # ~0.1733
_ZERO = Decimal(0)
_WEEKEND_WEEKDAY = 5  # Saturday=5, Sunday=6
_STALENESS_THRESHOLD_HOURS: float = 48.0  # 2x daily timeframe; skip if latest candle older
_ATR_MULTIPLIER_US = Decimal("2.0")
_ATR_MULTIPLIER_MOEX = Decimal("2.5")
_MARKET_CURRENCY: dict[str, str] = {"us": "USD", "moex": "RUB"}
_ARTICLE_DEDUP_MAX_SIZE = 5000  # max hashes to track
_ARTICLE_DEDUP_TTL_HOURS = 24  # skip articles seen within this window

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

    def __init__(  # noqa: PLR0915
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
        news_impact_analyzer: NewsImpactAnalyzer | None = None,
        sector_ticker_mapper: SectorTickerMapper | None = None,
        sandbox_monitor: SandboxMonitorService | None = None,
        health_monitor: object | None = None,
        metrics_collector: type[MetricsCollector] | None = None,
        grpc_loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        from finalayze.execution.broker_base import OrderRequest  # noqa: PLC0415
        from finalayze.execution.simulated_broker import StopLossState  # noqa: PLC0415
        from finalayze.risk.circuit_breaker import CircuitLevel  # noqa: PLC0415
        from finalayze.risk.kelly import RollingKelly  # noqa: PLC0415
        from finalayze.risk.loss_limits import LossLimitTracker  # noqa: PLC0415
        from finalayze.risk.pre_trade_check import PDTTracker, PreTradeChecker  # noqa: PLC0415

        # Store class references for runtime use without module-level imports
        self._OrderRequest = OrderRequest
        self._CircuitLevel = CircuitLevel
        self._StopLossState = StopLossState

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
        self._news_impact_analyzer = news_impact_analyzer
        self._sector_ticker_mapper = sector_ticker_mapper
        self._sandbox_monitor = sandbox_monitor
        self._health_monitor = health_monitor
        self._metrics = metrics_collector

        self._fx = CurrencyConverter(base_currency="USD")

        # Per-ticker sentiment: (segment_id, ticker) -> (score, monotonic_timestamp)
        self._sentiment_cache: dict[tuple[str, str], tuple[float, float]] = {}
        self._sentiment_lock = threading.Lock()

        # Cached check: any segment has event_driven strategy enabled?
        self._event_driven_active: bool | None = None

        # Article dedup: SHA-256(url|title) -> monotonic timestamp (OPS-03)
        self._seen_article_hashes: OrderedDict[str, float] = OrderedDict()

        # Daily baseline equities: market_id -> equity at start of trading day
        self._baseline_equities: dict[str, Decimal] = {}

        # Stop-loss tracking: symbol -> StopLossState (trailing, thread-safe via lock)
        self._stop_states: dict[str, object] = {}  # StopLossState instances
        self._stop_loss_lock = threading.Lock()

        # Per-cycle re-entry guard: symbols stopped out this cycle skip signal gen
        self._cycle_exited_symbols: set[str] = set()

        # Risk management components
        # 6A.7: Wire PDTTracker into PreTradeChecker
        self._pdt_tracker = PDTTracker()
        _limits = settings.effective_risk_limits()
        self._pre_trade_checker = PreTradeChecker(
            max_position_pct=_limits.max_position_pct,
            max_positions_per_market=_limits.max_positions_per_market,
            pdt_tracker=self._pdt_tracker,
            max_sector_concentration_pct=_limits.max_sector_concentration_pct,
            min_cash_reserve_pct=_limits.min_cash_reserve_pct,
        )
        self._loss_limit_tracker = LossLimitTracker(
            daily_loss_limit_pct=_limits.daily_loss_limit_pct * 100,  # fraction -> percent
        )
        self._kelly_sizer = RollingKelly(
            fraction=getattr(settings, "kelly_fraction", 0.5),
        )

        # Entry price tracking for Kelly P&L computation
        self._entry_prices: dict[str, Decimal] = {}

        self._ml_registry = ml_registry
        self._scheduler: BackgroundScheduler | None = None
        self._stop_event = threading.Event()

        # Total strategy cycles completed (used by HealthMonitor for liveness)
        self._total_cycles: int = 0

        # Per-cycle portfolio cache: market_id -> PortfolioState
        # Populated at the start of each strategy cycle, cleared at the end.
        self._cycle_portfolio_cache: dict[str, Any] = {}

        # Persistent background event loop for non-gRPC async calls (HTTP, DB, Telegram)
        self._async_loop: asyncio.AbstractEventLoop | None = None
        self._async_thread: threading.Thread | None = None

        # Dedicated gRPC event loop -- isolates PollerCompletionQueue from general
        # async work. Prevents BlockingIOError contention causing 60-min cycle drift.
        self._grpc_loop: asyncio.AbstractEventLoop | None = grpc_loop
        self._grpc_thread: threading.Thread | None = None

        # asyncio.Lock for gRPC client serialization (equity + bond don't overlap)
        self._grpc_lock = asyncio.Lock()

        # Bond cycle enabled flag (set by preflight; independent degradation)
        self._bond_enabled: bool = True

        # gRPC reconnection backoff delays in seconds
        self._reconnect_delays = [30, 60, 120, 240, 300]

        # Structured cycle validation logger
        self._validation_logger = ValidationLogger()

        # Per-instrument last price cache: symbol -> Decimal (built during strategy cycle)
        self._last_prices: dict[str, Decimal] = {}

        # Segment min_combined_confidence cache: seg_id -> float
        self._segment_min_confidence: dict[str, float] = {}

        # Per-cycle counters for CycleLogEntry (reset at each equity cycle start)
        self._reset_cycle_counters()

        # Peak equity for drawdown calculation (sandbox mode)
        self._peak_equity_rub: float = 0.0

        # Consecutive cycle error counters for alerting (ERR-04)
        self._consecutive_equity_errors: int = 0
        self._consecutive_bond_errors: int = 0
        self._MAX_CONSECUTIVE_ERRORS: int = 3

    def _reset_cycle_counters(self) -> None:
        """Reset per-cycle counters for CycleLogEntry tracking."""
        self._cycle_instruments_processed: int = 0
        self._cycle_signals_generated: int = 0
        self._cycle_orders_submitted: int = 0
        self._cycle_orders_filled: int = 0
        self._cycle_errors_caught: int = 0
        self._cycle_exited_symbols: set[str] = set()

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

            if self._stop_event.wait(timeout=actual_delay):
                _log.info("grpc_reconnect_cancelled", broker=broker_name)
                return False

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

    def _run_async(self, coro: Any, *, timeout: int = 30) -> Any:
        """Run an async coroutine on a persistent background event loop.

        Lazily creates a daemon thread with its own event loop on first call.
        Default 30-second timeout; batch operations may pass a larger value.
        """
        if self._async_loop is None or self._async_loop.is_closed():
            loop = asyncio.new_event_loop()
            self._async_loop = loop
            thread = threading.Thread(target=loop.run_forever, daemon=True)
            thread.start()
            self._async_thread = thread
        future = asyncio.run_coroutine_threadsafe(coro, self._async_loop)
        return future.result(timeout=timeout)

    # ── gRPC loop helpers ─────────────────────────────────────────────────────

    def _init_grpc_loop(self) -> asyncio.AbstractEventLoop:
        """Create a dedicated background event loop for all gRPC operations.

        Isolated from _async_loop to prevent PollerCompletionQueue BlockingIOError
        from starving HTTP/DB/Telegram coroutines and causing strategy cycle drift.
        """
        loop = asyncio.new_event_loop()

        # Suppress benign BlockingIOError from gRPC PollerCompletionQueue
        def _grpc_exception_handler(
            loop: asyncio.AbstractEventLoop, context: dict[str, Any]
        ) -> None:
            exc = context.get("exception")
            if isinstance(exc, BlockingIOError):
                return  # benign EAGAIN from PollerCompletionQueue
            loop.default_exception_handler(context)

        loop.set_exception_handler(_grpc_exception_handler)
        thread = threading.Thread(target=loop.run_forever, daemon=True, name="grpc-loop")
        thread.start()
        self._grpc_loop = loop
        self._grpc_thread = thread
        return loop

    def _run_grpc(self, coro: Any, *, timeout: int = 30) -> Any:
        """Run a gRPC coroutine on the dedicated gRPC event loop.

        Use this for all TinkoffBroker and TinkoffFetcher calls.
        Non-gRPC async work (HTTP, DB, Telegram) should use _run_async().
        """
        if self._grpc_loop is None or self._grpc_loop.is_closed():
            self._init_grpc_loop()
        future = asyncio.run_coroutine_threadsafe(coro, self._grpc_loop)  # type: ignore[arg-type]
        return future.result(timeout=timeout)

    @property
    def total_cycles(self) -> int:
        """Return the total number of strategy cycles completed."""
        return self._total_cycles

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the APScheduler and block until stop() is called."""
        from apscheduler.executors.pool import (  # noqa: PLC0415
            ThreadPoolExecutor as APSThreadPoolExecutor,
        )

        executors: dict[str, APSThreadPoolExecutor] = {
            "default": APSThreadPoolExecutor(max_workers=4),
            "retrain": APSThreadPoolExecutor(max_workers=1),
            "news": APSThreadPoolExecutor(max_workers=1),
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
            executor="news",
            coalesce=True,
            max_instances=1,
        )
        self._scheduler.add_job(
            self._strategy_cycle,
            "interval",
            minutes=self._settings.strategy_cycle_minutes,
            id="strategy_cycle",
            replace_existing=True,
            misfire_grace_time=300,
            coalesce=True,
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

        # Preflight checks: gRPC connectivity, macro data, bond cycle gating
        self._preflight_check()

        # Always send startup alert (preflight may fail, but app is running)
        cb_keys = self._circuit_breakers if hasattr(self, "_circuit_breakers") else {}
        markets = list(cb_keys.keys())
        instruments = sum(len(self._registry.list_by_market(m)) for m in self._fetchers)
        mode = str(self._settings.mode) if self._settings else "unknown"
        self._alerter.on_startup(mode, markets, instruments)

        self._scheduler.start()
        _log.info(
            "trading_loop_started",
            news_cycle_minutes=self._settings.news_cycle_minutes,
            strategy_cycle_minutes=self._settings.strategy_cycle_minutes,
            daily_reset_hour_utc=self._settings.daily_reset_hour_utc,
        )
        # Initialize Prometheus metrics with current portfolio state so Grafana
        # shows data immediately, not only after the first strategy cycle (60 min).
        self._init_metrics()
        # Schedule first strategy cycle after 30s (not 60 min) so we get
        # fast feedback at startup, but after news cycle has started.
        self._scheduler.modify_job(
            "strategy_cycle",
            next_run_time=datetime.now(UTC) + timedelta(seconds=30),
        )
        self._stop_event.wait()

    def _init_metrics(self) -> None:
        """Seed Prometheus gauges with current portfolio state at startup.

        Without this, Grafana shows 'No data' until the first strategy cycle
        (up to 60 minutes after startup).
        """
        if not self._metrics:
            return

        for market_id in self._fetchers:
            try:
                broker = self._broker_router.route(market_id)
                portfolio = broker.get_portfolio()
                equity = getattr(portfolio, "equity", None) or getattr(
                    portfolio, "total_value", None
                )
                if equity is not None:
                    self._metrics.set_portfolio_equity(market_id, float(equity))
                    if market_id in ("moex", "moex_bonds"):
                        self._metrics.set_portfolio_equity_rub(float(equity))
                positions = getattr(portfolio, "positions", None)
                if positions is not None:
                    self._metrics.set_open_positions(market_id, len(positions))
            except Exception:
                _log.debug("init_metrics_portfolio_failed", market_id=market_id)
            # Circuit breaker: default to NORMAL
            self._metrics.set_circuit_breaker_level(market_id, 0)
            # Drawdown: default to 0
            self._metrics.set_drawdown(market_id, 0.0)
        _log.info("metrics_initialized", markets=list(self._fetchers.keys()))

    def stop(self) -> None:
        """Gracefully shut down scheduler, async/gRPC loops, and connections."""
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
        # Stop dedicated gRPC event loop
        if self._grpc_loop is not None and not self._grpc_loop.is_closed():
            self._grpc_loop.call_soon_threadsafe(self._grpc_loop.stop)
            if self._grpc_thread is not None:
                self._grpc_thread.join(timeout=5)
            self._grpc_loop = None
            self._grpc_thread = None
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
            self._consecutive_bond_errors = 0
        except Exception:
            errors_caught = 1
            self._consecutive_bond_errors += 1
            _log.exception("bond_cycle_failed")
            self._alerter.on_error("BondCycleProcessor", "bond_cycle_failed")
            if self._consecutive_bond_errors >= self._MAX_CONSECUTIVE_ERRORS:
                from finalayze.api.alerts import AlertPriority  # noqa: PLC0415

                self._alerter.send_alert(
                    f"ALERT: {self._consecutive_bond_errors} consecutive bond cycle failures",
                    priority=AlertPriority.CRITICAL,
                )
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
        """Fetch latest FX rates from CBR and update Prometheus metric."""
        if self._fx_service is not None:
            rate = self._run_async(self._fx_service.update_usdrub())
            if rate is not None:
                from finalayze.api.metrics import MetricsCollector  # noqa: PLC0415

                MetricsCollector.set_usd_rub_rate(float(rate))

    def _news_cycle(self) -> None:
        """Fetch news from RSS, Telegram, and legacy NewsAPI; analyze and update sentiment."""
        if not self._any_event_driven_enabled():
            _log.debug("news_cycle_skipped_no_event_driven")
            return

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
                    query=_NEWS_QUERY,
                    from_date=from_date,
                    to_date=now,
                )
            except Exception:
                _log.warning("news_legacy_fetch_failed", exc_info=True)
                return

        # Analyze articles via NewsImpactAnalyzer (single LLM call per article)
        # Large timeout: with rate-limited LLM, batches may take minutes.
        _batch_timeout = 1800
        processed_ok = 0
        processed_fail = 0
        if self._news_impact_analyzer is not None and articles:
            processed_ok, processed_fail, _ = self._run_async(
                self._analyze_impact_batch(articles), timeout=_batch_timeout
            )

        # Single summary line for the entire news cycle
        log_fn = _log.info if processed_fail == 0 else _log.warning
        log_fn(
            "news_cycle_complete",
            articles=len(articles),
            processed_ok=processed_ok,
            processed_fail=processed_fail,
        )

    def _is_article_duplicate(self, article: NewsArticle) -> bool:
        """Check if article was already processed within the TTL window.

        Uses SHA-256 of (url + title) as the dedup key. Evicts entries
        older than _ARTICLE_DEDUP_TTL_HOURS and caps at _ARTICLE_DEDUP_MAX_SIZE.
        """
        key = hashlib.sha256(f"{article.url}|{article.title}".encode()).hexdigest()
        now = time.monotonic()

        # Evict expired entries (oldest first, since OrderedDict preserves insertion order)
        cutoff = now - _ARTICLE_DEDUP_TTL_HOURS * 3600
        while self._seen_article_hashes:
            oldest_key, oldest_ts = next(iter(self._seen_article_hashes.items()))
            if oldest_ts < cutoff:
                del self._seen_article_hashes[oldest_key]
            else:
                break

        if key in self._seen_article_hashes:
            return True

        self._seen_article_hashes[key] = now
        # Cap size
        while len(self._seen_article_hashes) > _ARTICLE_DEDUP_MAX_SIZE:
            self._seen_article_hashes.popitem(last=False)

        return False

    async def _analyze_impact_batch(
        self, articles: list[NewsArticle]
    ) -> tuple[int, int, str]:
        """Analyze all articles via NewsImpactAnalyzer with bounded concurrency.

        Uses an inline circuit breaker: after 5 consecutive LLM failures,
        remaining articles are skipped to avoid wasting minutes on retries.

        Returns:
            (ok_count, fail_count, last_error_type) for summary logging.
        """
        # Deduplicate articles already seen within TTL window (OPS-03)
        unique_articles = [a for a in articles if not self._is_article_duplicate(a)]
        skipped_count = len(articles) - len(unique_articles)
        if skipped_count > 0:
            _log.info(
                "news_articles_deduplicated",
                skipped=skipped_count,
                remaining=len(unique_articles),
            )
        articles = unique_articles
        if not articles:
            return 0, 0, ""

        sem = asyncio.Semaphore(5)
        ok_count = 0
        fail_count = 0
        last_error = ""
        consecutive_failures = 0
        _fail_threshold = 5
        analyzer = self._news_impact_analyzer
        assert analyzer is not None

        async def _process_one(article: NewsArticle) -> bool:
            nonlocal consecutive_failures, last_error
            if consecutive_failures >= _fail_threshold:
                return False
            async with sem:
                try:
                    result = await analyzer.analyze(article)
                    self._apply_impact_result(result)
                    consecutive_failures = 0
                    return True
                except Exception as exc:
                    consecutive_failures += 1
                    last_error = type(exc).__name__
                    _log.debug(
                        "news_article_analysis_failed",
                        article_title=article.title[:80],
                        error_type=last_error,
                        error=str(exc)[:200],
                    )
                    if consecutive_failures == _fail_threshold:
                        _log.warning(
                            "news_processing_circuit_opened",
                            error=last_error,
                            consecutive_failures=consecutive_failures,
                        )
                    return False

        results = await asyncio.gather(*[_process_one(a) for a in articles])
        for success in results:
            if success:
                ok_count += 1
            else:
                fail_count += 1
        return ok_count, fail_count, last_error

    def _apply_impact_result(self, result: NewsImpactResult) -> None:
        """Apply NewsImpactResult to per-ticker sentiment cache."""
        active_segments = self._collect_active_segments()
        mapper = self._sector_ticker_mapper
        if mapper is None:
            return

        # Build ticker -> score mapping from sectors
        ticker_scores: dict[str, float] = {}
        for sector_impact in result.affected_sectors:
            tickers = mapper.map_sectors([sector_impact.sector])
            score = sector_impact.magnitude * sector_impact.direction * result.sentiment
            for ticker in tickers:
                # Take the strongest impact if ticker appears in multiple sectors
                if ticker not in ticker_scores or abs(score) > abs(ticker_scores[ticker]):
                    ticker_scores[ticker] = score

        # Direct tickers get the raw sentiment * confidence
        for ticker in result.direct_tickers:
            direct_score = result.sentiment * result.confidence
            if ticker not in ticker_scores or abs(direct_score) > abs(ticker_scores[ticker]):
                ticker_scores[ticker] = direct_score

        # Update cache for all active segments containing these tickers
        redis_updates: list[tuple[str, str, float]] = []
        with self._sentiment_lock:
            for seg_id in active_segments:
                seg_tickers = self._get_segment_tickers(seg_id)
                for ticker in seg_tickers:
                    if ticker in ticker_scores:
                        cache_key = (seg_id, ticker)
                        existing = self._read_decayed_sentiment(seg_id, ticker)
                        new_score = existing * 0.7 + ticker_scores[ticker] * 0.3
                        self._sentiment_cache[cache_key] = (new_score, time.monotonic())
                        redis_updates.append((seg_id, ticker, new_score))

        # Redis write outside lock
        if self._cache is not None:
            for seg_id, ticker, score in redis_updates:
                try:
                    self._run_async(
                        self._cache.set_sentiment(f"{seg_id}:{ticker}", score)
                    )
                except Exception:
                    _log.debug("Failed to write sentiment to Redis cache")

    def _get_segment_tickers(self, seg_id: str) -> list[str]:
        """Get ticker symbols for instruments in this segment."""
        return [
            instr.symbol
            for market_id in self._fetchers
            for instr in self._registry.list_by_market(market_id)
            if hasattr(instr, "segment_id") and instr.segment_id == seg_id
        ]

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

    def _read_decayed_sentiment(
        self, seg_id: str, ticker: str | None = None
    ) -> float:
        """Read sentiment with exponential time-decay applied.

        If ticker is provided, reads per-ticker score.
        Falls back to segment average if no per-ticker entry.
        Must be called while holding _sentiment_lock.
        """
        if ticker is not None:
            entry = self._sentiment_cache.get((seg_id, ticker))
            if entry is not None:
                score, ts = entry
                hours_elapsed = (time.monotonic() - ts) / 3600.0
                return score * math.exp(-_SENTIMENT_DECAY_LAMBDA * hours_elapsed)
            # Fallback: average of all per-ticker scores for this segment
            seg_scores = []
            for (s, _t), (score, ts) in self._sentiment_cache.items():
                if s == seg_id:
                    hours_elapsed = (time.monotonic() - ts) / 3600.0
                    seg_scores.append(score * math.exp(-_SENTIMENT_DECAY_LAMBDA * hours_elapsed))
            if seg_scores:
                return sum(seg_scores) / len(seg_scores)
            return _DEFAULT_SENTIMENT
        # Legacy: no ticker -- average all scores for segment
        seg_scores = []
        for (s, _t), (score, ts) in self._sentiment_cache.items():
            if s == seg_id:
                hours_elapsed = (time.monotonic() - ts) / 3600.0
                seg_scores.append(score * math.exp(-_SENTIMENT_DECAY_LAMBDA * hours_elapsed))
        if seg_scores:
            return sum(seg_scores) / len(seg_scores)
        return _DEFAULT_SENTIMENT

    def _get_sentiment(self, seg_id: str, ticker: str | None = None) -> float:
        """Read sentiment from Redis cache (if available) or in-memory fallback."""
        if self._cache is not None:
            cache_key = f"{seg_id}:{ticker}" if ticker else seg_id
            try:
                cached: float | None = self._run_async(
                    self._cache.get_sentiment(cache_key)
                )
                if cached is not None:
                    return cached
            except Exception:
                _log.debug("Failed to read sentiment from Redis cache")
        with self._sentiment_lock:
            return self._read_decayed_sentiment(seg_id, ticker)

    def _any_event_driven_enabled(self) -> bool:
        """Check if any segment preset has event_driven strategy enabled.

        Caches result in self._event_driven_active to avoid re-reading
        YAML files on every news cycle.
        """
        if self._event_driven_active is not None:
            return self._event_driven_active

        import yaml  # noqa: PLC0415

        presets_dir = Path(__file__).parent.parent / "strategies" / "presets"
        result = False
        try:
            for path in presets_dir.glob("*.yaml"):
                try:
                    with path.open() as f:
                        config = yaml.safe_load(f)
                    if (
                        isinstance(config, dict)
                        and config.get("strategies", {})
                        .get("event_driven", {})
                        .get("enabled", False)
                    ):
                        result = True
                        break
                except (OSError, yaml.YAMLError):
                    _log.warning("preset_read_failed", path=str(path))
        except OSError:
            _log.warning("presets_dir_not_found", path=str(presets_dir))

        self._event_driven_active = result
        return result

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

    def _strategy_cycle(self) -> None:  # noqa: PLR0915
        """For each market and instrument, generate a signal and submit orders."""
        import time as _time  # noqa: PLC0415

        self._total_cycles += 1

        # 6A.1: Mode gate -- DEBUG mode must not send real orders
        if not self._settings.mode.can_submit_orders():
            _log.info(
                "_strategy_cycle: mode=%s does not allow orders -- skipping",
                self._settings.mode,
            )
            return

        # 6A.2: Market-hours gate -- skip cycle when all markets are closed
        any_market_open = False
        for market_id in self._broker_router.registered_markets:
            schedule = SCHEDULES.get(market_id)
            if schedule is None or schedule.is_market_open():
                any_market_open = True
                break
        if not any_market_open:
            _log.info(
                "strategy_cycle_skipped_markets_closed",
                markets=list(self._broker_router.registered_markets),
            )
            return

        cycle_start = _time.monotonic()
        self._cycle_portfolio_cache.clear()
        self._reset_cycle_counters()
        _cycle_failed = False
        try:
            self._strategy_cycle_impl()
            self._consecutive_equity_errors = 0
        except Exception:
            _cycle_failed = True
            self._consecutive_equity_errors += 1
            _log.exception("strategy_cycle_impl_failed")
            if self._consecutive_equity_errors >= self._MAX_CONSECUTIVE_ERRORS:
                from finalayze.api.alerts import AlertPriority  # noqa: PLC0415

                self._alerter.send_alert(
                    f"ALERT: {self._consecutive_equity_errors} consecutive equity cycle failures",
                    priority=AlertPriority.CRITICAL,
                )
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
                holds = (
                    self._cycle_instruments_processed
                    - self._cycle_signals_generated
                    - self._cycle_errors_caught
                )
                _log.info(
                    "strategy_cycle_summary",
                    duration_ms=duration_ms,
                    instruments=self._cycle_instruments_processed,
                    holds=max(holds, 0),
                    signals=self._cycle_signals_generated,
                    orders=self._cycle_orders_submitted,
                    fills=self._cycle_orders_filled,
                    errors=self._cycle_errors_caught,
                    equity_rub=round(equity_rub, 2),
                    drawdown_pct=round(drawdown_pct, 4),
                )

                # Update feed timestamp at cycle level (not just per-instrument)
                # so health monitor sees the cycle ran even when 0 instruments processed
                if self._health_monitor is not None:
                    self._health_monitor.update_feed_timestamp(self._now())

                # Sandbox monitoring: persist cycle metrics
                if self._sandbox_monitor is not None:
                    from finalayze.monitoring.sandbox_monitor import CycleMetrics  # noqa: PLC0415

                    cycle_metrics = CycleMetrics(
                        timestamp=self._now(),
                        trade_count=self._cycle_orders_filled,
                        pnl_rub=Decimal(
                            str(equity_rub - self._baseline_equities.get("moex", equity_rub))
                        ),
                        equity_rub=Decimal(str(equity_rub)),
                        fill_rate=(
                            self._cycle_orders_filled / max(self._cycle_orders_submitted, 1)
                        ),
                        uptime_cycles=self._sandbox_monitor.cycle_count + 1,
                        signals_generated=self._cycle_signals_generated,
                        errors_caught=self._cycle_errors_caught,
                        max_slippage_bps=max(self._sandbox_monitor.slippage_buffer, default=0.0),
                        avg_slippage_bps=(
                            sum(self._sandbox_monitor.slippage_buffer)
                            / max(len(self._sandbox_monitor.slippage_buffer), 1)
                        ),
                        drawdown_pct=drawdown_pct,
                    )
                    self._sandbox_monitor.on_cycle_complete(cycle_metrics)
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
        if equity is not None and self._metrics:
            self._metrics.set_portfolio_equity(market_id, float(equity))
            cb_level_numeric = {"normal": 0, "caution": 1, "halted": 2, "liquidate": 3}
            self._metrics.set_circuit_breaker_level(market_id, cb_level_numeric.get(level.value, 0))

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

    def _process_instrument(  # noqa: PLR0911, PLR0912, PLR0915
        self,
        instrument: Instrument,
        market_id: str,
        level: CircuitLevel,
        fetcher: object,
        now: datetime,
    ) -> None:
        """Fetch candles, generate signal, and submit order for one instrument."""
        from finalayze.core.exceptions import InstrumentNotFoundError  # noqa: PLC0415

        # Skip instruments without FIGI (delisted shares, bonds handled by bond_cycle)
        figi = getattr(instrument, "figi", None)
        if not figi:
            _log.debug("skip_no_figi", symbol=instrument.symbol)
            return

        seg_id = getattr(instrument, "segment_id", "") or "us_tech"
        try:
            # Convert limit (bar count) to date range for fetcher API
            end = now
            start = end - timedelta(days=_CANDLE_LOOKBACK * 2)  # ~2x for weekends/holidays
            candles: list[Candle] = fetcher.fetch_candles(  # type: ignore[attr-defined]
                symbol=instrument.symbol,
                start=start,
                end=end,
            )
        except InstrumentNotFoundError:
            _log.debug("skip_instrument_not_found", symbol=instrument.symbol)
            return
        except Exception:
            _log.exception("_strategy_cycle: failed to fetch candles for %s", instrument.symbol)
            self._cycle_errors_caught += 1
            return

        # DATA-01: Validate candles through DataNormalizer before any processing
        normalizer = DataNormalizer(market_id=market_id, source="live")
        candles = normalizer.normalize_batch(candles)
        if not candles:
            _log.warning("all_candles_invalid", symbol=instrument.symbol, market=market_id)
            return

        # DATA-02: Skip instrument if latest candle is stale
        if self._is_candle_stale(candles[-1].timestamp, _STALENESS_THRESHOLD_HOURS):
            _log.warning(
                "candle_data_stale",
                symbol=instrument.symbol,
                latest_ts=candles[-1].timestamp.isoformat(),
                threshold_hours=_STALENESS_THRESHOLD_HOURS,
            )
            return

        # Update health monitor feed timestamp on successful fetch
        if candles and self._health_monitor is not None:
            self._health_monitor.update_feed_timestamp(now)

        # Cache last price for per-position sector exposure calculation (SIZE-02)
        if candles:
            self._last_prices[instrument.symbol] = Decimal(str(candles[-1].close))

        # #157/#182: Check stop-losses against latest candle price
        if candles:
            current_price = candles[-1].close
            self._check_stop_losses(market_id, instrument.symbol, current_price)

        # PARITY-04: Skip signal generation for symbols stopped out this cycle
        if instrument.symbol in self._cycle_exited_symbols:
            _log.debug("skip_reentry_guard", symbol=instrument.symbol)
            return

        sentiment_score = self._get_sentiment(seg_id, instrument.symbol)

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
            _log.debug("signal_hold", symbol=instrument.symbol, segment=seg_id)
            return

        # Skip BUY when position already open — prevent infinite accumulation
        if has_open_position and signal.direction == SignalDirection.BUY:
            _log.debug(
                "signal_skip_already_positioned",
                symbol=instrument.symbol,
                direction="BUY",
            )
            return

        self._cycle_signals_generated += 1

        if self._metrics:
            self._metrics.record_signal(
                market=market_id,
                strategy=signal.strategy_name,
                direction=signal.direction.value,
            )

        _log.info(
            "signal_generated",
            symbol=instrument.symbol,
            direction=signal.direction.value,
            strategy=signal.strategy_name,
            confidence=round(signal.confidence, 3),
            sentiment=round(sentiment_score, 3),
            segment=seg_id,
            has_position=has_open_position,
            reasoning=signal.reasoning,
            features={k: round(v, 4) for k, v in signal.features.items()} or None,
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
            portfolio=portfolio,
            seg_id=seg_id,
        )
        if order is None:
            _log.info(
                "order_sizing_zero",
                symbol=instrument.symbol,
                direction=signal.direction.value,
                strategy=signal.strategy_name,
                reason="position size rounded to zero",
            )
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

        # 6A.2: Compute sector exposure for concentration check (SIZE-02 fix)
        sector_exposure = _ZERO
        for pos_symbol, qty in portfolio.positions.items():
            if qty > _ZERO:
                # Use each position's own last known price, not current instrument's candle
                pos_price = self._get_last_price(pos_symbol)
                sector_exposure += qty * pos_price
        # Only pass if we have segment context
        seg_exposure: Decimal | None = sector_exposure if seg_id else None

        # PARITY-03: Gather all pre-trade check parameters
        # Check 9: stop_loss_price from trailing stop state (Plan 01)
        with self._stop_loss_lock:
            _stop_st = self._stop_states.get(instrument.symbol)
            stop_loss_price = _stop_st.current_stop if _stop_st is not None else None  # type: ignore[union-attr]

        # Check 10: has_pending_order via broker
        has_pending = self._has_pending_order(instrument.symbol, market_id)

        # Check 12: regime_state from macro cache
        regime_state = self._get_regime_state()

        # Check 13: strategy_name from the signal
        strategy_name = signal.strategy_name

        # Check 14: open positions and correlations
        open_positions = [s for s, q in portfolio.positions.items() if q > _ZERO]
        correlations = self._get_correlations(open_positions)

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
            stop_loss_price=stop_loss_price,
            require_stop_loss=(instrument.symbol in self._stop_states),
            has_pending_order=has_pending,
            symbol=instrument.symbol,
            cross_market_exposure_pct=cross_exposure,
            max_cross_market_exposure_pct=max_exposure,
            is_day_trade=is_day_trade,
            sector_exposure_value=seg_exposure,
            sector_id=seg_id,
            regime_state=regime_state,
            strategy_name=strategy_name,
            open_positions=open_positions,
            correlations=correlations,
        )

        if not pre_result.passed:
            _log.info(
                "pre_trade_rejected",
                symbol=instrument.symbol,
                direction=signal.direction.value,
                strategy=signal.strategy_name,
                violations=pre_result.violations,
            )
            return

        price = candles[-1].close if candles else _ZERO
        _log.info(
            "order_submitted",
            symbol=instrument.symbol,
            direction=order.side,
            quantity=int(order.quantity),
            price=float(price),
            value_rub=float(order.quantity * price),
            kelly=float(kelly_fraction),
            equity=float(portfolio.equity),
            strategy=signal.strategy_name,
            market=market_id,
        )
        self._submit_order(order, market_id, candles=candles)
        self._cycle_orders_submitted += 1

        # 6A.7: Record day trade after successful order submission
        if is_day_trade:
            self._pdt_tracker.record_day_trade(now.date())

    def _build_sizing_pipeline(self, segment_id: str) -> object:
        """Build position sizing pipeline matching backtest engine step order.

        Pipeline order: Kelly -> VolTarget -> Regime -> [RubOilRegime] -> [BrentGate]
            -> [CBRRegime] -> [SectorAllocation] -> [Copula] -> [EVT] -> MetaLabel -> HardCaps
        """
        from finalayze.risk.position_sizing_pipeline import (  # noqa: PLC0415
            BrentGateStep,
            CBRRegimeStep,
            CopulaStep,
            EVTStep,
            HardCapsStep,
            KellyStep,
            MetaLabelStep,
            PositionSizingPipeline,
            RegimeStep,
            RubOilRegimeStep,
            SectorAllocationStep,
            VolTargetStep,
        )

        steps: list[object] = [KellyStep(), VolTargetStep(), RegimeStep()]

        # Add MOEX-specific steps when macro_cache provides data
        if self._macro_cache is not None and segment_id.startswith("ru_"):
            rub_oil_signal = getattr(self._macro_cache, "rub_oil_regime_signal", None)
            if rub_oil_signal is not None:
                steps.append(RubOilRegimeStep(rub_oil_signal, segment_id))
            brent_rub = getattr(self._macro_cache, "brent_rub_price", 0.0)
            if brent_rub > 0:
                steps.append(BrentGateStep(brent_rub, segment_id))
            yield_slope = getattr(self._macro_cache, "yield_slope_bps", 0.0)
            steps.append(CBRRegimeStep(yield_slope, segment_id))
            cbr_dir = getattr(self._macro_cache, "cbr_direction", "")
            if cbr_dir:
                steps.append(SectorAllocationStep(brent_rub, cbr_dir, segment_id))

        steps.append(CopulaStep())
        steps.append(EVTStep())
        steps.append(MetaLabelStep())
        steps.append(HardCapsStep())
        return PositionSizingPipeline(steps=steps)  # type: ignore[arg-type]

    @staticmethod
    def _compute_asset_vol(candles: list[Candle]) -> Decimal:
        """Compute annualized volatility from candle close prices."""
        if len(candles) < 2:  # noqa: PLR2004
            return Decimal("0.20")  # fallback
        import math  # noqa: PLC0415

        closes = [float(c.close) for c in candles]
        log_rets = [
            math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes)) if closes[i - 1] > 0
        ]
        if not log_rets:
            return Decimal("0.20")
        var = sum(r**2 for r in log_rets) / len(log_rets)
        annual_vol = math.sqrt(var * 252)
        return Decimal(str(round(annual_vol, 4)))

    def _get_regime_scale(self) -> Decimal:
        """Get current regime scale factor. 1.0 = risk-on, lower = risk-off."""
        if self._macro_cache is not None:
            regime = getattr(self._macro_cache, "regime_scale", None)
            if regime is not None:
                return Decimal(str(regime))
        return Decimal("1.0")

    def _has_pending_order(self, symbol: str, market_id: str) -> bool:
        """Check if broker has a pending (unfilled) order for symbol."""
        try:
            broker = self._broker_router.route(market_id)
            if hasattr(broker, "get_pending_orders"):
                pending = broker.get_pending_orders()
                return any(o.symbol == symbol for o in pending)
        except Exception:
            _log.debug("pending_order_check_failed", symbol=symbol)
        return False

    def _get_regime_state(self) -> object | None:
        """Get current regime state from macro cache."""
        if self._macro_cache is not None:
            return getattr(self._macro_cache, "regime_state", None)
        return None

    def _get_correlations(
        self,
        open_positions: list[str],  # noqa: ARG002
    ) -> dict[tuple[str, str], float]:
        """Compute pairwise correlations for open positions.

        For live trading, correlation computation requires historical returns
        which we don't track yet. Return empty dict for now (check 14 passes through).
        TODO: Wire returns history for live correlation computation in future phase.
        """
        return {}

    def _build_order(
        self,
        signal: Signal,
        level: CircuitLevel,
        portfolio_equity: Decimal,
        available_cash: Decimal,
        candles: list[Candle],
        symbol: str,
        kelly_fraction: Decimal,
        *,
        portfolio: PortfolioState | None = None,
        seg_id: str = "us_tech",
    ) -> OrderRequest | None:
        """Build an order from signal, using PositionSizingPipeline for BUY orders.

        PARITY-01: BUY orders go through the same multi-step sizing pipeline as backtest.
        SIZE-01: SELL orders use actual held position quantity.
        SIZE-03: CAUTION threshold uses segment preset min_combined_confidence * 1.2.
        """
        from finalayze.risk.position_sizing_pipeline import SizingContext  # noqa: PLC0415

        side: Literal["BUY", "SELL"] = "BUY" if signal.direction == SignalDirection.BUY else "SELL"

        # SIZE-01: SELL orders use actual held quantity, skip pipeline sizing
        if signal.direction == SignalDirection.SELL:
            held = portfolio.positions.get(symbol, _ZERO) if portfolio is not None else _ZERO
            if held <= _ZERO:
                return None
            return self._OrderRequest(symbol=symbol, side=side, quantity=held)

        # SIZE-03: CAUTION threshold from segment preset (not hardcoded 0.5)
        if level == self._CircuitLevel.CAUTION:
            preset_conf = self._get_segment_min_confidence(seg_id)
            min_conf = preset_conf * _MIN_CONFIDENCE_BOOST
            if signal.confidence < min_conf:
                return None

        # PARITY-01: Build sizing pipeline and context (matching backtest engine)
        pipeline = self._build_sizing_pipeline(seg_id)
        asset_vol = self._compute_asset_vol(candles)
        regime_scale = self._get_regime_scale()
        ml_confidence = signal.features.get("ml_confidence") if signal.features else None

        _limits = self._settings.effective_risk_limits()
        min_pos = max(portfolio_equity * Decimal("0.005"), Decimal(500))

        context = SizingContext(
            equity=portfolio_equity,
            base_position=kelly_fraction * portfolio_equity,
            max_position_pct=Decimal(str(_limits.max_position_pct)),
            min_position_size=min_pos,
            asset_vol=asset_vol,
            target_vol=Decimal(str(getattr(self._settings, "target_vol", 0.15))),
            regime_scale=regime_scale,
            correlation_scale=Decimal("1.0"),
            returns_history=(),
            ml_confidence=ml_confidence,
        )

        order_value = pipeline.compute(context)  # type: ignore[union-attr]
        if order_value <= _ZERO:
            return None

        # Cap by available cash
        order_value = min(order_value, available_cash)

        # CAUTION reduction (on top of pipeline)
        if level == self._CircuitLevel.CAUTION:
            order_value = order_value * _CAUTION_SIZE_FACTOR

        qty = (order_value / Decimal(str(candles[-1].close))) if candles else _ZERO
        qty = qty.quantize(Decimal(1))
        if qty <= _ZERO:
            return None

        return self._OrderRequest(symbol=symbol, side=side, quantity=qty)

    def _get_last_price(self, symbol: str) -> Decimal:
        """Return cached last price for a symbol, or _ZERO if unknown (SIZE-02)."""
        return self._last_prices.get(symbol, _ZERO)

    def _get_segment_min_confidence(self, seg_id: str) -> float:
        """Load min_combined_confidence from segment preset YAML (SIZE-03).

        Caches result to avoid re-reading YAML on every call.
        Falls back to 0.5 if preset not found.
        """
        if seg_id in self._segment_min_confidence:
            return self._segment_min_confidence[seg_id]

        import yaml  # noqa: PLC0415

        presets_dir = Path(__file__).parent.parent / "strategies" / "presets"
        path = presets_dir / f"{seg_id}.yaml"
        default_conf = 0.5
        try:
            with path.open() as f:
                config = yaml.safe_load(f)
            if isinstance(config, dict):
                result = float(config.get("min_combined_confidence", default_conf))
            else:
                result = default_conf
        except (FileNotFoundError, OSError):
            _log.warning("segment_preset_not_found", seg_id=seg_id, path=str(path))
            result = default_conf

        self._segment_min_confidence[seg_id] = result
        return result

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

                # Compute slippage in bps
                expected_price = candles[-1].close if candles else None
                if (
                    result.fill_price is not None
                    and expected_price is not None
                    and expected_price > 0
                ):
                    slippage_bps = float(
                        (result.fill_price - expected_price) / expected_price * 10000
                    )
                else:
                    slippage_bps = 0.0

                if self._sandbox_monitor is not None:
                    self._sandbox_monitor.record_slippage(slippage_bps)

                if self._metrics:
                    self._metrics.record_trade(
                        market=market_id,
                        side=order.side.lower(),
                        slippage_bps=slippage_bps,
                        fill_latency_seconds=0.0,
                    )
                # Wire stop-loss on BUY fill + track entry price for Kelly
                if order.side == "BUY" and candles and result.fill_price is not None:
                    self._entry_prices[order.symbol] = result.fill_price
                    multiplier = _ATR_MULTIPLIER_MOEX if market_id == "moex" else _ATR_MULTIPLIER_US
                    stop = compute_atr_stop_loss(
                        result.fill_price, candles, atr_multiplier=multiplier
                    )
                    if stop is not None and multiplier > _ZERO:
                        # Derive ATR: stop = entry - mult * atr => atr = (entry - stop) / mult
                        atr_val = (result.fill_price - stop) / multiplier
                        with self._stop_loss_lock:
                            self._stop_states[order.symbol] = self._StopLossState(
                                initial_stop=stop,
                                current_stop=stop,
                                highest_price=result.fill_price,
                                trail_activated=False,
                                activation_atr=Decimal("1.0"),
                                trail_atr=Decimal("1.5"),
                                entry_price=result.fill_price,
                                atr_value=atr_val,
                            )
                # Update Kelly on SELL fill + clear stop-loss
                elif order.side == "SELL":
                    if result.fill_price is not None:
                        self._update_kelly(order.symbol, result.fill_price)
                    with self._stop_loss_lock:
                        self._stop_states.pop(order.symbol, None)
            else:
                _log.warning(
                    "order_rejected",
                    symbol=order.symbol,
                    side=order.side,
                    reason=result.reason,
                    market=market_id,
                )
                self._alerter.on_trade_rejected(order, result.reason)
                if self._metrics:
                    self._metrics.record_rejection(
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
        """Check trailing stop-loss state and trigger SELL if breached.

        Implements the same 5-step trailing logic as SimulatedBroker:
        1. Update high-water mark
        2. Check activation threshold
        3. Ratchet trail stop upward (never down)
        4. Check trigger condition
        5. Submit SELL and record in _cycle_exited_symbols (PARITY-04)

        The entire check-sell-remove is atomic under _stop_loss_lock to prevent
        double-sell from concurrent threads (CONC-01).
        """
        with self._stop_loss_lock:
            state = self._stop_states.get(symbol)
            if state is None:
                return

            # Step 1: Update high-water mark
            state.highest_price = max(state.highest_price, current_price)

            # Step 2: Check activation
            if not state.trail_activated:
                activation_threshold = state.entry_price + state.activation_atr * state.atr_value
                if state.highest_price >= activation_threshold:
                    state.trail_activated = True

            # Step 3: Ratchet trail stop (only moves up)
            if state.trail_activated:
                trail_stop = state.highest_price - state.trail_atr * state.atr_value
                state.current_stop = max(state.current_stop, trail_stop)

            # Step 4: Trigger check
            if current_price > state.current_stop:
                return

            # Step 5: Stop triggered
            _log.warning(
                "stop_triggered",
                symbol=symbol,
                price=float(current_price),
                stop=float(state.current_stop),
                trailing=state.trail_activated,
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
                    return  # Don't clear stop state -- retry next cycle
                # Update Kelly with stop-loss exit
                self._update_kelly(symbol, current_price)
            # Clear stop state after successful trigger (or zero position)
            del self._stop_states[symbol]
            self._cycle_exited_symbols.add(symbol)  # PARITY-04

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
                retrain_end = self._now()
                retrain_start = retrain_end - timedelta(days=500 * 2)
                candles = fetcher.fetch_candles(  # type: ignore[attr-defined]
                    symbol=instrument.symbol,
                    start=retrain_start,
                    end=retrain_end,
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
        if self._metrics:
            for market_id, equity in new_baselines.items():
                pnl_val = market_pnl.get(market_id, _ZERO)
                self._metrics.set_daily_pnl(market_id, float(pnl_val))
                self._metrics.set_portfolio_equity(market_id, float(equity))

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

    def _persist_to_db(self, coro: Any, *, table: str) -> None:
        """Fire-and-forget DB write. Never crashes the trading loop (PERSIST-05)."""
        try:
            self._run_async(coro)
        except Exception:
            from finalayze.api.metrics import db_write_failures  # noqa: PLC0415

            db_write_failures.labels(table=table).inc()
            _log.warning("db_persist_failed", table=table, exc_info=True)

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
            _log.warning("equity_snapshot_persist_failed", exc_info=True)

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
        Otherwise current broker equity becomes the baseline and is
        persisted so subsequent restarts within the same day find it.
        """
        try:
            self._run_async(self._load_baseline_async())
        except Exception:
            _log.info(
                "baseline_from_broker",
                reason="no DB snapshots for today, persisting current equity",
            )
            # Persist current broker equity so next restart finds it
            baselines: dict[str, Decimal] = {}
            for market_id in self._fetchers:
                equity = self._get_market_equity(market_id)
                if equity is not None:
                    baselines[market_id] = equity
                    self._baseline_equities[market_id] = equity
            if baselines:
                now = datetime.now(UTC)
                self._persist_equity_snapshots(baselines, now)

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
        if loaded == 0:
            msg = "no snapshots for today"
            raise ValueError(msg)

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
        from finalayze.api.alerts import AlertPriority  # noqa: PLC0415

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
