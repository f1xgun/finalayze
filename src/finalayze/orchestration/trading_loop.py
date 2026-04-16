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
import math
import threading
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import structlog
from apscheduler.schedulers.background import BackgroundScheduler

from finalayze.core.schemas import SignalDirection
from finalayze.core.validation_logger import CycleLogEntry, ValidationLogger

try:
    from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
except ImportError:  # pragma: no cover
    SQLAlchemyJobStore = None
from finalayze.data.moex_calendar import is_moex_holiday
from finalayze.data.normalizer import DataNormalizer
from finalayze.markets.currency import CurrencyConverter
from finalayze.markets.schedule import SCHEDULES
from finalayze.orchestration.daily_reporting import DailyReportingService
from finalayze.orchestration.db_persistence import TradingPersistence
from finalayze.orchestration.ml_retraining import MLRetrainingService
from finalayze.orchestration.news_pipeline import NewsPipeline
from finalayze.orchestration.sentiment_manager import SentimentManager

if TYPE_CHECKING:
    from config.settings import Settings

    from finalayze.analysis.event_classifier import EventClassifier
    from finalayze.analysis.impact_estimator import ImpactEstimator
    from finalayze.analysis.news_analyzer import NewsAnalyzer
    from finalayze.analysis.news_impact_analyzer import NewsImpactAnalyzer
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
    from finalayze.execution.broker_base import OrderRequest
    from finalayze.execution.broker_router import BrokerRouter
    from finalayze.markets.fx_service import FXRateService
    from finalayze.markets.instruments import Instrument, InstrumentRegistry
    from finalayze.ml.registry import MLModelRegistry
    from finalayze.monitoring.health_monitor import HealthMonitor
    from finalayze.monitoring.sandbox_monitor import SandboxMonitorService
    from finalayze.orchestration.bond_cycle import BondCycleProcessor
    from finalayze.risk.circuit_breaker import (
        CircuitBreaker,
        CircuitLevel,
        CrossMarketCircuitBreaker,
    )
    from finalayze.risk.position_sizing_pipeline import PositionSizingPipeline
    from finalayze.risk.regime import RegimeState
    from finalayze.strategies.combiner import StrategyCombiner

# ── Constants ──────────────────────────────────────────────────────────────
_CANDLE_LOOKBACK = 210  # SMA-200 needs 200 bars + buffer; dual_momentum needs 126
_CAUTION_SIZE_FACTOR = Decimal("0.5")  # halve position size at CAUTION
_MIN_CONFIDENCE_BOOST = 1.2  # raise required confidence 20% at CAUTION
_DEFAULT_SENTIMENT = 0.0
_SENTIMENT_HALF_LIFE_HOURS = 4.0
_SENTIMENT_DECAY_LAMBDA = math.log(2) / _SENTIMENT_HALF_LIFE_HOURS  # ~0.1733
_ZERO = Decimal(0)
_WEEKEND_WEEKDAY = 5  # Saturday=5, Sunday=6
_STALENESS_THRESHOLD_HOURS: float = 72.0  # 3x daily; covers weekends + calendar-aware holidays
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
        health_monitor: HealthMonitor | None = None,
        metrics_collector: type[MetricsCollector] | None = None,
        grpc_loop: asyncio.AbstractEventLoop | None = None,
        kill_switch: object | None = None,
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
        self._kill_switch = kill_switch

        self._fx = CurrencyConverter(base_currency="USD")

        # Sentiment management (thread-safe via SentimentManager)
        self._sentiment_mgr = SentimentManager(
            registry=instrument_registry,
            market_ids=list(fetchers.keys()),
            cache=cache,
        )

        # Daily baseline equities: market_id -> equity at start of trading day
        self._baseline_equities: dict[str, Decimal] = {}

        # Stop-loss tracking: symbol -> StopLossState (trailing, thread-safe via lock)
        self._stop_states: dict[str, StopLossState] = {}
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

        # Position ownership tracking: symbol -> strategy_name that opened the position
        # Set on BUY fill, cleared on SELL fill and stop-loss trigger.
        # Used by PresetApplicator (Plan 38-01) to check for open positions before
        # disabling a strategy via auto-apply.
        self._entry_strategy: dict[str, str] = {}

        self._ml_registry = ml_registry
        self._ml_retrainer = MLRetrainingService(
            fetchers=fetchers,
            registry=instrument_registry,
            ml_registry=ml_registry,
            settings=settings,
            alerter=alerter,
            collect_segments_fn=self._sentiment_mgr.collect_active_segments,
            now_fn=self._now,
        )
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

        # Database persistence (fire-and-forget writes to avoid crashing trading loop)
        # Initialize with settings.database_url if available
        db_url = getattr(settings, "database_url", None)
        self._persistence = TradingPersistence(db_url, self._async_loop, settings)

        # Daily reporting service (extracted Phase 1.4)
        self._daily_reporter = DailyReportingService(
            broker_router=broker_router,
            circuit_breakers=circuit_breakers,
            cross_market_breaker=cross_market_breaker,
            loss_limit_tracker=self._loss_limit_tracker,
            alerter=alerter,
            persistence=self._persistence,
            bond_processor=bond_cycle_processor,
            fx_service=fx_service,
            metrics_collector=metrics_collector,
            settings=settings,
            now_fn=self._now,
        )

        # News pipeline service (extracted Phase 1.5)
        self._news_pipeline = NewsPipeline(
            rss_fetcher=rss_fetcher,
            telegram_reader=telegram_reader,
            news_fetcher=news_fetcher,
            news_impact_analyzer=news_impact_analyzer,
            sector_ticker_mapper=sector_ticker_mapper,
            sentiment_mgr=self._sentiment_mgr,
            persistence=self._persistence,
            registry=instrument_registry,
            cache=cache,
            settings=settings,
            alerter=alerter,
            async_loop_fn=self._run_async,
        )

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
        self._cycle_instruments_processed = 0
        self._cycle_signals_generated = 0
        self._cycle_orders_submitted = 0
        self._cycle_orders_filled = 0
        self._cycle_errors_caught = 0
        self._cycle_exited_symbols = set()
        self._cycle_dropped_no_bars = 0
        self._cycle_dropped_below_threshold = 0
        self._cycle_dropped_pre_trade = 0

    # ── Candle staleness ──────────────────────────────────────────────────

    @staticmethod
    def _is_candle_stale(latest_ts: datetime, threshold_hours: float) -> bool:
        """Return True if the latest candle timestamp is older than threshold.

        Calendar-aware: subtracts weekends and MOEX holidays from the age
        so that Monday mornings and post-New-Year cycles are not falsely
        flagged as stale.

        Args:
            latest_ts: Timestamp of the most recent candle (UTC).
            threshold_hours: Maximum acceptable age in hours.

        Returns:
            True if candle data is genuinely stale after accounting for
            non-trading days.
        """
        now = datetime.now(UTC)
        age = now - latest_ts
        # Quick path: if within threshold even counting all hours, not stale
        if age < timedelta(hours=threshold_hours):
            return False
        # Count non-trading days between latest_ts and now
        non_trading_days = 0
        check_date = latest_ts.date() + timedelta(days=1)
        end_date = now.date()
        while check_date <= end_date:
            if check_date.weekday() >= 5 or is_moex_holiday(check_date):
                non_trading_days += 1
            check_date += timedelta(days=1)
        # Subtract non-trading days from the age
        adjusted_age = age - timedelta(days=non_trading_days)
        return adjusted_age >= timedelta(hours=threshold_hours)

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
            self._persistence._async_loop = loop  # Update persistence's loop reference
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
        if self._kill_switch is not None and self._kill_switch.is_killed:
            raise RuntimeError("Kill switch active -- clear flag before restarting")

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
            self._news_pipeline.run_news_cycle,
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
                self._ml_retrainer.retrain_all,
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
                self._bond_processor.reconcile_with_broker()  # type: ignore[attr-defined]
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
                    signals_dropped_no_bars=self._cycle_dropped_no_bars,
                    signals_dropped_below_threshold=self._cycle_dropped_below_threshold,
                    signals_dropped_pre_trade=self._cycle_dropped_pre_trade,
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
                        pnl_rub=Decimal(str(equity_rub))
                        - self._baseline_equities.get("moex", Decimal(str(equity_rub))),
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
            self._cycle_dropped_no_bars += 1
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

        sentiment_score = self._sentiment_mgr.get_sentiment(seg_id, instrument.symbol)

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
            _log.info("signal_dropped_below_threshold", symbol=instrument.symbol, segment=seg_id)
            self._cycle_dropped_below_threshold += 1
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

        # Fire-and-forget signal persistence (PERSIST-02)
        self._persistence.persist_signal(signal)

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
            stop_loss_price = _stop_st.current_stop if _stop_st is not None else None

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
            self._cycle_dropped_pre_trade += 1
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
        self._submit_order(order, market_id, candles=candles, strategy_name=signal.strategy_name)
        self._cycle_orders_submitted += 1

        # 6A.7: Record day trade after successful order submission
        if is_day_trade:
            self._pdt_tracker.record_day_trade(now.date())

    def _build_sizing_pipeline(self, segment_id: str) -> PositionSizingPipeline:
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

    def _get_regime_state(self) -> RegimeState | None:
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

        order_value = pipeline.compute(context)
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

    def get_entry_strategies(self) -> dict[str, str]:
        """Return a snapshot of {symbol: strategy_name} for currently open positions.

        Used by PresetApplicator to check position ownership before disabling a
        strategy via auto-apply.  Returns a copy so callers cannot mutate internal state.
        """
        return dict(self._entry_strategy)

    def _submit_order(
        self,
        order: OrderRequest,
        market_id: str,
        candles: list[Candle] | None = None,
        strategy_name: str = "",
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

                # Fire-and-forget order persistence (PERSIST-01)
                self._persistence.persist_order(order, result, market_id)

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
                # Track position ownership for PresetApplicator (APPLY-03)
                if order.side == "BUY":
                    self._entry_strategy[order.symbol] = strategy_name
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
                    self._entry_strategy.pop(order.symbol, None)
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
            self._entry_strategy.pop(symbol, None)
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

    def _daily_reset(self) -> None:
        """Reset circuit breakers and send daily P&L summary.

        Delegates to DailyReportingService.daily_reset().
        """
        # Sync metrics reference and _now method (for test compatibility and consistency)
        self._daily_reporter._metrics = self._metrics
        self._daily_reporter._now = self._now
        updated_baselines = self._daily_reporter.daily_reset(self._baseline_equities)
        self._baseline_equities.update(updated_baselines)

    def _compute_top_movers(self) -> list[tuple[str, float]]:
        """Compute top 3 movers by absolute P&L % across all markets.

        Delegates to DailyReportingService._compute_top_movers().
        """
        return self._daily_reporter._compute_top_movers(self._baseline_equities)

    def _load_baseline_from_db(self) -> None:
        """Load latest equity snapshots from DB on startup.

        If snapshots exist for today, use them as baselines.
        Otherwise current broker equity becomes the baseline and is
        persisted so subsequent restarts within the same day find it.

        Delegates to DailyReportingService.load_baselines_from_db().
        """
        loaded = self._daily_reporter.load_baselines_from_db(list(self._fetchers.keys()))
        self._baseline_equities.update(loaded)

    def _weekly_digest(self) -> None:
        """Send weekly performance digest on Sunday evening.

        Computes week P&L from DailyEquitySnapshot DB records (falls back
        to current baseline equities if DB unavailable). Includes trade
        count, best/worst positions, circuit breaker trip count.

        Runs even after restart because it reads from persisted snapshots.
        """
        """Delegates to DailyReportingService.weekly_digest().
        """
        self._daily_reporter.weekly_digest(self._baseline_equities)

    def _liquidate_market(self, market_id: str) -> None:
        """Close all open positions in a market (L3 circuit breaker response).

        Delegates to DailyReportingService.liquidate_market().
        """
        self._daily_reporter.liquidate_market(market_id, self._baseline_equities)
