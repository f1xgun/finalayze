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
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Protocol, cast

import structlog
from apscheduler.schedulers.background import BackgroundScheduler

from finalayze.core.validation_logger import CycleLogEntry, ValidationLogger

try:
    from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
except ImportError:  # pragma: no cover
    SQLAlchemyJobStore = None
from finalayze.data.moex_calendar import is_moex_holiday
from finalayze.markets.currency import CurrencyConverter
from finalayze.markets.schedule import SCHEDULES
from finalayze.orchestration.anomaly_handler import AnomalyHandler
from finalayze.orchestration.async_runtime import AsyncRuntime
from finalayze.orchestration.broker_reconnect import (
    attempt_grpc_reconnect,
    reconcile_inflight_orders,
)
from finalayze.orchestration.sentiment_manager import SentimentManager
from finalayze.orchestration.signal_executor import (
    _CANDLE_LOOKBACK,  # noqa: F401  # re-export for tests
    _STALENESS_THRESHOLD_HOURS,  # noqa: F401  # re-export for tests
)

if TYPE_CHECKING:
    import uuid

    from config.settings import Settings

    from finalayze.analysis.event_classifier import EventClassifier
    from finalayze.analysis.impact_estimator import ImpactEstimator
    from finalayze.analysis.news_analyzer import NewsAnalyzer
    from finalayze.analysis.news_impact_analyzer import NewsImpactAnalyzer
    from finalayze.analysis.sector_ticker_mapper import SectorTickerMapper
    from finalayze.api.alerts import TelegramAlerter
    from finalayze.api.metrics import MetricsCollector
    from finalayze.core.schemas import FundamentalSnapshot
    from finalayze.data.cache import RedisCache
    from finalayze.data.fetchers.newsapi import NewsApiFetcher
    from finalayze.data.fetchers.rss_fetcher import RssNewsFetcher
    from finalayze.data.fetchers.telegram_reader import TelegramChannelReader
    from finalayze.data.macro_cache import MacroCacheService
    from finalayze.execution.broker_router import BrokerRouter
    from finalayze.markets.fx_service import FXRateService
    from finalayze.markets.instruments import InstrumentRegistry
    from finalayze.ml.registry import MLModelRegistry
    from finalayze.monitoring.health_monitor import HealthMonitor
    from finalayze.monitoring.sandbox_monitor import SandboxMonitorService
    from finalayze.orchestration.bond_cycle import BondCycleProcessor
    from finalayze.risk.circuit_breaker import (
        CircuitBreaker,
        CircuitLevel,
        CrossMarketCircuitBreaker,
    )
    from finalayze.strategies.combiner import StrategyCombiner


class _SupportsFetchFundamentals(Protocol):
    """Structural type for a fetcher exposing the Phase-59 fundamentals call.

    The heterogeneous ``self._fetchers`` map is typed ``dict[str, object]``; the
    MOEX entry (TinkoffFetcher) provides ``fetch_fundamentals``. Casting to this
    Protocol narrows the type for the capture cycle without importing the concrete
    data-tier class into orchestration.
    """

    def fetch_fundamentals(self, symbol: str) -> FundamentalSnapshot | None: ...


# ── Constants ──────────────────────────────────────────────────────────────
_ZERO = Decimal(0)
_WEEKEND_WEEKDAY = 5  # Saturday=5, Sunday=6
_MARKET_CURRENCY: dict[str, str] = {"us": "USD", "moex": "RUB"}

# US market hours in UTC: 9:30-16:00 ET = 14:30-21:00 UTC
_US_OPEN_UTC = (14, 30)
_US_CLOSE_UTC = (21, 0)
# MOEX market hours in UTC: 10:00-18:45 MSK = 07:00-15:45 UTC
_MOEX_OPEN_UTC = (7, 0)
_MOEX_CLOSE_UTC = (15, 45)

_MAX_ARTICLES_PER_CYCLE = 20  # budget cap: prevent LLM cost explosion on busy news days

SOURCE_CREDIBILITY: dict[str, float] = {
    "rbc": 0.8,
    "interfax": 0.8,
    "tass": 0.8,
    "moex_iss": 0.8,
    "reuters": 0.8,
    "telegram": 0.7,
}

_log = structlog.get_logger()


def get_credibility(source: str) -> float:
    """Return credibility score for a news source. Default 0.5 for unknown."""
    return SOURCE_CREDIBILITY.get(source.lower(), 0.5)


def validate_tickers(
    tickers: list[str],
    registry: object,
    market_id: str,
) -> list[str]:
    """Filter tickers to only those in the instrument registry."""
    from finalayze.core.exceptions import InstrumentNotFoundError  # noqa: PLC0415

    valid: list[str] = []
    for ticker in tickers:
        try:
            registry.get(ticker, market_id)  # type: ignore[attr-defined]
            valid.append(ticker)
        except InstrumentNotFoundError:
            _log.warning("entity_not_in_registry", ticker=ticker, market_id=market_id)
    return valid


@dataclass(frozen=True)
class TradingLoopDeps:
    """Typed bundle of every collaborator TradingLoop needs to run.

    Replaces a 30-parameter ctor. Fields kept in original ctor order for
    diff readability; required fields first, optional fields (default None)
    grouped at the end.
    """

    settings: Settings
    fetchers: dict[str, object]
    news_fetcher: NewsApiFetcher
    news_analyzer: NewsAnalyzer
    event_classifier: EventClassifier
    impact_estimator: ImpactEstimator
    strategy: StrategyCombiner
    broker_router: BrokerRouter
    circuit_breakers: dict[str, CircuitBreaker]
    cross_market_breaker: CrossMarketCircuitBreaker
    alerter: TelegramAlerter
    instrument_registry: InstrumentRegistry
    cache: RedisCache | None = None
    ml_registry: MLModelRegistry | None = None
    fx_service: FXRateService | None = None
    bond_cycle_processor: BondCycleProcessor | None = None
    macro_cache: MacroCacheService | None = None
    rss_fetcher: RssNewsFetcher | None = None
    telegram_reader: TelegramChannelReader | None = None
    news_impact_analyzer: NewsImpactAnalyzer | None = None
    sector_ticker_mapper: SectorTickerMapper | None = None
    sandbox_monitor: SandboxMonitorService | None = None
    health_monitor: HealthMonitor | None = None
    metrics_collector: type[MetricsCollector] | None = None
    grpc_loop: asyncio.AbstractEventLoop | None = field(default=None)
    kill_switch: object | None = None
    meta_agent_runner: object | None = None


class TradingLoop:
    """Schedules and runs the news, strategy, and daily-reset cycles.

    Designed for TEST / SANDBOX modes. Will gate on WorkMode in real mode.
    """

    # Class-level defaults so MagicMock(spec=TradingLoop) recognizes these attrs,
    # and so post-construction wiring from bootstrap.py (e.g. `loop._alerter_ref`)
    # type-checks. All sub-component fields are overridden in __init__.
    _news_pipeline: Any = None
    _signal_executor: Any = None
    _position_tracker: Any = None
    _daily_reporter: Any = None
    _ml_retraining: Any = None
    _sentiment_mgr: Any = None
    _persistence: Any = None
    _llm_client: Any = None
    _scheduler: BackgroundScheduler | None = None
    _async_runtime: AsyncRuntime | None = None
    # Post-construction wiring slot used by api/lifespan to reach the alerter
    # without traversing every sub-component (bootstrap.py sets this after
    # constructing the loop and circuit breakers).
    _alerter_ref: Any = None

    # ── Event loop property proxies (delegate to AsyncRuntime) ──────────────
    # These properties allow tests and internal code to read/write loop attributes
    # directly on TradingLoop while delegating to the single source of truth in
    # AsyncRuntime. This preserves backward compatibility without mirroring state.
    # If _async_runtime is None (e.g., when created with object.__new__ in tests),
    # the setters will create a lazy AsyncRuntime instance.

    @property
    def _async_loop(self) -> asyncio.AbstractEventLoop | None:
        """Get the async event loop from AsyncRuntime."""
        if self._async_runtime is None:
            return None
        return self._async_runtime.async_loop

    @_async_loop.setter
    def _async_loop(self, value: asyncio.AbstractEventLoop | None) -> None:
        """Set the async event loop on AsyncRuntime."""
        # Lazy-create AsyncRuntime if needed (for tests using object.__new__)
        if self._async_runtime is None:
            self._async_runtime = AsyncRuntime()
        self._async_runtime.async_loop = value

    @property
    def _async_thread(self) -> threading.Thread | None:
        """Get the async thread from AsyncRuntime."""
        if self._async_runtime is None:
            return None
        return self._async_runtime.async_thread

    @_async_thread.setter
    def _async_thread(self, value: threading.Thread | None) -> None:
        """Set the async thread on AsyncRuntime."""
        # Lazy-create AsyncRuntime if needed (for tests using object.__new__)
        if self._async_runtime is None:
            self._async_runtime = AsyncRuntime()
        self._async_runtime.async_thread = value

    @property
    def _grpc_loop(self) -> asyncio.AbstractEventLoop | None:
        """Get the gRPC event loop from AsyncRuntime."""
        if self._async_runtime is None:
            return None
        return self._async_runtime.grpc_loop

    @_grpc_loop.setter
    def _grpc_loop(self, value: asyncio.AbstractEventLoop | None) -> None:
        """Set the gRPC event loop on AsyncRuntime."""
        # Lazy-create AsyncRuntime if needed (for tests using object.__new__)
        if self._async_runtime is None:
            self._async_runtime = AsyncRuntime()
        self._async_runtime.grpc_loop = value

    @property
    def _grpc_thread(self) -> threading.Thread | None:
        """Get the gRPC thread from AsyncRuntime."""
        if self._async_runtime is None:
            return None
        return self._async_runtime.grpc_thread

    @_grpc_thread.setter
    def _grpc_thread(self, value: threading.Thread | None) -> None:
        """Set the gRPC thread on AsyncRuntime."""
        # Lazy-create AsyncRuntime if needed (for tests using object.__new__)
        if self._async_runtime is None:
            self._async_runtime = AsyncRuntime()
        self._async_runtime.grpc_thread = value

    def __init__(self, deps: TradingLoopDeps) -> None:  # noqa: PLR0915
        from finalayze.execution.broker_base import OrderRequest  # noqa: PLC0415
        from finalayze.risk.circuit_breaker import CircuitLevel  # noqa: PLC0415
        from finalayze.risk.kelly import RollingKelly  # noqa: PLC0415
        from finalayze.risk.loss_limits import LossLimitTracker  # noqa: PLC0415
        from finalayze.risk.pre_trade_check import PDTTracker, PreTradeChecker  # noqa: PLC0415

        # Unpack deps into locals so the body below is unchanged.
        settings = deps.settings
        fetchers = deps.fetchers
        news_fetcher = deps.news_fetcher
        news_analyzer = deps.news_analyzer
        event_classifier = deps.event_classifier
        impact_estimator = deps.impact_estimator
        strategy = deps.strategy
        broker_router = deps.broker_router
        circuit_breakers = deps.circuit_breakers
        cross_market_breaker = deps.cross_market_breaker
        alerter = deps.alerter
        instrument_registry = deps.instrument_registry
        cache = deps.cache
        ml_registry = deps.ml_registry
        fx_service = deps.fx_service
        bond_cycle_processor = deps.bond_cycle_processor
        macro_cache = deps.macro_cache
        rss_fetcher = deps.rss_fetcher
        telegram_reader = deps.telegram_reader
        news_impact_analyzer = deps.news_impact_analyzer
        sector_ticker_mapper = deps.sector_ticker_mapper
        sandbox_monitor = deps.sandbox_monitor
        health_monitor = deps.health_monitor
        metrics_collector = deps.metrics_collector
        grpc_loop = deps.grpc_loop
        kill_switch = deps.kill_switch
        meta_agent_runner = deps.meta_agent_runner

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
        # Phase 58-02-07: optional MetaAgentRunner injected at bootstrap.
        # Default None keeps every existing call-site source-compatible.
        self._meta_agent_runner = meta_agent_runner

        self._fx = CurrencyConverter(base_currency="USD")

        # Sentiment management (thread-safe via SentimentManager)
        self._sentiment_mgr = SentimentManager(
            registry=instrument_registry,
            market_ids=list(fetchers.keys()),
            cache=cache,
        )

        # Daily baseline equities: market_id -> equity at start of trading day
        self._baseline_equities: dict[str, Decimal] = {}

        # Position tracking extracted to PositionTracker (Phase 1.6)
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

        # AsyncRuntime: manages persistent event loops for async and gRPC operations.
        # Inject a callback to wire the newly created async loop to persistence.
        def _on_async_loop_created(loop: asyncio.AbstractEventLoop) -> None:
            if hasattr(self, "_persistence") and self._persistence is not None:
                self._persistence._async_loop = loop

        self._async_runtime = AsyncRuntime(
            grpc_loop=grpc_loop,
            on_async_loop_created=_on_async_loop_created,
        )

        # Construct all 6 service collaborators in dependency order (STOP-03).
        # TradingPersistence must be first, then PositionTracker, then the rest.
        from finalayze.orchestration.loop_services import build_loop_services  # noqa: PLC0415

        _services = build_loop_services(
            deps,
            sentiment_mgr=self._sentiment_mgr,
            kelly_sizer=self._kelly_sizer,
            pre_trade_checker=self._pre_trade_checker,
            loss_limit_tracker=self._loss_limit_tracker,
            now_fn=self._now,
            run_async_fn=self._run_async,
        )
        self._persistence = _services.persistence
        self._position_tracker = _services.position_tracker
        self._signal_executor = _services.signal_executor
        self._ml_retrainer = _services.ml_retrainer
        self._daily_reporter = _services.daily_reporter
        self._news_pipeline = _services.news_pipeline

        # Anomaly handler extracted to improve modularity (Phase 57-04 D-04)
        # Reads _llm_client lazily via lambda so late initialization is supported
        self._anomaly_handler = AnomalyHandler(
            alerter,
            lambda: getattr(self, "_llm_client", None),
        )

        self._ml_registry = ml_registry
        self._scheduler: BackgroundScheduler | None = None
        self._stop_event = threading.Event()

        # Total strategy cycles completed (used by HealthMonitor for liveness)
        self._total_cycles: int = 0

        # ALRT-01 D-07: monotonic cycle counter, incremented at the top of every
        # _strategy_cycle_impl invocation. NEVER reset (initialised exactly once
        # here). Stamped onto StopLossState.entry_cycle_index at register_entry
        # so check_stop_losses can compute hold_bars on trigger.
        self._cycle_count: int = 0

        # Per-cycle portfolio cache: market_id -> PortfolioState
        # Populated at the start of each strategy cycle, cleared at the end.
        self._cycle_portfolio_cache: dict[str, Any] = {}

        # asyncio.Lock for gRPC client serialization (equity + bond don't overlap)
        self._grpc_lock = asyncio.Lock()

        # Bond cycle enabled flag (set by preflight; independent degradation)
        self._bond_enabled: bool = True

        # gRPC reconnection backoff delays in seconds
        self._reconnect_delays = [30.0, 60.0, 120.0, 240.0, 300.0]

        # Structured cycle validation logger
        self._validation_logger = ValidationLogger()

        # Per-instrument last price cache: symbol -> Decimal (built during strategy cycle)
        self._last_prices: dict[str, Decimal] = {}

        # Fundamental-capture liveness markers (Phase 63 CAPTURE-01 / D-03).
        # Set by _fundamental_capture_cycle after each successful run; read by the
        # freshness check (plan 63-03) to alert when the "data clock" stops. None
        # until the first run so freshness has a defined initial state.
        self._last_fundamental_capture_at: datetime | None = None
        self._last_fundamental_coverage_ratio: float | None = None

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
        self._position_tracker.reset_cycle_exits()
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
            if check_date.weekday() >= _WEEKEND_WEEKDAY or is_moex_holiday(check_date):
                non_trading_days += 1
            check_date += timedelta(days=1)
        # Subtract non-trading days from the age
        adjusted_age = age - timedelta(days=non_trading_days)
        return adjusted_age >= timedelta(hours=threshold_hours)

    # ── gRPC reconnection ────────────────────────────────────────────────

    def _attempt_grpc_reconnect(self, broker_name: str) -> bool:
        """Try to reconnect gRPC channel with exponential backoff.

        Thin delegator to broker_reconnect.attempt_grpc_reconnect.
        Handles early-exit for non-Tinkoff brokers before forwarding
        attributes to avoid test-compatibility issues when mocking.
        """
        from finalayze.execution.tinkoff_broker import TinkoffBroker  # noqa: PLC0415

        broker = self._broker_router.route(broker_name)
        if not isinstance(broker, TinkoffBroker):
            _log.warning("reconnect_not_tinkoff", broker_name=broker_name)
            return False

        return attempt_grpc_reconnect(
            broker_router=self._broker_router,
            alerter=self._alerter,
            stop_event=self._stop_event,
            reconnect_delays=self._reconnect_delays,
            broker_name=broker_name,
        )

    # ── In-flight order reconciliation ───────────────────────────────────

    def _reconcile_inflight_orders(self) -> None:
        """Query open orders from all TinkoffBrokers, cancel stale ones, log fills.

        Thin delegator to broker_reconnect.reconcile_inflight_orders.
        Forwards self's attributes to the pure function.
        """
        reconcile_inflight_orders(
            broker_router=self._broker_router,
            circuit_breakers=self._circuit_breakers,
        )

    # ── Async helper ────────────────────────────────────────────────────────

    def _run_async(self, coro: Any, *, timeout: int = 30) -> Any:
        """Run an async coroutine on a persistent background event loop.

        Delegates to AsyncRuntime.run_async. Lazily creates a daemon thread
        with its own event loop on first call. The callback registered with
        AsyncRuntime automatically wires the loop to persistence.

        Default 30-second timeout; batch operations may pass a larger value.
        """
        assert self._async_runtime is not None
        return self._async_runtime.run_async(coro, timeout=timeout)

    # ── gRPC loop helpers ─────────────────────────────────────────────────────

    def _init_grpc_loop(self) -> asyncio.AbstractEventLoop:
        """Create a dedicated background event loop for all gRPC operations.

        Delegates to AsyncRuntime.init_grpc_loop. Isolated from async_loop
        to prevent PollerCompletionQueue BlockingIOError from starving
        HTTP/DB/Telegram coroutines and causing strategy cycle drift.
        """
        assert self._async_runtime is not None
        return self._async_runtime.init_grpc_loop()

    def _run_grpc(self, coro: Any, *, timeout: int = 30) -> Any:
        """Run a gRPC coroutine on the dedicated gRPC event loop.

        Delegates to AsyncRuntime.run_grpc. Use this for all TinkoffBroker
        and TinkoffFetcher calls. Non-gRPC async work (HTTP, DB, Telegram)
        should use _run_async().
        """
        assert self._async_runtime is not None
        return self._async_runtime.run_grpc(coro, timeout=timeout)

    @property
    def total_cycles(self) -> int:
        """Return the total number of strategy cycles completed."""
        return self._total_cycles

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def _setup_scheduler(self) -> None:
        """Construct APScheduler and register all scheduled jobs.

        Initializes:
          - BackgroundScheduler with thread pool executors (default, retrain, news)
          - News cycle job (interval-based)
          - Strategy cycle job (interval-based)
          - Daily reset job (cron-based)
          - Portfolio review job (cron-based at 15:50 UTC)
          - Meta-agent job (if enabled)
          - ML retrain job (if enabled)
          - FX update job (if enabled)
          - Bond cycle jobs: macro_refresh, bond_cycle, cbr_day_refresh (if bond cycle enabled)
          - Weekly digest job (cron-based on Sunday)

        All jobs use replace_existing=True so multiple calls to start() are safe.
        """
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
        # ALRT-04 (Phase 57 D-01, Pitfall 6): register portfolio review cron.
        # Fires at 15:50 UTC = 18:50 MSK, post-MOEX-close. The method body
        # at _portfolio_review_cycle exists since Phase 52, but PR #223
        # decomposition lost the add_job registration — re-added here.
        from apscheduler.triggers.cron import CronTrigger as _PRCronTrigger  # noqa: PLC0415

        self._scheduler.add_job(
            self._portfolio_review_cycle,
            _PRCronTrigger(hour=15, minute=50, timezone="UTC"),
            id="portfolio_review",
            replace_existing=True,
        )
        # CAPTURE-01 (Phase 63 D-01, Pitfall 6): register the daily MOEX
        # fundamental-capture cron alongside its method (land both together).
        # Default 07:00 UTC = 10:00 MSK pre-open; coalesce + max_instances=1
        # prevent a slow full-universe run from stacking (T-63-05).
        self._scheduler.add_job(
            self._fundamental_capture_cycle,
            _PRCronTrigger(
                hour=getattr(self._settings, "fundamental_capture_hour_utc", 7),
                minute=0,
                timezone="UTC",
            ),
            id="fundamental_capture",
            replace_existing=True,
            coalesce=True,
            max_instances=1,
        )
        # CAPTURE-03 (Phase 63 D-03, Pitfall 6): register the freshness/coverage
        # monitor alongside its method. Runs daily a few hours after the capture job
        # (default 09:00 UTC = 12:00 MSK) so a stalled "data clock" or a gRPC-wide
        # coverage collapse is surfaced the same morning via an IMPORTANT alert.
        _freshness_hour = getattr(self._settings, "fundamental_freshness_hour_utc", 9)
        if not isinstance(_freshness_hour, int):
            _freshness_hour = 9  # MagicMock-settings unit harness: use the default hour
        self._scheduler.add_job(
            self._fundamental_freshness_cycle,
            _PRCronTrigger(hour=_freshness_hour, minute=0, timezone="UTC"),
            id="fundamental_freshness",
            replace_existing=True,
        )
        # Phase 58-02-07: meta-agent (cron-driven autonomous monitor).
        # Guarded by meta_agent_enabled (SPEC AC #6 — disabled → no job).
        if getattr(self._settings, "meta_agent_enabled", False):
            from finalayze.api.v1.meta_agent import (  # noqa: PLC0415
                set_runner as _set_meta_runner,
            )
            from finalayze.meta_agent.scheduler import (  # noqa: PLC0415
                register_meta_agent_job,
            )

            register_meta_agent_job(
                self._scheduler,
                settings=self._settings,
                runner=self._meta_agent_runner,
                async_loop=self._async_loop,
            )
            _set_meta_runner(self._meta_agent_runner)  # type: ignore[arg-type]
            # Phase 58-05-06 (META-08, SPEC AC #15): launch the killswitch
            # env-var poller on the persistent async loop. The poller
            # MUST run on the same loop where ``await proc.wait()`` was
            # set up by spawn_readonly / spawn_fix — see PLAN body Risks.
            # Lazy-init the async loop now so the poller has a live loop
            # to attach to (mirrors the lazy init in _run_async()).
            # Always ensure async_loop is initialized when meta_agent is enabled
            # so persist_decision / persist_alert work from the very first tick.
            # ensure_async_loop lazily creates the loop + fires the persistence callback.
            if self._async_loop is None or self._async_loop.is_closed():
                assert self._async_runtime is not None
                self._async_runtime.ensure_async_loop()
            # Phase 58-05-06 (META-08, SPEC AC #15): launch killswitch poller.
            if (
                self._meta_agent_runner is not None
                and getattr(self._meta_agent_runner, "killswitch", None) is not None
                and self._async_loop is not None
            ):
                _ks = self._meta_agent_runner.killswitch  # type: ignore[attr-defined]
                asyncio.run_coroutine_threadsafe(_ks.start(), self._async_loop)
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

    def start(self) -> None:
        """Start the APScheduler and block until stop() is called."""
        if (
            self._kill_switch is not None
            and hasattr(self._kill_switch, "is_killed")
            and self._kill_switch.is_killed
        ):
            raise RuntimeError("Kill switch active -- clear flag before restarting")

        self._setup_scheduler()
        assert self._scheduler is not None

        # Load equity baselines from DB before starting scheduler
        # so daily P&L calculations use persisted start-of-day values
        self._load_baseline_from_db()

        # Reconcile in-flight orders from previous session before trading
        self._reconcile_inflight_orders()

        # Restore stop-loss state from DB so existing positions are protected
        self._restore_stop_states_from_db()

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
        # Phase 58-05-06 (META-08, SPEC AC #15): cancel the meta-agent
        # killswitch poller BEFORE shutting down the async loop (otherwise
        # the cancel coroutine has nowhere to run). Best-effort — never
        # raises, mirroring the rest of the stop() shutdown sequence.
        if (
            self._meta_agent_runner is not None
            and getattr(self._meta_agent_runner, "killswitch", None) is not None
            and self._async_loop is not None
            and not self._async_loop.is_closed()
        ):
            try:
                ks = self._meta_agent_runner.killswitch  # type: ignore[attr-defined]
                future = asyncio.run_coroutine_threadsafe(ks.stop(), self._async_loop)
                future.result(timeout=5)
            except Exception:
                _log.debug("meta_agent_killswitch_stop_failed_during_shutdown")
        if self._scheduler is not None:
            self._scheduler.shutdown(wait=True)
        # Close Redis and FX connections on the async loop before stopping it
        if self._async_loop is not None and not self._async_loop.is_closed():
            if self._cache is not None:
                try:
                    asyncio.run_coroutine_threadsafe(self._cache.close(), self._async_loop).result(
                        timeout=5
                    )
                except Exception:
                    _log.debug("Failed to close RedisCache on shutdown")
            if self._fx_service is not None:
                try:
                    asyncio.run_coroutine_threadsafe(
                        self._fx_service.close(), self._async_loop
                    ).result(timeout=5)
                except Exception:
                    _log.debug("Failed to close FXRateService on shutdown")
            # Dispose the per-loop DB engine on its OWN loop before that loop is
            # torn down, so its asyncpg pool is returned to PostgreSQL instead of
            # being orphaned on every stop/start (audit 2026-06-28, HIGH leak).
            persistence = getattr(self, "_persistence", None)
            if persistence is not None:
                try:
                    asyncio.run_coroutine_threadsafe(
                        persistence.dispose_all(), self._async_loop
                    ).result(timeout=5)
                except Exception:
                    _log.debug("Failed to dispose persistence engines on shutdown")
        # Shut down both event loops via AsyncRuntime's public method
        assert self._async_runtime is not None
        self._async_runtime.shutdown()
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

    def _fundamental_capture_cycle(self) -> None:
        """Daily: capture fundamentals for the MOEX universe -> idempotent persist (CAPTURE-01).

        Synchronous cycle (mirrors _fx_update_cycle): runs on the APScheduler thread
        executor. ``fetch_fundamentals`` is already synchronous and rate-limited, so it
        is called directly per symbol (no _run_grpc wrapper — RESEARCH Pitfall 6). One
        failing/empty symbol must NOT abort the run (per-symbol try/except, D-02). Each
        non-None snapshot is persisted through the fire-and-forget guard so a table-absent
        DB never crashes the loop (D-04). On completion the liveness markers the freshness
        check (plan 63-03) reads are set.
        """
        raw_fetcher = self._fetchers.get("moex")
        if raw_fetcher is None or self._persistence is None:
            _log.info(
                "fundamental_capture_skipped",
                reason="no moex fetcher or persistence",
            )
            return
        fetcher = cast("_SupportsFetchFundamentals", raw_fetcher)

        universe = self._registry.list_by_market("moex")
        universe_size = len(universe)
        captured = 0
        for instr in universe:
            try:
                snap = fetcher.fetch_fundamentals(instr.symbol)
            except Exception:  # degrade per-symbol (D-02): one bad symbol must not abort the run
                _log.warning(
                    "fundamental_capture_fetch_failed",
                    symbol=instr.symbol,
                    exc_info=True,
                )
                continue
            if snap is None:
                continue
            self._persistence.persist_fundamental_snapshot(snap)
            captured += 1

        self._last_fundamental_capture_at = self._now()
        self._last_fundamental_coverage_ratio = captured / max(1, universe_size)
        _log.info(
            "fundamental_capture_done",
            captured=captured,
            universe=universe_size,
            coverage_ratio=round(self._last_fundamental_coverage_ratio, 4),
        )

    def _fundamental_freshness_cycle(self) -> None:
        """Daily: alert when the fundamental-capture data clock has stopped (CAPTURE-03).

        D-03 refinement: staleness is *job-run liveness* — the age of the last
        successful capture RUN (``_last_fundamental_capture_at``), NOT the as_of-age of
        the newest snapshot. Point-in-time fundamentals legitimately stay constant for
        weeks, so as_of-age would false-alarm; a stopped scheduler is the real risk.

        Two independent alert conditions, either trips a single IMPORTANT alert:
          * Stale run: marker is None (never ran) OR run-age > fundamental_staleness_hours.
          * Low coverage: ratio is None OR < fundamental_coverage_floor — catches a
            gRPC-wide outage the per-symbol degrade (D-02) would silently swallow.

        A healthy run (recent AND coverage above floor) produces NO alert. The message
        carries only counts/ages/thresholds — never a token or DB URL (T-63-08).
        """
        if self._alerter is None:
            return

        _default_staleness_hours = 36
        _default_coverage_floor = 0.5

        def _num(name: str, default: float) -> float:
            # Read a real numeric setting; a MagicMock-settings test (no real value set)
            # yields an auto-attribute whose __float__ returns junk, so accept only a
            # genuine int/float (bool excluded) and otherwise fall back to the default.
            value = getattr(self._settings, name, default)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return default
            return float(value)

        staleness_hours = _num("fundamental_staleness_hours", _default_staleness_hours)
        coverage_floor = _num("fundamental_coverage_floor", _default_coverage_floor)

        last_run = self._last_fundamental_capture_at
        coverage = self._last_fundamental_coverage_ratio

        reasons: list[str] = []

        if last_run is None:
            reasons.append(f"no successful capture run yet (threshold {staleness_hours}h)")
        else:
            age_hours = (self._now() - last_run).total_seconds() / 3600.0
            if age_hours > staleness_hours:
                reasons.append(
                    f"last capture run was {age_hours:.1f}h ago (threshold {staleness_hours}h)"
                )

        if coverage is None:
            reasons.append(f"no coverage ratio recorded (floor {coverage_floor:.0%})")
        elif coverage < coverage_floor:
            reasons.append(f"coverage {coverage:.0%} below floor {coverage_floor:.0%}")

        if not reasons:
            return

        from finalayze.api.alerts import AlertPriority  # noqa: PLC0415

        message = "Fundamental capture freshness alert: " + "; ".join(reasons)
        self._alerter.send_alert(message, priority=AlertPriority.IMPORTANT)
        _log.warning("fundamental_freshness_alert", reasons=reasons)

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
        now_for_schedule = self._now()
        for market_id in self._broker_router.registered_markets:
            schedule = SCHEDULES.get(market_id)
            if schedule is None or schedule.is_market_open(now_for_schedule):
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
        # ALRT-01 D-07: monotonic cycle counter — bumped FIRST so any code
        # below (halt, breaker, snapshot) sees the new index. Mirror onto
        # PositionTracker so check_stop_losses can compute hold_bars without
        # an extension to its 3-param public signature (revision B3).
        self._cycle_count += 1
        if self._position_tracker is not None:
            self._position_tracker.set_current_cycle(self._cycle_count)
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

        # STOP-03: per-cycle snapshot of all active stop-loss states (D-04).
        # Wrapped in its own try/except so a DB failure here NEVER propagates
        # to _strategy_cycle's outer catch — that would inflate
        # _consecutive_equity_errors (see 54-RESEARCH.md §Open Questions #3).
        try:
            self._position_tracker.snapshot_all_stops_to_db(
                market_ids=self._build_symbol_to_market_map(),
                prices=dict(self._last_prices),
                now=now,
            )
        except Exception:
            # PERSIST-05: must NEVER affect _consecutive_equity_errors.
            _log.warning("stop_snapshot_write_failed", exc_info=True)

        # Phase 56 EQTY-01: per-cycle equity snapshot (D-01, D-02 Route B, D-03).
        # Wrapped in its own try/except so a DB failure here NEVER propagates
        # to _strategy_cycle's outer catch — that would inflate
        # _consecutive_equity_errors (mirrors STOP-03 pattern above).
        # Placement is intentional: AFTER the per-market loop and AFTER halt
        # early-returns at lines ~1257 and ~1264 — halted cycles do not snapshot
        # (matches D-01 "after each strategy cycle completes" — see 56-RESEARCH
        # Pitfall 6).
        try:
            self._daily_reporter.persist_cycle_snapshot(now)
        except Exception:
            # PERSIST-05: must NEVER affect _consecutive_equity_errors.
            _log.warning("equity_snapshot_persist_failed", exc_info=True)

    def _build_symbol_to_market_map(self) -> dict[str, str]:
        """Resolve each open-position symbol to its market_id via ``broker_router``.

        Runs once per strategy cycle (~5-15 min), not per fill — O(markets x
        positions) is acceptable here. If cycle latency becomes a concern, we
        can cache ``symbol -> market_id`` in ``PositionTracker`` at
        ``register_entry`` time (the relation is 1:1).
        """
        result: dict[str, str] = {}
        try:
            for mid in self._broker_router.registered_markets:
                broker = self._broker_router.route(mid)
                for sym in broker.get_positions():
                    result[sym] = mid
        except Exception:
            _log.debug("symbol_to_market_map_failed", exc_info=True)
        return result

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

        # Get cached portfolio once per market for all instruments
        portfolio = self._get_cached_portfolio(market_id)
        equity = market_equities.get(market_id, _ZERO)
        broker = self._broker_router.route(market_id)
        portfolio_obj = broker.get_portfolio()
        cash = Decimal(str(portfolio_obj.cash)) if portfolio_obj else _ZERO

        for instrument in instruments:
            # Process via SignalExecutor and aggregate cycle stats
            stats = self._signal_executor.process_instrument(
                instrument, market_id, level, fetcher, now, equity, cash, portfolio
            )
            self._cycle_signals_generated += stats.signals_generated
            self._cycle_orders_submitted += stats.orders_submitted
            self._cycle_orders_filled += stats.orders_filled
            self._cycle_errors_caught += stats.errors_caught
            self._cycle_dropped_no_bars += stats.dropped_no_bars
            self._cycle_dropped_below_threshold += stats.dropped_below_threshold
            self._cycle_dropped_pre_trade += stats.dropped_pre_trade

        # Update Prometheus metrics after processing all instruments
        market_equity = market_equities.get(market_id)
        if market_equity is not None and self._metrics:
            self._metrics.set_portfolio_equity(market_id, float(market_equity))
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

    def get_entry_strategies(self) -> dict[str, str]:
        """Return a snapshot of {symbol: strategy_name} for currently open positions."""
        result: dict[str, str] = self._signal_executor.get_entry_strategies()
        return result

    # ── Delegation shims for backward-compat with tests ──────────────────
    def _build_order(self, *args: object, **kwargs: object) -> object:
        """Delegate to SignalExecutor._build_order (test compat)."""
        return self._signal_executor._build_order(*args, **kwargs)

    def _submit_order(self, *args: object, **kwargs: object) -> object:
        """Delegate to SignalExecutor._submit_order (test compat)."""
        return self._signal_executor._submit_order(*args, **kwargs)

    # ---- Persistence delegation shims (test compat) ----

    def _persist_to_db(self, *args: object, **kwargs: object) -> None:
        """Delegate to TradingPersistence._persist_to_db (test compat)."""
        self._persistence._persist_to_db(*args, **kwargs)

    async def _persist_signal_async(self, *args: object, **kwargs: object) -> None:
        """Delegate to TradingPersistence._persist_signal_async (test compat)."""
        await self._persistence._persist_signal_async(*args, **kwargs)

    async def _persist_order_async(self, *args: object, **kwargs: object) -> None:
        """Delegate to TradingPersistence._persist_order_async (test compat)."""
        await self._persistence._persist_order_async(*args, **kwargs)

    async def _persist_news_article_async(self, article: object, impact_result: object) -> None:
        """Delegate to TradingPersistence._persist_news_article_async (test compat)."""
        await self._persistence._persist_news_article_async(article, impact_result)

    async def _persist_sentiment_batch_async(self, *args: object, **kwargs: object) -> None:
        """Delegate to TradingPersistence._persist_sentiment_batch_async (test compat)."""
        await self._persistence._persist_sentiment_batch_async(*args, **kwargs)

    # ---- Anomaly enrichment (test compat) ----

    async def _handle_anomaly_async(
        self,
        symbol: str,
        market_id: str,
        anomaly: object,
        raw_text: str,
    ) -> uuid.UUID | None:
        """Delegate to AnomalyHandler.handle (test compat)."""
        # Lazy-initialize handler for tests using object.__new__()
        if not hasattr(self, "_anomaly_handler"):
            self._anomaly_handler = AnomalyHandler(
                self._alerter,
                lambda: getattr(self, "_llm_client", None),
            )
        return await self._anomaly_handler.handle(symbol, market_id, anomaly, raw_text)

    async def _enrich_anomaly_async(
        self,
        symbol: str,
        market_id: str,
        anomaly: object,
        *,
        parent_id: uuid.UUID | None = None,
    ) -> None:
        """Delegate to AnomalyHandler.enrich (test compat)."""
        # Lazy-initialize handler for tests using object.__new__()
        if not hasattr(self, "_anomaly_handler"):
            self._anomaly_handler = AnomalyHandler(
                self._alerter,
                lambda: getattr(self, "_llm_client", None),
            )
        await self._anomaly_handler.enrich(symbol, market_id, anomaly, parent_id=parent_id)

    # ---- Process instrument delegation (test compat) ----

    def _process_instrument(self, *args: object, **kwargs: object) -> object:
        """Delegate to SignalExecutor.process_instrument (test compat).

        Provides default values for equity/cash/portfolio when not supplied,
        so callers using the old 5-arg signature still work.
        """
        # If called with old 5-arg signature (instrument, market_id, level, fetcher, now),
        # provide defaults for the new equity/cash/portfolio params
        _old_sig_max = 5
        if "equity" not in kwargs and len(args) <= _old_sig_max:
            # Try to get real portfolio from cache
            market_id = args[1] if len(args) > 1 else kwargs.get("market_id", "")
            portfolio = self._get_cached_portfolio(str(market_id))
            equity = getattr(portfolio, "equity", _ZERO) if portfolio else _ZERO
            cash = getattr(portfolio, "cash", _ZERO) if portfolio else _ZERO
            kwargs.setdefault("equity", equity)
            kwargs.setdefault("cash", cash)
            kwargs.setdefault("portfolio", portfolio)
        return self._signal_executor.process_instrument(*args, **kwargs)

    # ---- Portfolio review (re-added after decomposition) ----

    def _portfolio_review_cycle(self) -> None:
        """APScheduler callback -- dispatches async review without blocking."""
        if not hasattr(self, "_llm_client") or self._llm_client is None:
            _log.info("portfolio_review_skipped", reason="no LLM client configured")
            return
        if self._async_loop is None or self._async_loop.is_closed():
            return
        asyncio.run_coroutine_threadsafe(
            self._run_portfolio_review_async(),
            self._async_loop,
        )

    async def _run_portfolio_review_async(self) -> None:
        """Fire-and-forget async portfolio review -- never raises, never blocks.

        ALRT-04 (Phase 57 D-01): the LLM advisory result is merged with a
        deterministic ``compute_daily_recap`` payload so the single daily
        Telegram message carries today's realized P&L, positions opened/
        closed, and equity change vs previous close (Option A
        consolidation — no new message type added).
        """
        # ALRT-04 (D-01): require persistence to be wired; short-circuit
        # if not. _get_bg_session_factory lives on TradingPersistence, not
        # TradingLoop — tests/backtest paths that don't inject persistence
        # must noop cleanly here (the LLM advisory still depends on the
        # session factory for the recap merge).
        if self._persistence is None:
            _log.warning("portfolio_review_skipped_no_persistence")
            return
        try:
            from finalayze.analysis.portfolio_review_agent import (  # noqa: PLC0415
                PORTFOLIO_REVIEW_SYSTEM_PROMPT,
                REVIEW_LLM_TIMEOUT,
                PortfolioReviewResult,
                build_review_prompt,
                compute_daily_recap,
                format_review_telegram,
            )
            from finalayze.api.alerts import AlertPriority  # noqa: PLC0415

            portfolio_data = self._gather_portfolio_data()
            prompt = build_review_prompt(portfolio_data)
            assert self._llm_client is not None
            result = await asyncio.wait_for(
                self._llm_client.parse_structured(
                    prompt=prompt,
                    system=PORTFOLIO_REVIEW_SYSTEM_PROMPT,
                    response_model=PortfolioReviewResult,
                ),
                timeout=REVIEW_LLM_TIMEOUT,
            )
            # ALRT-04 (D-01): merge deterministic daily recap into advisory
            # LLM result. Failure to compute recap MUST NOT block the
            # advisory message — log and continue with the unmerged result.
            try:
                factory = self._persistence._get_bg_session_factory()
                async with factory() as session:
                    recap = await compute_daily_recap(
                        session,
                        datetime.now(tz=UTC),
                    )
                # model_copy with update keyword — Pydantic v2 idiom,
                # preserves frozen-model immutability.
                result = result.model_copy(update=recap)
            except Exception:
                _log.warning(
                    "portfolio_review_recap_merge_failed",
                    exc_info=True,
                )
            message = format_review_telegram(result)
            await self._alerter.send_async(
                message,
                alert_type="daily_summary",
                priority=AlertPriority.INFO,
            )
        except Exception:
            _log.warning("portfolio_review_llm_failure")

    def _gather_portfolio_data(self) -> dict[str, object]:
        """Collect portfolio state from all configured markets."""
        data: dict[str, object] = {}
        for market_id in self._circuit_breakers:
            try:
                broker = self._broker_router.route(market_id)
                portfolio = broker.get_portfolio()
                positions = broker.get_positions()
                data[market_id] = {
                    "equity": portfolio.equity,
                    "cash": portfolio.cash,
                    "positions": positions,
                }
            except Exception:
                _log.warning("portfolio_review_broker_error", market_id=market_id)
        return data

    # ---- ML retraining delegation (test compat) ----

    def _retrain_cycle(self) -> None:
        """Delegate to MLRetrainingService.retrain_all (test compat)."""
        self._ml_retraining.retrain_all()

    def _retrain_segment(self, *args: object, **kwargs: object) -> None:
        """Delegate to MLRetrainingService._retrain_segment (test compat)."""
        self._ml_retraining._retrain_segment(*args, **kwargs)

    # ---- News cycle delegation (test compat) ----

    def _news_cycle(self) -> None:
        """Delegate to NewsPipeline.run_news_cycle (test compat)."""
        self._news_pipeline.run_news_cycle()

    def _daily_reset(self) -> None:
        """Reset circuit breakers and send daily P&L summary.

        Delegates to DailyReportingService.daily_reset().
        """
        # Sync metrics reference and _now method (for test compatibility and consistency)
        self._daily_reporter._metrics = self._metrics
        self._daily_reporter._now = self._now
        updated_baselines = self._daily_reporter.daily_reset(self._baseline_equities)
        self._baseline_equities.update(updated_baselines)
        # S4.1: stock-side reconcile + equity-drift check (wires S3.2). Bonds
        # have their own reconcile via bond_processor.reconcile_with_broker.
        self._run_stock_reconcile()

    def _run_stock_reconcile(self) -> None:
        """S4.1: per-market stock reconcile + equity-drift via the daily hook.

        Iterates equity markets (skips bond-only markets — those reconcile
        through ``bond_processor.reconcile_with_broker``). For each market:

        1. ``reconcile_stocks`` flags broker positions the tracker doesn't
           know about + tracker entries the broker no longer reports.
           ``apply=False`` — the daily hook only *alerts*; an operator
           decides whether to clear stale state (a manual SELL might be
           intentional during a halt, for instance).
        2. ``reconcile_equity_drift`` compares broker-reported equity vs
           locally-computed MTM. Out-of-tolerance gaps imply coupon accrual,
           stale prices, or a real divergence — alerts via Telegram.

        Failures per market are caught + logged so a single broker outage
        doesn't suppress reconcile across the rest of the portfolio.
        """
        # Lazy import — keeps trading_loop import-time graph thin.
        from finalayze.orchestration.equity_reconcile import (  # noqa: PLC0415
            reconcile_equity_drift,
            reconcile_stocks,
        )

        last_prices: dict[str, Any] = getattr(self._signal_executor, "_last_prices", {})
        for market_id in self._circuit_breakers:
            try:
                broker = self._broker_router.route(market_id)
                portfolio = broker.get_portfolio()
            except Exception:
                _log.exception("stock_reconcile_broker_fetch_failed", market_id=market_id)
                continue

            try:
                reconcile_stocks(
                    broker,
                    self._position_tracker,
                    market_id=market_id,
                    registry=self._registry,
                    alerter=self._alerter,
                    apply=False,
                )
            except Exception:
                _log.exception("stock_reconcile_failed", market_id=market_id)

            try:
                reconcile_equity_drift(
                    portfolio,
                    last_prices,
                    alerter=self._alerter,
                    market_id=market_id,
                )
            except Exception:
                _log.exception("equity_drift_check_failed", market_id=market_id)

    def _compute_top_movers(self) -> list[tuple[str, float]]:
        """Compute top 3 movers by absolute P&L % across all markets.

        Delegates to DailyReportingService._compute_top_movers().
        """
        result: list[tuple[str, float]] = self._daily_reporter._compute_top_movers(
            self._baseline_equities
        )
        return result

    def _restore_stop_states_from_db(self) -> None:
        """Re-hydrate PositionTracker stop-loss state from the last DB snapshot.

        Called on startup after _reconcile_inflight_orders and before
        _preflight_check, so the first strategy cycle has ATR stops for all
        positions that existed before the restart.

        Positions that appear in the broker but have no DB snapshot are logged
        as warnings — they remain unprotected until the next BUY fill would
        re-establish a stop (or until a future orphan-recovery pass is added).
        """
        if self._persistence is None:
            return
        all_states = self._persistence.load_stop_snapshots()
        if not all_states:
            _log.info("stop_restore_no_snapshots_found")
            return

        # Reconcile against broker: only restore for symbols still open
        open_symbols: set[str] = set()
        for market_id in list(self._circuit_breakers.keys()):
            try:
                broker = self._broker_router.route(market_id)
                positions = broker.get_positions()
                open_symbols.update(positions.keys())
            except Exception:
                _log.warning("stop_restore_broker_fetch_failed", market=market_id)

        filtered = {sym: st for sym, st in all_states.items() if sym in open_symbols}
        orphaned = open_symbols - set(all_states.keys())

        if filtered:
            self._position_tracker.restore_stops(filtered)
        if orphaned:
            _log.warning(
                "stop_restore_orphaned_positions",
                symbols=sorted(orphaned),
                reason="no DB snapshot — no stop protection until next BUY",
            )

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
