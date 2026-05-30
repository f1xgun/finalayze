"""Factory for constructing TradingLoop service collaborators in dependency order.

Extracted from TradingLoop.__init__ (increment #2 of god-object decomposition).
Encapsulates the construction of 6 service instances with their cross-wire dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from finalayze.orchestration.daily_reporting import DailyReportingService
from finalayze.orchestration.db_persistence import TradingPersistence
from finalayze.orchestration.ml_retraining import MLRetrainingService
from finalayze.orchestration.news_pipeline import NewsPipeline
from finalayze.orchestration.position_manager import PositionTracker
from finalayze.orchestration.signal_executor import SignalExecutor

if TYPE_CHECKING:
    from collections.abc import Callable
    from datetime import datetime

    # Import TradingLoopDeps ONLY for type annotations (avoid runtime circular import)
    from finalayze.orchestration.trading_loop import TradingLoopDeps


@dataclass(frozen=True)
class LoopServices:
    """Immutable bundle of the 6 service collaborators constructed together.

    All services share the same TradingPersistence instance and are wired
    with cross-dependencies in order of construction.
    """

    persistence: TradingPersistence
    position_tracker: PositionTracker
    signal_executor: SignalExecutor
    ml_retrainer: MLRetrainingService
    daily_reporter: DailyReportingService
    news_pipeline: NewsPipeline


def build_loop_services(
    deps: TradingLoopDeps,
    *,
    sentiment_mgr: Any,
    kelly_sizer: Any,
    pre_trade_checker: Any,
    loss_limit_tracker: Any,
    now_fn: Callable[[], datetime],
    run_async_fn: Callable[..., Any],
) -> LoopServices:
    """Construct the 6 service collaborators in strict dependency order.

    Order is critical (see STOP-03 comment in original code):
    1. TradingPersistence (no deps)
    2. PositionTracker (depends on persistence)
    3. SignalExecutor (depends on persistence, position_tracker, sentiment_mgr)
    4. MLRetrainingService (uses sentiment_mgr.collect_active_segments)
    5. DailyReportingService (depends on persistence)
    6. NewsPipeline (depends on persistence, sentiment_mgr, async_loop_fn)

    Args:
        deps: TradingLoopDeps unpacked bundle of all collaborators
        sentiment_mgr: SentimentManager instance (already constructed in TradingLoop)
        kelly_sizer: RollingKelly instance (already constructed in TradingLoop)
        pre_trade_checker: PreTradeChecker instance (already constructed in TradingLoop)
        loss_limit_tracker: LossLimitTracker instance (already constructed in TradingLoop)
        now_fn: Callable returning current UTC datetime (typically self._now)
        run_async_fn: Callable to run async coroutines (typically self._run_async)

    Returns:
        LoopServices with all 6 services ready for use.
    """
    # ── 1. TradingPersistence (must be first; no dependencies) ──
    db_url = getattr(deps.settings, "database_url", None)
    persistence = TradingPersistence(db_url, None, deps.settings)

    # ── 2. PositionTracker (depends on persistence) ──
    position_tracker = PositionTracker(
        kelly_sizer=kelly_sizer,
        broker_router=deps.broker_router,
        alerter=deps.alerter,
        persistence=persistence,
    )

    # ── 3. SignalExecutor (depends on persistence, position_tracker, sentiment_mgr) ──
    signal_executor = SignalExecutor(
        strategy=deps.strategy,
        broker_router=deps.broker_router,
        position_tracker=position_tracker,
        sentiment_mgr=sentiment_mgr,
        persistence=persistence,
        pre_trade_checker=pre_trade_checker,
        loss_limit_tracker=loss_limit_tracker,
        macro_cache=deps.macro_cache,
        health_monitor=deps.health_monitor,
        sandbox_monitor=deps.sandbox_monitor,
        metrics=deps.metrics_collector,
        alerter=deps.alerter,
        registry=deps.instrument_registry,
        ml_registry=deps.ml_registry,
        settings=deps.settings,
    )

    # ── 4. MLRetrainingService (uses sentiment_mgr.collect_active_segments) ──
    ml_retrainer = MLRetrainingService(
        fetchers=deps.fetchers,
        registry=deps.instrument_registry,
        ml_registry=deps.ml_registry,
        settings=deps.settings,
        alerter=deps.alerter,
        collect_segments_fn=sentiment_mgr.collect_active_segments,
        now_fn=now_fn,
    )

    # ── 5. DailyReportingService (depends on persistence) ──
    daily_reporter = DailyReportingService(
        broker_router=deps.broker_router,
        circuit_breakers=deps.circuit_breakers,
        cross_market_breaker=deps.cross_market_breaker,
        loss_limit_tracker=loss_limit_tracker,
        alerter=deps.alerter,
        persistence=persistence,
        bond_processor=deps.bond_cycle_processor,
        fx_service=deps.fx_service,
        metrics_collector=deps.metrics_collector,
        settings=deps.settings,
        now_fn=now_fn,
    )

    # ── 6. NewsPipeline (depends on persistence, sentiment_mgr, async_loop_fn) ──
    news_pipeline = NewsPipeline(
        rss_fetcher=deps.rss_fetcher,
        telegram_reader=deps.telegram_reader,
        news_fetcher=deps.news_fetcher,
        news_impact_analyzer=deps.news_impact_analyzer,
        sector_ticker_mapper=deps.sector_ticker_mapper,
        sentiment_mgr=sentiment_mgr,
        persistence=persistence,
        registry=deps.instrument_registry,
        cache=deps.cache,
        settings=deps.settings,
        alerter=deps.alerter,
        async_loop_fn=run_async_fn,
    )

    return LoopServices(
        persistence=persistence,
        position_tracker=position_tracker,
        signal_executor=signal_executor,
        ml_retrainer=ml_retrainer,
        daily_reporter=daily_reporter,
        news_pipeline=news_pipeline,
    )
