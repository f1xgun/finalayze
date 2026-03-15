"""FastAPI application entry point.

Layer 6 -- API / Dashboard layer.
"""

from __future__ import annotations

import threading
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import structlog
from config.logging import setup_logging
from config.settings import get_settings
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from finalayze.api.v1.router import api_router
from finalayze.core.modes import WorkMode

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

_settings = get_settings()
setup_logging(_settings.mode)
log = structlog.get_logger()

# Module-level reference for graceful shutdown
_trading_loop_instance: object | None = None
_trading_loop_thread: threading.Thread | None = None


@asynccontextmanager
async def lifespan(_application: FastAPI) -> AsyncIterator[None]:
    """Start TradingLoop in background thread for sandbox/real modes, shut down on exit."""
    global _trading_loop_instance, _trading_loop_thread  # noqa: PLW0603

    log.info("finalayze started", mode=_settings.mode.value)

    if _settings.mode in (WorkMode.SANDBOX, WorkMode.REAL):
        try:
            _trading_loop_instance = _build_trading_loop(_settings)
            if _trading_loop_instance is not None:
                # Wire real health probes
                from finalayze.api.v1.system import set_tinkoff_broker  # noqa: PLC0415
                from finalayze.execution.tinkoff_broker import TinkoffBroker  # noqa: PLC0415

                broker_router = getattr(_trading_loop_instance, "_broker_router", None)
                if broker_router is not None:
                    try:
                        moex_broker = broker_router.route("moex")
                        if isinstance(moex_broker, TinkoffBroker):
                            set_tinkoff_broker(moex_broker)
                            log.info("tinkoff_broker_wired_to_health")
                    except Exception:
                        log.debug("tinkoff_broker_health_wire_skipped", exc_info=True)

                _trading_loop_thread = threading.Thread(
                    target=_trading_loop_instance.start,  # type: ignore[union-attr]
                    daemon=True,
                    name="trading-loop",
                )
                _trading_loop_thread.start()
                log.info("trading_loop_started_in_background")
        except Exception:
            log.exception("trading_loop_build_failed")

    yield

    # Shutdown
    if _trading_loop_instance is not None:
        try:
            _trading_loop_instance.stop()  # type: ignore[union-attr]
            log.info("trading_loop_stopped")
        except Exception:
            log.debug("trading_loop_stop_failed", exc_info=True)
    if _trading_loop_thread is not None and _trading_loop_thread.is_alive():
        _trading_loop_thread.join(timeout=10)


def _build_trading_loop(settings: object) -> object | None:
    """Build TradingLoop with all dependencies. Returns None on failure.

    Lazy imports to maintain dependency layering (Layer 6).
    """
    try:
        from finalayze.analysis.event_classifier import EventClassifier  # noqa: PLC0415
        from finalayze.analysis.impact_estimator import ImpactEstimator  # noqa: PLC0415
        from finalayze.analysis.news_analyzer import NewsAnalyzer  # noqa: PLC0415
        from finalayze.core.alerts import TelegramAlerter  # noqa: PLC0415
        from finalayze.core.trading_loop import TradingLoop  # noqa: PLC0415
        from finalayze.data.fetchers.newsapi import NewsApiFetcher  # noqa: PLC0415
        from finalayze.execution.broker_router import BrokerRouter  # noqa: PLC0415
        from finalayze.markets.instruments import InstrumentRegistry  # noqa: PLC0415
        from finalayze.risk.circuit_breaker import (  # noqa: PLC0415
            CircuitBreaker,
            CrossMarketCircuitBreaker,
        )
        from finalayze.analysis.llm_client import AnthropicClient  # noqa: PLC0415
        from finalayze.strategies.combiner import StrategyCombiner  # noqa: PLC0415

        # Build minimal dependencies -- actual wiring depends on available services
        alerter = TelegramAlerter(
            getattr(settings, "telegram_bot_token", "") or "",
            getattr(settings, "telegram_chat_id", "") or "",
        )
        registry = InstrumentRegistry()
        broker_router = BrokerRouter(brokers={})
        circuit_breakers: dict[str, CircuitBreaker] = {}
        cross_market_breaker = CrossMarketCircuitBreaker()
        combiner = StrategyCombiner(strategies={})
        news_fetcher = NewsApiFetcher(api_key=getattr(settings, "newsapi_api_key", "") or "")
        llm_client = AnthropicClient(
            api_key=getattr(settings, "anthropic_api_key", "") or "",
            model=getattr(settings, "llm_model", "claude-sonnet-4-20250514"),
        )
        news_analyzer = NewsAnalyzer(llm_client=llm_client)
        event_classifier = EventClassifier(llm_client=llm_client)
        impact_estimator = ImpactEstimator()

        loop = TradingLoop(
            settings=settings,  # type: ignore[arg-type]
            fetchers={},
            news_fetcher=news_fetcher,
            news_analyzer=news_analyzer,
            event_classifier=event_classifier,
            impact_estimator=impact_estimator,
            strategy=combiner,
            broker_router=broker_router,
            circuit_breakers=circuit_breakers,
            cross_market_breaker=cross_market_breaker,
            alerter=alerter,
            instrument_registry=registry,
        )
        log.info("trading_loop_built")
        return loop
    except Exception:
        log.exception("trading_loop_build_failed")
        return None


def create_app() -> FastAPI:
    """Construct and configure the FastAPI application."""
    application = FastAPI(title="Finalayze", version="0.1.0", lifespan=lifespan)
    settings = get_settings()
    allowed_origins = settings.cors_origins or []
    application.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "Authorization", "X-API-Key"],
    )
    application.include_router(api_router, prefix="/api/v1")

    # Mount Telegram webhook router when bot token and webhook secret are configured
    if settings.telegram_bot_token and settings.telegram_webhook_secret:
        from finalayze.api.v1.telegram import create_telegram_router  # noqa: PLC0415
        from finalayze.core.alerts import TelegramAlerter  # noqa: PLC0415
        from finalayze.core.telegram_bot import TelegramBotHandler  # noqa: PLC0415

        alerter = TelegramAlerter(settings.telegram_bot_token, settings.telegram_chat_id)
        bot_handler = TelegramBotHandler(
            alerter=alerter,
            broker_router=None,  # type: ignore[arg-type] -- wired in TradingLoop startup
            circuit_breakers={},
            settings=settings,  # type: ignore[arg-type]
        )
        telegram_router = create_telegram_router(bot_handler, settings.telegram_webhook_secret)
        application.include_router(telegram_router)
        log.info("telegram_webhook_mounted", path="/api/telegram/webhook")

    # Prometheus HTTP metrics -- no auth (internal network only)
    Instrumentator().instrument(application).expose(
        application, endpoint="/metrics", include_in_schema=False
    )
    return application


app = create_app()
