"""FastAPI application entry point.

Layer 6 -- API / Dashboard layer.
"""

from __future__ import annotations

import asyncio
import threading
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

import structlog
from config.logging import setup_logging
from config.settings import get_settings
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from finalayze.api.v1.router import api_router
from finalayze.bootstrap import build_trading_loop
from finalayze.core.modes import WorkMode

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

_settings = get_settings()
setup_logging(_settings.mode)
log = structlog.get_logger()

# Module-level reference for graceful shutdown
_trading_loop_instance: Any | None = None
_trading_loop_thread: threading.Thread | None = None

# Module-level reference for Telegram bot handler (wired in lifespan)
_bot_handler_instance: Any | None = None


@asynccontextmanager
async def lifespan(_application: FastAPI) -> AsyncIterator[None]:  # noqa: PLR0912, PLR0915
    """Start TradingLoop in background thread for sandbox/real modes, shut down on exit."""
    global _trading_loop_instance, _trading_loop_thread  # noqa: PLW0603

    log.info("finalayze started", mode=_settings.mode.value)

    # Suppress benign gRPC BlockingIOError on uvicorn's event loop.
    # gRPC PollerCompletionQueue callbacks may leak here from health checks.
    _main_loop = asyncio.get_running_loop()

    def _grpc_exception_handler(
        loop: asyncio.AbstractEventLoop, context: dict[str, object]
    ) -> None:
        exc = context.get("exception")
        if isinstance(exc, BlockingIOError):
            return
        loop.default_exception_handler(context)

    _main_loop.set_exception_handler(_grpc_exception_handler)

    if _settings.mode in (WorkMode.SANDBOX, WorkMode.REAL):
        try:
            _trading_loop_instance = build_trading_loop(_settings)
            if _trading_loop_instance is not None:
                # Wire real health probes
                from finalayze.api.v1.system import (  # noqa: PLC0415
                    set_health_monitor,
                    set_kill_switch,
                    set_tinkoff_broker,
                )
                from finalayze.execution.tinkoff_broker import TinkoffBroker  # noqa: PLC0415
                from finalayze.monitoring.health_monitor import HealthMonitor  # noqa: PLC0415

                broker_router = getattr(_trading_loop_instance, "_broker_router", None)
                if broker_router is not None:
                    # Expose to portfolio/positions API endpoints
                    _application.state.broker_router = broker_router
                    # STOP-01: expose PositionTracker so /portfolio/positions can
                    # read stop-loss state off-lock via get_stop_state.
                    position_tracker = getattr(
                        _trading_loop_instance, "_position_tracker", None
                    )
                    if position_tracker is not None:
                        _application.state.position_tracker = position_tracker
                    try:
                        moex_broker = broker_router.route("moex")
                        if isinstance(moex_broker, TinkoffBroker):
                            set_tinkoff_broker(moex_broker)
                            log.info("tinkoff_broker_wired_to_health")
                    except Exception:
                        log.debug("tinkoff_broker_health_wire_skipped", exc_info=True)

                # Wire KillSwitch to REST API
                kill_switch = getattr(_trading_loop_instance, "_kill_switch", None)
                if kill_switch is not None:
                    set_kill_switch(kill_switch)
                    log.info("kill_switch_wired_to_rest_api")

                # Wire KillSwitch and GoNoGoReporter to Telegram bot handler
                if _bot_handler_instance is not None:
                    if kill_switch is not None:
                        _bot_handler_instance._kill_switch = kill_switch
                        log.info("kill_switch_wired_to_telegram_bot")

                    try:
                        from pathlib import Path as _Path  # noqa: PLC0415

                        from finalayze.monitoring.go_no_go import (  # noqa: PLC0415
                            GateThresholds,
                            GoNoGoReporter,
                        )

                        _gate_cfg = _Path("config/gate_thresholds.yaml")
                        if _gate_cfg.exists():
                            _thresholds = GateThresholds.from_yaml(_gate_cfg)
                            go_no_go_reporter = GoNoGoReporter(_thresholds, market_id="moex")
                            _bot_handler_instance._go_no_go_reporter = go_no_go_reporter
                            log.info("go_no_go_reporter_wired_to_telegram_bot")

                            from finalayze.api.v1.sandbox import (  # noqa: PLC0415
                                set_go_no_go_reporter as set_sandbox_reporter,
                            )

                            set_sandbox_reporter(go_no_go_reporter)
                            log.info("go_no_go_reporter_wired_to_sandbox_endpoint")
                    except Exception:
                        log.debug("go_no_go_reporter_wire_failed", exc_info=True)

                    if broker_router is not None:
                        _bot_handler_instance._broker_router = broker_router
                        circuit_breakers_ref = getattr(
                            _trading_loop_instance, "_circuit_breakers", {}
                        )
                        _bot_handler_instance._circuit_breakers = circuit_breakers_ref
                        _bot_handler_instance._trading_loop = _trading_loop_instance
                        log.info("bot_handler_fully_wired")

                # Wire GoNoGoReporter to sandbox endpoint even without Telegram bot
                if _bot_handler_instance is None:
                    try:
                        from pathlib import Path as _Path2  # noqa: PLC0415

                        from finalayze.api.v1.sandbox import (  # noqa: PLC0415
                            set_go_no_go_reporter as set_sandbox_reporter,
                        )
                        from finalayze.monitoring.go_no_go import (  # noqa: PLC0415
                            GateThresholds,
                            GoNoGoReporter,
                        )

                        _gate_cfg2 = _Path2("config/gate_thresholds.yaml")
                        if _gate_cfg2.exists():
                            _thresholds2 = GateThresholds.from_yaml(_gate_cfg2)
                            go_no_go_reporter = GoNoGoReporter(_thresholds2, market_id="moex")
                            set_sandbox_reporter(go_no_go_reporter)
                            log.info("go_no_go_reporter_wired_to_sandbox_endpoint")
                    except Exception:
                        log.debug("go_no_go_reporter_sandbox_wire_failed", exc_info=True)

                # Create and wire HealthMonitor
                alerter_ref = getattr(_trading_loop_instance, "_alerter_ref", None)
                if broker_router is not None and alerter_ref is not None:
                    health_monitor = HealthMonitor(
                        broker_router=broker_router,
                        trading_loop=_trading_loop_instance,
                        alerter=alerter_ref,
                        strategy_cycle_minutes=_settings.strategy_cycle_minutes,
                        # Feed freshness must exceed strategy cycle interval + buffer
                        # to avoid false stale alerts between cycles
                        feed_freshness_minutes=_settings.strategy_cycle_minutes + 15,
                    )
                    set_health_monitor(health_monitor)
                    # Wire health monitor into trading loop for feed timestamp updates
                    if _trading_loop_instance is not None:
                        _trading_loop_instance._health_monitor = health_monitor
                    health_monitor.start()
                    log.info("health_monitor_started")

                _trading_loop_thread = threading.Thread(
                    target=_trading_loop_instance.start,
                    daemon=True,
                    name="trading-loop",
                )
                _trading_loop_thread.start()
                log.info("trading_loop_started_in_background")
        except Exception:
            log.exception("trading_loop_build_failed")

    yield

    # Shutdown
    # Stop health monitor first
    if _trading_loop_instance is not None:
        _hm = getattr(_trading_loop_instance, "_health_monitor_ref", None)
        if _hm is None:
            # Check if we stored it via set_health_monitor
            try:
                from finalayze.api.v1.system import _health_monitor as _hm_ref  # noqa: PLC0415

                if _hm_ref is not None:
                    _hm_ref.stop()
                    log.info("health_monitor_stopped")
            except Exception:
                log.debug("health_monitor_stop_failed", exc_info=True)

        try:
            _trading_loop_instance.stop()
            log.info("trading_loop_stopped")
        except Exception:
            log.debug("trading_loop_stop_failed", exc_info=True)
    if _trading_loop_thread is not None and _trading_loop_thread.is_alive():
        _trading_loop_thread.join(timeout=10)

    # Close TelegramAlerter httpx clients to prevent resource leaks
    # Trading loop alerter (first instance)
    if _trading_loop_instance is not None:
        alerter_ref = getattr(_trading_loop_instance, "_alerter_ref", None)
        if alerter_ref is not None and hasattr(alerter_ref, "close"):
            try:
                await alerter_ref.close()
                log.info("trading_loop_alerter_closed")
            except Exception:
                log.debug("trading_loop_alerter_close_failed", exc_info=True)

    # Bot handler alerter (second instance)
    if _bot_handler_instance is not None:
        bot_alerter = getattr(_bot_handler_instance, "_alerter", None)
        if bot_alerter is not None and hasattr(bot_alerter, "close"):
            try:
                await bot_alerter.close()
                log.info("bot_handler_alerter_closed")
            except Exception:
                log.debug("bot_handler_alerter_close_failed", exc_info=True)


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
        from finalayze.api.alerts import TelegramAlerter  # noqa: PLC0415
        from finalayze.api.telegram_bot import TelegramBotHandler  # noqa: PLC0415
        from finalayze.api.v1.telegram import create_telegram_router  # noqa: PLC0415

        alerter = TelegramAlerter(settings.telegram_bot_token, settings.telegram_chat_id)
        global _bot_handler_instance  # noqa: PLW0603
        bot_handler = TelegramBotHandler(
            alerter=alerter,
            broker_router=None,  # type: ignore[arg-type]  # wired in TradingLoop startup
            circuit_breakers={},
            settings=settings,
        )
        _bot_handler_instance = bot_handler
        telegram_router = create_telegram_router(bot_handler, settings.telegram_webhook_secret)
        application.include_router(telegram_router)
        log.info("telegram_webhook_mounted", path="/api/telegram/webhook")

    # Prometheus HTTP metrics -- no auth (internal network only)
    Instrumentator().instrument(application).expose(
        application, endpoint="/metrics", include_in_schema=False
    )
    return application


app = create_app()
