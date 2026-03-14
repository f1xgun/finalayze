"""FastAPI application entry point.

Layer 6 -- API / Dashboard layer.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import structlog
from config.logging import setup_logging
from config.settings import get_settings
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from finalayze.api.v1.router import api_router

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

_settings = get_settings()
setup_logging(_settings.mode)
log = structlog.get_logger()


@asynccontextmanager
async def lifespan(_application: FastAPI) -> AsyncIterator[None]:
    """Emit a startup message."""
    log.info("finalayze started", mode=_settings.mode.value)
    yield


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
        from finalayze.core.telegram_bot import TelegramBotHandler  # noqa: PLC0415

        from finalayze.core.alerts import TelegramAlerter  # noqa: PLC0415

        alerter = TelegramAlerter(settings.telegram_bot_token, settings.telegram_chat_id)
        bot_handler = TelegramBotHandler(
            alerter=alerter,
            broker_router=None,  # type: ignore[arg-type] — wired in TradingLoop startup
            circuit_breakers={},
            settings=settings,  # type: ignore[arg-type]
        )
        telegram_router = create_telegram_router(
            bot_handler, settings.telegram_webhook_secret
        )
        application.include_router(telegram_router)
        log.info("telegram_webhook_mounted", path="/api/telegram/webhook")

    # Prometheus HTTP metrics — no auth (internal network only)
    Instrumentator().instrument(application).expose(
        application, endpoint="/metrics", include_in_schema=False
    )
    return application


app = create_app()
