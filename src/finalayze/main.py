"""FastAPI application entry point.

Layer 6 -- API / Dashboard layer.
"""

from __future__ import annotations

import os
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
    allowed_origins = os.getenv("FINALAYZE_CORS_ORIGINS", "*").split(",")
    application.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "Authorization", "X-API-Key"],
    )
    application.include_router(api_router, prefix="/api/v1")
    # Prometheus HTTP metrics — no auth (internal network only)
    Instrumentator().instrument(application).expose(
        application, endpoint="/metrics", include_in_schema=False
    )
    return application


app = create_app()
