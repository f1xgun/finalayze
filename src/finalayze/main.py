"""FastAPI application entry point.

Layer 6 -- API / Dashboard layer.
"""

from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from finalayze.api.v1.router import api_router


def create_app() -> FastAPI:
    """Construct and configure the FastAPI application."""
    application = FastAPI(title="Finalayze", version="0.1.0")
    allowed_origins = os.getenv("FINALAYZE_CORS_ORIGINS", "*").split(",")
    application.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "Authorization"],
    )
    application.include_router(api_router, prefix="/api/v1")
    return application


app = create_app()
