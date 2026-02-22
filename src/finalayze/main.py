"""FastAPI application entry point.

Layer 6 -- API / Dashboard layer.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from finalayze.api.v1.router import api_router


def create_app() -> FastAPI:
    """Construct and configure the FastAPI application."""
    application = FastAPI(title="Finalayze", version="0.1.0")
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    application.include_router(api_router, prefix="/api/v1")
    return application


app = create_app()
