"""API v1 router: aggregates all sub-routers for version 1 of the API.

Layer 6 -- API layer.
"""

from __future__ import annotations

from fastapi import APIRouter

from finalayze.api.v1.portfolio import router as portfolio_router
from finalayze.api.v1.system import router as system_router

api_router = APIRouter()
api_router.include_router(system_router)
api_router.include_router(portfolio_router)
