from __future__ import annotations

from fastapi import APIRouter

from finalayze.api.v1.portfolio import router as portfolio_router
from finalayze.api.v1.system import router as system_router
from finalayze.api.v1.trades import router as trades_router

api_router = APIRouter()
api_router.include_router(system_router)
api_router.include_router(portfolio_router)
api_router.include_router(trades_router)
