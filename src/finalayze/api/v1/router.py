from __future__ import annotations

from fastapi import APIRouter

from finalayze.api.v1.ml import router as ml_router
from finalayze.api.v1.news import router as news_router
from finalayze.api.v1.portfolio import router as portfolio_router
from finalayze.api.v1.risk import router as risk_router
from finalayze.api.v1.signals import router as signals_router
from finalayze.api.v1.system import router as system_router
from finalayze.api.v1.trades import router as trades_router

api_router = APIRouter()
api_router.include_router(system_router)
api_router.include_router(portfolio_router)
api_router.include_router(trades_router)
api_router.include_router(signals_router)
api_router.include_router(risk_router)
api_router.include_router(ml_router)
api_router.include_router(news_router)
