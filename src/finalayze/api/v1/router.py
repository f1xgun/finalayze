from __future__ import annotations

from fastapi import APIRouter

from finalayze.api.v1.alerts import router as alerts_router
from finalayze.api.v1.debates import router as debates_router
from finalayze.api.v1.experiments import router as experiments_router
from finalayze.api.v1.meta_agent import router as meta_agent_router
from finalayze.api.v1.ml import router as ml_router
from finalayze.api.v1.news import router as news_router
from finalayze.api.v1.portfolio import router as portfolio_router
from finalayze.api.v1.risk import router as risk_router
from finalayze.api.v1.saa import router as saa_router
from finalayze.api.v1.sandbox import router as sandbox_router
from finalayze.api.v1.signals import router as signals_router
from finalayze.api.v1.system import router as system_router
from finalayze.api.v1.trades import router as trades_router

api_router = APIRouter()
api_router.include_router(system_router)
api_router.include_router(portfolio_router)
api_router.include_router(trades_router)
api_router.include_router(signals_router)
api_router.include_router(risk_router)
api_router.include_router(saa_router)
api_router.include_router(ml_router)
api_router.include_router(news_router)
api_router.include_router(sandbox_router)
api_router.include_router(debates_router)
api_router.include_router(experiments_router)
api_router.include_router(alerts_router)
api_router.include_router(meta_agent_router)

# Telegram webhook router is mounted in main.py create_app() when
# telegram_bot_token and telegram_webhook_secret are configured.
# See: finalayze.api.v1.telegram.create_telegram_router
