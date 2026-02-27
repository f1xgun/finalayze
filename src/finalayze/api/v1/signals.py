"""Signals and strategy performance endpoints (Layer 6)."""

from __future__ import annotations

from config.settings import Settings
from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict

from finalayze.api.v1.auth import require_api_key

_settings = Settings()
router = APIRouter(
    tags=["signals"],
    dependencies=[Depends(require_api_key(_settings.api_key))],
)


class SignalEntry(BaseModel):
    model_config = ConfigDict(frozen=True)
    id: str
    symbol: str
    market_id: str
    segment_id: str
    strategy: str
    direction: str
    confidence: float
    created_at: str


class SignalsResponse(BaseModel):
    model_config = ConfigDict(frozen=True)
    signals: list[SignalEntry]


class StrategyPerf(BaseModel):
    model_config = ConfigDict(frozen=True)
    strategy: str
    market_id: str
    win_rate: float | None
    profit_factor: float | None
    trades_today: int
    last_signal_at: str | None


class StrategiesResponse(BaseModel):
    model_config = ConfigDict(frozen=True)
    strategies: list[StrategyPerf]


@router.get("/signals", response_model=SignalsResponse)
async def list_signals(
    market: str | None = None,  # noqa: ARG001
    segment: str | None = None,  # noqa: ARG001
    limit: int = 50,  # noqa: ARG001
) -> SignalsResponse:
    return SignalsResponse(signals=[])


@router.get("/strategies/performance", response_model=StrategiesResponse)
async def strategies_performance() -> StrategiesResponse:
    return StrategiesResponse(strategies=[])
