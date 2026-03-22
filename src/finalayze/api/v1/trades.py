"""Trades endpoints (Layer 6)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict

from finalayze.api.v1.auth import api_key_auth

router = APIRouter(
    prefix="/trades",
    tags=["trades"],
    dependencies=[Depends(api_key_auth)],
)


class TradeEntry(BaseModel):
    model_config = ConfigDict(frozen=True)
    id: str
    symbol: str
    market_id: str
    side: str
    quantity: float
    fill_price: float | None
    slippage_bps: float | None
    timestamp: str


class TradesResponse(BaseModel):
    model_config = ConfigDict(frozen=True)
    trades: list[TradeEntry]
    total: int


class TradeAnalytics(BaseModel):
    model_config = ConfigDict(frozen=True)
    period_days: int
    total_trades: int
    avg_slippage_bps: float | None
    avg_fill_latency_ms: float | None
    rejection_rate_pct: float | None


@router.get("", response_model=TradesResponse)
async def list_trades(
    market: str | None = None,  # noqa: ARG001
    symbol: str | None = None,  # noqa: ARG001
    limit: int = 100,  # noqa: ARG001
) -> TradesResponse:
    """Trade history. Not yet implemented."""
    raise HTTPException(status_code=501, detail="Not yet implemented")


@router.get("/analytics", response_model=TradeAnalytics)
async def trade_analytics(
    market: str | None = None,  # noqa: ARG001
    period: int = 7,  # noqa: ARG001
) -> TradeAnalytics:
    """Slippage and fill latency stats. Not yet implemented."""
    raise HTTPException(status_code=501, detail="Not yet implemented")


@router.get("/{trade_id}", response_model=TradeEntry)
async def get_trade(trade_id: str) -> TradeEntry:  # noqa: ARG001
    """Single trade detail. Not yet implemented."""
    raise HTTPException(status_code=501, detail="Not yet implemented")
