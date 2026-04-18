"""Signals and strategy performance endpoints (Layer 6)."""

from __future__ import annotations

import structlog
from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict

from finalayze.api.v1.auth import api_key_auth

_log = structlog.get_logger()

router = APIRouter(
    tags=["signals"],
    dependencies=[Depends(api_key_auth)],
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
    market: str | None = None,
    segment: str | None = None,
    limit: int = 50,
) -> SignalsResponse:
    """List recent trading signals from the database."""
    try:
        from sqlalchemy import select, text  # noqa: PLC0415

        from finalayze.core.db import get_async_session_factory  # noqa: PLC0415
        from finalayze.core.models import SignalModel  # noqa: PLC0415

        async with get_async_session_factory()() as session:
            stmt = select(SignalModel).order_by(text("created_at desc")).limit(limit)
            if market:
                stmt = stmt.where(SignalModel.market_id == market)
            if segment:
                stmt = stmt.where(SignalModel.segment_id == segment)
            result = await session.execute(stmt)
            rows = result.scalars().all()

        signals = [
            SignalEntry(
                id=str(r.id),
                symbol=r.symbol,
                market_id=r.market_id,
                segment_id=r.segment_id,
                strategy=r.strategy_name,
                direction=r.direction,
                confidence=float(r.confidence),
                created_at=r.created_at.isoformat(),
            )
            for r in rows
        ]
        return SignalsResponse(signals=signals)
    except Exception as exc:
        _log.warning("signals_query_failed", error=str(exc))
        return SignalsResponse(signals=[])


@router.get("/strategies/performance", response_model=StrategiesResponse)
async def strategies_performance() -> StrategiesResponse:
    """Strategy performance from recent signals. Returns empty if no data."""
    try:
        from sqlalchemy import func, select  # noqa: PLC0415

        from finalayze.core.db import get_async_session_factory  # noqa: PLC0415
        from finalayze.core.models import SignalModel  # noqa: PLC0415

        async with get_async_session_factory()() as session:
            stmt = select(
                SignalModel.strategy_name,
                SignalModel.market_id,
                func.count().label("signal_count"),
                func.max(SignalModel.created_at).label("last_signal"),
            ).group_by(SignalModel.strategy_name, SignalModel.market_id)
            result = await session.execute(stmt)
            rows = result.all()

        strategies = [
            StrategyPerf(
                strategy=r.strategy_name,
                market_id=r.market_id,
                win_rate=None,
                profit_factor=None,
                trades_today=int(r.signal_count),
                last_signal_at=r.last_signal.isoformat() if r.last_signal else None,
            )
            for r in rows
        ]
        return StrategiesResponse(strategies=strategies)
    except Exception as exc:
        _log.warning("strategies_performance_failed", error=str(exc))
        return StrategiesResponse(strategies=[])
