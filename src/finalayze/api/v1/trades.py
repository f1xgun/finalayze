"""Trades endpoints (Layer 6)."""

from __future__ import annotations

import structlog
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict

from finalayze.api.v1.auth import api_key_auth

_log = structlog.get_logger()

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
    market: str | None = None,
    symbol: str | None = None,
    limit: int = 100,
) -> TradesResponse:
    """Trade history from orders table (filled orders)."""
    try:
        from sqlalchemy import func, select, text  # noqa: PLC0415

        from finalayze.core.db import get_async_session_factory  # noqa: PLC0415
        from finalayze.core.models import OrderModel  # noqa: PLC0415

        async with get_async_session_factory()() as session:
            stmt = (
                select(OrderModel)
                .where(OrderModel.status == "filled")
                .order_by(text("filled_at desc nulls last"))
                .limit(limit)
            )
            if market:
                stmt = stmt.where(OrderModel.market_id == market)
            if symbol:
                stmt = stmt.where(OrderModel.symbol == symbol)

            result = await session.execute(stmt)
            rows = result.scalars().all()

            # Total count
            count_stmt = (
                select(func.count()).select_from(OrderModel).where(OrderModel.status == "filled")
            )
            total = (await session.execute(count_stmt)).scalar() or 0

        trades = [
            TradeEntry(
                id=str(r.id),
                symbol=r.symbol,
                market_id=r.market_id,
                side=r.side,
                quantity=float(r.filled_quantity),
                fill_price=float(r.filled_avg_price) if r.filled_avg_price else None,
                slippage_bps=None,
                timestamp=(r.filled_at or r.submitted_at or "").isoformat()  # type: ignore[union-attr]
                if hasattr(r.filled_at or r.submitted_at, "isoformat")
                else "",
            )
            for r in rows
        ]
        return TradesResponse(trades=trades, total=int(total))
    except Exception as exc:
        _log.warning("trades_query_failed", error=str(exc))
        return TradesResponse(trades=[], total=0)


@router.get("/analytics", response_model=TradeAnalytics)
async def trade_analytics(
    market: str | None = None,
    period: int = 7,
) -> TradeAnalytics:
    """Trade analytics from orders table."""
    try:
        from datetime import UTC, datetime, timedelta  # noqa: PLC0415

        from sqlalchemy import func, select  # noqa: PLC0415

        from finalayze.core.db import get_async_session_factory  # noqa: PLC0415
        from finalayze.core.models import OrderModel  # noqa: PLC0415

        cutoff = datetime.now(UTC) - timedelta(days=period)
        async with get_async_session_factory()() as session:
            stmt = (
                select(func.count())
                .select_from(OrderModel)
                .where(
                    OrderModel.status == "filled",
                    OrderModel.submitted_at >= cutoff,
                )
            )
            if market:
                stmt = stmt.where(OrderModel.market_id == market)
            total = (await session.execute(stmt)).scalar() or 0

        return TradeAnalytics(
            period_days=period,
            total_trades=int(total),
            avg_slippage_bps=None,
            avg_fill_latency_ms=None,
            rejection_rate_pct=None,
        )
    except Exception as exc:
        _log.warning("trade_analytics_failed", error=str(exc))
        return TradeAnalytics(
            period_days=period,
            total_trades=0,
            avg_slippage_bps=None,
            avg_fill_latency_ms=None,
            rejection_rate_pct=None,
        )


@router.get("/{trade_id}", response_model=TradeEntry)
async def get_trade(trade_id: str) -> TradeEntry:
    """Single trade detail."""
    try:
        from sqlalchemy import select  # noqa: PLC0415

        from finalayze.core.db import get_async_session_factory  # noqa: PLC0415
        from finalayze.core.models import OrderModel  # noqa: PLC0415

        async with get_async_session_factory()() as session:
            import uuid  # noqa: PLC0415

            stmt = select(OrderModel).where(OrderModel.id == uuid.UUID(trade_id))
            result = await session.execute(stmt)
            r = result.scalar_one_or_none()

        if r is None:
            raise HTTPException(status_code=404, detail=f"Trade {trade_id!r} not found")

        return TradeEntry(
            id=str(r.id),
            symbol=r.symbol,
            market_id=r.market_id,
            side=r.side,
            quantity=float(r.filled_quantity),
            fill_price=float(r.filled_avg_price) if r.filled_avg_price else None,
            slippage_bps=None,
            timestamp=(r.filled_at or r.submitted_at or "").isoformat()  # type: ignore[union-attr]
            if hasattr(r.filled_at or r.submitted_at, "isoformat")
            else "",
        )
    except HTTPException:
        raise
    except Exception as exc:
        _log.warning("get_trade_failed", error=str(exc))
        raise HTTPException(status_code=500, detail="Internal error") from exc
