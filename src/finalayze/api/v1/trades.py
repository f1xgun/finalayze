"""Trades endpoints (Layer 6)."""

from __future__ import annotations

from decimal import Decimal

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
    """Trade analytics for the last ``period_days`` (D-13 default 30).

    Phase 55 is realized-only per CONTEXT.md 2026-04-18 amendment — D-02
    (unrealized P&L) deferred to Phase 56. All win/PF metrics are computed
    from FIFO-paired closed trades via ``_fifo.fifo_pair``.

    Sign convention (D-08): ``avg_slippage_bps`` positive = adverse for trader.
    Null-fallback (D-07): a field is ``None`` when it cannot be computed
    (e.g. ``profit_factor`` when ``gross_loss == 0`` per Pitfall 1).
    """

    model_config = ConfigDict(frozen=True)
    period_days: int
    total_trades: int
    win_rate: float | None
    avg_win: float | None
    avg_loss: float | None
    profit_factor: float | None
    avg_slippage_bps: float | None
    avg_fill_latency_ms: float | None
    rejection_rate_pct: float | None


def _slippage_for(row: object) -> float | None:
    """Compute slippage_bps for a row using its eagerly-loaded signal.

    Returns None when fill_price is missing, the signal is not joined, or the
    signal has no signal_price (D-07). Keeps the list/detail/analytics paths
    in sync — all three call this helper.

    ``row`` is typed as ``object`` to support both SQLAlchemy ``OrderModel``
    and test-side ``SimpleNamespace`` fixtures; fields are duck-typed.
    """
    from finalayze.api.v1._slippage import compute_slippage_bps  # noqa: PLC0415

    fill = getattr(row, "filled_avg_price", None)
    signal = getattr(row, "signal", None)
    side = getattr(row, "side", None)
    if fill is None or signal is None or side is None:
        return None
    signal_price = getattr(signal, "signal_price", None)
    return compute_slippage_bps(Decimal(fill), signal_price, side)


@router.get("", response_model=TradesResponse)
async def list_trades(
    market: str | None = None,
    symbol: str | None = None,
    limit: int = 100,
) -> TradesResponse:
    """Trade history from orders table (filled orders).

    Eagerly loads ``OrderModel.signal`` and populates ``slippage_bps`` per
    row via ``_slippage.compute_slippage_bps`` (TRAD-01 read path).
    """
    try:
        from sqlalchemy import func, select, text  # noqa: PLC0415
        from sqlalchemy.orm import selectinload  # noqa: PLC0415

        from finalayze.core.db import get_async_session_factory  # noqa: PLC0415
        from finalayze.core.models import OrderModel  # noqa: PLC0415

        async with get_async_session_factory()() as session:
            stmt = (
                select(OrderModel)
                .options(selectinload(OrderModel.signal))
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
                slippage_bps=_slippage_for(r),
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
async def trade_analytics(  # noqa: PLR0915  # end-to-end FIFO+analytics handler; splitting obscures flow
    market: str | None = None,
    period: int = 30,
) -> TradeAnalytics:
    """Trade analytics: realized P&L stats from FIFO-paired filled orders.

    Phase 55 is realized-only — D-02 (unrealized P&L) deferred to Phase 56
    per CONTEXT.md amendment 2026-04-18.

    Win condition (D-03): ``pnl > commission_cost + slippage_cost`` where the
    commission source is ``Settings.default_commission_bps_{us,moex}``
    (RESEARCH.md Open Q1) and slippage cost is
    ``Settings.default_slippage_cost_bps``. Market routing uses the CLOSING
    order's ``market_id`` so a pair is credited to the market that closed it.

    Default period is 30 days (D-13). Period boundary uses a date-truncated
    UTC-midnight cutoff to avoid the "period=7 missed 6h of day 1" off-by-one
    (Pitfall 4).

    ``avg_slippage_bps`` averages over filled rows with a resolvable
    ``signal.signal_price`` only (D-07). ``profit_factor`` is ``None`` when
    ``gross_loss == 0`` (Pitfall 1).
    """
    try:
        from datetime import UTC, datetime, timedelta  # noqa: PLC0415

        from config.settings import Settings  # noqa: PLC0415
        from sqlalchemy import select  # noqa: PLC0415
        from sqlalchemy.orm import selectinload  # noqa: PLC0415

        from finalayze.api.v1._fifo import fifo_pair  # noqa: PLC0415
        from finalayze.core.db import get_async_session_factory  # noqa: PLC0415
        from finalayze.core.models import OrderModel  # noqa: PLC0415

        settings = Settings()
        cutoff = (datetime.now(UTC) - timedelta(days=period)).replace(
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
        )

        async with get_async_session_factory()() as session:
            stmt = (
                select(OrderModel)
                .options(selectinload(OrderModel.signal))
                .where(
                    OrderModel.status == "filled",
                    OrderModel.filled_at.isnot(None),
                    OrderModel.filled_at >= cutoff,
                )
                .order_by(OrderModel.symbol.asc(), OrderModel.filled_at.asc())
            )
            if market:
                stmt = stmt.where(OrderModel.market_id == market)
            rows = list((await session.execute(stmt)).scalars().all())

        _bps = Decimal(10000)
        wins = 0
        losses = 0
        gross_win = Decimal(0)
        gross_loss = Decimal(0)
        total_win_amount = Decimal(0)
        total_loss_amount = Decimal(0)

        # Route commission bps using the closing order's market_id. We look up
        # the (symbol, exit_ts) tuple since fifo_pair does not carry market_id
        # on PairedTrade (the helper is market-agnostic).
        market_id_by_close_ts: dict[tuple[str, object], str] = {
            (r.symbol, r.filled_at): r.market_id for r in rows if r.filled_at is not None
        }

        for pair in fifo_pair(rows):
            m = market_id_by_close_ts.get((pair.symbol, pair.exit_ts), "us")
            is_moex = str(m).lower() == "moex" or str(m).lower().startswith("ru")
            commission_bps = Decimal(
                str(
                    settings.default_commission_bps_moex
                    if is_moex
                    else settings.default_commission_bps_us,
                ),
            )
            slippage_cost_bps = Decimal(str(settings.default_slippage_cost_bps))
            avg_price = (pair.entry_price + pair.exit_price) / Decimal(2)
            notional = avg_price * pair.quantity
            cost_threshold = (notional * commission_bps / _bps) + (
                notional * slippage_cost_bps / _bps
            )
            pnl = (pair.exit_price - pair.entry_price) * pair.quantity

            if pnl > cost_threshold:
                wins += 1
                gross_win += pnl
                total_win_amount += pnl
            else:
                losses += 1
                if pnl < 0:
                    gross_loss += abs(pnl)
                total_loss_amount += pnl

        total_trades = wins + losses
        win_rate = Decimal(wins) / Decimal(total_trades) if total_trades > 0 else None
        avg_win = total_win_amount / Decimal(wins) if wins > 0 else None
        avg_loss = total_loss_amount / Decimal(losses) if losses > 0 else None
        profit_factor = gross_win / gross_loss if gross_loss > 0 else None

        # avg_slippage: mean across filled rows in the window with non-null signal_price
        slippage_values: list[float] = []
        for r in rows:
            s = _slippage_for(r)
            if s is not None:
                slippage_values.append(s)
        avg_slippage = sum(slippage_values) / len(slippage_values) if slippage_values else None

        return TradeAnalytics(
            period_days=period,
            total_trades=total_trades,
            win_rate=float(win_rate) if win_rate is not None else None,
            avg_win=float(avg_win) if avg_win is not None else None,
            avg_loss=float(avg_loss) if avg_loss is not None else None,
            profit_factor=float(profit_factor) if profit_factor is not None else None,
            avg_slippage_bps=avg_slippage,
            avg_fill_latency_ms=None,
            rejection_rate_pct=None,
        )
    except Exception as exc:
        _log.warning("trade_analytics_failed", error=str(exc))
        return TradeAnalytics(
            period_days=period,
            total_trades=0,
            win_rate=None,
            avg_win=None,
            avg_loss=None,
            profit_factor=None,
            avg_slippage_bps=None,
            avg_fill_latency_ms=None,
            rejection_rate_pct=None,
        )


@router.get("/{trade_id}", response_model=TradeEntry)
async def get_trade(trade_id: str) -> TradeEntry:
    """Single trade detail.

    Eagerly loads ``OrderModel.signal`` so ``slippage_bps`` is populated
    consistently with ``list_trades`` (TRAD-01 read path).
    """
    try:
        import uuid  # noqa: PLC0415

        from sqlalchemy import select  # noqa: PLC0415
        from sqlalchemy.orm import selectinload  # noqa: PLC0415

        from finalayze.core.db import get_async_session_factory  # noqa: PLC0415
        from finalayze.core.models import OrderModel  # noqa: PLC0415

        async with get_async_session_factory()() as session:
            stmt = (
                select(OrderModel)
                .options(selectinload(OrderModel.signal))
                .where(OrderModel.id == uuid.UUID(trade_id))
            )
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
            slippage_bps=_slippage_for(r),
            timestamp=(r.filled_at or r.submitted_at or "").isoformat()  # type: ignore[union-attr]
            if hasattr(r.filled_at or r.submitted_at, "isoformat")
            else "",
        )
    except HTTPException:
        raise
    except Exception as exc:
        _log.warning("get_trade_failed", error=str(exc))
        raise HTTPException(status_code=500, detail="Internal error") from exc
