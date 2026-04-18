"""Portfolio endpoints (Layer 6)."""

from __future__ import annotations

import asyncio
from decimal import Decimal
from typing import Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from finalayze.api.v1.auth import api_key_auth
from finalayze.markets.instruments import build_default_registry
from finalayze.markets.registry import default_registry

# Symbol → segment_id mapping (populated lazily from config)
_symbol_to_segment: dict[str, str] = {}


def _get_segment_for_symbol(symbol: str) -> str:
    """Resolve symbol to segment_id using config/segments.py."""
    if not _symbol_to_segment:
        try:
            from config.segments import DEFAULT_SEGMENTS  # noqa: PLC0415

            for seg in DEFAULT_SEGMENTS:
                for sym in seg.symbols:
                    _symbol_to_segment[sym] = seg.segment_id
        except Exception:  # noqa: S110
            pass  # segments config unavailable
    return _symbol_to_segment.get(symbol, "")


def _build_stop_fields(
    symbol: str,
    current_price: float,
    position_tracker: Any | None,
) -> dict[str, Any]:
    """Build the 8 D-02 stop fields for a position (STOP-01).

    Returns all-null dict when no tracker or no active stop. Uses Decimal
    arithmetic for activation_threshold to match PositionTracker formula
    (54-RESEARCH Pitfall 6).

    Return type is ``dict[str, Any]`` because PositionDetail's stop fields
    are a heterogenous mix of ``float | None`` and ``bool | None``; a narrower
    union confuses mypy when splatting with ``**`` into the constructor.
    """
    empty: dict[str, Any] = {
        "stop_price": None,
        "distance_pct": None,
        "distance_atr": None,
        "atr_value": None,
        "entry_price": None,
        "highest_price": None,
        "trail_activated": None,
        "activation_threshold": None,
    }
    if position_tracker is None:
        return empty
    state = position_tracker.get_stop_state(symbol)
    if state is None:
        return empty
    current_dec = Decimal(str(current_price))
    dist_pct: float | None
    dist_atr: float | None
    if current_dec > 0 and state.atr_value > 0:
        dist_pct = float((current_dec - state.current_stop) / current_dec)
        dist_atr = float((current_dec - state.current_stop) / state.atr_value)
    else:
        dist_pct = None
        dist_atr = None
    # Decimal arithmetic to match PositionTracker (Pitfall 6)
    activation_threshold_dec = state.entry_price + state.activation_atr * state.atr_value
    return {
        "stop_price": float(state.current_stop),
        "distance_pct": dist_pct,
        "distance_atr": dist_atr,
        "atr_value": float(state.atr_value),
        "entry_price": float(state.entry_price),
        "highest_price": float(state.highest_price),
        "trail_activated": bool(state.trail_activated),
        "activation_threshold": float(activation_threshold_dec),
    }


_log = structlog.get_logger()

router = APIRouter(
    prefix="/portfolio",
    tags=["portfolio"],
    dependencies=[Depends(api_key_auth)],
)


class MarketPortfolio(BaseModel):
    model_config = ConfigDict(frozen=True)
    market_id: str
    equity_usd: float
    cash_usd: float
    positions_value_usd: float
    daily_pnl_usd: float
    daily_pnl_pct: float


class PortfolioResponse(BaseModel):
    model_config = ConfigDict(frozen=True)
    total_equity_usd: float
    total_cash_usd: float
    daily_pnl_usd: float
    daily_pnl_pct: float
    markets: list[MarketPortfolio]


class PositionDetail(BaseModel):
    model_config = ConfigDict(frozen=True)
    symbol: str
    market_id: str
    segment_id: str
    quantity: float
    avg_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    # --- STOP-01 D-02 stop-loss state (all nullable; null when no active stop, D-03) ---
    stop_price: float | None = Field(
        default=None,
        description="Current trailing stop-loss price. Null if no active stop.",
    )
    distance_pct: float | None = Field(
        default=None,
        description=(
            "Distance from current price to stop, as fraction of current price: "
            "(current_price - stop_price) / current_price. Positive when price is "
            "above stop (normal long position). Null if no active stop."
        ),
    )
    distance_atr: float | None = Field(
        default=None,
        description=(
            "ATR-normalized distance: (current_price - stop_price) / atr_value. "
            "Used by the risk heatmap (D-10): green > 1.5, yellow 0.5-1.5, red < 0.5. "
            "Null if no active stop."
        ),
    )
    atr_value: float | None = Field(
        default=None,
        description=(
            "ATR at entry time, cached in PositionTracker (constant for position lifetime)."
        ),
    )
    entry_price: float | None = Field(
        default=None,
        description="Fill price when position was opened.",
    )
    highest_price: float | None = Field(
        default=None,
        description="High-water mark since entry, used for trailing activation.",
    )
    trail_activated: bool | None = Field(
        default=None,
        description="True once highest_price reached entry + activation_atr * atr_value.",
    )
    activation_threshold: float | None = Field(
        default=None,
        description=(
            "Price level that activates trailing: entry_price + activation_atr * atr_value. "
            "Once highest_price crosses this, trail_activated becomes True."
        ),
    )


class PositionsResponse(BaseModel):
    model_config = ConfigDict(frozen=True)
    positions: list[PositionDetail]


class StopEventEntry(BaseModel):
    model_config = ConfigDict(frozen=True)
    timestamp: str
    event_type: str
    current_stop: float | None
    entry_price: float | None
    highest_price: float | None
    current_price: float | None
    atr_value: float | None
    trail_activated: bool | None


class StopHistoryResponse(BaseModel):
    model_config = ConfigDict(frozen=True)
    symbol: str
    events: list[StopEventEntry]


class SnapshotEntry(BaseModel):
    model_config = ConfigDict(frozen=True)
    timestamp: str
    market_id: str
    equity: float
    drawdown_pct: float


class HistoryResponse(BaseModel):
    model_config = ConfigDict(frozen=True)
    snapshots: list[SnapshotEntry]


class PerformanceResponse(BaseModel):
    model_config = ConfigDict(frozen=True)
    sharpe_30d: float | None
    sortino_30d: float | None
    max_drawdown_pct: float | None
    win_rate: float | None
    profit_factor: float | None
    avg_win_loss_ratio: float | None


def _empty_portfolio() -> PortfolioResponse:
    return PortfolioResponse(
        total_equity_usd=0.0,
        total_cash_usd=0.0,
        daily_pnl_usd=0.0,
        daily_pnl_pct=0.0,
        markets=[],
    )


@router.get("", response_model=PortfolioResponse)
async def get_portfolio(request: Request) -> PortfolioResponse:
    """Unified portfolio across all markets in base currency (USD)."""
    broker_router = getattr(request.app.state, "broker_router", None)
    if broker_router is None:
        return _empty_portfolio()

    registry = default_registry()
    markets: list[MarketPortfolio] = []
    registered = broker_router.registered_markets
    for market_def in registry.list_markets():
        market_id = market_def.id
        if market_id not in registered:
            continue
        try:
            broker = broker_router.route(market_id)
            loop = asyncio.get_running_loop()
            p = await loop.run_in_executor(None, broker.get_portfolio)
            equity = float(p.equity)
            cash = float(p.cash)
            markets.append(
                MarketPortfolio(
                    market_id=market_id,
                    equity_usd=equity,
                    cash_usd=cash,
                    positions_value_usd=equity - cash,
                    daily_pnl_usd=0.0,
                    daily_pnl_pct=0.0,
                )
            )
        except Exception as exc:
            _log.warning("Failed to fetch portfolio for market %s: %s", market_id, exc)

    total = sum(m.equity_usd for m in markets)
    return PortfolioResponse(
        total_equity_usd=total,
        total_cash_usd=sum(m.cash_usd for m in markets),
        daily_pnl_usd=0.0,
        daily_pnl_pct=0.0,
        markets=markets,
    )


@router.get("/positions", response_model=PositionsResponse)
async def get_positions(request: Request) -> PositionsResponse:
    """All open positions with unrealized P&L and trailing stop-loss state (STOP-01)."""
    broker_router = getattr(request.app.state, "broker_router", None)
    if broker_router is None:
        return PositionsResponse(positions=[])

    # STOP-01: PositionTracker provides the stop-loss state (None in TEST/DEBUG modes)
    position_tracker = getattr(request.app.state, "position_tracker", None)

    market_registry = default_registry()
    instrument_registry = build_default_registry()
    positions: list[PositionDetail] = []
    registered = broker_router.registered_markets
    for market_def in market_registry.list_markets():
        market_id = market_def.id
        if market_id not in registered:
            continue
        try:
            broker = broker_router.route(market_id)
            # Use enriched positions if available (TinkoffBroker)
            detail_fn = getattr(broker, "get_positions_detail", None)
            if detail_fn is not None:
                for p in detail_fn():
                    figi = p.get("figi", "")
                    display_symbol = str(figi)
                    try:
                        inst = instrument_registry.get_by_figi(figi)
                        display_symbol = inst.symbol
                    except Exception:  # noqa: S110
                        pass
                    stop_fields = _build_stop_fields(
                        display_symbol,
                        float(p.get("current_price", 0)),
                        position_tracker,
                    )
                    positions.append(
                        PositionDetail(
                            symbol=display_symbol,
                            market_id=market_id,
                            segment_id=_get_segment_for_symbol(display_symbol),
                            quantity=float(p.get("quantity", 0)),
                            avg_price=float(p.get("avg_price", 0)),
                            current_price=float(p.get("current_price", 0)),
                            market_value=float(p.get("market_value", 0)),
                            unrealized_pnl=float(p.get("unrealized_pnl", 0)),
                            unrealized_pnl_pct=float(p.get("unrealized_pnl_pct", 0)),
                            **stop_fields,
                        )
                    )
            else:
                # Fallback for brokers without get_positions_detail
                raw = broker.get_positions()
                for key, qty in raw.items():
                    if qty > Decimal(0):
                        display_symbol = key
                        try:
                            inst = instrument_registry.get_by_figi(key)
                            display_symbol = inst.symbol
                        except Exception:  # noqa: S110
                            pass
                        stop_fields = _build_stop_fields(display_symbol, 0.0, position_tracker)
                        positions.append(
                            PositionDetail(
                                symbol=display_symbol,
                                market_id=market_id,
                                segment_id=_get_segment_for_symbol(display_symbol),
                                quantity=float(qty),
                                avg_price=0.0,
                                current_price=0.0,
                                market_value=0.0,
                                unrealized_pnl=0.0,
                                unrealized_pnl_pct=0.0,
                                **stop_fields,
                            )
                        )
        except Exception as exc:
            _log.warning("Failed to fetch positions for market %s: %s", market_id, exc)

    return PositionsResponse(positions=positions)


@router.get(
    "/positions/{symbol}/stop-history",
    response_model=StopHistoryResponse,
)
async def get_stop_history(
    symbol: str,
    days: int = 30,
) -> StopHistoryResponse:
    """Return stop-loss state history from stop_loss_events for a symbol (last N days).

    Reads from the async session factory bound to the uvicorn event loop
    (NOT the trading-loop background factory -- Pitfall 2).
    Empty list if no history exists yet (not a 404).
    """
    from datetime import UTC, datetime, timedelta  # noqa: PLC0415

    from sqlalchemy import select  # noqa: PLC0415

    from finalayze.core.db import get_async_session_factory  # noqa: PLC0415
    from finalayze.core.models import StopLossEventModel  # noqa: PLC0415

    cutoff = datetime.now(UTC) - timedelta(days=days)
    try:
        factory = get_async_session_factory()
        async with factory() as session:
            stmt = (
                select(StopLossEventModel)
                .where(
                    StopLossEventModel.symbol == symbol,
                    StopLossEventModel.timestamp >= cutoff,
                )
                .order_by(StopLossEventModel.timestamp.asc())
            )
            result = await session.execute(stmt)
            rows = result.scalars().all()
    except Exception as exc:
        # DB unavailable (test mode or infra issue) -> empty list, not 500.
        # Mirrors get_portfolio_history:397-399 pattern.
        _log.warning("stop_history_failed", symbol=symbol, error=str(exc))
        return StopHistoryResponse(symbol=symbol, events=[])
    events = [
        StopEventEntry(
            timestamp=r.timestamp.isoformat(),
            event_type=r.event_type,
            current_stop=float(r.current_stop) if r.current_stop is not None else None,
            entry_price=float(r.entry_price) if r.entry_price is not None else None,
            highest_price=float(r.highest_price) if r.highest_price is not None else None,
            current_price=float(r.current_price) if r.current_price is not None else None,
            atr_value=float(r.atr_value) if r.atr_value is not None else None,
            trail_activated=r.trail_activated,
        )
        for r in rows
    ]
    return StopHistoryResponse(symbol=symbol, events=events)


@router.get("/positions/{symbol}", response_model=PositionDetail)
async def get_position(symbol: str, request: Request) -> PositionDetail:
    """Return detail for a single open position. Returns 404 if not found."""
    broker_router: Any = getattr(request.app.state, "broker_router", None)
    if broker_router is None:
        raise HTTPException(status_code=404, detail=f"Position {symbol!r} not found")
    # TODO: wire to real broker_router
    raise HTTPException(status_code=404, detail=f"Position {symbol!r} not found")


@router.get("/history", response_model=HistoryResponse)
async def get_portfolio_history() -> HistoryResponse:
    """Equity curve from sandbox_metrics table (last 30 days)."""
    try:
        from datetime import UTC, datetime, timedelta  # noqa: PLC0415

        from sqlalchemy import select, text  # noqa: PLC0415

        from finalayze.core.db import get_async_session_factory  # noqa: PLC0415
        from finalayze.core.models import SandboxMetricRow  # noqa: PLC0415

        cutoff = datetime.now(UTC) - timedelta(days=30)
        async with get_async_session_factory()() as session:
            stmt = (
                select(SandboxMetricRow)
                .where(SandboxMetricRow.timestamp >= cutoff)
                .order_by(text("timestamp asc"))
            )
            result = await session.execute(stmt)
            rows = result.scalars().all()

        snapshots = [
            SnapshotEntry(
                timestamp=r.timestamp.isoformat(),
                market_id=r.market_id,
                equity=float(r.equity_rub),
                drawdown_pct=float(r.drawdown_pct) if r.drawdown_pct is not None else 0.0,
            )
            for r in rows
        ]
        return HistoryResponse(snapshots=snapshots)
    except Exception as exc:
        _log.warning("portfolio_history_failed", error=str(exc))
        return HistoryResponse(snapshots=[])


@router.get("/performance", response_model=PerformanceResponse)
async def get_performance() -> PerformanceResponse:
    """Rolling 30-day performance metrics. Stub."""
    return PerformanceResponse(
        sharpe_30d=None,
        sortino_30d=None,
        max_drawdown_pct=None,
        win_rate=None,
        profit_factor=None,
        avg_win_loss_ratio=None,
    )
