"""Portfolio endpoints (Layer 6)."""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict

from finalayze.api.v1.auth import api_key_auth
from finalayze.markets.registry import default_registry

_log = logging.getLogger(__name__)

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
    market_value_usd: float
    unrealized_pnl_usd: float
    unrealized_pnl_pct: float
    stop_distance_atr: float | None


class PositionsResponse(BaseModel):
    model_config = ConfigDict(frozen=True)
    positions: list[PositionDetail]


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
    for market_def in registry.list_markets():
        market_id = market_def.id
        try:
            broker = broker_router.route(market_id)
            p = broker.get_portfolio()
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
    """All open positions with unrealized P&L."""
    broker_router = getattr(request.app.state, "broker_router", None)
    if broker_router is None:
        return PositionsResponse(positions=[])

    registry = default_registry()
    positions: list[PositionDetail] = []
    for market_def in registry.list_markets():
        market_id = market_def.id
        try:
            broker = broker_router.route(market_id)
            raw = broker.get_positions()
            for symbol, qty in raw.items():
                if qty > Decimal(0):
                    positions.append(
                        PositionDetail(
                            symbol=symbol,
                            market_id=market_id,
                            segment_id="",
                            quantity=float(qty),
                            market_value_usd=0.0,
                            unrealized_pnl_usd=0.0,
                            unrealized_pnl_pct=0.0,
                            stop_distance_atr=None,
                        )
                    )
        except Exception as exc:
            _log.warning("Failed to fetch positions for market %s: %s", market_id, exc)

    return PositionsResponse(positions=positions)


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
    """Equity curve from portfolio_snapshots table (last 30 days). Stub."""
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
