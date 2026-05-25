"""Portfolio endpoints (Layer 6)."""

from __future__ import annotations

import asyncio
from datetime import datetime
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
                if not seg.enabled:
                    continue
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
    sharpe_30d: float | None = Field(
        default=None,
        description=(
            "Annualised Sharpe ratio over the requested window (default 30d). "
            "Multi-currency portfolios mix RUB and USD equities directly; "
            "FX-adjusted variant is out of scope (D-11). Null when "
            "n_snapshots < 3."
        ),
    )
    sortino_30d: float | None = Field(
        default=None,
        description=(
            "Annualised Sortino ratio over the requested window. "
            "PerformanceAnalyzer returns 0 on negative-mean returns — "
            "that is a meaningful 'losing portfolio' value, not 'no data'. "
            "Null only when n_snapshots < 3 (D-12 + Open Q4)."
        ),
    )
    max_drawdown_pct: float | None = Field(
        default=None,
        description=(
            "Portfolio-aggregate max drawdown over the window. "
            "Distinct from per-market drawdown_pct shown on /history "
            "(see Pitfall 4). Null when n_snapshots < 2."
        ),
    )
    win_rate: float | None = Field(
        default=None,
        description=(
            "FIFO-paired win rate (wins / total paired trades). Reuses "
            "api/v1/_fifo.fifo_pair — single source of truth shared with "
            "/trades/analytics (D-10). Null when n_paired_trades == 0."
        ),
    )
    profit_factor: float | None = Field(
        default=None,
        description=(
            "Gross profit / gross loss across FIFO-paired trades. "
            "Null when n_paired_trades == 0 or gross_loss == 0."
        ),
    )
    avg_win_loss_ratio: float | None = Field(
        default=None,
        description=(
            "Average winning trade P&L / average losing trade P&L. "
            "Null when there are no losses or no wins."
        ),
    )
    n_snapshots: int = Field(
        default=0,
        description=(
            "Count of daily_equity_snapshots in the window. Drives null "
            "logic for sharpe_30d / sortino_30d / max_drawdown_pct "
            "(Pitfall 5 / Open Q3 — separate from n_paired_trades)."
        ),
    )
    n_paired_trades: int = Field(
        default=0,
        description=(
            "Count of FIFO-paired round-trip trades in the window. "
            "Drives null logic for win_rate / profit_factor / "
            "avg_win_loss_ratio (Pitfall 5 / Open Q3 — separate from "
            "n_snapshots)."
        ),
    )
    current_drawdown_pct: float | None = Field(
        default=None,
        description=(
            "Current drawdown from the period high-water mark: "
            "(peak_equity - latest_equity) / peak_equity. "
            "Unlike max_drawdown_pct (worst historical point), this resets "
            "when the portfolio recovers above its previous peak. "
            "Null when n_snapshots < 2."
        ),
    )


def _empty_portfolio() -> PortfolioResponse:
    return PortfolioResponse(
        total_equity_usd=0.0,
        total_cash_usd=0.0,
        daily_pnl_usd=0.0,
        daily_pnl_pct=0.0,
        markets=[],
    )


# EQTY-02 D-05 hybrid fallback threshold. If daily_equity_snapshots returns
# fewer than this many rows in the requested window, the handler falls back
# to sandbox_metrics so day-1 of the EQTY-01 writer rollout never produces
# an empty chart.
_MIN_ROWS_FOR_PRIMARY = 5


def _build_history_with_drawdown(
    rows: list[tuple[datetime, str, Decimal]],
) -> list[SnapshotEntry]:
    """Per-market running-peak drawdown (EQTY-02 D-07).

    Walks rows in timestamp order, tracks per-market peak, and emits
    ``(peak - equity) / peak`` per row. Per-market scope is intentional:
    each market's drawdown is independent (market A's higher equity does
    NOT influence market B's peak).

    This is distinct from ``PerformanceResponse.max_drawdown_pct`` (Plan
    56-04), which aggregates across markets via summed equity. See
    Pitfall 4 in 56-RESEARCH.md.
    """
    peaks: dict[str, Decimal] = {}
    out: list[SnapshotEntry] = []
    for ts, market_id, equity in rows:
        prev_peak = peaks.get(market_id, equity)
        peak = max(prev_peak, equity)
        peaks[market_id] = peak
        dd = float((peak - equity) / peak) if peak > 0 else 0.0
        out.append(
            SnapshotEntry(
                timestamp=ts.isoformat(),
                market_id=market_id,
                equity=float(equity),
                drawdown_pct=dd,
            )
        )
    return out


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
async def get_portfolio_history(
    days: int = 30,
    market_id: str | None = None,
) -> HistoryResponse:
    """Equity curve from daily_equity_snapshots with sandbox_metrics fallback.

    EQTY-02 D-05: primary source is `daily_equity_snapshots` (populated by
    Plan 56-02's per-cycle writer). When fewer than ``_MIN_ROWS_FOR_PRIMARY``
    rows fall inside the requested window, the handler falls back to
    `sandbox_metrics` so operators never see an empty chart on day-1 of
    the writer rollout.

    Query params (D-06):
    * ``days`` — window length in days (default 30).
    * ``market_id`` — restrict to a single market (omitted = all markets).

    ``drawdown_pct`` is computed server-side per market via running peak
    (D-07); see ``_build_history_with_drawdown``. Per-market scope is
    distinct from ``PerformanceResponse.max_drawdown_pct`` (Plan 56-04),
    which aggregates across markets.

    Source choice is logged via structlog under event
    ``portfolio_history_served`` with keys ``history_source`` and
    ``row_count`` (D-08; no ``?source=`` override).
    """
    from datetime import UTC, timedelta  # noqa: PLC0415

    from sqlalchemy import select, text  # noqa: PLC0415

    from finalayze.core.db import get_async_session_factory  # noqa: PLC0415
    from finalayze.core.models import (  # noqa: PLC0415
        DailyEquitySnapshot,
        SandboxMetricRow,
    )

    cutoff = datetime.now(UTC) - timedelta(days=days)

    try:
        async with get_async_session_factory()() as session:
            # PRIMARY: daily_equity_snapshots
            stmt = (
                select(DailyEquitySnapshot)
                .where(DailyEquitySnapshot.timestamp >= cutoff)
                .order_by(text("timestamp asc"))
            )
            if market_id:
                stmt = stmt.where(DailyEquitySnapshot.market_id == market_id)
            primary_rows = (await session.execute(stmt)).scalars().all()

            if len(primary_rows) >= _MIN_ROWS_FOR_PRIMARY:
                source = "daily_equity_snapshots"
                snapshots = _build_history_with_drawdown(
                    [(r.timestamp, r.market_id, Decimal(r.equity)) for r in primary_rows]
                )
            else:
                # FALLBACK: sandbox_metrics (preserves drawdown_pct from the
                # writer-side column when present; older rows may be null).
                source = "sandbox_metrics"
                fallback_stmt = (
                    select(SandboxMetricRow)
                    .where(SandboxMetricRow.timestamp >= cutoff)
                    .order_by(text("timestamp asc"))
                )
                if market_id:
                    fallback_stmt = fallback_stmt.where(SandboxMetricRow.market_id == market_id)
                fallback_rows = (await session.execute(fallback_stmt)).scalars().all()
                snapshots = [
                    SnapshotEntry(
                        timestamp=r.timestamp.isoformat(),
                        market_id=r.market_id,
                        equity=float(r.equity_rub),
                        drawdown_pct=float(r.drawdown_pct) if r.drawdown_pct is not None else 0.0,
                    )
                    for r in fallback_rows
                ]

        _log.info(
            "portfolio_history_served",
            history_source=source,
            row_count=len(snapshots),
            days=days,
            market_id=market_id,
        )
        return HistoryResponse(snapshots=snapshots)
    except Exception as exc:
        _log.warning("portfolio_history_failed", error=str(exc))
        return HistoryResponse(snapshots=[])


@router.get("/performance", response_model=PerformanceResponse)
async def get_performance(days: int = 30) -> PerformanceResponse:
    """Rolling N-day Sharpe/Sortino/MaxDD + FIFO-derived win/PF metrics (PERF-01).

    Reuses ``backtest.performance.PerformanceAnalyzer`` for the snapshot-derived
    metrics (D-09 — Layer 6 importing Layer 4 is permitted) and the shared
    ``api/v1/_fifo.fifo_pair`` helper for win_rate / profit_factor /
    avg_win_loss_ratio (D-10 — single source of truth with /trades/analytics).

    Per-metric null gating is on COUNT (n_snapshots / n_paired_trades), NOT
    on metric value (D-12 + Open Q4): a Sortino of 0 from negative-mean
    returns is a meaningful 'losing portfolio' signal, distinct from
    'no data'.
    """
    from datetime import UTC, datetime, timedelta  # noqa: PLC0415

    from sqlalchemy import select, text  # noqa: PLC0415

    from finalayze.api.v1._fifo import fifo_pair  # noqa: PLC0415
    from finalayze.api.v1._perf import equity_snapshots_to_portfolio_states  # noqa: PLC0415
    from finalayze.backtest.performance import PerformanceAnalyzer  # noqa: PLC0415
    from finalayze.core.db import get_async_session_factory  # noqa: PLC0415
    from finalayze.core.models import DailyEquitySnapshot, OrderModel  # noqa: PLC0415

    cutoff = datetime.now(UTC) - timedelta(days=days)
    # PerformanceAnalyzer requires ≥3 snapshots for Sharpe/Sortino
    # (backtest/performance.py:178); ≥2 for MaxDD (line 141).
    _MIN_SNAPSHOTS_FOR_SHARPE = 3  # noqa: N806
    _MIN_SNAPSHOTS_FOR_DD = 2  # noqa: N806

    try:
        async with get_async_session_factory()() as session:
            equity_rows = (
                (
                    await session.execute(
                        select(DailyEquitySnapshot)
                        .where(DailyEquitySnapshot.timestamp >= cutoff)
                        .order_by(text("timestamp asc"))
                    )
                )
                .scalars()
                .all()
            )

            order_rows = (
                (
                    await session.execute(
                        select(OrderModel)
                        .where(OrderModel.status == "filled")
                        .where(OrderModel.filled_at >= cutoff)
                        .order_by(text("symbol asc, filled_at asc"))
                    )
                )
                .scalars()
                .all()
            )

        portfolio_snapshots = equity_snapshots_to_portfolio_states(equity_rows)
        n_snapshots = len(portfolio_snapshots)

        # D-09: reuse PerformanceAnalyzer (Layer 4) — Layer 6 import permitted
        sharpe_dec, _, _, _ = PerformanceAnalyzer._compute_sharpe_with_significance(
            portfolio_snapshots
        )
        sortino_dec = PerformanceAnalyzer().sortino_ratio(portfolio_snapshots)
        max_dd_dec = PerformanceAnalyzer._compute_max_drawdown(portfolio_snapshots)

        # D-10: FIFO-derived win_rate / profit_factor / avg_win_loss_ratio
        # via the shared api/v1/_fifo.fifo_pair helper (single source of truth
        # with /trades/analytics).
        paired = list(fifo_pair(order_rows))
        n_paired = len(paired)
        wins = [p for p in paired if (p.exit_price - p.entry_price) * p.quantity > 0]
        losses = [p for p in paired if (p.exit_price - p.entry_price) * p.quantity < 0]
        gross_profit = sum(
            ((p.exit_price - p.entry_price) * p.quantity for p in wins),
            Decimal(0),
        )
        gross_loss = -sum(
            ((p.exit_price - p.entry_price) * p.quantity for p in losses),
            Decimal(0),
        )
        win_rate = (Decimal(len(wins)) / Decimal(n_paired)) if n_paired > 0 else None
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else None
        avg_win = (gross_profit / Decimal(len(wins))) if wins else None
        avg_loss = (gross_loss / Decimal(len(losses))) if losses else None
        avg_win_loss_ratio = (
            (avg_win / avg_loss)
            if (avg_win is not None and avg_loss is not None and avg_loss > 0)
            else None
        )

        # Current drawdown: running peak → latest equity (resets on recovery).
        current_dd: float | None = None
        if n_snapshots >= _MIN_SNAPSHOTS_FOR_DD:
            running_peak = float(portfolio_snapshots[0].equity)
            latest_dd = 0.0
            for ps in portfolio_snapshots:
                eq = float(ps.equity)
                running_peak = max(running_peak, eq)
                latest_dd = (running_peak - eq) / running_peak if running_peak > 0 else 0.0
            current_dd = latest_dd

        # D-12 (per Open Q3 + Q4): null logic gates on COUNT, not metric value.
        # PerformanceAnalyzer.sortino_ratio returns Decimal(0) on negative-mean
        # returns — that is a meaningful "0" (negative-portfolio period), NOT
        # "no data". Honor it.
        return PerformanceResponse(
            sharpe_30d=(float(sharpe_dec) if n_snapshots >= _MIN_SNAPSHOTS_FOR_SHARPE else None),
            sortino_30d=(float(sortino_dec) if n_snapshots >= _MIN_SNAPSHOTS_FOR_SHARPE else None),
            max_drawdown_pct=(float(max_dd_dec) if n_snapshots >= _MIN_SNAPSHOTS_FOR_DD else None),
            current_drawdown_pct=current_dd,
            win_rate=float(win_rate) if win_rate is not None else None,
            profit_factor=float(profit_factor) if profit_factor is not None else None,
            avg_win_loss_ratio=(
                float(avg_win_loss_ratio) if avg_win_loss_ratio is not None else None
            ),
            n_snapshots=n_snapshots,
            n_paired_trades=n_paired,
        )
    except Exception as exc:
        _log.warning("performance_failed", error=str(exc))
        return PerformanceResponse(
            sharpe_30d=None,
            sortino_30d=None,
            max_drawdown_pct=None,
            current_drawdown_pct=None,
            win_rate=None,
            profit_factor=None,
            avg_win_loss_ratio=None,
            n_snapshots=0,
            n_paired_trades=0,
        )
