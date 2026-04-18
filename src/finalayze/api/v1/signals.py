"""Signals and strategy performance endpoints (Layer 6)."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

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
    """Per-(strategy, market, segment) performance row.

    Schema extension per Phase 55 D-16 — consumed by the Plan-05 heatmap
    without adding a second endpoint.

    Fields:
        strategy: Strategy name (e.g. "momentum", "mean_reversion").
        market_id: Market identifier ("us", "moex").
        segment_id: Market segment (e.g. "ru_blue_chips"). Added per D-16.
        win_rate: Fraction of paired trades whose P&L > commission+slippage
            (D-03 cost-threshold win). Null when `trades_count < 5` (D-15)
            or when no paired trades exist for this group.
        profit_factor: gross_win / gross_loss on paired trades. Null when
            `trades_count < 5` or no losing trades exist in the window.
        trades_count: Number of closed FIFO pairs attributed to this group
            via the CLOSING order's `signal.strategy_name` (D-04).
        signal_count: Number of signals emitted by this strategy for this
            (market, segment) within the period. Distinct from trades_count.
        last_signal_at: ISO-8601 timestamp of the most recent signal, or None.
    """

    model_config = ConfigDict(frozen=True)
    strategy: str
    market_id: str
    segment_id: str
    win_rate: float | None
    profit_factor: float | None
    trades_count: int
    signal_count: int
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


_MIN_TRADES = 5  # D-15 sample-size gate — below this, win_rate / PF are null.
_BPS = Decimal(10000)

# Aggregation bucket shape: wins, losses, gross_win, gross_loss, trades_count.
_AggBucket = dict[str, Decimal | int]
_AggKey = tuple[str, str, str]


def _empty_bucket() -> _AggBucket:
    return {
        "wins": 0,
        "losses": 0,
        "gross_win": Decimal(0),
        "gross_loss": Decimal(0),
        "trades_count": 0,
    }


def _cost_bps(market_id: str, settings: Any) -> tuple[Decimal, Decimal]:
    """Return (commission_bps, slip_bps) per market per D-03 config rate."""
    is_moex = str(market_id).lower() == "moex" or str(market_id).lower().startswith("ru")
    commission = Decimal(
        str(
            settings.default_commission_bps_moex if is_moex else settings.default_commission_bps_us,
        ),
    )
    slip = Decimal(str(settings.default_slippage_cost_bps))
    return commission, slip


def _update_bucket(bucket: _AggBucket, pnl: Decimal, cost: Decimal) -> None:
    """Apply D-03 cost-threshold win test to a single paired trade."""
    bucket["trades_count"] = int(bucket["trades_count"]) + 1
    if pnl > cost:
        bucket["wins"] = int(bucket["wins"]) + 1
        bucket["gross_win"] = Decimal(bucket["gross_win"]) + pnl
    else:
        bucket["losses"] = int(bucket["losses"]) + 1
        if pnl < 0:
            bucket["gross_loss"] = Decimal(bucket["gross_loss"]) + abs(pnl)


def _aggregate_pairs(
    orders: list[Any],
    settings: Any,
) -> dict[_AggKey, _AggBucket]:
    """FIFO-pair the orders and aggregate per (strategy, market, segment).

    Per D-04, attribution uses the CLOSING order's `signal.strategy_name`.
    Orphan closes (`signal_id=None` or no loaded signal relationship) are
    silently skipped per Pitfall 5.
    """
    from finalayze.api.v1._fifo import fifo_pair  # noqa: PLC0415

    sig_attr_by_id: dict[object, _AggKey] = {}
    for o in orders:
        s = getattr(o, "signal", None)
        if s is None:
            continue
        sig_attr_by_id[s.id] = (s.strategy_name, s.market_id, s.segment_id)

    pair_agg: dict[_AggKey, _AggBucket] = {}
    for pair in fifo_pair(orders):
        if pair.closing_signal_id is None:
            continue
        attr = sig_attr_by_id.get(pair.closing_signal_id)
        if attr is None:
            continue
        commission_bps, slip_bps = _cost_bps(attr[1], settings)
        avg_price = (pair.entry_price + pair.exit_price) / Decimal(2)
        notional = avg_price * pair.quantity
        cost = (notional * commission_bps / _BPS) + (notional * slip_bps / _BPS)
        pnl = (pair.exit_price - pair.entry_price) * pair.quantity
        bucket = pair_agg.setdefault(attr, _empty_bucket())
        _update_bucket(bucket, pnl, cost)
    return pair_agg


def _finalize_metrics(
    agg: _AggBucket,
) -> tuple[int, float | None, float | None]:
    """Apply D-15 sample gate and compute win_rate / profit_factor."""
    trades_count = int(agg["trades_count"])
    wins = int(agg["wins"])
    losses = int(agg["losses"])
    if trades_count < _MIN_TRADES:
        return trades_count, None, None
    total = wins + losses
    win_rate_f = float(Decimal(wins) / Decimal(total)) if total > 0 else None
    gl = Decimal(agg["gross_loss"])
    pf_f = float(Decimal(agg["gross_win"]) / gl) if gl > 0 else None
    return trades_count, win_rate_f, pf_f


async def _fetch_perf_rows(
    *,
    market: str | None,
    cutoff: Any,
) -> tuple[list[Any], list[Any]]:
    """Run the two DB reads: signals group-by and filled-orders scan.

    Returns `(sig_rows, orders)`. `sig_rows` are labeled aggregation rows
    `(strategy_name, market_id, segment_id, sig_count, last_signal)`;
    `orders` are `OrderModel` instances with `.signal` eagerly loaded.
    """
    from sqlalchemy import func, select  # noqa: PLC0415
    from sqlalchemy.orm import selectinload  # noqa: PLC0415

    from finalayze.core.db import get_async_session_factory  # noqa: PLC0415
    from finalayze.core.models import OrderModel, SignalModel  # noqa: PLC0415

    async with get_async_session_factory()() as session:
        sig_stmt = (
            select(
                SignalModel.strategy_name,
                SignalModel.market_id,
                SignalModel.segment_id,
                func.count().label("sig_count"),
                func.max(SignalModel.created_at).label("last_signal"),
            )
            .where(SignalModel.created_at >= cutoff)
            .group_by(
                SignalModel.strategy_name,
                SignalModel.market_id,
                SignalModel.segment_id,
            )
        )
        if market:
            sig_stmt = sig_stmt.where(SignalModel.market_id == market)
        sig_rows = (await session.execute(sig_stmt)).all()

        order_stmt = (
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
            order_stmt = order_stmt.where(OrderModel.market_id == market)
        orders = list((await session.execute(order_stmt)).scalars().all())
    return list(sig_rows), orders


def _assemble_rows(
    sig_rows: list[Any],
    pair_agg: dict[_AggKey, _AggBucket],
) -> list[StrategyPerf]:
    """Merge signal group-by rows with pair aggregates into response rows."""
    all_keys: set[_AggKey] = set(pair_agg.keys())
    sig_lookup: dict[_AggKey, tuple[int, Any]] = {}
    for r in sig_rows:
        key_r = (r.strategy_name, r.market_id, r.segment_id)
        sig_lookup[key_r] = (int(r.sig_count), r.last_signal)
        all_keys.add(key_r)

    rows: list[StrategyPerf] = []
    for key in all_keys:
        strategy, mkt, seg = key
        sig_count, last_signal = sig_lookup.get(key, (0, None))
        agg = pair_agg.get(key, _empty_bucket())
        trades_count, win_rate_f, pf_f = _finalize_metrics(agg)
        rows.append(
            StrategyPerf(
                strategy=strategy,
                market_id=mkt,
                segment_id=seg,
                win_rate=win_rate_f,
                profit_factor=pf_f,
                trades_count=trades_count,
                signal_count=sig_count,
                last_signal_at=last_signal.isoformat() if last_signal is not None else None,
            ),
        )
    return rows


@router.get("/strategies/performance", response_model=StrategiesResponse)
async def strategies_performance(
    market: str | None = None,
    period: int = 30,
) -> StrategiesResponse:
    """Per-strategy, per-segment performance. Realized-only (D-02 deferred → Phase 56).

    Joins `signals ⨝ orders` on `OrderModel.signal_id`, FIFO-pairs filled orders
    per-symbol (D-01), and credits each pair's P&L to the closing order's
    `signal.strategy_name` (D-04). Applies the D-03 cost-threshold win test
    (`pnl > commission + slippage`) and the D-15 sample-size gate (`N >= 5`).

    Decisions referenced:
        D-02 deferred — Phase 55 is realized-only (CONTEXT.md amendment 2026-04-18).
        D-04 — strategy attribution uses closing order's signal.strategy_name.
        D-13 — default period is 30 days.
        D-15 — win_rate/profit_factor null when trades_count < 5.
        D-16 — response row includes segment_id + trades_count so the dashboard
               can render a Strategy x Segment heatmap without a new endpoint.

    Returns `StrategiesResponse(strategies=[])` on any exception — read-only
    analytics endpoints degrade gracefully rather than 500 on DB blips.
    """
    try:
        from datetime import UTC, datetime, timedelta  # noqa: PLC0415

        from config.settings import Settings  # noqa: PLC0415

        settings = Settings()
        cutoff = (datetime.now(UTC) - timedelta(days=period)).replace(
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
        )
        sig_rows, orders = await _fetch_perf_rows(market=market, cutoff=cutoff)
        pair_agg = _aggregate_pairs(orders, settings)
        return StrategiesResponse(strategies=_assemble_rows(sig_rows, pair_agg))
    except Exception as exc:
        _log.warning("strategies_performance_failed", error=str(exc))
        return StrategiesResponse(strategies=[])
