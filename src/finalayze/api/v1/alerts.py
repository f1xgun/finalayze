"""Alerts list endpoint (Layer 6, ALRT-03).

Phase 57-04 — paginated read-only endpoint over the ``alerts`` hypertable
(Plan 57-01 schema). Operators consume this via the Streamlit /alerts
page (Plan 57-04 Task 2) for a queryable Telegram-replay dashboard.

D-16 filters: ``alert_type`` (multi), ``symbol``, ``priority`` (uppercase
``CRITICAL``/``IMPORTANT``/``INFO`` matching ``AlertPriority.name`` per
Phase 57-02 revision Mi5), ``since``/``until``.
D-18 pagination: ``page`` (>=1), ``page_size`` (default 50, max 200).
Graceful degradation: DB failure returns ``{alerts: [], total: 0}`` rather
than 500 (mirrors the /portfolio/history hybrid pattern from Phase 56-03).
"""

from __future__ import annotations

from datetime import datetime  # noqa: TC003 — required at runtime by FastAPI/Pydantic Query
from typing import Annotated, Any

import structlog
from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, ConfigDict

from finalayze.api.v1.auth import api_key_auth

_log = structlog.get_logger()

router = APIRouter(
    prefix="/alerts",
    tags=["alerts"],
    dependencies=[Depends(api_key_auth)],
)

_DEFAULT_PAGE_SIZE = 50
_MAX_PAGE_SIZE = 200


class AlertEntry(BaseModel):
    """One alert row, serialised for the dashboard."""

    model_config = ConfigDict(frozen=True)
    id: str
    timestamp: str
    alert_type: str
    priority: str
    symbol: str | None
    market_id: str | None
    message: str
    parent_id: str | None
    delivery_status: str


class AlertsResponse(BaseModel):
    """Paginated envelope: alerts + total + page + page_size."""

    model_config = ConfigDict(frozen=True)
    alerts: list[AlertEntry]
    total: int
    page: int
    page_size: int


@router.get("", response_model=AlertsResponse)
async def list_alerts(
    page: int = Query(1, ge=1),
    page_size: int = Query(_DEFAULT_PAGE_SIZE, ge=1, le=_MAX_PAGE_SIZE),
    alert_type: Annotated[list[str] | None, Query()] = None,
    symbol: str | None = None,
    priority: str | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
) -> AlertsResponse:
    """Paginated alert history (ALRT-03, D-15/D-16/D-18).

    Returns a JSON envelope with the paginated alert window plus the
    unfiltered-by-pagination ``total`` count for the matching filters.

    Empty/error response shape: ``{alerts: [], total: 0, page: N,
    page_size: M}`` — used both when no rows match and when the DB query
    raises (graceful degradation; failure logged at warning level).
    """
    try:
        from sqlalchemy import func, select  # noqa: PLC0415

        from finalayze.core.db import get_async_session_factory  # noqa: PLC0415
        from finalayze.core.models import AlertModel  # noqa: PLC0415

        async with get_async_session_factory()() as session:
            stmt = select(AlertModel)
            count_stmt = select(func.count()).select_from(AlertModel)
            # SQLAlchemy ColumnElement comparisons typecheck loosely; collect
            # them as Any so mypy doesn't complain about BinaryExpression vs
            # ColumnElement covariance on `filters.append(...)`.
            filters: list[Any] = []
            if alert_type:
                filters.append(AlertModel.alert_type.in_(alert_type))
            if symbol:
                filters.append(AlertModel.symbol == symbol)
            if priority:
                filters.append(AlertModel.priority == priority)
            if since:
                filters.append(AlertModel.timestamp >= since)
            if until:
                filters.append(AlertModel.timestamp <= until)
            for f in filters:
                stmt = stmt.where(f)
                count_stmt = count_stmt.where(f)
            stmt = (
                stmt.order_by(AlertModel.timestamp.desc())
                .limit(page_size)
                .offset((page - 1) * page_size)
            )
            result = await session.execute(stmt)
            rows = result.scalars().all()
            total = (await session.execute(count_stmt)).scalar() or 0

        entries = [
            AlertEntry(
                id=str(r.id),
                timestamp=r.timestamp.isoformat(),
                alert_type=r.alert_type,
                priority=r.priority,
                symbol=r.symbol,
                market_id=r.market_id,
                message=r.message,
                parent_id=str(r.parent_id) if r.parent_id else None,
                delivery_status=r.delivery_status,
            )
            for r in rows
        ]
        return AlertsResponse(
            alerts=entries,
            total=int(total),
            page=page,
            page_size=page_size,
        )
    except Exception as exc:
        _log.warning("alerts_query_failed", error=str(exc))
        return AlertsResponse(
            alerts=[],
            total=0,
            page=page,
            page_size=page_size,
        )
