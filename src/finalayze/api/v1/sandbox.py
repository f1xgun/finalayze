"""Sandbox go/no-go gate and metrics REST endpoints (Layer 6).

Exposes GoNoGoReporter evaluation and sandbox metric queries over HTTP
with Pydantic response models.
The reporter instance is injected via ``set_go_no_go_reporter()`` during lifespan.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict
from sqlalchemy import select

from finalayze.api.v1.auth import api_key_auth
from finalayze.core.db import get_db
from finalayze.core.models import SandboxMetricRow

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from finalayze.monitoring.go_no_go import GoNoGoReporter

_log = structlog.get_logger()

# Module-level singleton (set via set_go_no_go_reporter in main.py lifespan)
_go_no_go_reporter: GoNoGoReporter | None = None

router = APIRouter(tags=["sandbox"])


def set_go_no_go_reporter(reporter: GoNoGoReporter) -> None:
    """Set the GoNoGoReporter instance for the sandbox endpoint."""
    global _go_no_go_reporter  # noqa: PLW0603
    _go_no_go_reporter = reporter


# ── Response models ────────────────────────────────────────────────────────────


class CriterionResponse(BaseModel):
    """Single gate criterion result."""

    model_config = ConfigDict(frozen=True)
    name: str
    passed: bool
    actual: float
    threshold: float
    unit: str
    critical: bool


class GoNoGoResponse(BaseModel):
    """Full go/no-go gate evaluation response."""

    model_config = ConfigDict(frozen=True)
    verdict: str
    criteria: list[CriterionResponse]
    sandbox_days: int
    evaluated_at: str
    reason: str


# ── Endpoint ───────────────────────────────────────────────────────────────────


@router.get(
    "/sandbox/gonogo",
    response_model=GoNoGoResponse,
    dependencies=[Depends(api_key_auth)],
)
async def sandbox_gonogo(
    session: AsyncSession = Depends(get_db),  # noqa: B008
) -> GoNoGoResponse:
    """Evaluate sandbox go/no-go gate criteria.

    Returns structured pass/fail report with per-criterion breakdown.
    Requires X-API-Key authentication.
    """
    if _go_no_go_reporter is None:
        raise HTTPException(status_code=503, detail="GoNoGoReporter not configured")

    report = await _go_no_go_reporter.evaluate(session)

    return GoNoGoResponse(
        verdict=report.verdict.value,
        criteria=[
            CriterionResponse(
                name=c.name,
                passed=c.passed,
                actual=c.actual,
                threshold=c.threshold,
                unit=c.unit,
                critical=c.critical,
            )
            for c in report.criteria
        ],
        sandbox_days=report.sandbox_days,
        evaluated_at=report.evaluated_at.isoformat(),
        reason=report.reason,
    )


# ── Metrics response model ────────────────────────────────────────────────────


class SandboxMetricResponse(BaseModel):
    """Single sandbox metric row serialized for API response."""

    model_config = ConfigDict(frozen=True)
    timestamp: str
    market_id: str
    trade_count: int
    pnl_rub: float | None
    equity_rub: float
    fill_rate: float | None
    uptime_cycles: int
    signals_generated: int
    errors_caught: int
    max_slippage_bps: float | None
    avg_slippage_bps: float | None
    drawdown_pct: float | None


# ── Metrics endpoint ──────────────────────────────────────────────────────────


@router.get(
    "/sandbox/metrics",
    response_model=list[SandboxMetricResponse],
    dependencies=[Depends(api_key_auth)],
)
async def sandbox_metrics(
    days: int = Query(default=7, ge=1, le=365),
    market_id: str = Query(default="moex"),
    session: AsyncSession = Depends(get_db),  # noqa: B008
) -> list[SandboxMetricResponse]:
    """Return sandbox metric rows filtered by date range and market.

    Query parameters:
    - **days**: Number of days to look back (default 7).
    - **market_id**: Market identifier to filter by (default "moex").

    Requires X-API-Key authentication.
    """
    cutoff = datetime.now(UTC) - timedelta(days=days)
    stmt = (
        select(SandboxMetricRow)
        .where(
            SandboxMetricRow.market_id == market_id,
            SandboxMetricRow.timestamp >= cutoff,
        )
        .order_by(SandboxMetricRow.timestamp)
    )
    result = await session.execute(stmt)
    rows = result.scalars().all()

    return [
        SandboxMetricResponse(
            timestamp=row.timestamp.isoformat(),
            market_id=row.market_id,
            trade_count=row.trade_count,
            pnl_rub=float(row.pnl_rub) if row.pnl_rub is not None else None,
            equity_rub=float(row.equity_rub),
            fill_rate=float(row.fill_rate) if row.fill_rate is not None else None,
            uptime_cycles=row.uptime_cycles,
            signals_generated=row.signals_generated,
            errors_caught=row.errors_caught,
            max_slippage_bps=(
                float(row.max_slippage_bps) if row.max_slippage_bps is not None else None
            ),
            avg_slippage_bps=(
                float(row.avg_slippage_bps) if row.avg_slippage_bps is not None else None
            ),
            drawdown_pct=float(row.drawdown_pct) if row.drawdown_pct is not None else None,
        )
        for row in rows
    ]
