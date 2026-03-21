"""Sandbox go/no-go gate REST endpoint (Layer 6).

Exposes GoNoGoReporter evaluation over HTTP with Pydantic response models.
The reporter instance is injected via ``set_go_no_go_reporter()`` during lifespan.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict

from finalayze.api.v1.auth import api_key_auth
from finalayze.core.db import get_db

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
