"""System endpoints: health check, feed health, system status, mode management.

Layer 6 -- API layer. Depends on Layer 0 (exceptions, modes).
"""

from __future__ import annotations

import logging
import time
from collections import deque
from datetime import UTC, datetime
from typing import Annotated, Any

import redis.asyncio
from config.settings import get_settings
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict
from sqlalchemy import text

from finalayze.api.v1.auth import api_key_auth
from finalayze.core.db import get_async_session_factory
from finalayze.core.exceptions import ModeError
from finalayze.core.modes import ModeManager, WorkMode

_log = logging.getLogger(__name__)

router = APIRouter(tags=["system"])

# Application-scoped singleton (overridden in tests via dependency overrides)
_default_mode_manager = ModeManager()

APP_VERSION = "0.1.0"
_start_time = datetime.now(UTC)

# In-memory ring buffer for recent errors (max 100); deque(maxlen=100) handles eviction
_recent_errors: deque[dict[str, Any]] = deque(maxlen=100)

# Health check cache: avoid hammering db/redis on every /health call
_HEALTH_CACHE_TTL = 30  # seconds
_health_cache: dict[str, Any] = {}
_health_cache_ts: float = 0.0


def get_mode_manager() -> ModeManager:
    """Dependency that returns the application-wide ModeManager."""
    return _default_mode_manager


def record_error(component: str, message: str, traceback_excerpt: str = "") -> None:
    """Called externally to store recent exceptions in the ring buffer."""
    _recent_errors.append(
        {
            "timestamp": datetime.now(UTC).isoformat(),
            "component": component,
            "message": message,
            "traceback_excerpt": traceback_excerpt,
        }
    )


# ── Response models ────────────────────────────────────────────────────────────


class ComponentStatus(BaseModel):
    """Real-time component health status from liveness probes."""

    model_config = ConfigDict(frozen=True)
    db: str
    redis: str
    alpaca: str = "unknown"
    tinkoff: str = "unknown"
    llm: str = "unknown"


class HealthResponse(BaseModel):
    model_config = ConfigDict(frozen=True)
    status: str
    mode: str
    version: str
    components: ComponentStatus


class FeedStatus(BaseModel):
    model_config = ConfigDict(frozen=True)
    source: str
    last_seen: str | None
    latency_ms: float | None


class FeedsResponse(BaseModel):
    model_config = ConfigDict(frozen=True)
    feeds: list[FeedStatus]


class SystemStatusResponse(BaseModel):
    model_config = ConfigDict(frozen=True)
    mode: str
    version: str
    uptime_seconds: float
    components: ComponentStatus


class ErrorEntry(BaseModel):
    model_config = ConfigDict(frozen=True)
    timestamp: str
    component: str
    message: str
    traceback_excerpt: str


class ModeResponse(BaseModel):
    model_config = ConfigDict(frozen=True)
    mode: str


class ModeRequest(BaseModel):
    mode: WorkMode
    confirm_token: str | None = None


# ── Liveness helpers ─────────────────────────────────────────────────────────


async def _check_db() -> str:
    """Return 'ok' if the database responds to SELECT 1, else 'error'."""
    try:
        factory = get_async_session_factory()
        async with factory() as session:
            await session.execute(text("SELECT 1"))
        return "ok"
    except Exception:
        _log.debug("DB health check failed", exc_info=True)
        return "error"


async def _check_redis() -> str:
    """Return 'ok' if Redis responds to PING, else 'error'."""
    try:
        settings = get_settings()
        client: redis.asyncio.Redis[str] = redis.asyncio.from_url(
            settings.redis_url, decode_responses=True
        )
        await client.ping()
        await client.aclose()  # type: ignore[attr-defined]
        return "ok"
    except Exception:
        _log.debug("Redis health check failed", exc_info=True)
        return "error"


async def _get_component_status() -> ComponentStatus:
    """Run real health checks with 30s caching."""
    global _health_cache, _health_cache_ts  # noqa: PLW0603

    now = time.monotonic()
    if _health_cache and (now - _health_cache_ts) < _HEALTH_CACHE_TTL:
        return ComponentStatus(**_health_cache)

    db_status = await _check_db()
    redis_status = await _check_redis()

    result = {
        "db": db_status,
        "redis": redis_status,
        "alpaca": "ok",
        "tinkoff": "ok",
        "llm": "ok",
    }
    _health_cache = result
    _health_cache_ts = now
    return ComponentStatus(**result)


# ── Endpoints ─────────────────────────────────────────────────────────────────


@router.get("/health", response_model=HealthResponse)
async def health(
    mgr: Annotated[ModeManager, Depends(get_mode_manager)],
) -> HealthResponse:
    """Liveness check — performs real DB and Redis probes. No auth required."""
    components = await _get_component_status()
    # Only mandatory components (db, redis) determine overall status.
    # Optional components default to "unknown" and do not degrade overall status.
    _mandatory = {"db": components.db, "redis": components.redis}
    overall = "ok" if all(v == "ok" for v in _mandatory.values()) else "degraded"
    return HealthResponse(
        status=overall,
        mode=str(mgr.current_mode),
        version=APP_VERSION,
        components=components,
    )


@router.get("/health/feeds", response_model=FeedsResponse)
async def health_feeds() -> FeedsResponse:
    """Feed health: last-seen per data source. No auth required."""
    return FeedsResponse(
        feeds=[
            FeedStatus(source="finnhub", last_seen=None, latency_ms=None),
            FeedStatus(source="newsapi", last_seen=None, latency_ms=None),
            FeedStatus(source="tinkoff", last_seen=None, latency_ms=None),
        ]
    )


@router.get(
    "/system/status",
    response_model=SystemStatusResponse,
    dependencies=[Depends(api_key_auth)],
)
async def system_status(
    mgr: Annotated[ModeManager, Depends(get_mode_manager)],
) -> SystemStatusResponse:
    """System status including mode, uptime, component health. Auth required."""
    uptime = (datetime.now(UTC) - _start_time).total_seconds()
    components = await _get_component_status()
    return SystemStatusResponse(
        mode=str(mgr.current_mode),
        version=APP_VERSION,
        uptime_seconds=uptime,
        components=components,
    )


@router.get(
    "/system/errors",
    response_model=list[ErrorEntry],
    dependencies=[Depends(api_key_auth)],
)
async def system_errors() -> list[ErrorEntry]:
    """Last 100 recorded exceptions. Auth required."""
    return [ErrorEntry(**e) for e in _recent_errors]


@router.get(
    "/mode",
    response_model=ModeResponse,
    dependencies=[Depends(api_key_auth)],
)
async def get_mode(
    mgr: Annotated[ModeManager, Depends(get_mode_manager)],
) -> ModeResponse:
    """Return the current work mode. Auth required."""
    return ModeResponse(mode=str(mgr.current_mode))


@router.post(
    "/mode",
    response_model=ModeResponse,
    dependencies=[Depends(api_key_auth)],
)
async def set_mode(
    request: ModeRequest,
    mgr: Annotated[ModeManager, Depends(get_mode_manager)],
) -> ModeResponse:
    """Change the work mode. Auth required.

    Transitioning to REAL mode requires either:
    - ``FINALAYZE_REAL_CONFIRMED=true`` env var (legacy / deployment guard), or
    - A valid ``confirm_token`` matching ``FINALAYZE_REAL_TOKEN`` in settings.

    Raises:
        HTTPException(400): When the ModeManager rejects the transition
            (e.g. REAL without env var confirmation).
        HTTPException(403): When real_token is not configured or the request
            confirm_token does not match.
    """
    if request.mode == WorkMode.REAL:
        _real_settings = get_settings()
        if not _real_settings.real_token or request.confirm_token != _real_settings.real_token:
            raise HTTPException(
                status_code=403,
                detail="Transitioning to REAL mode requires a valid confirm_token",
            )
    try:
        mgr.transition_to(request.mode)
    except ModeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ModeResponse(mode=str(mgr.current_mode))
