"""System endpoints: health check, feed health, system status, mode management.

Layer 6 -- API layer. Depends on Layer 0 (exceptions, modes).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Any

from config.settings import Settings
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict

from finalayze.api.v1.auth import require_api_key
from finalayze.core.exceptions import ModeError
from finalayze.core.modes import ModeManager, WorkMode

_settings = Settings()
router = APIRouter(tags=["system"])

# Application-scoped singleton (overridden in tests via dependency overrides)
_default_mode_manager = ModeManager()

APP_VERSION = "0.1.0"
_start_time = datetime.now(UTC)

# In-memory ring buffer for recent errors (max 100)
_recent_errors: list[dict[str, Any]] = []
_MAX_ERRORS = 100


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
    if len(_recent_errors) > _MAX_ERRORS:
        _recent_errors.pop(0)


# ── Response models ────────────────────────────────────────────────────────────


class ComponentStatus(BaseModel):
    model_config = ConfigDict(frozen=True)
    db: str = "ok"
    redis: str = "ok"
    alpaca: str = "ok"
    tinkoff: str = "ok"
    llm: str = "ok"


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


# ── Endpoints ─────────────────────────────────────────────────────────────────


@router.get("/health", response_model=HealthResponse)
async def health(
    mgr: Annotated[ModeManager, Depends(get_mode_manager)],
) -> HealthResponse:
    """Liveness check. No auth required."""
    components = ComponentStatus()
    overall = "ok" if all(v == "ok" for v in components.model_dump().values()) else "degraded"
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
    dependencies=[Depends(require_api_key(_settings.api_key))],
)
async def system_status(
    mgr: Annotated[ModeManager, Depends(get_mode_manager)],
) -> SystemStatusResponse:
    """System status including mode, uptime, component health. Auth required."""
    uptime = (datetime.now(UTC) - _start_time).total_seconds()
    return SystemStatusResponse(
        mode=str(mgr.current_mode),
        version=APP_VERSION,
        uptime_seconds=uptime,
        components=ComponentStatus(),
    )


@router.get(
    "/system/errors",
    response_model=list[ErrorEntry],
    dependencies=[Depends(require_api_key(_settings.api_key))],
)
async def system_errors() -> list[ErrorEntry]:
    """Last 100 recorded exceptions. Auth required."""
    return [ErrorEntry(**e) for e in _recent_errors]


@router.get(
    "/mode",
    response_model=ModeResponse,
    dependencies=[Depends(require_api_key(_settings.api_key))],
)
async def get_mode(
    mgr: Annotated[ModeManager, Depends(get_mode_manager)],
) -> ModeResponse:
    """Return the current work mode. Auth required."""
    return ModeResponse(mode=str(mgr.current_mode))


@router.post(
    "/mode",
    response_model=ModeResponse,
    dependencies=[Depends(require_api_key(_settings.api_key))],
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
        settings = Settings()
        if not settings.real_token or request.confirm_token != settings.real_token:
            raise HTTPException(
                status_code=403,
                detail="Transitioning to REAL mode requires a valid confirm_token",
            )
    try:
        mgr.transition_to(request.mode)
    except ModeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ModeResponse(mode=str(mgr.current_mode))
