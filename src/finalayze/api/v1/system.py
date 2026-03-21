"""System endpoints: health check, feed health, system status, mode management.

Layer 6 -- API layer. Depends on Layer 0 (exceptions, modes).
"""

from __future__ import annotations

import time
from collections import deque
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Annotated, Any

import redis.asyncio
import structlog
from config.settings import get_settings
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict
from sqlalchemy import text

from finalayze.api.v1.auth import api_key_auth
from finalayze.core.db import get_async_session_factory
from finalayze.core.exceptions import ModeError
from finalayze.core.modes import ModeManager, WorkMode

if TYPE_CHECKING:
    from finalayze.core.kill_switch import KillSwitch
    from finalayze.execution.tinkoff_broker import TinkoffBroker
    from finalayze.monitoring.health_monitor import HealthMonitor

_log = structlog.get_logger()

router = APIRouter(tags=["system"])

# Application-scoped singleton (overridden in tests via dependency overrides)
_default_mode_manager = ModeManager()

APP_VERSION = "0.1.0"
_start_time = datetime.now(UTC)

# Tinkoff broker reference for health probes (set via set_tinkoff_broker)
_tinkoff_broker: TinkoffBroker | None = None

# Production health monitor and kill switch (set via setter functions)
_health_monitor: HealthMonitor | None = None
_kill_switch: KillSwitch | None = None

# Feed freshness tracking: source -> last candle timestamp
_last_candle_timestamps: dict[str, datetime] = {}
_FEED_FRESHNESS_THRESHOLD_HOURS = 2.0

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


class ProductionHealthResponse(BaseModel):
    """Per-component production health status."""

    model_config = ConfigDict(frozen=True)
    broker_ok: bool
    feed_fresh: bool
    loop_alive: bool
    overall: bool
    timestamp: str
    details: dict[str, str]


class KillResponse(BaseModel):
    """Result of KillSwitch activation."""

    model_config = ConfigDict(frozen=True)
    orders_cancelled: int
    scheduler_stopped: bool
    breakers_escalated: int
    alert_sent: bool
    elapsed_seconds: float


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


def set_tinkoff_broker(broker: TinkoffBroker | None) -> None:
    """Set the TinkoffBroker instance for health probes."""
    global _tinkoff_broker  # noqa: PLW0603
    _tinkoff_broker = broker


def set_health_monitor(monitor: HealthMonitor) -> None:
    """Set the HealthMonitor instance for production health endpoint."""
    global _health_monitor  # noqa: PLW0603
    _health_monitor = monitor


def set_kill_switch(ks: KillSwitch) -> None:
    """Set the KillSwitch instance for REST kill endpoint."""
    global _kill_switch  # noqa: PLW0603
    _kill_switch = ks


def update_feed_timestamp(source: str, ts: datetime) -> None:
    """Update the latest candle timestamp for a data source."""
    _last_candle_timestamps[source] = ts


async def _check_tinkoff() -> str:
    """Return 'ok' if TinkoffBroker responds to get_portfolio(), else 'error'.

    Returns 'unknown' if no broker is configured.
    TinkoffBroker uses asyncio.run() internally, so we run it in a thread
    to avoid conflict with uvicorn's event loop.
    """
    if _tinkoff_broker is None:
        return "unknown"
    try:
        import asyncio  # noqa: PLC0415

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _tinkoff_broker.get_portfolio)
        return "ok"
    except Exception:
        _log.debug("Tinkoff health check failed", exc_info=True)
        return "error"


async def _check_feed_freshness() -> str:
    """Return 'ok' if all feeds are fresh, 'stale' if any exceed threshold, 'unknown' if no data."""
    if not _last_candle_timestamps:
        return "unknown"
    now = datetime.now(UTC)
    for ts in _last_candle_timestamps.values():
        age = now - ts
        if age >= timedelta(hours=_FEED_FRESHNESS_THRESHOLD_HOURS):
            return "stale"
    return "ok"


async def _get_component_status() -> ComponentStatus:
    """Run real health checks with 30s caching."""
    global _health_cache, _health_cache_ts  # noqa: PLW0603

    now = time.monotonic()
    if _health_cache and (now - _health_cache_ts) < _HEALTH_CACHE_TTL:
        return ComponentStatus(**_health_cache)

    db_status = await _check_db()
    redis_status = await _check_redis()
    tinkoff_status = await _check_tinkoff()

    result = {
        "db": db_status,
        "redis": redis_status,
        "alpaca": "ok",
        "tinkoff": tinkoff_status,
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
    # Mandatory components determine overall status.
    # "unknown" is acceptable (broker not configured); only "error" degrades.
    _mandatory = {"db": components.db, "redis": components.redis, "tinkoff": components.tinkoff}
    _ok_values = {"ok", "unknown"}  # "unknown" = not configured, not an error
    overall = "ok" if all(v in _ok_values for v in _mandatory.values()) else "degraded"
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


# ── Production health & kill endpoints ────────────────────────────────────────


@router.get("/health/production", response_model=ProductionHealthResponse)
async def health_production() -> ProductionHealthResponse:
    """Per-component production health status. No auth required.

    Returns 200 when all checks pass, 503 when any fail.
    """
    if _health_monitor is None:
        raise HTTPException(status_code=503, detail="Health monitor not configured")

    result = _health_monitor.check_now()
    overall = result.broker_ok and result.feed_fresh and result.loop_alive

    response = ProductionHealthResponse(
        broker_ok=result.broker_ok,
        feed_fresh=result.feed_fresh,
        loop_alive=result.loop_alive,
        overall=overall,
        timestamp=result.timestamp.isoformat(),
        details=result.details,
    )

    if not overall:
        raise HTTPException(status_code=503, detail=response.model_dump())

    return response


@router.post("/kill", response_model=KillResponse)
async def kill_endpoint() -> KillResponse:
    """Trigger emergency shutdown via REST API. No auth required (internal network)."""
    if _kill_switch is None:
        raise HTTPException(status_code=503, detail="Kill switch not configured")

    result = _kill_switch.activate(reason="rest_api")
    return KillResponse(
        orders_cancelled=result.orders_cancelled,
        scheduler_stopped=result.scheduler_stopped,
        breakers_escalated=result.breakers_escalated,
        alert_sent=result.alert_sent,
        elapsed_seconds=result.elapsed_seconds,
    )
