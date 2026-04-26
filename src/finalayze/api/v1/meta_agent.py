"""Meta-agent control endpoints (Layer 6, Phase 58-01, META-08 surface).

Currently exposes:
  - ``GET /api/v1/meta-agent/status`` — operator visibility for the
    APScheduler tick state, dry-run flag, last run timestamp, and the
    in-flight subprocess registry. The registry is read from the wired
    ``MetaAgentRunner`` instance and tolerates an empty / uninitialised
    registry — Plan 58-05 wires the killswitch counters.

``POST /api/v1/meta-agent/disable`` lands in Plan 58-05 alongside the
killswitch + abort wiring.

Pattern source: ``api/v1/system.py:213-216`` for the module-level
singleton + setter pattern; ``api/v1/alerts.py:28-32`` for the router
header + auth dependency.
"""

from __future__ import annotations

from datetime import datetime  # noqa: TC003 — required at runtime by Pydantic
from typing import TYPE_CHECKING

import structlog
from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict

from finalayze.api.v1.auth import api_key_auth

if TYPE_CHECKING:
    from finalayze.meta_agent.runner import MetaAgentRunner

_log = structlog.get_logger()

router = APIRouter(
    prefix="/meta-agent",
    tags=["meta-agent"],
    dependencies=[Depends(api_key_auth)],
)


class InflightSpawns(BaseModel):
    """Counts of currently-running spawns by type. Plan 58-05 populates."""

    model_config = ConfigDict(frozen=True)
    investigate: int
    fix: int


class MetaAgentStatus(BaseModel):
    """Five-field envelope returned by ``GET /status`` (SPEC §AC #16)."""

    model_config = ConfigDict(frozen=True)
    enabled: bool
    dry_run: bool
    last_run_ts: datetime | None
    scheduler_active: bool
    inflight_spawns: InflightSpawns


# Module-level singleton (mirrors ``api/v1/system.py:_kill_switch``).
_runner: MetaAgentRunner | None = None


def set_runner(runner: MetaAgentRunner | None) -> None:
    """Wire the MetaAgentRunner instance into this router.

    Called from main.py / bootstrap.py after the runner has been built.
    Passing ``None`` clears the wiring (used by tests for isolation).
    """
    global _runner  # noqa: PLW0603 — mirrors api/v1/system.py pattern
    _runner = runner


@router.get("/status", response_model=MetaAgentStatus)
async def get_meta_agent_status() -> MetaAgentStatus:
    """Operator-visibility status for the meta-agent (SPEC §Requirement 8)."""
    if _runner is None:
        return MetaAgentStatus(
            enabled=False,
            dry_run=True,
            last_run_ts=None,
            scheduler_active=False,
            inflight_spawns=InflightSpawns(investigate=0, fix=0),
        )
    snap = _runner.status_snapshot()
    inflight_raw = snap.get("inflight_spawns") or {}
    return MetaAgentStatus(
        enabled=bool(snap.get("enabled", False)),
        dry_run=bool(snap.get("dry_run", True)),
        last_run_ts=snap.get("last_run_ts"),
        scheduler_active=bool(snap.get("scheduler_active", False)),
        inflight_spawns=InflightSpawns(
            investigate=int(inflight_raw.get("investigate", 0)),
            fix=int(inflight_raw.get("fix", 0)),
        ),
    )
