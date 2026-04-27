"""Meta-agent control endpoints (Layer 6, Phase 58-01 + 58-05, META-08 surface).

Exposes:
  - ``GET /api/v1/meta-agent/status`` (Plan 58-01 Task 11) — operator
    visibility for the APScheduler tick state, dry-run flag, last run
    timestamp, and the in-flight subprocess registry. ``inflight_spawns``
    is populated live from ``meta_agent.spawner.inflight_count_by_type()``
    (Plan 58-05 Task 05).
  - ``POST /api/v1/meta-agent/disable`` (Plan 58-05 Task 04) — single-action
    killswitch trigger. Aborts every in-flight spawn via SIGTERM→3s→SIGKILL
    and removes the meta_agent APScheduler job. Returns within 5 wall-clock
    seconds (SPEC §Requirement 8).

Pattern source: ``api/v1/system.py:213-216`` for the module-level
singleton + setter pattern; ``api/v1/system.py:430-443`` for the
``POST /kill`` exemplar mirrored by ``POST /disable``;
``api/v1/alerts.py:28-32`` for the router header + auth dependency.
"""

from __future__ import annotations

from datetime import datetime  # noqa: TC003 — required at runtime by Pydantic
from typing import TYPE_CHECKING, Literal

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
    """Counts of currently-running spawns by type."""

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


class DisableResponse(BaseModel):
    """Result of the killswitch invocation (SPEC §Requirement 8)."""

    model_config = ConfigDict(frozen=True)
    status: Literal["disabled"]
    aborted_spawns: int
    job_removed: bool


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


@router.post("/disable", response_model=DisableResponse)
async def disable_meta_agent() -> DisableResponse:
    """SPEC §Requirement 8: single-action killswitch.

    Behaviour:
      1. Resolve the wired ``Killswitch`` instance from ``_runner``.
         If the runner / killswitch is not wired, return a no-op
         response (aborted_spawns=0, job_removed=False) — the killswitch
         path must remain idempotent and never raise.
      2. Call ``killswitch.abort_all_inflight()`` to terminate every
         entry in ``spawner._inflight_handles`` via SIGTERM→3s→SIGKILL.
      3. Call ``killswitch.remove_job()`` to remove the meta_agent
         APScheduler job. Idempotent on JobLookupError.
      4. Return a 200 ``DisableResponse`` with the abort + remove counts.

    Wall-clock budget: ≤ 5 s end-to-end (SPEC line 75).
    """
    if _runner is None or getattr(_runner, "killswitch", None) is None:
        _log.warning("meta_agent_disabled_via_api_no_runner")
        return DisableResponse(
            status="disabled",
            aborted_spawns=0,
            job_removed=False,
        )
    killswitch = _runner.killswitch
    aborted = await killswitch.abort_all_inflight()
    job_removed = killswitch.remove_job()
    _log.warning(
        "meta_agent_disabled_via_api",
        aborted_spawns=aborted,
        job_removed=job_removed,
    )
    return DisableResponse(
        status="disabled",
        aborted_spawns=aborted,
        job_removed=job_removed,
    )
