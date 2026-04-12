"""Debates REST endpoints (Layer 6).

ORCH-02: Manual pipeline invocation via REST without writing Python.
Provides endpoints for creating debates from conflicting AgentOutputs,
listing debates, and retrieving debate details.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

from finalayze.api.v1.auth import api_key_auth
from finalayze.core.debate_manager import DebateManager
from finalayze.core.schemas import AgentOutput
from finalayze.orchestration.agent_orchestrator import AgentOrchestrator

router = APIRouter(
    prefix="/debates",
    tags=["debates"],
    dependencies=[Depends(api_key_auth)],
)


# ── Request / Response models ─────────────────────────────────────────────────


class CreateDebateRequest(BaseModel):
    """Request body for POST /debates."""

    model_config = ConfigDict(frozen=True)

    outputs: list[AgentOutput]


class CreateDebateResponse(BaseModel):
    """Response for POST /debates."""

    model_config = ConfigDict(frozen=True)

    debate_id: str | None
    conflicts_found: int


class DebateListResponse(BaseModel):
    """Response for GET /debates."""

    model_config = ConfigDict(frozen=True)

    debate_ids: list[str]


class DebateDetailResponse(BaseModel):
    """Response for GET /debates/{debate_id}."""

    model_config = ConfigDict(frozen=True)

    debate_id: str
    topic: str
    status: str
    created: str
    agents: list[str]
    resolution: str | None
    experiment_id: str | None
    has_arbiter_report: bool


# ── Endpoints ─────────────────────────────────────────────────────────────────


@router.post("", response_model=CreateDebateResponse)
async def create_debate(req: CreateDebateRequest) -> JSONResponse:
    """Run the conflict-to-debate pipeline on the provided agent outputs.

    Returns 201 if at least one conflict was detected and a debate was created.
    Returns 200 if no conflicts were found (no debate created).
    """
    # Fresh AgentOrchestrator per request avoids stale ConflictDetector state
    orch = AgentOrchestrator()
    debate_ids = orch.run(list(req.outputs))

    payload = CreateDebateResponse(
        debate_id=debate_ids[0] if debate_ids else None,
        conflicts_found=len(debate_ids),
    )
    status_code = 201 if debate_ids else 200
    return JSONResponse(content=payload.model_dump(), status_code=status_code)


@router.get("", response_model=DebateListResponse)
async def list_debates() -> DebateListResponse:
    """Return the list of all debate IDs."""
    # Fresh DebateManager per request avoids CWD-relative path issues
    dm = DebateManager()
    return DebateListResponse(debate_ids=dm.list_debates())


@router.get("/{debate_id}", response_model=DebateDetailResponse)
async def get_debate(debate_id: str) -> DebateDetailResponse:
    """Return debate detail for the given debate_id.

    Raises:
        HTTPException(404): if the debate does not exist.
    """
    dm = DebateManager()
    try:
        state = dm.read_debate(debate_id)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404, detail=f"Debate not found: {debate_id!r}"
        ) from exc

    return DebateDetailResponse(
        debate_id=state.debate_id,
        topic=state.topic,
        status=str(state.status),
        created=state.created,
        agents=list(state.agents),
        resolution=state.resolution,
        experiment_id=state.experiment_id,
        has_arbiter_report=state.arbiter_report is not None,
    )
