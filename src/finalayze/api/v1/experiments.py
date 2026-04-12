"""Experiments REST endpoints (Layer 6).

ORCH-02: Manual pipeline invocation via REST without writing Python.
Provides READ-ONLY endpoints for listing experiments and retrieving
experiment details. Write operations are deferred to Phase 38.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict

from finalayze.api.v1.auth import api_key_auth
from finalayze.core.experiment_manager import ExperimentManager

router = APIRouter(
    prefix="/experiments",
    tags=["experiments"],
    dependencies=[Depends(api_key_auth)],
)


# ── Response models ───────────────────────────────────────────────────────────


class ExperimentListResponse(BaseModel):
    """Response for GET /experiments."""

    model_config = ConfigDict(frozen=True)

    experiment_ids: list[str]


class ExperimentDetailResponse(BaseModel):
    """Response for GET /experiments/{experiment_id}."""

    model_config = ConfigDict(frozen=True)

    experiment_id: str
    hypothesis: str
    status: str
    verdict: str | None
    reasoning: str | None
    debate_id: str | None
    created: str
    success_criteria: dict[str, Any]
    results: list[dict[str, Any]]
    preset_overrides: dict[str, Any] | None


# ── Endpoints ─────────────────────────────────────────────────────────────────


@router.get("", response_model=ExperimentListResponse)
async def list_experiments() -> ExperimentListResponse:
    """Return the list of all experiment IDs."""
    # Fresh ExperimentManager per request avoids CWD-relative path issues
    em = ExperimentManager()
    return ExperimentListResponse(experiment_ids=em.list_experiments())


@router.get("/{experiment_id}", response_model=ExperimentDetailResponse)
async def get_experiment(experiment_id: str) -> ExperimentDetailResponse:
    """Return experiment detail for the given experiment_id.

    Raises:
        HTTPException(404): if the experiment does not exist.
    """
    em = ExperimentManager()
    try:
        state = em.read_experiment(experiment_id)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404, detail=f"Experiment not found: {experiment_id!r}"
        ) from exc

    return ExperimentDetailResponse(
        experiment_id=state.experiment_id,
        hypothesis=state.hypothesis,
        status=str(state.status),
        verdict=state.verdict,
        reasoning=state.reasoning,
        debate_id=state.debate_id,
        created=state.created,
        success_criteria={
            "metric": state.success_criteria.metric,
            "threshold": state.success_criteria.threshold,
            "operator": state.success_criteria.operator,
        },
        results=[
            {
                "run_name": r.run_name,
                "iteration_name": r.iteration_name,
                "metrics": dict(r.metrics),
            }
            for r in state.results
        ],
        preset_overrides=dict(state.preset_overrides) if state.preset_overrides else None,
    )
