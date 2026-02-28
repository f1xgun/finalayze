"""ML model status endpoints (Layer 6)."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict

from finalayze.api.v1.auth import api_key_auth

router = APIRouter(
    prefix="/ml",
    tags=["ml"],
    dependencies=[Depends(api_key_auth)],
)


class ModelStatus(BaseModel):
    model_config = ConfigDict(frozen=True)
    segment_id: str
    model_type: str
    last_retrain: str | None
    prediction_latency_p50_ms: float | None
    is_stale: bool


class MLStatusResponse(BaseModel):
    model_config = ConfigDict(frozen=True)
    models: list[ModelStatus]


@router.get("/status", response_model=MLStatusResponse)
async def ml_status() -> MLStatusResponse:
    return MLStatusResponse(models=[])
