"""ML model status endpoints (Layer 6)."""

from __future__ import annotations

from config.settings import Settings
from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict

from finalayze.api.v1.auth import require_api_key

_settings = Settings()
router = APIRouter(
    prefix="/ml",
    tags=["ml"],
    dependencies=[Depends(require_api_key(_settings.api_key))],
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
