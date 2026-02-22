"""System endpoints: health check and mode management.

Layer 6 -- API layer. Depends on Layer 0 (exceptions, modes).

The ``ModeManager`` is provided via FastAPI dependency injection so that
tests can substitute a fresh instance without module reloading.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from finalayze.core.exceptions import ModeError
from finalayze.core.modes import ModeManager, WorkMode

router = APIRouter(tags=["system"])

# Application-scoped singleton (overridden in tests via dependency overrides)
_default_mode_manager = ModeManager()

APP_VERSION = "0.1.0"


def get_mode_manager() -> ModeManager:
    """Dependency that returns the application-wide ModeManager."""
    return _default_mode_manager


class HealthResponse(BaseModel):
    """Response model for the health endpoint."""

    status: str
    mode: str
    version: str


class ModeResponse(BaseModel):
    """Response model for mode endpoints."""

    mode: str


class ModeRequest(BaseModel):
    """Request model for changing the work mode."""

    mode: WorkMode


@router.get("/health", response_model=HealthResponse)
async def health(
    mgr: Annotated[ModeManager, Depends(get_mode_manager)],
) -> HealthResponse:
    """Return the current system health and operational mode."""
    return HealthResponse(
        status="ok",
        mode=str(mgr.current_mode),
        version=APP_VERSION,
    )


@router.get("/mode", response_model=ModeResponse)
async def get_mode(
    mgr: Annotated[ModeManager, Depends(get_mode_manager)],
) -> ModeResponse:
    """Return the current work mode."""
    return ModeResponse(mode=str(mgr.current_mode))


@router.post("/mode", response_model=ModeResponse)
async def set_mode(
    request: ModeRequest,
    mgr: Annotated[ModeManager, Depends(get_mode_manager)],
) -> ModeResponse:
    """Change the work mode.

    Raises:
        HTTPException(400): When transitioning to REAL mode without
            ``FINALAYZE_REAL_CONFIRMED=true`` environment variable set.
    """
    try:
        mgr.transition_to(request.mode)
    except ModeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ModeResponse(mode=str(mgr.current_mode))
