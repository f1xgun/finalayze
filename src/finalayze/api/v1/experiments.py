"""Experiments REST endpoints (Layer 6).

ORCH-02: Manual pipeline invocation via REST without writing Python.
Provides endpoints for listing experiments, retrieving experiment details,
and applying accepted verdicts to strategy YAML presets (Phase 38).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict

from finalayze.api.alerts import TelegramAlerter
from finalayze.api.v1.auth import api_key_auth
from finalayze.core.experiment_manager import ExperimentManager
from finalayze.risk.circuit_breaker import CircuitBreaker

_log = logging.getLogger(__name__)

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


class ApplyResultResponse(BaseModel):
    """Response for POST /experiments/{experiment_id}/apply."""

    model_config = ConfigDict(frozen=True)

    experiment_id: str
    applied: bool
    backup_path: str | None
    verdict: str
    reason: str


class ApplyRequest(BaseModel):
    """Request body for POST /experiments/{experiment_id}/apply."""

    model_config = ConfigDict(frozen=True)

    market_id: str = "moex"


# Phase 38 limitation: REST endpoint lacks access to TradingLoop runtime state.
# - circuit_breakers=_get_circuit_breakers(): Creates a new CircuitBreaker("moex")
#   at NORMAL level. This is independent of TradingLoop circuit breaker state —
#   full integration requires sharing state via Redis or DB (future enhancement).
#   The key improvement is: the code path exercises a real CircuitBreaker object
#   instead of the previous empty dict which silently skipped the gate.
# - entry_strategy_getter=lambda: {}: Position ownership check (APPLY-01
#   step 8) cannot detect open positions from REST. TradingLoop callers
#   will inject their real get_entry_strategies() method.
# - combiner=None: No StrategyCombiner instance available in REST context.
#   Cache invalidation is skipped; combiner reads from disk on every call
#   anyway, so this is safe.
_PRESETS_DIR = Path("src/finalayze/strategies/presets")


def _make_alerter() -> TelegramAlerter:
    """Return a TelegramAlerter configured from settings.

    When telegram_bot_token is empty string, TelegramAlerter already returns
    immediately on all send_* calls without making any network requests — so
    no special no-op class is needed.
    """
    import config.settings as _settings_mod  # noqa: PLC0415

    s = _settings_mod.get_settings()
    return TelegramAlerter(bot_token=s.telegram_bot_token, chat_id=s.telegram_chat_id)


def _get_circuit_breakers() -> dict[str, CircuitBreaker]:
    """Return a dict with a fresh CircuitBreaker for the MOEX market.

    NOTE: This CircuitBreaker starts at NORMAL level (no drawdown history).
    It is independent of the TradingLoop runtime state. Full real-time
    integration requires persisting circuit breaker state to Redis or the DB,
    which is out of scope for REST gap closure.
    """
    _log.warning(
        "REST circuit_breakers is independent of TradingLoop state — "
        "starts at NORMAL level with no drawdown history"
    )
    cb = CircuitBreaker("moex")
    return {"moex": cb}


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


@router.post("/{experiment_id}/apply", response_model=ApplyResultResponse)
async def apply_experiment(
    experiment_id: str,
    body: ApplyRequest | None = None,
) -> ApplyResultResponse:
    """Apply an accepted experiment verdict to the strategy YAML preset.

    Invokes PresetApplicator with Phase 38 limitations documented above:
    circuit breaker starts at NORMAL (independent of TradingLoop state),
    position ownership check is skipped (no REST access to open positions),
    and combiner cache invalidation is skipped.

    Args:
        experiment_id: The experiment to apply.
        body: Optional request body with market_id (defaults to 'moex').

    Returns:
        ApplyResultResponse with applied status and backup path.

    Raises:
        HTTPException(404): if the experiment does not exist.
        HTTPException(409): if a safety gate blocks the apply.
        HTTPException(422): if preset_overrides fail validation.
    """
    from finalayze.core.db import get_async_session_factory  # noqa: PLC0415
    from finalayze.orchestration.preset_applicator import (  # noqa: PLC0415
        PresetApplicator,
        PresetApplyBlockedError,
        PresetValidationError,
        SandboxGate,
    )

    market_id = body.market_id if body is not None else "moex"

    em = ExperimentManager()
    applicator = PresetApplicator(
        circuit_breakers=_get_circuit_breakers(),
        alerter=_make_alerter(),
        experiment_manager=em,
        presets_dir=_PRESETS_DIR,
        sandbox_gate=SandboxGate(),
        entry_strategy_getter=lambda: {},
        combiner=None,
    )

    factory = get_async_session_factory()
    try:
        async with factory() as session:
            result = await applicator.apply_verdict(experiment_id, market_id, session)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404, detail=f"Experiment not found: {experiment_id!r}"
        ) from exc
    except PresetApplyBlockedError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except PresetValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return ApplyResultResponse(
        experiment_id=result.experiment_id,
        applied=result.applied,
        backup_path=result.backup_path,
        verdict=result.verdict,
        reason=result.reason,
    )
