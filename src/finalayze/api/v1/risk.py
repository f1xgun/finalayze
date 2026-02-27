"""Risk endpoints (Layer 6)."""

from __future__ import annotations

from config.settings import Settings
from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, ConfigDict

from finalayze.api.v1.auth import require_api_key
from finalayze.risk.circuit_breaker import CircuitLevel

_settings = Settings()
router = APIRouter(
    prefix="/risk",
    tags=["risk"],
    dependencies=[Depends(require_api_key(_settings.api_key))],
)

_LEVEL_MAP: dict[CircuitLevel, tuple[int, str]] = {
    CircuitLevel.NORMAL: (0, "NORMAL"),
    CircuitLevel.CAUTION: (1, "CAUTION"),
    CircuitLevel.HALTED: (2, "HALTED"),
    CircuitLevel.LIQUIDATE: (3, "LIQUIDATE"),
}

_LEVEL_LIST: list[CircuitLevel] = [
    CircuitLevel.NORMAL,
    CircuitLevel.CAUTION,
    CircuitLevel.HALTED,
    CircuitLevel.LIQUIDATE,
]


class MarketRiskStatus(BaseModel):
    model_config = ConfigDict(frozen=True)
    market_id: str
    circuit_breaker_level: int
    level_label: str
    level_since: str | None


class RiskStatusResponse(BaseModel):
    model_config = ConfigDict(frozen=True)
    markets: list[MarketRiskStatus]
    cross_market_halted: bool


class SegmentExposure(BaseModel):
    model_config = ConfigDict(frozen=True)
    segment_id: str
    market_id: str
    value_usd: float
    pct_of_portfolio: float


class ExposureResponse(BaseModel):
    model_config = ConfigDict(frozen=True)
    segments: list[SegmentExposure]
    total_invested_pct: float


class OverrideRequest(BaseModel):
    market_id: str
    level: int


class OverrideResponse(BaseModel):
    model_config = ConfigDict(frozen=True)
    market_id: str
    level: int
    applied: bool


@router.get("/status", response_model=RiskStatusResponse)
async def risk_status(request: Request) -> RiskStatusResponse:
    circuit_breakers = getattr(request.app.state, "circuit_breakers", {})
    markets: list[MarketRiskStatus] = []
    for market_id in ["us", "moex"]:
        cb = circuit_breakers.get(market_id)
        if cb is not None:
            lvl_int, lvl_label = _LEVEL_MAP.get(cb.current_level, (0, "NORMAL"))
            markets.append(
                MarketRiskStatus(
                    market_id=market_id,
                    circuit_breaker_level=lvl_int,
                    level_label=lvl_label,
                    level_since=None,
                )
            )
        else:
            markets.append(
                MarketRiskStatus(
                    market_id=market_id,
                    circuit_breaker_level=0,
                    level_label="NORMAL",
                    level_since=None,
                )
            )
    return RiskStatusResponse(markets=markets, cross_market_halted=False)


@router.get("/exposure", response_model=ExposureResponse)
async def risk_exposure() -> ExposureResponse:
    return ExposureResponse(segments=[], total_invested_pct=0.0)


@router.post("/override", response_model=OverrideResponse)
async def risk_override(req: OverrideRequest, request: Request) -> OverrideResponse:
    circuit_breakers = getattr(request.app.state, "circuit_breakers", {})
    cb = circuit_breakers.get(req.market_id)
    if cb is None:
        return OverrideResponse(market_id=req.market_id, level=req.level, applied=False)
    if 0 <= req.level < len(_LEVEL_LIST):
        cb._level = _LEVEL_LIST[req.level]
    return OverrideResponse(market_id=req.market_id, level=req.level, applied=True)
