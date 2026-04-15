"""Advisory portfolio review agent (Layer 3).

Provides the PortfolioReviewResult schema, sub-schemas, prompt builder,
and Telegram formatter for daily LLM portfolio analysis.

This module is advisory-only: it has no write path to the order pipeline.
The schema is structurally incompatible with Signal (no direction/confidence,
uses ticker/market instead of symbol/market_id) and OrderRequest (no side).

Safety enforced at three levels:
1. Schema design -- forbidden fields absent by construction
2. Module-level assertion -- _FORBIDDEN_FIELDS checked at import time
3. Code-grep test -- zero order-pipeline references in this file

See docs/architecture/DEPENDENCY_LAYERS.md for layering rules.
"""

from __future__ import annotations

from datetime import datetime  # noqa: TC003
from decimal import Decimal  # noqa: TC003

from pydantic import BaseModel, ConfigDict, Field

# ── Sub-schemas (all frozen Pydantic v2 BaseModel) ─────────────────────────


class PositionSummary(BaseModel):
    """Advisory summary of one open position."""

    model_config = ConfigDict(frozen=True)

    ticker: str  # NOT "symbol" — avoids Signal field name
    market: str  # NOT "market_id" — avoids Signal field name
    quantity: Decimal
    unrealized_pnl: Decimal
    pct_of_portfolio: float  # 0.0-1.0


class ConcentrationWarning(BaseModel):
    """A concentration risk flag."""

    model_config = ConfigDict(frozen=True)

    ticker: str
    market: str
    concentration_pct: float  # e.g. 0.25 = 25% of portfolio
    warning_level: str  # "HIGH" | "MEDIUM" — NOT a trade direction


class CatalystEvent(BaseModel):
    """Upcoming event that may affect a position."""

    model_config = ConfigDict(frozen=True)

    ticker: str
    event_type: str  # "earnings" | "cbr_meeting" | "dividend" | "other"
    expected_date: str  # ISO date string


# ── Main schema ────────────────────────────────────────────────────────────


class PortfolioReviewResult(BaseModel):
    """Advisory-only portfolio analysis from LLM.

    SAFETY INVARIANT: This schema has no field named 'direction', 'confidence',
    or 'side', and does not have a (symbol, market_id) pair — making it
    structurally incompatible with Signal and OrderRequest.
    """

    model_config = ConfigDict(frozen=True)

    reviewed_at: datetime
    positions: list[PositionSummary] = Field(default_factory=list)
    concentration_warnings: list[ConcentrationWarning] = Field(default_factory=list)
    catalyst_events: list[CatalystEvent] = Field(default_factory=list)
    overall_assessment: str  # brief narrative, NOT a trade recommendation
    risk_score: float  # 0.0-1.0 advisory risk level, NOT confidence


# ── Module-level safety assertion (PFRA-03) ────────────────────────────────

_FORBIDDEN_FIELDS = {"direction", "confidence", "side"}
_review_fields = set(PortfolioReviewResult.model_fields)
assert not (_review_fields & _FORBIDDEN_FIELDS), (
    f"PortfolioReviewResult has forbidden trade-directive fields: "
    f"{_review_fields & _FORBIDDEN_FIELDS}"
)
