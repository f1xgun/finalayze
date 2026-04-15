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

# ── Constants ─────────────────────────────────────────────────────────────

PORTFOLIO_REVIEW_SYSTEM_PROMPT: str = (
    "You are a portfolio risk analyst. Analyze the given portfolio positions "
    "and return a structured JSON matching the PortfolioReviewResult schema.\n\n"
    "Required JSON fields:\n"
    '  "reviewed_at": ISO 8601 datetime string\n'
    '  "positions": list of {ticker, market, quantity, unrealized_pnl, pct_of_portfolio}\n'
    '  "concentration_warnings": list of {ticker, market, concentration_pct, warning_level}\n'
    '  "catalyst_events": list of {ticker, event_type, expected_date}\n'
    '  "overall_assessment": brief narrative (NOT a trade recommendation)\n'
    '  "risk_score": float 0.0-1.0 advisory risk level\n\n'
    "Do NOT give trade directives. Do NOT include direction, confidence, or side fields.\n"
    "Focus on concentration risk, upcoming catalysts, and overall risk assessment."
)

REVIEW_LLM_TIMEOUT: float = 60.0


# ── Prompt builder ────────────────────────────────────────────────────────


def build_review_prompt(portfolio_data: dict[str, object]) -> str:
    """Build an LLM prompt from portfolio data gathered by TradingLoop.

    Args:
        portfolio_data: Dict keyed by market_id, values are dicts with
            equity (Decimal), cash (Decimal), positions (dict[str, Decimal]).

    Returns:
        Formatted prompt string for LLM portfolio review.
    """
    if not portfolio_data:
        return (
            "Portfolio Review Request\n\n"
            "No positions are currently open across any market.\n"
            "Provide a brief assessment noting the empty portfolio state."
        )

    lines: list[str] = ["Portfolio Review Request\n"]

    for market_id, market_info in portfolio_data.items():
        if not isinstance(market_info, dict):
            continue

        equity = market_info.get("equity", 0)
        cash = market_info.get("cash", 0)
        positions = market_info.get("positions", {})

        lines.append(f"Market: {market_id}")
        lines.append(f"  Equity: {equity}")
        lines.append(f"  Cash: {cash}")

        if isinstance(positions, dict) and positions:
            lines.append("  Positions:")
            for ticker, qty in positions.items():
                lines.append(f"    {ticker}: {qty} shares")
        else:
            lines.append("  Positions: none")

        lines.append("")

    lines.append(
        "Analyze the above portfolio. Identify concentration risks, "
        "upcoming catalysts, and provide an overall risk assessment."
    )

    return "\n".join(lines)


# ── Telegram formatter ────────────────────────────────────────────────────


def format_review_telegram(result: PortfolioReviewResult) -> str:
    """Format a PortfolioReviewResult into a structured Telegram message.

    Uses plain text (not Markdown) for robustness in Telegram rendering.
    Sections: header, positions, concentration warnings, catalysts, assessment.

    Args:
        result: The portfolio review result from LLM analysis.

    Returns:
        Multi-section plain text message suitable for Telegram.
    """
    sections: list[str] = []

    # Header
    reviewed_str = result.reviewed_at.strftime("%Y-%m-%d %H:%M UTC")
    sections.append(f"Portfolio Review -- {reviewed_str}")
    sections.append("")

    # Positions
    if result.positions:
        sections.append(f"Positions ({len(result.positions)} open)")
        for pos in result.positions:
            pnl_sign = "+" if pos.unrealized_pnl >= 0 else ""
            pct_display = f"{pos.pct_of_portfolio:.0%}"
            sections.append(
                f"  {pos.ticker} [{pos.market}] x{pos.quantity} | "
                f"PnL: {pnl_sign}{pos.unrealized_pnl} | {pct_display}"
            )
    else:
        sections.append("Positions: No open positions")

    sections.append("")

    # Concentration warnings
    if result.concentration_warnings:
        sections.append("Concentration Risk")
        sections.extend(
            f"  {warn.ticker} [{warn.market}]: {warn.concentration_pct:.0%} -- {warn.warning_level}"
            for warn in result.concentration_warnings
        )
    else:
        sections.append("Concentration Risk: No warnings")

    sections.append("")

    # Catalyst events
    if result.catalyst_events:
        sections.append("Upcoming Catalysts")
        sections.extend(
            f"  {cat.ticker} -- {cat.event_type} ({cat.expected_date})"
            for cat in result.catalyst_events
        )
    else:
        sections.append("Upcoming Catalysts: None")

    sections.append("")

    # Assessment and risk score
    sections.append(f"Assessment: {result.overall_assessment}")
    sections.append(f"Risk Score: {result.risk_score:.2f}")

    return "\n".join(sections)
