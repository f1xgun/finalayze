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

from datetime import datetime, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING

import structlog
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

_log = structlog.get_logger()

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

    # ALRT-04 (Phase 57 D-01): deterministic daily recap fields.
    # MUST remain Optional — additive-only per Pitfall 4 so existing
    # kwargs-only callers (LLM parse_structured + tests) keep working.
    total_realized_pnl: Decimal | None = None
    positions_opened_today: int | None = None
    positions_closed_today: int | None = None
    equity_change_pct: float | None = None  # vs previous close
    equity_change_amount: Decimal | None = None
    previous_close_equity: Decimal | None = None


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

    # ALRT-04 (Phase 57 D-01): Daily Recap section.
    # Emitted only when at least one recap field is populated. HTML
    # ampersand escape (&amp;) matches Plan 02 Task 2 convention so the
    # message renders cleanly with parse_mode='HTML' on Telegram.
    if (
        result.total_realized_pnl is not None
        or result.equity_change_pct is not None
        or result.positions_opened_today is not None
        or result.positions_closed_today is not None
    ):
        sections.append("")
        sections.append("Daily Recap")
        if result.total_realized_pnl is not None:
            sign = "+" if result.total_realized_pnl >= 0 else ""
            sections.append(
                f"  Realized P&amp;L: {sign}{result.total_realized_pnl:.2f}",
            )
        if result.positions_opened_today is not None:
            sections.append(f"  Opened: {result.positions_opened_today}")
        if result.positions_closed_today is not None:
            sections.append(f"  Closed: {result.positions_closed_today}")
        if result.equity_change_pct is not None:
            sections.append(
                f"  Equity change: {result.equity_change_pct:+.2%}",
            )

    return "\n".join(sections)


# ── Daily recap helper (ALRT-04, Phase 57 D-01) ───────────────────────────


async def compute_daily_recap(
    session: AsyncSession,
    now: datetime,
) -> dict[str, object]:
    """Compute deterministic daily-recap values for ALRT-04.

    Returns a dict with keys matching ``PortfolioReviewResult`` recap
    fields. Never raises — returns ``None``/zero defaults on query
    failure so the LLM advisory message is never blocked.

    Realized P&L is derived from ``fifo_pair(orders)``: sum of
    ``(exit_price - entry_price) * quantity`` across closed round-trips
    today. ``DailyEquitySnapshot`` is used ONLY for equity_change (column
    name: ``equity`` — there is no ``total_equity`` or ``realized_pnl``
    column on this table). Equity sums across markets — currency mixing
    (RUB + USD) is acceptable for the directional-change signal but
    documented in the SUMMARY.
    """
    from sqlalchemy import and_, func, select  # noqa: PLC0415

    from finalayze.api.v1._fifo import fifo_pair  # noqa: PLC0415
    from finalayze.core.models import (  # noqa: PLC0415
        DailyEquitySnapshot,
        OrderModel,
    )

    today_start = datetime(
        now.year,
        now.month,
        now.day,
        tzinfo=now.tzinfo,
    )
    yesterday_start = today_start - timedelta(days=1)

    result: dict[str, object] = {
        "total_realized_pnl": None,
        "positions_opened_today": 0,
        "positions_closed_today": 0,
        "equity_change_pct": None,
        "equity_change_amount": None,
        "previous_close_equity": None,
    }

    try:
        # Positions opened/closed today from OrderModel.
        buy_count = (
            await session.execute(
                select(func.count())
                .select_from(OrderModel)
                .where(
                    and_(
                        OrderModel.filled_at >= today_start,
                        OrderModel.side == "BUY",
                    ),
                ),
            )
        ).scalar() or 0
        sell_count = (
            await session.execute(
                select(func.count())
                .select_from(OrderModel)
                .where(
                    and_(
                        OrderModel.filled_at >= today_start,
                        OrderModel.side == "SELL",
                    ),
                ),
            )
        ).scalar() or 0
        result["positions_opened_today"] = int(buy_count)
        result["positions_closed_today"] = int(sell_count)

        # Total realized P&L via FIFO pairing of today's orders.
        # PairedTrade has NO realized_pnl field — compute inline as
        # (exit_price - entry_price) * quantity. fifo_pair returns an
        # Iterator, so materialise with list() (defensive — keeps the
        # pattern safe against future double-iteration changes).
        today_orders_rows = (
            (
                await session.execute(
                    select(OrderModel)
                    .where(OrderModel.filled_at >= today_start)
                    .order_by(OrderModel.filled_at.asc()),
                )
            )
            .scalars()
            .all()
        )
        if today_orders_rows:
            paired = list(fifo_pair(list(today_orders_rows)))
            realized = sum(
                ((p.exit_price - p.entry_price) * p.quantity for p in paired),
                Decimal(0),
            )
            result["total_realized_pnl"] = realized

        # Equity change: yesterday end-of-day vs today's latest snapshot.
        # DailyEquitySnapshot column is `equity` (NOT `total_equity`).
        today_equity = (
            await session.execute(
                select(func.sum(DailyEquitySnapshot.equity)).where(
                    DailyEquitySnapshot.timestamp >= today_start,
                ),
            )
        ).scalar()
        yesterday_equity = (
            await session.execute(
                select(func.sum(DailyEquitySnapshot.equity)).where(
                    and_(
                        DailyEquitySnapshot.timestamp >= yesterday_start,
                        DailyEquitySnapshot.timestamp < today_start,
                    ),
                ),
            )
        ).scalar()
        if today_equity is not None and yesterday_equity is not None and yesterday_equity != 0:
            change_amt = Decimal(str(today_equity)) - Decimal(
                str(yesterday_equity),
            )
            result["equity_change_amount"] = change_amt
            result["previous_close_equity"] = Decimal(str(yesterday_equity))
            result["equity_change_pct"] = float(
                change_amt / Decimal(str(yesterday_equity)),
            )
    except Exception:
        _log.warning("compute_daily_recap_failed", exc_info=True)
    return result
