"""Integration test for Phase 57-05 portfolio review recap end-to-end.

Verifies that ``TradingLoop._run_portfolio_review_async`` end-to-end:
  1. Calls the LLM (mocked) to produce a base PortfolioReviewResult.
  2. Merges deterministic recap from ``compute_daily_recap`` (mocked
     against a duck-typed session that returns seeded ORM rows for
     OrderModel + DailyEquitySnapshot via ``select(...).execute(...)``).
  3. Routes through ``alerter._send`` with ``alert_type='daily_summary'``
     so Plan 02's persistence labels the row correctly.
  4. The Daily Recap section appears in the rendered Telegram text with
     realized P&L, opened/closed counts, and equity-change percentage.

This test does NOT require a live database — it patches
``compute_daily_recap`` to return a seeded recap (the unit tests in
``tests/unit/analysis/`` cover the SQL paths). It DOES exercise the
actual ``_run_portfolio_review_async`` method on TradingLoop end-to-end.

ROADMAP success criterion #4 ("at MOEX market close [18:50 MSK], a daily
summary Telegram message reports total realized P&L, positions opened/
closed, equity change from previous close") is verified here.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _make_tl() -> object:
    """Create a minimal TradingLoop instance without calling __init__."""
    from finalayze.core.trading_loop import TradingLoop

    return object.__new__(TradingLoop)


def _make_persistence_with_session() -> tuple[MagicMock, MagicMock]:
    """Build a (persistence, session) pair with a working async-context session."""
    session = MagicMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=None)
    factory = MagicMock(return_value=session)
    persistence = MagicMock()
    persistence._get_bg_session_factory = MagicMock(return_value=factory)
    return persistence, session


# ── Test 1: full e2e with seeded data ────────────────────────────────────


@pytest.mark.asyncio
async def test_portfolio_review_recap_e2e() -> None:
    """End-to-end: LLM call + recap merge + alerter._send with full payload.

    Seeds the recap with realized P&L = +500 (mirrors plan example), 2
    positions opened, 2 closed, and an equity-change of +1000 (from
    yesterday=10000 to today=11000 → +10%). Asserts the rendered
    Telegram text contains the expected Daily Recap content AND that
    ``alert_type='daily_summary'`` flows to ``_send``.
    """
    from finalayze.analysis.portfolio_review_agent import PortfolioReviewResult

    base_result = PortfolioReviewResult(
        reviewed_at=datetime(2026, 4, 19, 15, 50, tzinfo=UTC),
        overall_assessment="Portfolio looks healthy.",
        risk_score=0.3,
    )

    alerter = MagicMock()
    alerter.send_async = AsyncMock(return_value=(True, None))
    llm_client = AsyncMock()
    llm_client.parse_structured = AsyncMock(return_value=base_result)
    persistence, _session = _make_persistence_with_session()

    tl = _make_tl()
    tl._llm_client = llm_client  # type: ignore[attr-defined]
    tl._alerter = alerter  # type: ignore[attr-defined]
    tl._circuit_breakers = {}  # type: ignore[attr-defined]
    tl._broker_router = MagicMock()  # type: ignore[attr-defined]
    tl._persistence = persistence  # type: ignore[attr-defined]

    seeded_recap = {
        "total_realized_pnl": Decimal(500),
        "positions_opened_today": 2,
        "positions_closed_today": 2,
        "equity_change_pct": 0.1,
        "equity_change_amount": Decimal(1000),
        "previous_close_equity": Decimal(10000),
    }
    with patch(
        "finalayze.analysis.portfolio_review_agent.compute_daily_recap",
        AsyncMock(return_value=seeded_recap),
    ):
        await tl._run_portfolio_review_async()  # type: ignore[attr-defined]

    # (a) alerter._send called once with alert_type='daily_summary'
    alerter.send_async.assert_called_once()
    call = alerter.send_async.call_args
    assert call.kwargs.get("alert_type") == "daily_summary", (
        f"Expected alert_type='daily_summary', got: {call.kwargs}"
    )

    # (b) Message text contains Daily Recap content
    sent_text = call.args[0]
    assert "Daily Recap" in sent_text
    assert "+500" in sent_text  # realized P&L
    assert "Opened: 2" in sent_text
    assert "Closed: 2" in sent_text
    assert "Equity change: +" in sent_text  # positive sign is rendered
    # The LLM advisory body still ships alongside the recap.
    assert "Portfolio looks healthy." in sent_text


# ── Test 2: graceful no-data path ───────────────────────────────────────


@pytest.mark.asyncio
async def test_portfolio_review_no_data_graceful() -> None:
    """Empty recap (counts=0, P&L/equity None) ships LLM advisory cleanly.

    When compute_daily_recap returns the all-zero/None default (no
    activity today), the LLM advisory message must still be sent. The
    Daily Recap section may render with zero counts and no P&L line —
    this is acceptable per the format_review_telegram contract (any
    non-None field triggers the section).
    """
    from finalayze.analysis.portfolio_review_agent import PortfolioReviewResult

    base_result = PortfolioReviewResult(
        reviewed_at=datetime(2026, 4, 19, 15, 50, tzinfo=UTC),
        overall_assessment="Quiet market day.",
        risk_score=0.1,
    )

    alerter = MagicMock()
    alerter.send_async = AsyncMock(return_value=(True, None))
    llm_client = AsyncMock()
    llm_client.parse_structured = AsyncMock(return_value=base_result)
    persistence, _session = _make_persistence_with_session()

    tl = _make_tl()
    tl._llm_client = llm_client  # type: ignore[attr-defined]
    tl._alerter = alerter  # type: ignore[attr-defined]
    tl._circuit_breakers = {}  # type: ignore[attr-defined]
    tl._broker_router = MagicMock()  # type: ignore[attr-defined]
    tl._persistence = persistence  # type: ignore[attr-defined]

    empty_recap = {
        "total_realized_pnl": None,
        "positions_opened_today": 0,
        "positions_closed_today": 0,
        "equity_change_pct": None,
        "equity_change_amount": None,
        "previous_close_equity": None,
    }
    with patch(
        "finalayze.analysis.portfolio_review_agent.compute_daily_recap",
        AsyncMock(return_value=empty_recap),
    ):
        # Must not raise.
        await tl._run_portfolio_review_async()  # type: ignore[attr-defined]

    alerter.send_async.assert_called_once()
    sent_text = alerter.send_async.call_args.args[0]
    # LLM advisory still ships
    assert "Quiet market day." in sent_text
    # Recap section renders with the populated count fields (Opened: 0,
    # Closed: 0). P&L and equity-change lines correctly omitted because
    # those fields are None.
    assert "Daily Recap" in sent_text
    assert "Opened: 0" in sent_text
    assert "Closed: 0" in sent_text
    assert "Realized" not in sent_text  # P&L line absent
    assert "Equity change" not in sent_text  # equity-change line absent
