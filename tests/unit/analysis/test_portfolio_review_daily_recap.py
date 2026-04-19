"""Tests for ALRT-04 daily recap fields on PortfolioReviewResult.

Phase 57 Plan 05 (D-01 Option A): the v10.0 PortfolioReviewAgent's
single advisory Telegram message is extended with a deterministic
"Daily Recap" block (realized P&L, positions opened/closed, equity
change vs previous close).

Tests cover:
  1-3) PortfolioReviewResult schema extension (additive, all Optional).
  4-5) format_review_telegram emits a Daily Recap section.
  6-7) compute_daily_recap helper queries OrderModel + DailyEquitySnapshot.

The recap fields are Optional and default to None — must remain
backward-compatible with existing call sites (Pitfall 4).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest

from finalayze.analysis.portfolio_review_agent import (
    PortfolioReviewResult,
    format_review_telegram,
)

if TYPE_CHECKING:
    pass


# ── Test 1-3: PortfolioReviewResult schema extension ─────────────────────


class TestRecapFieldsSchema:
    """6 Optional recap fields are added to PortfolioReviewResult, default None."""

    def test_recap_fields_default_to_none(self) -> None:
        """All 6 recap fields default to None when not supplied."""
        result = PortfolioReviewResult(
            reviewed_at=datetime.now(tz=UTC),
            overall_assessment="ok",
            risk_score=0.5,
        )
        assert result.total_realized_pnl is None
        assert result.positions_opened_today is None
        assert result.positions_closed_today is None
        assert result.equity_change_pct is None
        assert result.equity_change_amount is None
        assert result.previous_close_equity is None

    def test_recap_fields_accept_values(self) -> None:
        """All 6 recap fields round-trip when explicitly supplied."""
        result = PortfolioReviewResult(
            reviewed_at=datetime(2026, 4, 19, 15, 50, tzinfo=UTC),
            overall_assessment="Daily summary",
            risk_score=0.4,
            total_realized_pnl=Decimal("1250.50"),
            positions_opened_today=3,
            positions_closed_today=2,
            equity_change_pct=0.021,
            equity_change_amount=Decimal("2800"),
            previous_close_equity=Decimal("130000"),
        )
        assert result.total_realized_pnl == Decimal("1250.50")
        assert result.positions_opened_today == 3
        assert result.positions_closed_today == 2
        assert result.equity_change_pct == pytest.approx(0.021)
        assert result.equity_change_amount == Decimal("2800")
        assert result.previous_close_equity == Decimal("130000")

    def test_existing_signature_backward_compat(self) -> None:
        """Existing keyword construction (no recap fields) still works.

        Mirrors the construction patterns in
        tests/unit/test_portfolio_review_agent.py and
        tests/unit/test_portfolio_review_integration.py.
        """
        # Mirror test_portfolio_review_agent.py:117 keyword construction.
        result = PortfolioReviewResult(
            reviewed_at=datetime.now(tz=UTC),
            positions=[],
            concentration_warnings=[],
            catalyst_events=[],
            overall_assessment="Portfolio is well-balanced.",
            risk_score=0.3,
        )
        assert result.overall_assessment == "Portfolio is well-balanced."
        assert result.risk_score == pytest.approx(0.3)
        # Recap fields silently default to None — additive, no breakage.
        assert result.total_realized_pnl is None
        assert result.positions_opened_today is None


# ── Test 4-5: format_review_telegram Daily Recap section ─────────────────


def _base_result() -> PortfolioReviewResult:
    """Return a minimal valid result with no recap fields populated."""
    return PortfolioReviewResult(
        reviewed_at=datetime(2026, 4, 19, 15, 50, tzinfo=UTC),
        overall_assessment="No notable risks.",
        risk_score=0.2,
    )


class TestFormatTelegramRecapSection:
    """format_review_telegram emits Daily Recap iff any recap field is set."""

    def test_format_review_telegram_no_recap_omits_section(self) -> None:
        """When all recap fields are None, no Daily Recap section appears."""
        msg = format_review_telegram(_base_result())
        assert "Daily Recap" not in msg

    def test_format_review_telegram_with_recap_includes_section(self) -> None:
        """When recap fields populated, Daily Recap section is present."""
        result = PortfolioReviewResult(
            reviewed_at=datetime(2026, 4, 19, 15, 50, tzinfo=UTC),
            overall_assessment="Daily summary",
            risk_score=0.4,
            total_realized_pnl=Decimal("1250.50"),
            positions_opened_today=3,
            positions_closed_today=2,
            equity_change_pct=0.021,
            equity_change_amount=Decimal("2800"),
            previous_close_equity=Decimal("130000"),
        )
        msg = format_review_telegram(result)
        assert "Daily Recap" in msg
        # Realized P&L renders with sign and 2-decimal formatting
        assert "+1250.50" in msg
        # Position counts render
        assert "Opened: 3" in msg
        assert "Closed: 2" in msg
        # Equity change renders as percent with sign
        assert "+2.10%" in msg


# ── Test 6-7: compute_daily_recap helper ─────────────────────────────────


@pytest.fixture()
def now_utc() -> datetime:
    """A fixed 'now' anchored to 2026-04-19 15:50 UTC."""
    return datetime(2026, 4, 19, 15, 50, tzinfo=UTC)


def _mock_session_with_results(*, scalar_returns: list) -> AsyncMock:
    """Build an AsyncSession-like mock returning the given scalars in order."""
    session = MagicMock()
    # Each .execute() call must return an object with .scalar() (for counts /
    # sums) and .scalars().all() (for ORM rows). We dispatch based on the
    # configured side_effect list.
    results = list(scalar_returns)

    def _execute(_stmt: object) -> MagicMock:
        # Pop the next scalar value; wrap it in a result-like mock.
        if results:
            current = results.pop(0)
        else:
            current = None
        result = MagicMock()
        if isinstance(current, list):
            # ORM row collection (for .scalars().all())
            result.scalars.return_value.all.return_value = current
            result.scalar.return_value = None
        else:
            result.scalar.return_value = current
            result.scalars.return_value.all.return_value = []
        return result

    session.execute = AsyncMock(side_effect=_execute)
    return session


class TestComputeDailyRecap:
    """compute_daily_recap returns dict of recap values; never raises."""

    @pytest.mark.asyncio
    async def test_compute_daily_recap_empty_db(self, now_utc: datetime) -> None:
        """Empty DB returns counts=0, P&L/equity values None."""
        from finalayze.analysis.portfolio_review_agent import compute_daily_recap

        # Order: buy_count, sell_count, today_orders_rows([]), today_eq, yest_eq
        session = _mock_session_with_results(
            scalar_returns=[0, 0, [], None, None],
        )

        recap = await compute_daily_recap(session, now_utc)
        assert recap["positions_opened_today"] == 0
        assert recap["positions_closed_today"] == 0
        assert recap["total_realized_pnl"] is None
        assert recap["equity_change_amount"] is None
        assert recap["equity_change_pct"] is None
        assert recap["previous_close_equity"] is None

    @pytest.mark.asyncio
    async def test_compute_daily_recap_with_orders_and_snapshots(
        self,
        now_utc: datetime,
    ) -> None:
        """With seeded orders + 2-day snapshots, returns correct values.

        Orders: 2 BUY @100 x5, 2 SELL @110 x5 → 2 paired trades, each
        realized = (110-100)*5 = 50, total = 100.
        Equity snapshots: yesterday=10000, today=10200 → +200, +2%.
        """
        from finalayze.analysis.portfolio_review_agent import compute_daily_recap

        today_start = datetime(
            now_utc.year, now_utc.month, now_utc.day, tzinfo=now_utc.tzinfo,
        )
        yesterday_start = today_start - timedelta(days=1)
        # Build duck-typed OrderModel-like rows for fifo_pair.
        # fifo_pair requires: status, side, symbol, filled_quantity,
        # filled_avg_price, filled_at, signal_id.
        def _ord(side: str, price: Decimal, ts: datetime) -> SimpleNamespace:
            return SimpleNamespace(
                status="filled",
                side=side,
                symbol="SBER",
                filled_quantity=Decimal(5),
                filled_avg_price=price,
                filled_at=ts,
                signal_id=None,
            )

        orders = [
            _ord("BUY", Decimal(100), today_start + timedelta(hours=1)),
            _ord("SELL", Decimal(110), today_start + timedelta(hours=2)),
            _ord("BUY", Decimal(100), today_start + timedelta(hours=3)),
            _ord("SELL", Decimal(110), today_start + timedelta(hours=4)),
        ]
        # Order: buy_count, sell_count, today_orders_rows, today_eq, yest_eq
        session = _mock_session_with_results(
            scalar_returns=[2, 2, orders, Decimal("10200"), Decimal("10000")],
        )

        recap = await compute_daily_recap(session, now_utc)
        assert recap["positions_opened_today"] == 2
        assert recap["positions_closed_today"] == 2
        assert recap["total_realized_pnl"] == Decimal("100")
        assert recap["equity_change_amount"] == Decimal("200")
        assert recap["previous_close_equity"] == Decimal("10000")
        assert recap["equity_change_pct"] == pytest.approx(0.02)
        # Make sure yesterday_start is referenced for clarity (avoid linter
        # complaints about unused locals while keeping the docstring honest).
        assert yesterday_start < today_start
