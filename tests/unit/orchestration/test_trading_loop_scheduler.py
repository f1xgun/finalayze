"""Tests for ALRT-04 portfolio review APScheduler cron + recap merge.

Phase 57 Plan 05 wires two pieces:

1. ``TradingLoop._start_scheduler`` (the body of ``start()``) registers
   ``_portfolio_review_cycle`` as a cron job at 15:50 UTC (= 18:50 MSK,
   post-MOEX-close). Without this registration, the method existed but
   was dead code (Pitfall 6 — Phase 52 added the body, PR #223 lost
   the ``add_job``).

2. ``_run_portfolio_review_async`` merges the deterministic
   ``compute_daily_recap`` output into the LLM advisory result via
   ``model_copy(update=recap)``. The merge is best-effort — if recap
   computation fails, the LLM advisory still ships (D-01 Option A).

Tests use the ``BackgroundScheduler`` patch pattern from
``tests/unit/test_trading_loop_jobstore.py`` to avoid a real cron loop.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _make_trading_loop() -> object:
    """Create a TradingLoop with all deps mocked (jobstore-pattern parity)."""
    from finalayze.core.trading_loop import TradingLoop

    settings = MagicMock()
    settings.mode = MagicMock()
    settings.mode.value = "sandbox"
    settings.mode.can_submit_orders.return_value = True
    settings.news_cycle_minutes = 30
    settings.strategy_cycle_minutes = 15
    settings.daily_reset_hour_utc = 0
    settings.max_position_pct = 0.10
    settings.max_positions_per_market = 10
    settings.daily_loss_limit_pct = 0.05
    settings.kelly_fraction = 0.5
    settings.database_url = "postgresql+asyncpg://user:pass@localhost/db"
    settings.ml_enabled = False
    settings.bond_cycle_enabled = False
    settings.weekly_digest_hour_utc = 16

    return TradingLoop(
        settings=settings,
        fetchers={},
        news_fetcher=MagicMock(),
        news_analyzer=MagicMock(),
        event_classifier=MagicMock(),
        impact_estimator=MagicMock(),
        strategy=MagicMock(),
        broker_router=MagicMock(),
        circuit_breakers={},
        cross_market_breaker=MagicMock(),
        alerter=MagicMock(),
        instrument_registry=MagicMock(),
    )


# ── Tests 1-2: APScheduler cron registration ────────────────────────────


class TestPortfolioReviewCronRegistration:
    """start() registers _portfolio_review_cycle at 15:50 UTC."""

    @patch("finalayze.core.trading_loop.BackgroundScheduler")
    def test_portfolio_review_cron_registered(
        self,
        mock_scheduler_cls: MagicMock,
    ) -> None:
        """An add_job call registers _portfolio_review_cycle with id='portfolio_review'."""
        mock_scheduler = MagicMock()
        mock_scheduler_cls.return_value = mock_scheduler
        loop = _make_trading_loop()
        with (
            patch.object(loop, "_load_baseline_from_db"),
            patch.object(loop, "_reconcile_inflight_orders"),
            patch.object(loop, "_stop_event") as mock_stop,
        ):
            mock_stop.wait.side_effect = lambda: None
            loop.start()  # type: ignore[union-attr]

        # Find the portfolio_review add_job call.
        portfolio_calls = [
            call
            for call in mock_scheduler.add_job.call_args_list
            if (call.kwargs or {}).get("id") == "portfolio_review"
        ]
        assert len(portfolio_calls) == 1, (
            f"Expected exactly one portfolio_review add_job, got "
            f"{len(portfolio_calls)}: "
            f"{[c.kwargs.get('id') for c in mock_scheduler.add_job.call_args_list]}"
        )
        call = portfolio_calls[0]
        # First positional arg should be the bound _portfolio_review_cycle.
        assert call.args[0] == loop._portfolio_review_cycle  # type: ignore[attr-defined]
        # CronTrigger should be the second positional arg with hour=15, minute=50.
        trigger = call.args[1]
        # Inspect the trigger's fields — APScheduler stores them as a tuple.
        # Robust check: stringified trigger contains "hour='15'" and "minute='50'".
        trigger_str = str(trigger)
        assert "hour='15'" in trigger_str, (
            f"Expected hour=15 in trigger, got: {trigger_str}"
        )
        assert "minute='50'" in trigger_str, (
            f"Expected minute=50 in trigger, got: {trigger_str}"
        )

    @patch("finalayze.core.trading_loop.BackgroundScheduler")
    def test_portfolio_review_cron_uses_replace_existing(
        self,
        mock_scheduler_cls: MagicMock,
    ) -> None:
        """add_job for portfolio_review uses replace_existing=True."""
        mock_scheduler = MagicMock()
        mock_scheduler_cls.return_value = mock_scheduler
        loop = _make_trading_loop()
        with (
            patch.object(loop, "_load_baseline_from_db"),
            patch.object(loop, "_reconcile_inflight_orders"),
            patch.object(loop, "_stop_event") as mock_stop,
        ):
            mock_stop.wait.side_effect = lambda: None
            loop.start()  # type: ignore[union-attr]

        portfolio_calls = [
            call
            for call in mock_scheduler.add_job.call_args_list
            if (call.kwargs or {}).get("id") == "portfolio_review"
        ]
        assert len(portfolio_calls) == 1
        assert portfolio_calls[0].kwargs.get("replace_existing") is True


# ── Tests 3-5: _run_portfolio_review_async recap merge ──────────────────


def _make_tl_for_async() -> object:
    """Create a bare TradingLoop instance (no __init__) for async-method tests."""
    from finalayze.core.trading_loop import TradingLoop

    return object.__new__(TradingLoop)


class TestRunPortfolioReviewAsyncMerge:
    """_run_portfolio_review_async merges compute_daily_recap into LLM result."""

    @pytest.mark.asyncio
    async def test_portfolio_review_async_merges_recap(self) -> None:
        """Recap dict is merged into PortfolioReviewResult via model_copy."""
        from finalayze.analysis.portfolio_review_agent import PortfolioReviewResult

        base_result = PortfolioReviewResult(
            reviewed_at=datetime(2026, 4, 19, 15, 50, tzinfo=UTC),
            overall_assessment="Looking healthy.",
            risk_score=0.3,
        )

        alerter = MagicMock()
        alerter._send = AsyncMock()
        llm_client = AsyncMock()
        llm_client.parse_structured = AsyncMock(return_value=base_result)

        # Persistence mock with a session factory that returns an async-context-
        # manager session (we don't actually exercise the DB; compute_daily_recap
        # is patched below so the session can be a no-op).
        session = MagicMock()
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=None)
        factory = MagicMock(return_value=session)
        persistence = MagicMock()
        persistence._get_bg_session_factory = MagicMock(return_value=factory)

        tl = _make_tl_for_async()
        tl._llm_client = llm_client  # type: ignore[attr-defined]
        tl._alerter = alerter  # type: ignore[attr-defined]
        tl._circuit_breakers = {}  # type: ignore[attr-defined]
        tl._broker_router = MagicMock()  # type: ignore[attr-defined]
        tl._persistence = persistence  # type: ignore[attr-defined]

        recap = {
            "total_realized_pnl": Decimal("100"),
            "positions_opened_today": 2,
            "positions_closed_today": 1,
            "equity_change_pct": 0.01,
            "equity_change_amount": Decimal("100"),
            "previous_close_equity": Decimal("10000"),
        }
        with patch(
            "finalayze.analysis.portfolio_review_agent.compute_daily_recap",
            AsyncMock(return_value=recap),
        ):
            await tl._run_portfolio_review_async()  # type: ignore[attr-defined]

        alerter._send.assert_called_once()
        sent_text = alerter._send.call_args[0][0]
        assert "Daily Recap" in sent_text
        assert "100" in sent_text
        assert "Opened: 2" in sent_text

    @pytest.mark.asyncio
    async def test_portfolio_review_async_recap_failure_does_not_block_llm_message(
        self,
    ) -> None:
        """compute_daily_recap raising does not stop the LLM advisory message."""
        from finalayze.analysis.portfolio_review_agent import PortfolioReviewResult

        base_result = PortfolioReviewResult(
            reviewed_at=datetime(2026, 4, 19, 15, 50, tzinfo=UTC),
            overall_assessment="LLM still wants to talk.",
            risk_score=0.4,
        )

        alerter = MagicMock()
        alerter._send = AsyncMock()
        llm_client = AsyncMock()
        llm_client.parse_structured = AsyncMock(return_value=base_result)

        session = MagicMock()
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=None)
        factory = MagicMock(return_value=session)
        persistence = MagicMock()
        persistence._get_bg_session_factory = MagicMock(return_value=factory)

        tl = _make_tl_for_async()
        tl._llm_client = llm_client  # type: ignore[attr-defined]
        tl._alerter = alerter  # type: ignore[attr-defined]
        tl._circuit_breakers = {}  # type: ignore[attr-defined]
        tl._broker_router = MagicMock()  # type: ignore[attr-defined]
        tl._persistence = persistence  # type: ignore[attr-defined]

        with patch(
            "finalayze.analysis.portfolio_review_agent.compute_daily_recap",
            AsyncMock(side_effect=RuntimeError("boom")),
        ):
            await tl._run_portfolio_review_async()  # type: ignore[attr-defined]

        # LLM advisory must still ship (no Daily Recap section).
        alerter._send.assert_called_once()
        sent_text = alerter._send.call_args[0][0]
        assert "LLM still wants to talk." in sent_text
        assert "Daily Recap" not in sent_text

    @pytest.mark.asyncio
    async def test_portfolio_review_async_no_persistence_guard(self) -> None:
        """When _persistence is None, method returns early without crash."""
        alerter = MagicMock()
        alerter._send = AsyncMock()
        llm_client = AsyncMock()
        llm_client.parse_structured = AsyncMock()

        tl = _make_tl_for_async()
        tl._llm_client = llm_client  # type: ignore[attr-defined]
        tl._alerter = alerter  # type: ignore[attr-defined]
        tl._circuit_breakers = {}  # type: ignore[attr-defined]
        tl._broker_router = MagicMock()  # type: ignore[attr-defined]
        tl._persistence = None  # type: ignore[attr-defined]

        # Must not raise.
        await tl._run_portfolio_review_async()  # type: ignore[attr-defined]

        # No LLM call, no telegram send — early return.
        llm_client.parse_structured.assert_not_called()
        alerter._send.assert_not_called()
