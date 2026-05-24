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
    # Phase 58-02-07: meta-agent default-disabled so existing trading-loop
    # registration tests keep passing without injecting a MetaAgentRunner
    # (per PATTERNS.md "EXTENSION: trading_loop.py" + 58-02-07 plan body).
    settings.meta_agent_enabled = False

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
        assert "hour='15'" in trigger_str, f"Expected hour=15 in trigger, got: {trigger_str}"
        assert "minute='50'" in trigger_str, f"Expected minute=50 in trigger, got: {trigger_str}"

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
        alerter.send_async = AsyncMock(return_value=(True, None))
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
            "total_realized_pnl": Decimal(100),
            "positions_opened_today": 2,
            "positions_closed_today": 1,
            "equity_change_pct": 0.01,
            "equity_change_amount": Decimal(100),
            "previous_close_equity": Decimal(10000),
        }
        with patch(
            "finalayze.analysis.portfolio_review_agent.compute_daily_recap",
            AsyncMock(return_value=recap),
        ):
            await tl._run_portfolio_review_async()  # type: ignore[attr-defined]

        alerter.send_async.assert_called_once()
        sent_text = alerter.send_async.call_args[0][0]
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
        alerter.send_async = AsyncMock(return_value=(True, None))
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
        alerter.send_async.assert_called_once()
        sent_text = alerter.send_async.call_args[0][0]
        assert "LLM still wants to talk." in sent_text
        assert "Daily Recap" not in sent_text

    @pytest.mark.asyncio
    async def test_portfolio_review_async_no_persistence_guard(self) -> None:
        """When _persistence is None, method returns early without crash."""
        alerter = MagicMock()
        alerter.send_async = AsyncMock(return_value=(True, None))
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
        alerter.send_async.assert_not_called()


# ── Phase 58-02-07: meta-agent APScheduler job registration ─────────────────


class TestMetaAgentJobRegistration:
    """Phase 58-02-07: TradingLoop.start() registers the meta-agent job
    only when ``settings.meta_agent_enabled is True``. SPEC AC #6 + #18.
    """

    @patch("finalayze.core.trading_loop.BackgroundScheduler")
    def test_meta_agent_job_registered_when_enabled(
        self,
        mock_scheduler_cls: MagicMock,
    ) -> None:
        """meta_agent_enabled=True → register_meta_agent_job adds an id='meta_agent' job."""
        from finalayze.meta_agent.runner import MetaAgentRunner

        mock_scheduler = MagicMock()
        mock_scheduler_cls.return_value = mock_scheduler
        loop = _make_trading_loop()
        # Enable the meta-agent + inject a runner so the wiring branch fires.
        loop._settings.meta_agent_enabled = True  # type: ignore[attr-defined]
        loop._settings.meta_agent_dry_run = True  # type: ignore[attr-defined]
        loop._settings.meta_agent_interval_minutes = 30  # type: ignore[attr-defined]
        runner = MagicMock(spec=MetaAgentRunner)
        loop._meta_agent_runner = runner  # type: ignore[attr-defined]

        with (
            patch.object(loop, "_load_baseline_from_db"),
            patch.object(loop, "_reconcile_inflight_orders"),
            patch.object(loop, "_stop_event") as mock_stop,
        ):
            mock_stop.wait.side_effect = lambda: None
            loop.start()  # type: ignore[union-attr]

        meta_calls = [
            call
            for call in mock_scheduler.add_job.call_args_list
            if (call.kwargs or {}).get("id") == "meta_agent"
        ]
        assert len(meta_calls) == 1, (
            f"Expected exactly one meta_agent add_job call, got "
            f"{len(meta_calls)}: "
            f"{[c.kwargs.get('id') for c in mock_scheduler.add_job.call_args_list]}"
        )

    @patch("finalayze.core.trading_loop.BackgroundScheduler")
    def test_meta_agent_job_not_registered_when_disabled(
        self,
        mock_scheduler_cls: MagicMock,
    ) -> None:
        """meta_agent_enabled=False → no add_job with id='meta_agent'."""
        mock_scheduler = MagicMock()
        mock_scheduler_cls.return_value = mock_scheduler
        loop = _make_trading_loop()
        loop._settings.meta_agent_enabled = False  # type: ignore[attr-defined]

        with (
            patch.object(loop, "_load_baseline_from_db"),
            patch.object(loop, "_reconcile_inflight_orders"),
            patch.object(loop, "_stop_event") as mock_stop,
        ):
            mock_stop.wait.side_effect = lambda: None
            loop.start()  # type: ignore[union-attr]

        meta_calls = [
            call
            for call in mock_scheduler.add_job.call_args_list
            if (call.kwargs or {}).get("id") == "meta_agent"
        ]
        assert len(meta_calls) == 0, (
            f"Expected zero meta_agent add_job calls when disabled, got {len(meta_calls)}"
        )


# ── Phase 58-05-06: TradingLoop wires killswitch start/stop ─────────────────


class TestMetaAgentKillswitchWiring:
    """Phase 58-05-06: TradingLoop.start() launches the meta-agent
    killswitch poller (env-var watcher) when meta_agent_enabled=True
    AND a killswitch is wired on the runner. ``stop()`` cancels the
    poller cleanly.

    SPEC AC #15 + #18.
    """

    @patch("finalayze.core.trading_loop.BackgroundScheduler")
    def test_meta_agent_killswitch_started_when_enabled(
        self,
        mock_scheduler_cls: MagicMock,
    ) -> None:
        """meta_agent_enabled=True with a wired killswitch on the runner →
        TradingLoop.start() schedules ``runner.killswitch.start()`` onto
        the persistent async loop.
        """
        from finalayze.meta_agent.killswitch import Killswitch
        from finalayze.meta_agent.runner import MetaAgentRunner

        mock_scheduler = MagicMock()
        mock_scheduler_cls.return_value = mock_scheduler
        loop = _make_trading_loop()
        loop._settings.meta_agent_enabled = True  # type: ignore[attr-defined]
        loop._settings.meta_agent_dry_run = True  # type: ignore[attr-defined]
        loop._settings.meta_agent_interval_minutes = 30  # type: ignore[attr-defined]

        # Build runner whose .killswitch is a Killswitch-shaped MagicMock.
        runner = MagicMock(spec=MetaAgentRunner)
        ks = MagicMock(spec=Killswitch)
        ks.start = AsyncMock()
        ks.stop = AsyncMock()
        runner.killswitch = ks
        loop._meta_agent_runner = runner  # type: ignore[attr-defined]

        with (
            patch.object(loop, "_load_baseline_from_db"),
            patch.object(loop, "_reconcile_inflight_orders"),
            patch.object(loop, "_stop_event") as mock_stop,
            patch("asyncio.run_coroutine_threadsafe") as mock_run_threadsafe,
        ):
            mock_stop.wait.side_effect = lambda: None
            loop.start()  # type: ignore[union-attr]

        # The wiring path scheduled killswitch.start() onto the async loop
        # via run_coroutine_threadsafe. The first positional arg of one of
        # those calls must be the coroutine returned by ks.start().
        assert ks.start.call_count == 1, (
            f"Expected exactly one ks.start() invocation, got {ks.start.call_count}"
        )
        # And run_coroutine_threadsafe was called with the resulting coro.
        ks_start_calls = [
            call
            for call in mock_run_threadsafe.call_args_list
            if call.args and getattr(call.args[0], "__class__", None).__name__ == "coroutine"
        ]
        # At least one run_coroutine_threadsafe call dispatched a coroutine.
        assert len(ks_start_calls) >= 1, (
            f"Expected ≥1 run_coroutine_threadsafe(coro) call, got "
            f"{[c.args for c in mock_run_threadsafe.call_args_list]}"
        )

    @patch("finalayze.core.trading_loop.BackgroundScheduler")
    def test_meta_agent_killswitch_not_started_when_disabled(
        self,
        mock_scheduler_cls: MagicMock,
    ) -> None:
        """meta_agent_enabled=False → no killswitch.start() call."""
        from finalayze.meta_agent.killswitch import Killswitch
        from finalayze.meta_agent.runner import MetaAgentRunner

        mock_scheduler = MagicMock()
        mock_scheduler_cls.return_value = mock_scheduler
        loop = _make_trading_loop()
        loop._settings.meta_agent_enabled = False  # type: ignore[attr-defined]

        runner = MagicMock(spec=MetaAgentRunner)
        ks = MagicMock(spec=Killswitch)
        ks.start = AsyncMock()
        ks.stop = AsyncMock()
        runner.killswitch = ks
        loop._meta_agent_runner = runner  # type: ignore[attr-defined]

        with (
            patch.object(loop, "_load_baseline_from_db"),
            patch.object(loop, "_reconcile_inflight_orders"),
            patch.object(loop, "_stop_event") as mock_stop,
        ):
            mock_stop.wait.side_effect = lambda: None
            loop.start()  # type: ignore[union-attr]

        ks.start.assert_not_called()

    @patch("finalayze.core.trading_loop.BackgroundScheduler")
    def test_meta_agent_killswitch_stopped_on_trading_loop_stop(
        self,
        mock_scheduler_cls: MagicMock,
    ) -> None:
        """TradingLoop.stop() cancels the killswitch poller via
        run_coroutine_threadsafe(ks.stop()) when a killswitch is wired.
        """
        from finalayze.meta_agent.killswitch import Killswitch
        from finalayze.meta_agent.runner import MetaAgentRunner

        mock_scheduler = MagicMock()
        mock_scheduler_cls.return_value = mock_scheduler
        loop = _make_trading_loop()
        loop._settings.meta_agent_enabled = True  # type: ignore[attr-defined]
        loop._settings.meta_agent_dry_run = True  # type: ignore[attr-defined]
        loop._settings.meta_agent_interval_minutes = 30  # type: ignore[attr-defined]

        runner = MagicMock(spec=MetaAgentRunner)
        ks = MagicMock(spec=Killswitch)
        ks.start = AsyncMock()
        ks.stop = AsyncMock()
        runner.killswitch = ks
        loop._meta_agent_runner = runner  # type: ignore[attr-defined]

        # Pre-set _async_loop so stop() doesn't try to enter the live-loop
        # close path on a None/closed loop.
        fake_async_loop = MagicMock()
        fake_async_loop.is_closed.return_value = False
        loop._async_loop = fake_async_loop  # type: ignore[attr-defined]

        with patch("asyncio.run_coroutine_threadsafe") as mock_run_threadsafe:
            loop.stop()  # type: ignore[union-attr]

        # ks.stop was invoked (the coroutine was created).
        assert ks.stop.call_count >= 1, (
            f"TradingLoop.stop() must invoke killswitch.stop(); got call_count={ks.stop.call_count}"
        )
        # And dispatched onto the async loop.
        assert mock_run_threadsafe.call_count >= 1
