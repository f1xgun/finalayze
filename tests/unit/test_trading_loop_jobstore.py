"""Tests for APScheduler SQLAlchemyJobStore configuration in TradingLoop."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestTradingLoopJobStore:
    """TradingLoop.start() configures persistent job store when DB is available."""

    def _make_trading_loop(self) -> object:
        """Create a TradingLoop with all deps mocked."""
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
        settings.meta_agent_enabled = False
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

    @patch("finalayze.core.trading_loop.BackgroundScheduler")
    def test_jobs_have_stable_ids(self, mock_scheduler_cls: MagicMock) -> None:
        """All add_job calls use id= and replace_existing=True."""
        mock_scheduler = MagicMock()
        mock_scheduler_cls.return_value = mock_scheduler

        loop = self._make_trading_loop()

        # Patch _load_baseline_from_db and _reconcile_inflight_orders
        with (
            patch.object(loop, "_load_baseline_from_db"),
            patch.object(loop, "_reconcile_inflight_orders"),
            patch.object(loop, "_stop_event") as mock_stop,
        ):
            mock_stop.wait.side_effect = lambda: None  # Don't block

            loop.start()  # type: ignore[union-attr]

        # Check all add_job calls have id and replace_existing
        for call in mock_scheduler.add_job.call_args_list:
            kwargs = call.kwargs or {}
            # Some calls may use positional args for trigger type
            assert "id" in kwargs, f"Missing id in add_job call: {call}"
            assert kwargs.get("replace_existing") is True, (
                f"Missing replace_existing=True in add_job call: {call}"
            )

    @patch("finalayze.core.trading_loop.BackgroundScheduler")
    def test_jobstore_fallback_to_memory(self, mock_scheduler_cls: MagicMock) -> None:
        """When SQLAlchemyJobStore import fails, falls back to MemoryJobStore."""
        mock_scheduler = MagicMock()
        mock_scheduler_cls.return_value = mock_scheduler

        loop = self._make_trading_loop()

        with (
            patch.object(loop, "_load_baseline_from_db"),
            patch.object(loop, "_reconcile_inflight_orders"),
            patch.object(loop, "_stop_event") as mock_stop,
            patch(
                "finalayze.core.trading_loop.SQLAlchemyJobStore",
                side_effect=ImportError("No module named 'psycopg2'"),
            ),
        ):
            mock_stop.wait.side_effect = lambda: None
            loop.start()  # type: ignore[union-attr]

        # Scheduler should still be created (with default MemoryJobStore)
        mock_scheduler_cls.assert_called_once()
        # jobstores kwarg should NOT contain "default" (fallback = no explicit jobstore)
        call_kwargs = mock_scheduler_cls.call_args.kwargs
        if "jobstores" in call_kwargs:
            assert "default" not in call_kwargs["jobstores"]

    @patch("finalayze.core.trading_loop.BackgroundScheduler")
    def test_jobstore_uses_memory(self, mock_scheduler_cls: MagicMock) -> None:
        """TradingLoop uses MemoryJobStore (SQLAlchemy jobstore dropped due to pickling)."""
        mock_scheduler = MagicMock()
        mock_scheduler_cls.return_value = mock_scheduler

        loop = self._make_trading_loop()

        with (
            patch.object(loop, "_load_baseline_from_db"),
            patch.object(loop, "_reconcile_inflight_orders"),
            patch.object(loop, "_stop_event") as mock_stop,
        ):
            mock_stop.wait.side_effect = lambda: None
            loop.start()  # type: ignore[union-attr]

        # BackgroundScheduler should be created without explicit jobstores
        call_kwargs = mock_scheduler_cls.call_args.kwargs
        assert "jobstores" not in call_kwargs

    @patch("finalayze.core.trading_loop.BackgroundScheduler")
    def test_expected_job_ids(self, mock_scheduler_cls: MagicMock) -> None:
        """Verify expected job IDs are present."""
        mock_scheduler = MagicMock()
        mock_scheduler_cls.return_value = mock_scheduler

        loop = self._make_trading_loop()

        with (
            patch.object(loop, "_load_baseline_from_db"),
            patch.object(loop, "_reconcile_inflight_orders"),
            patch.object(loop, "_stop_event") as mock_stop,
        ):
            mock_stop.wait.side_effect = lambda: None
            loop.start()  # type: ignore[union-attr]

        job_ids = {
            call.kwargs["id"]
            for call in mock_scheduler.add_job.call_args_list
            if "id" in (call.kwargs or {})
        }
        assert "news_cycle" in job_ids
        assert "strategy_cycle" in job_ids
        assert "daily_reset" in job_ids
        assert "weekly_digest" in job_ids
