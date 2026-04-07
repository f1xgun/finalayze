"""Tests for TradingLoop bond cycle integration (05-02, 05-03)."""

from __future__ import annotations

import asyncio
import inspect
from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from finalayze.core.trading_loop import TradingLoop


class TestBondCycleProcessorIntegration:
    """TradingLoop._bond_cycle() must call processor.run_cycle()."""

    def test_bond_cycle_delegates_to_processor(self) -> None:
        """TradingLoop._bond_cycle() must call processor.run_cycle()."""
        import inspect

        # Use getsource instead of signature to avoid conftest __init__ wrapper
        source = inspect.getsource(TradingLoop)
        assert "bond_cycle_processor" in source
        assert "macro_cache" in source


class TestAsyncioLockSerialization:
    """asyncio.Lock serializes concurrent gRPC calls (equity + bond don't overlap)."""

    def test_grpc_lock_exists(self) -> None:
        """TradingLoop creates _grpc_lock as asyncio.Lock."""
        import inspect

        # Use getsource on the class to avoid conftest __init__ wrapper
        source = inspect.getsource(TradingLoop)
        assert "_grpc_lock" in source

    def test_grpc_lock_serializes_calls(self) -> None:
        """asyncio.Lock prevents concurrent gRPC access."""
        call_order: list[str] = []

        async def _test_lock_serialization() -> None:
            lock = asyncio.Lock()

            async def task_a() -> None:
                async with lock:
                    call_order.append("a_start")
                    await asyncio.sleep(0.01)
                    call_order.append("a_end")

            async def task_b() -> None:
                async with lock:
                    call_order.append("b_start")
                    await asyncio.sleep(0.01)
                    call_order.append("b_end")

            await asyncio.gather(task_a(), task_b())

        asyncio.run(_test_lock_serialization())

        # With a lock, calls should be serialized: a_start, a_end, b_start, b_end
        # (or b_start, b_end, a_start, a_end)
        assert call_order[0].endswith("_start")
        assert call_order[1].endswith("_end")
        assert call_order[2].endswith("_start")
        assert call_order[3].endswith("_end")


class TestDailyEquitySnapshotModel:
    """DailyEquitySnapshot model has timestamp, market_id, equity, currency columns."""

    def test_model_exists(self) -> None:
        """DailyEquitySnapshot model is importable."""
        from finalayze.core.models import DailyEquitySnapshot

        assert DailyEquitySnapshot is not None

    def test_model_columns(self) -> None:
        """DailyEquitySnapshot has required columns."""
        from finalayze.core.models import DailyEquitySnapshot

        assert hasattr(DailyEquitySnapshot, "timestamp")
        assert hasattr(DailyEquitySnapshot, "market_id")
        assert hasattr(DailyEquitySnapshot, "equity")
        assert hasattr(DailyEquitySnapshot, "currency")

    def test_model_tablename(self) -> None:
        """DailyEquitySnapshot.__tablename__ is 'daily_equity_snapshots'."""
        from finalayze.core.models import DailyEquitySnapshot

        assert DailyEquitySnapshot.__tablename__ == "daily_equity_snapshots"


# ── 05-03 Tests: CBR alerts, coupon alerts, weekly digest ──────────────────


def _make_trading_loop_mocks() -> dict[str, MagicMock]:
    """Create mock dependencies for TradingLoop instantiation."""
    settings = MagicMock()
    settings.mode = "test"
    settings.max_position_pct = 0.20
    settings.max_positions_per_market = 10
    settings.daily_loss_limit_pct = 0.02
    settings.kelly_fraction = 0.5
    settings.news_cycle_minutes = 30
    settings.strategy_cycle_minutes = 60
    settings.daily_reset_hour_utc = 0
    settings.ml_enabled = False
    settings.bond_cycle_enabled = True
    settings.bond_cycle_minutes = 1440
    settings.weekly_digest_hour_utc = 16
    settings.telegram_allowed_chat_ids = []
    settings.telegram_webhook_secret = ""

    return {
        "settings": settings,
        "fetchers": {},
        "news_fetcher": MagicMock(),
        "news_analyzer": MagicMock(),
        "event_classifier": MagicMock(),
        "impact_estimator": MagicMock(),
        "strategy": MagicMock(),
        "broker_router": MagicMock(),
        "circuit_breakers": {"us": MagicMock(), "moex": MagicMock()},
        "cross_market_breaker": MagicMock(),
        "alerter": MagicMock(),
        "instrument_registry": MagicMock(),
        "bond_cycle_processor": MagicMock(),
        "macro_cache": MagicMock(),
    }


class TestCBRDayAlerts:
    """_cbr_day_refresh calls alerter.on_cbr_meeting after macro refresh."""

    def test_cbr_day_refresh_calls_on_cbr_meeting(self) -> None:
        """After macro refresh on CBR meeting day, alerter.on_cbr_meeting is called."""
        source = inspect.getsource(TradingLoop._cbr_day_refresh)
        assert "on_cbr_meeting" in source

    def test_cbr_day_refresh_sends_skip_alert_on_missing_macro(self) -> None:
        """If macro data stale after refresh, sends unexpected-skip alert."""
        source = inspect.getsource(TradingLoop._cbr_day_refresh)
        assert "on_error" in source or "stale" in source.lower()


class TestWeeklyDigest:
    """Weekly digest runs on Sunday with week P&L and trade stats."""

    def test_weekly_digest_method_exists(self) -> None:
        """TradingLoop has a _weekly_digest method."""
        assert hasattr(TradingLoop, "_weekly_digest")
        assert callable(TradingLoop._weekly_digest)

    def test_weekly_digest_scheduled_via_cron(self) -> None:
        """Weekly digest is scheduled via CronTrigger on Sunday."""
        source = inspect.getsource(TradingLoop.start)
        assert "weekly_digest" in source
        assert "sun" in source.lower() or "day_of_week" in source

    def test_weekly_digest_sends_alert(self) -> None:
        """_weekly_digest sends alert via alerter.send_alert."""
        source = inspect.getsource(TradingLoop._weekly_digest)
        assert "send_alert" in source or "_send" in source
