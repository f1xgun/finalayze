"""Tests for fire-and-forget DB persistence in TradingLoop.

Covers _persist_to_db helper: exception swallowing, logging, counter increment,
and isolation from _consecutive_equity_errors.
"""

from __future__ import annotations

import asyncio
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from finalayze.orchestration.trading_loop import TradingLoop


def _make_loop() -> TradingLoop:
    """Create a minimal TradingLoop with mocked dependencies."""
    settings = MagicMock()
    settings.mode = MagicMock()
    settings.mode.value = "sandbox"
    settings.effective_risk_limits.return_value = MagicMock(
        max_position_pct=Decimal("0.1"),
        max_positions_per_market=10,
        max_sector_concentration_pct=Decimal("0.3"),
        min_cash_reserve_pct=Decimal("0.1"),
        daily_loss_limit_pct=0.02,
    )
    settings.kelly_fraction = 0.5

    loop = TradingLoop(
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
    return loop


class TestPersistToDb:
    """Tests for _persist_to_db fire-and-forget helper."""

    def test_catches_exceptions_and_does_not_reraise(self) -> None:
        """_persist_to_db must swallow any exception from the coroutine."""
        loop = _make_loop()

        async def failing_coro() -> None:
            msg = "DB connection failed"
            raise RuntimeError(msg)

        # Must not raise
        loop._persist_to_db(failing_coro(), table="orders")

    def test_logs_db_persist_failed_on_error(self) -> None:
        """_persist_to_db emits 'db_persist_failed' log with table name."""
        loop = _make_loop()

        async def failing_coro() -> None:
            msg = "timeout"
            raise RuntimeError(msg)

        with patch("finalayze.orchestration.trading_loop._log") as mock_log:
            loop._persist_to_db(failing_coro(), table="signals")
            mock_log.warning.assert_called_once()
            call_args = mock_log.warning.call_args
            assert call_args[0][0] == "db_persist_failed"
            assert call_args[1]["table"] == "signals"

    def test_increments_db_write_failures_counter_on_error(self) -> None:
        """db_write_failures Prometheus counter incremented on failure."""
        loop = _make_loop()

        async def failing_coro() -> None:
            msg = "insert failed"
            raise RuntimeError(msg)

        with patch("finalayze.api.metrics.db_write_failures") as mock_counter:
            mock_labels = MagicMock()
            mock_counter.labels.return_value = mock_labels
            loop._persist_to_db(failing_coro(), table="orders")
            mock_counter.labels.assert_called_once_with(table="orders")
            mock_labels.inc.assert_called_once()

    def test_does_not_increment_consecutive_equity_errors(self) -> None:
        """DB failures must NOT touch _consecutive_equity_errors."""
        loop = _make_loop()
        loop._consecutive_equity_errors = 0

        async def failing_coro() -> None:
            msg = "boom"
            raise RuntimeError(msg)

        loop._persist_to_db(failing_coro(), table="orders")
        assert loop._consecutive_equity_errors == 0

    def test_successful_call_does_not_increment_counter(self) -> None:
        """Successful persistence does not touch the failure counter."""
        loop = _make_loop()

        async def ok_coro() -> None:
            pass

        with patch("finalayze.api.metrics.db_write_failures") as mock_counter:
            loop._persist_to_db(ok_coro(), table="signals")
            mock_counter.labels.assert_not_called()
