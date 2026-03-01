"""Tests for TradingLoop._run_async — persistent background event loop (5.4)."""

from __future__ import annotations

import asyncio
import threading
from unittest.mock import MagicMock

import pytest


def _make_loop() -> MagicMock:
    """Create a minimal TradingLoop-like object with _run_async bound."""
    from finalayze.core.trading_loop import TradingLoop

    settings = MagicMock()
    settings.news_cycle_minutes = 30
    settings.strategy_cycle_minutes = 60
    settings.daily_reset_hour_utc = 0
    settings.mode = "test"
    settings.max_position_pct = 0.20
    settings.max_positions_per_market = 10
    settings.daily_loss_limit_pct = 0.05
    settings.kelly_fraction = 0.5

    loop = MagicMock(spec=TradingLoop)
    loop._async_loop = None
    loop._async_thread = None
    loop._run_async = TradingLoop._run_async.__get__(loop)
    return loop


class TestRunAsync:
    def test_reuses_same_loop(self) -> None:
        """Multiple _run_async calls should reuse the same event loop."""
        loop = _make_loop()

        async def simple() -> int:
            return 42

        result1 = loop._run_async(simple())
        first_async_loop = loop._async_loop

        result2 = loop._run_async(simple())
        second_async_loop = loop._async_loop

        assert result1 == 42
        assert result2 == 42
        assert first_async_loop is second_async_loop

    def test_propagates_exceptions(self) -> None:
        """Exceptions from coroutines should propagate to the caller."""
        loop = _make_loop()

        async def fail() -> None:
            msg = "boom"
            raise ValueError(msg)

        with pytest.raises(ValueError, match="boom"):
            loop._run_async(fail())

    def test_works_from_thread(self) -> None:
        """_run_async should work when called from a non-main thread."""
        loop = _make_loop()
        results: list[int] = []
        errors: list[Exception] = []

        async def compute() -> int:
            await asyncio.sleep(0.01)
            return 99

        def worker() -> None:
            try:
                results.append(loop._run_async(compute()))
            except Exception as exc:
                errors.append(exc)

        t = threading.Thread(target=worker)
        t.start()
        t.join(timeout=10)

        assert not errors, f"Unexpected errors: {errors}"
        assert results == [99]

    def test_cleanup_stops_loop(self) -> None:
        """After stopping, the async loop should be stopped."""
        from finalayze.core.trading_loop import TradingLoop

        loop = _make_loop()
        loop._stop_event = threading.Event()
        loop._scheduler = None
        loop._cache = None
        loop._event_bus = None
        loop._fx_service = None
        loop.stop = TradingLoop.stop.__get__(loop)

        async def simple() -> int:
            return 1

        loop._run_async(simple())
        assert loop._async_loop is not None
        assert loop._async_loop.is_running()

        loop.stop()
        # After stop, the loop thread should have been joined
        assert loop._async_thread is not None
