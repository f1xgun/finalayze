"""Tests for 6D.2 (stop-loss lock), 6D.3 (sentiment lock scope), 6D.5 (shutdown cleanup)."""

from __future__ import annotations

import asyncio
import threading
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _make_trading_loop(**overrides: object) -> object:
    """Create a TradingLoop with minimal mock dependencies."""
    from finalayze.core.trading_loop import TradingLoop

    defaults = {
        "settings": MagicMock(
            max_position_pct=0.20,
            max_positions_per_market=10,
            daily_loss_limit_pct=0.02,
            kelly_fraction=0.5,
            news_cycle_minutes=30,
            strategy_cycle_minutes=60,
            daily_reset_hour_utc=0,
            ml_enabled=False,
        ),
        "fetchers": {},
        "news_fetcher": MagicMock(),
        "news_analyzer": MagicMock(),
        "event_classifier": MagicMock(),
        "impact_estimator": MagicMock(),
        "strategy": MagicMock(),
        "broker_router": MagicMock(),
        "circuit_breakers": {},
        "cross_market_breaker": MagicMock(),
        "alerter": MagicMock(),
        "instrument_registry": MagicMock(),
        "cache": None,
        "ml_registry": None,
        "event_bus": None,
        "fx_service": None,
    }
    defaults.update(overrides)
    return TradingLoop(**defaults)  # type: ignore[arg-type]


class TestStopLossLock:
    """6D.2: Verify _stop_loss_lock protects _stop_loss_prices."""

    def test_stop_loss_lock_exists(self) -> None:
        loop = _make_trading_loop()
        assert hasattr(loop, "_stop_loss_lock")
        assert isinstance(loop._stop_loss_lock, type(threading.Lock()))

    def test_concurrent_writes_no_crash(self) -> None:
        """Concurrent writes from two threads should not raise."""
        loop = _make_trading_loop()
        errors: list[Exception] = []

        def writer(start: int) -> None:
            try:
                for i in range(100):
                    with loop._stop_loss_lock:
                        loop._stop_loss_prices[f"SYM_{start}_{i}"] = Decimal(str(i))
            except Exception as exc:
                errors.append(exc)

        t1 = threading.Thread(target=writer, args=(0,))
        t2 = threading.Thread(target=writer, args=(1,))
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        assert errors == []
        # Both threads wrote 100 entries each
        assert len(loop._stop_loss_prices) == 200


class TestSentimentLockScope:
    """6D.3: Verify Redis writes happen outside _sentiment_lock."""

    def test_redis_write_outside_lock(self) -> None:
        """The _sentiment_lock should not be held during Redis cache writes."""
        mock_cache = MagicMock()
        lock_held_during_redis_write: list[bool] = []

        loop = _make_trading_loop(cache=mock_cache)

        call_count = 0

        def tracking_run_async(coro: object) -> object:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call is _analyze_article -- return mock sentiment/event
                return (MagicMock(score=0.5), MagicMock())
            # Subsequent calls are Redis cache writes
            locked = loop._sentiment_lock.locked()
            lock_held_during_redis_write.append(locked)
            return None

        loop._run_async = tracking_run_async  # type: ignore[assignment]

        # Simulate an impact
        impact = MagicMock()
        impact.segment_id = "us_tech"
        impact.sentiment = 0.8

        loop._impact_estimator.estimate.return_value = [impact]
        loop._process_news_article(MagicMock())

        # The Redis write should have happened with lock NOT held
        assert any(not held for held in lock_held_during_redis_write)


class TestShutdownCleanup:
    """6D.5: Verify stop() closes cache and event_bus."""

    def test_stop_closes_redis_cache(self) -> None:
        mock_cache = AsyncMock()
        loop = _make_trading_loop(cache=mock_cache)

        # Create a real async loop for the test
        real_loop = asyncio.new_event_loop()
        t = threading.Thread(target=real_loop.run_forever, daemon=True)
        t.start()
        loop._async_loop = real_loop
        loop._async_thread = t

        loop.stop()
        mock_cache.close.assert_called_once()

    def test_stop_closes_event_bus(self) -> None:
        mock_bus = AsyncMock()
        loop = _make_trading_loop(event_bus=mock_bus)

        real_loop = asyncio.new_event_loop()
        t = threading.Thread(target=real_loop.run_forever, daemon=True)
        t.start()
        loop._async_loop = real_loop
        loop._async_thread = t

        loop.stop()
        mock_bus.close.assert_called_once()

    def test_stop_without_cache_or_bus(self) -> None:
        """stop() should not crash when cache and event_bus are None."""
        loop = _make_trading_loop()
        loop.stop()  # Should not raise
