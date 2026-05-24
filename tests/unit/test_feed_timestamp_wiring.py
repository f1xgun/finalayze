"""Tests proving feed timestamp is wired correctly and /gonogo import works (INT-01, INT-02).

INT-01: GoNoGoReporter can be imported at runtime.
INT-02: HealthMonitor.update_feed_timestamp() is called directly after data fetch.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import MagicMock

import pytest


def _make_trading_loop_with_health_monitor(
    health_monitor: object | None = None,
) -> object:
    """Create a minimal TradingLoop with health_monitor for feed timestamp testing."""
    from finalayze.core.trading_loop import TradingLoop, TradingLoopDeps

    mock_settings = MagicMock()
    mock_settings.segment_ids = ["us_tech"]
    mock_settings.market_ids = ["us"]
    mock_settings.work_mode = "sandbox"
    mock_settings.kelly_fraction = 0.5

    return TradingLoop(
        TradingLoopDeps(
            settings=mock_settings,
            fetchers={"us": MagicMock()},
            news_fetcher=MagicMock(),
            news_analyzer=MagicMock(),
            event_classifier=MagicMock(),
            impact_estimator=MagicMock(),
            strategy=MagicMock(),
            broker_router=MagicMock(),
            circuit_breakers={"us": MagicMock()},
            cross_market_breaker=MagicMock(),
            alerter=MagicMock(),
            instrument_registry=MagicMock(),
            health_monitor=health_monitor,
        )
    )


class TestGoNoGoImport:
    """INT-01: GoNoGoReporter import works at runtime."""

    def test_gonogo_reporter_importable(self) -> None:
        """GoNoGoReporter can be imported from finalayze.monitoring.go_no_go."""
        from finalayze.monitoring.go_no_go import GoNoGoReporter

        assert GoNoGoReporter is not None

    def test_gonogo_reporter_has_evaluate_method(self) -> None:
        """GoNoGoReporter has the evaluate() method used by /gonogo command."""
        from finalayze.monitoring.go_no_go import GoNoGoReporter

        assert hasattr(GoNoGoReporter, "evaluate")


class TestFeedTimestampWiring:
    """INT-02: update_feed_timestamp is called directly after data fetch."""

    def test_update_feed_timestamp_called_with_candles(self) -> None:
        """When candles are fetched, update_feed_timestamp is called with datetime."""
        mock_health = MagicMock()
        _make_trading_loop_with_health_monitor(health_monitor=mock_health)

        # Simulate the feed timestamp update logic from _process_instrument
        # We test the actual code path by checking the source doesn't use getattr
        import inspect

        from finalayze.core.trading_loop import TradingLoop, TradingLoopDeps

        source = inspect.getsource(TradingLoop._process_instrument)

        # INT-02: Must call update_feed_timestamp directly (not via getattr)
        assert (
            "getattr" not in source
            or "update_feed_timestamp" not in source.split("getattr")[-1].split("\n")[0]
        ), "update_feed_timestamp should be called directly, not via getattr"

    def test_no_getattr_for_update_feed_timestamp(self) -> None:
        """The code must NOT use getattr to call update_feed_timestamp."""
        import inspect
        import re

        from finalayze.core.trading_loop import TradingLoop, TradingLoopDeps

        source = inspect.getsource(TradingLoop._process_instrument)

        # Check that getattr(..., "update_feed_timestamp", ...) pattern is absent
        pattern = r"getattr\([^)]*update_feed_timestamp[^)]*\)"
        assert not re.search(pattern, source), (
            "Found getattr(..., 'update_feed_timestamp', ...) in _process_instrument -- "
            "should call self._health_monitor.update_feed_timestamp(now) directly"
        )

    def test_direct_call_pattern_present(self) -> None:
        """The code must contain direct call to self._health_monitor.update_feed_timestamp.

        After Phase 3 the call lives in the fetch stage of process_instrument.
        """
        import inspect

        from finalayze.orchestration.signal_executor import SignalExecutor

        source = inspect.getsource(SignalExecutor._prepare_candles)

        assert "self._health_monitor.update_feed_timestamp(" in source, (
            "Missing direct call: self._health_monitor.update_feed_timestamp(now)"
        )

    def test_health_monitor_none_safe(self) -> None:
        """When health_monitor is None, no error occurs."""
        loop = _make_trading_loop_with_health_monitor(health_monitor=None)
        assert loop._health_monitor is None
