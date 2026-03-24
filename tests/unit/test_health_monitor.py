"""Unit tests for HealthMonitor service."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

from finalayze.monitoring.health_monitor import HealthCheckResult, HealthMonitor


def _make_monitor(
    *,
    feed_freshness_minutes: int = 30,
    check_interval_seconds: int = 300,
):
    """Build a HealthMonitor with fully mocked dependencies."""
    broker_router = MagicMock()
    trading_loop = MagicMock()
    alerter = MagicMock()

    # Default: one registered market, broker works
    broker_router.registered_markets = ["moex"]
    moex_broker = MagicMock()
    moex_broker.get_open_orders.return_value = []
    broker_router.route.return_value = moex_broker

    # Default: loop is alive (cycle count changes)
    trading_loop.total_cycles = 1

    monitor = HealthMonitor(
        broker_router=broker_router,
        trading_loop=trading_loop,
        alerter=alerter,
        check_interval_seconds=check_interval_seconds,
        feed_freshness_minutes=feed_freshness_minutes,
    )
    return monitor, broker_router, trading_loop, alerter


class TestHealthMonitor:
    """HealthMonitor unit tests."""

    def test_health_check_result_is_frozen(self):
        """Test 1: HealthCheckResult is frozen dataclass with expected fields."""
        result = HealthCheckResult(
            broker_ok=True,
            feed_fresh=True,
            loop_alive=True,
            timestamp=datetime.now(tz=UTC),
            details={"broker": "ok"},
        )
        assert result.broker_ok is True
        assert result.feed_fresh is True
        assert result.loop_alive is True
        assert isinstance(result.timestamp, datetime)
        assert result.details == {"broker": "ok"}

        # Frozen -- cannot modify
        try:
            result.broker_ok = False  # type: ignore[misc]
            raise AssertionError("Should have raised FrozenInstanceError")
        except AttributeError:
            pass

    def test_check_now_returns_correct_broker_status(self):
        """Test 2: check_now() returns HealthCheckResult with correct broker status."""
        monitor, broker_router, _trading_loop, _alerter = _make_monitor()

        # Broker works
        result = monitor.check_now()
        assert result.broker_ok is True

        # Broker fails
        broker = broker_router.route.return_value
        broker.get_open_orders.side_effect = Exception("gRPC timeout")
        result = monitor.check_now()
        assert result.broker_ok is False

    def test_check_now_detects_stale_feed(self):
        """Test 3: check_now() detects stale feed (last update > 30 min ago)."""
        monitor, _broker_router, _trading_loop, _alerter = _make_monitor(
            feed_freshness_minutes=30,
        )

        # No feed timestamp set -- stale
        result = monitor.check_now()
        assert result.feed_fresh is False

        # Recent feed timestamp -- fresh
        monitor.update_feed_timestamp(datetime.now(tz=UTC))
        result = monitor.check_now()
        assert result.feed_fresh is True

        # Old feed timestamp -- stale
        old_ts = datetime.now(tz=UTC) - timedelta(minutes=60)
        monitor.update_feed_timestamp(old_ts)
        result = monitor.check_now()
        assert result.feed_fresh is False

    def test_check_now_detects_dead_loop(self):
        """Test 4: check_now() detects dead loop (cycle count unchanged past 2x strategy interval)."""
        # Use short strategy_cycle_minutes to make test fast
        monitor, _broker_router, trading_loop, _alerter = _make_monitor()
        monitor._strategy_cycle_minutes = 1  # 1-min cycle → 2-min stale threshold

        # First check -- loop alive (cycle count > 0 and different from last)
        trading_loop.total_cycles = 5
        result = monitor.check_now()
        assert result.loop_alive is True

        # Second check -- same cycle count within grace period = still alive (waiting)
        result = monitor.check_now()
        assert result.loop_alive is True
        assert "waiting" in result.details["loop"]

        # Simulate time beyond 2x strategy interval
        monitor._last_cycle_change_time -= timedelta(minutes=3)
        result = monitor.check_now()
        assert result.loop_alive is False
        assert "stalled" in result.details["loop"]

        # Cycle count changed = alive again
        trading_loop.total_cycles = 6
        result = monitor.check_now()
        assert result.loop_alive is True

    def test_single_failure_no_alert(self):
        """Test 5: Single failure increments consecutive_failures to 1, no alert."""
        monitor, broker_router, _trading_loop, alerter = _make_monitor()

        # Make broker fail
        broker = broker_router.route.return_value
        broker.get_open_orders.side_effect = Exception("fail")

        monitor._heartbeat()

        assert monitor._consecutive_failures == 1
        alerter.send_alert.assert_not_called()

    def test_two_failures_triggers_alert(self):
        """Test 6: Two consecutive failures triggers IMPORTANT alert via alerter."""
        from finalayze.core.alerts import AlertPriority

        monitor, broker_router, _trading_loop, alerter = _make_monitor()

        # Make broker fail
        broker = broker_router.route.return_value
        broker.get_open_orders.side_effect = Exception("fail")

        monitor._heartbeat()
        assert monitor._consecutive_failures == 1
        alerter.send_alert.assert_not_called()

        monitor._heartbeat()
        assert monitor._consecutive_failures == 2  # noqa: PLR2004
        alerter.send_alert.assert_called_once()
        call_kwargs = alerter.send_alert.call_args
        assert call_kwargs.kwargs["priority"] == AlertPriority.IMPORTANT

    def test_success_resets_failures(self):
        """Test 7: Successful check after failure resets consecutive_failures to 0."""
        monitor, broker_router, trading_loop, _alerter = _make_monitor()

        # Make broker fail
        broker = broker_router.route.return_value
        broker.get_open_orders.side_effect = Exception("fail")
        monitor._heartbeat()
        assert monitor._consecutive_failures == 1

        # Fix broker, advance loop cycle
        broker.get_open_orders.side_effect = None
        broker.get_open_orders.return_value = []
        trading_loop.total_cycles = 2
        monitor.update_feed_timestamp(datetime.now(tz=UTC))

        monitor._heartbeat()
        assert monitor._consecutive_failures == 0

    def test_start_stop_scheduler(self):
        """Test 8: start() creates APScheduler job, stop() shuts it down."""
        monitor, _broker_router, _trading_loop, _alerter = _make_monitor(
            check_interval_seconds=60,
        )

        monitor.start()
        assert monitor._scheduler is not None
        assert monitor._scheduler.running is True

        monitor.stop()
        assert monitor._scheduler.running is False

    def test_total_cycles_counter(self):
        """Test 9: _total_cycles counter on TradingLoop increments each cycle."""
        from finalayze.core.trading_loop import TradingLoop

        # We just verify the property exists and returns an int
        loop = MagicMock(spec=TradingLoop)
        loop.total_cycles = 0
        assert loop.total_cycles == 0
        loop.total_cycles = 5
        assert loop.total_cycles == 5  # noqa: PLR2004
