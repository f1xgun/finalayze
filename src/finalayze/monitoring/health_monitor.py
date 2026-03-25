"""HealthMonitor -- periodic health checks for production liveness (Layer 6).

Checks broker connectivity, feed freshness, and trading loop liveness on a
configurable interval (default: 5 minutes). Alerts on 2 consecutive failures
via TelegramAlerter with IMPORTANT priority.

See docs/architecture/DEPENDENCY_LAYERS.md for layering rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import structlog
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

if TYPE_CHECKING:
    from finalayze.api.alerts import TelegramAlerter
    from finalayze.orchestration.trading_loop import TradingLoop
    from finalayze.execution.broker_router import BrokerRouter

_log = structlog.get_logger()


@dataclass(frozen=True)
class HealthCheckResult:
    """Immutable result of a single health check cycle."""

    broker_ok: bool
    feed_fresh: bool
    loop_alive: bool
    timestamp: datetime
    details: dict[str, str]


class HealthMonitor:
    """Periodic health monitor with 2-miss alerting.

    Constructor-injected dependencies keep this class testable with mocks.
    Uses TYPE_CHECKING guard to avoid importing from upper layers at module level.
    """

    def __init__(
        self,
        broker_router: BrokerRouter,
        trading_loop: TradingLoop,
        alerter: TelegramAlerter,
        check_interval_seconds: int = 300,
        feed_freshness_minutes: int = 30,
        strategy_cycle_minutes: int = 60,
    ) -> None:
        self._broker_router = broker_router
        self._trading_loop = trading_loop
        self._alerter = alerter
        self._check_interval_seconds = check_interval_seconds
        self._feed_freshness_minutes = feed_freshness_minutes
        self._strategy_cycle_minutes = strategy_cycle_minutes

        self._consecutive_failures: int = 0
        self._last_cycle_count: int = 0
        self._last_cycle_change_time: datetime = datetime.now(tz=UTC)
        self._last_result: HealthCheckResult | None = None
        self._last_feed_timestamp: datetime | None = None
        self._scheduler: BackgroundScheduler | None = None

    def check_now(self) -> HealthCheckResult:
        """Run all health checks and return a structured result.

        Checks:
          1. Broker connectivity: call get_open_orders on first registered market
          2. Feed freshness: last feed timestamp within feed_freshness_minutes
          3. Loop liveness: total_cycles changed since last check
        """
        details: dict[str, str] = {}

        # Check 1: Broker connectivity
        broker_ok = False
        try:
            markets = self._broker_router.registered_markets
            if markets:
                broker = self._broker_router.route(markets[0])
                broker.get_open_orders()
                broker_ok = True
                details["broker"] = "ok"
            else:
                broker_ok = True
                details["broker"] = "no markets registered"
        except Exception as exc:
            details["broker"] = f"error: {exc}"

        # Check 2: Feed freshness
        # During market closed hours (MOEX: 18:50-06:50 UTC, weekends) feed is
        # not updated because no strategy cycles run.  Treat feed as fresh
        # during off-hours to avoid false stale alerts overnight.
        feed_fresh = False
        now = datetime.now(tz=UTC)
        market_open = self._is_market_hours(now)
        if self._last_feed_timestamp is not None:
            age = now - self._last_feed_timestamp
            if age <= timedelta(minutes=self._feed_freshness_minutes):
                feed_fresh = True
                details["feed"] = f"fresh (age: {age.total_seconds():.0f}s)"
            elif not market_open:
                feed_fresh = True
                details["feed"] = f"off-hours (age: {age.total_seconds():.0f}s, market closed)"
            else:
                details["feed"] = f"stale (age: {age.total_seconds():.0f}s)"
        else:
            if not market_open:
                feed_fresh = True
                details["feed"] = "off-hours (no feed yet, market closed)"
            else:
                details["feed"] = "no feed timestamp set"

        # Check 3: Loop liveness
        # Compare cycle count change against strategy cycle interval, not health
        # check interval.  Strategy cycles run every strategy_cycle_minutes (e.g.
        # 60 min) while health checks run every 5 min.  Only flag as stalled when
        # no new cycle has completed for 2x the strategy interval.
        current_cycles = self._trading_loop.total_cycles
        now = datetime.now(tz=UTC)
        loop_alive = False
        if current_cycles != self._last_cycle_count:
            loop_alive = True
            self._last_cycle_change_time = now
            details["loop"] = f"alive (cycles: {current_cycles})"
        elif self._last_cycle_count == 0 and current_cycles == 0:
            # First check, no cycles yet -- treat as alive (not started yet)
            loop_alive = True
            details["loop"] = "not started yet"
        else:
            # Cycle count unchanged -- stalled only if too long since last change
            stale_threshold = timedelta(minutes=self._strategy_cycle_minutes * 2)
            elapsed = now - self._last_cycle_change_time
            if elapsed <= stale_threshold:
                loop_alive = True
                remaining = stale_threshold - elapsed
                details["loop"] = (
                    f"waiting (cycles: {current_cycles}, "
                    f"next expected in {remaining.total_seconds():.0f}s)"
                )
            else:
                details["loop"] = f"stalled at {current_cycles} cycles"
        self._last_cycle_count = current_cycles

        result = HealthCheckResult(
            broker_ok=broker_ok,
            feed_fresh=feed_fresh,
            loop_alive=loop_alive,
            timestamp=datetime.now(tz=UTC),
            details=details,
        )
        self._last_result = result
        return result

    def _heartbeat(self) -> None:
        """Run health check and manage failure counter / alerting."""
        from finalayze.api.alerts import AlertPriority  # noqa: PLC0415

        result = self.check_now()
        all_ok = result.broker_ok and result.feed_fresh and result.loop_alive

        if all_ok:
            self._consecutive_failures = 0
            _log.debug("health_check_ok", details=result.details)
        else:
            self._consecutive_failures += 1
            _log.warning(
                "health_check_failed",
                consecutive_failures=self._consecutive_failures,
                broker_ok=result.broker_ok,
                feed_fresh=result.feed_fresh,
                loop_alive=result.loop_alive,
                details=result.details,
            )
            _alert_threshold = 2
            if self._consecutive_failures >= _alert_threshold:
                failed_checks = []
                if not result.broker_ok:
                    failed_checks.append("broker")
                if not result.feed_fresh:
                    failed_checks.append("feed")
                if not result.loop_alive:
                    failed_checks.append("loop")
                message = (
                    f"Health check failed {self._consecutive_failures}x: "
                    f"{', '.join(failed_checks)}\n"
                    f"Details: {result.details}"
                )
                self._alerter.send_alert(message, priority=AlertPriority.IMPORTANT)

    def start(self) -> None:
        """Start the periodic health check scheduler."""
        self._scheduler = BackgroundScheduler()
        self._scheduler.add_job(
            self._heartbeat,
            trigger=IntervalTrigger(seconds=self._check_interval_seconds),
            id="health_monitor_heartbeat",
            replace_existing=True,
        )
        self._scheduler.start()

    def stop(self) -> None:
        """Stop the periodic health check scheduler."""
        if self._scheduler is not None and self._scheduler.running:
            self._scheduler.shutdown(wait=False)

    @property
    def last_result(self) -> HealthCheckResult | None:
        """Return the most recent health check result, or None if never checked."""
        return self._last_result

    def update_feed_timestamp(self, ts: datetime) -> None:
        """Update the last known feed data timestamp."""
        self._last_feed_timestamp = ts

    @staticmethod
    def _is_market_hours(now: datetime) -> bool:
        """Check if MOEX is likely open (Mon-Fri, 07:00-18:50 UTC).

        Approximate — does not account for MOEX holidays or transferred
        weekends, but prevents false stale alerts overnight and on weekends.
        """
        if now.weekday() >= 5:  # Saturday=5, Sunday=6  # noqa: PLR2004
            return False
        hour = now.hour
        return 7 <= hour < 19  # noqa: PLR2004
