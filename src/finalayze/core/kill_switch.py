"""KillSwitch orchestrator for emergency shutdown (Layer 0/6 boundary).

Provides a single ``activate()`` call that:
  1. Cancels all open orders across all markets
  2. Stops the TradingLoop scheduler
  3. Escalates all circuit breakers to LIQUIDATE
  4. Writes a persistent kill flag file (blocks restart)
  5. Sends a CRITICAL Telegram alert

The entire sequence targets <30s completion.

See docs/architecture/DEPENDENCY_LAYERS.md for layering rules.
"""

from __future__ import annotations

import contextlib
import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from finalayze.api.alerts import TelegramAlerter
    from finalayze.execution.broker_router import BrokerRouter
    from finalayze.orchestration.trading_loop import TradingLoop
    from finalayze.risk.circuit_breaker import CircuitBreaker

_log = structlog.get_logger()


def _default_flag_path() -> Path:
    """Durable default location for the kill flag (audit 2026-06-28).

    The kill flag must survive a process restart so a killed system stays killed.
    ``/tmp`` is cleared on reboot (and on tmpfs, on restart) on many systems, so it
    is not durable. Honour ``FINALAYZE_KILL_FLAG_PATH`` when set; otherwise default
    to ``~/.finalayze/killed`` (the container home is a persistent path, unlike
    ``/tmp``). The parent directory is created on write.
    """
    override = os.environ.get("FINALAYZE_KILL_FLAG_PATH")
    if override:
        return Path(override)
    return Path.home() / ".finalayze" / "killed"


@dataclass(frozen=True)
class KillSwitchResult:
    """Immutable result of a kill switch activation."""

    orders_cancelled: int
    scheduler_stopped: bool
    breakers_escalated: int
    alert_sent: bool
    elapsed_seconds: float


class KillSwitch:
    """Emergency shutdown orchestrator.

    Constructor-injected dependencies keep this class testable with mocks.
    Uses TYPE_CHECKING guard to avoid importing from upper layers at module level.
    """

    def __init__(
        self,
        broker_router: BrokerRouter,
        trading_loop: TradingLoop,
        circuit_breakers: dict[str, CircuitBreaker],
        alerter: TelegramAlerter,
        flag_path: Path | None = None,
    ) -> None:
        self._broker_router = broker_router
        self._trading_loop = trading_loop
        self._circuit_breakers = circuit_breakers
        self._alerter = alerter
        self._flag_path = flag_path or _default_flag_path()

    def activate(self, reason: str = "manual") -> KillSwitchResult:
        """Execute full emergency shutdown sequence.

        Steps run in order; each step is wrapped in try/except so a single
        failure never aborts the remaining steps.

        Args:
            reason: Human-readable reason for the kill (logged and persisted).

        Returns:
            KillSwitchResult summarizing what was done and how long it took.
        """
        from finalayze.api.alerts import AlertPriority  # noqa: PLC0415
        from finalayze.risk.circuit_breaker import CircuitLevel  # noqa: PLC0415

        start = time.monotonic()

        # Step 1: Cancel all open orders across all markets
        orders_cancelled = 0
        for market_id in self._broker_router.registered_markets:
            try:
                broker = self._broker_router.route(market_id)
                open_orders = broker.get_open_orders()  # type: ignore[attr-defined]
                for order in open_orders:
                    try:
                        broker.cancel_order_safe(order.order_id)  # type: ignore[attr-defined]
                        orders_cancelled += 1
                    except Exception:
                        _log.warning(
                            "kill_switch_cancel_failed",
                            market=market_id,
                            order_id=order.order_id,
                        )
            except Exception:
                _log.warning("kill_switch_market_cancel_failed", market=market_id)

        # Step 2: Stop TradingLoop
        scheduler_stopped = False
        try:
            self._trading_loop.stop()
            scheduler_stopped = True
        except Exception:
            _log.warning("kill_switch_stop_loop_failed")

        # Step 3: Escalate all circuit breakers to LIQUIDATE
        breakers_escalated = 0
        for cb in self._circuit_breakers.values():
            try:
                cb.override_level(CircuitLevel.LIQUIDATE)
                breakers_escalated += 1
            except Exception:
                _log.warning("kill_switch_breaker_escalation_failed")

        # Step 4: Write persistent flag file
        try:
            timestamp = datetime.now(tz=UTC).isoformat()
            self._flag_path.parent.mkdir(parents=True, exist_ok=True)
            self._flag_path.write_text(f"killed:{reason}:{timestamp}")
        except Exception:
            _log.warning("kill_switch_flag_write_failed")

        # Step 5: Send CRITICAL alert
        elapsed = time.monotonic() - start

        alert_sent = False
        try:
            message = (
                f"KILL SWITCH ACTIVATED: {reason}\n"
                f"Orders cancelled: {orders_cancelled}\n"
                f"Scheduler stopped: {scheduler_stopped}\n"
                f"Breakers escalated: {breakers_escalated}\n"
                f"Elapsed: {elapsed:.2f}s"
            )
            self._alerter.send_alert(message, priority=AlertPriority.CRITICAL)
            alert_sent = True
        except Exception:
            _log.warning("kill_switch_alert_failed")

        _log.critical(
            "kill_switch_activated",
            reason=reason,
            orders_cancelled=orders_cancelled,
            scheduler_stopped=scheduler_stopped,
            breakers_escalated=breakers_escalated,
            elapsed_seconds=elapsed,
        )

        # Rebuild result with actual alert_sent status
        return KillSwitchResult(
            orders_cancelled=orders_cancelled,
            scheduler_stopped=scheduler_stopped,
            breakers_escalated=breakers_escalated,
            alert_sent=alert_sent,
            elapsed_seconds=elapsed,
        )

    @property
    def is_killed(self) -> bool:
        """Return True if the persistent kill flag file exists."""
        return self._flag_path.exists()

    def clear_flag(self) -> None:
        """Remove the persistent kill flag file if it exists."""
        with contextlib.suppress(FileNotFoundError):
            self._flag_path.unlink()
