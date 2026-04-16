"""Sandbox monitor service: per-cycle metric collection and DB persistence.

Captures CycleMetrics after each TradingLoop cycle, persists to TimescaleDB
``sandbox_metrics`` hypertable, and delegates anomaly checks to AnomalyDetector.
"""

from __future__ import annotations

import asyncio
import concurrent.futures  # noqa: TC003 - used at runtime for Future type
import threading
from dataclasses import dataclass
from datetime import datetime  # noqa: TC003
from decimal import Decimal
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from finalayze.api.alerts import TelegramAlerter

_log = structlog.get_logger()


@dataclass(frozen=True)
class CycleMetrics:
    """Immutable snapshot of one trading-loop cycle's key metrics."""

    timestamp: datetime
    trade_count: int
    pnl_rub: Decimal
    equity_rub: Decimal
    fill_rate: float
    uptime_cycles: int
    signals_generated: int
    errors_caught: int
    max_slippage_bps: float
    avg_slippage_bps: float
    drawdown_pct: float


class SandboxMonitorService:
    """Collects per-cycle metrics and persists them to the database.

    Parameters
    ----------
    alerter:
        Optional TelegramAlerter for anomaly notifications.
    market_id:
        Market identifier used when persisting rows (default ``"moex"``).
    """

    _persist_loop: asyncio.AbstractEventLoop | None = None
    _persist_lock: threading.Lock = threading.Lock()

    def __init__(
        self,
        alerter: TelegramAlerter | None = None,
        market_id: str = "moex",
    ) -> None:
        from finalayze.markets.schedule import SCHEDULES  # noqa: PLC0415
        from finalayze.monitoring.anomaly_detector import AnomalyDetector  # noqa: PLC0415

        self._market_id = market_id
        self._anomaly_detector = AnomalyDetector(
            alerter=alerter,
            market_schedule=SCHEDULES.get(market_id),
        )
        self._cycle_count: int = 0
        self._slippage_buffer: list[float] = []

    # ── Public API ────────────────────────────────────────────────────────────

    def record_slippage(self, slippage_bps: float) -> None:
        """Accumulate a per-order slippage value (basis points)."""
        self._slippage_buffer.append(slippage_bps)

    def on_cycle_complete(self, metrics: CycleMetrics) -> None:
        """Handle end-of-cycle: persist, check anomalies, reset buffer."""
        self._cycle_count += 1
        self._persist_metrics(metrics)
        self._anomaly_detector.check(metrics)
        self._slippage_buffer.clear()

    @property
    def cycle_count(self) -> int:
        """Number of completed cycles recorded so far."""
        return self._cycle_count

    @property
    def slippage_buffer(self) -> list[float]:
        """Current accumulated slippage values (cleared each cycle)."""
        return list(self._slippage_buffer)

    # ── Persistence ───────────────────────────────────────────────────────────

    def _persist_metrics(self, metrics: CycleMetrics) -> None:
        """Fire-and-forget async persistence wrapped for sync callers."""
        self._run_async_safe(self._persist_metrics_async(metrics))

    async def _persist_metrics_async(self, metrics: CycleMetrics) -> None:
        """Write a ``SandboxMetricRow`` to TimescaleDB."""
        from finalayze.core.db import get_async_session_factory  # noqa: PLC0415
        from finalayze.core.models import SandboxMetricRow  # noqa: PLC0415

        factory = get_async_session_factory()
        async with factory() as session:
            row = SandboxMetricRow(
                timestamp=metrics.timestamp,
                market_id=self._market_id,
                trade_count=metrics.trade_count,
                pnl_rub=metrics.pnl_rub,
                equity_rub=metrics.equity_rub,
                fill_rate=Decimal(str(metrics.fill_rate)),
                uptime_cycles=metrics.uptime_cycles,
                signals_generated=metrics.signals_generated,
                errors_caught=metrics.errors_caught,
                max_slippage_bps=Decimal(str(metrics.max_slippage_bps)),
                avg_slippage_bps=Decimal(str(metrics.avg_slippage_bps)),
                drawdown_pct=Decimal(str(metrics.drawdown_pct)),
            )
            session.add(row)
            await session.commit()
        _log.info("sandbox_metrics_persisted", market_id=self._market_id)

    def _get_persist_loop(self) -> asyncio.AbstractEventLoop:
        """Return (lazily-created) background event loop for persistence."""
        if self.__class__._persist_loop is None or self.__class__._persist_loop.is_closed():
            with self._persist_lock:
                if self.__class__._persist_loop is None or self.__class__._persist_loop.is_closed():
                    loop = asyncio.new_event_loop()
                    t = threading.Thread(
                        target=loop.run_forever,
                        daemon=True,
                        name="sandbox-persist",
                    )
                    t.start()
                    self.__class__._persist_loop = loop
        return self.__class__._persist_loop

    def _run_async_safe(self, coro: object) -> None:
        """Run a coroutine from sync context via background event loop.

        Safe to call from APScheduler thread context (no ``asyncio.run()``).
        Errors are caught and logged, never raised.
        """
        try:
            loop = self._get_persist_loop()
            future: concurrent.futures.Future[None] = asyncio.run_coroutine_threadsafe(
                coro,  # type: ignore[arg-type]
                loop,
            )
            _PERSIST_TIMEOUT_S = 10  # noqa: N806
            future.result(timeout=_PERSIST_TIMEOUT_S)
        except Exception:
            _log.debug("sandbox_metrics_persist_failed", exc_info=True)
