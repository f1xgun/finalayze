"""Anomaly detector for sandbox monitoring metrics.

Checks each cycle's metrics against z-score and threshold rules,
fires Telegram alerts with per-metric cooldown to avoid spam.
"""

from __future__ import annotations

import statistics
import time
from collections import deque
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from finalayze.api.alerts import TelegramAlerter
    from finalayze.monitoring.sandbox_monitor import CycleMetrics

_log = structlog.get_logger()


class AnomalyDetector:
    """Detect anomalous sandbox metrics and fire alerts.

    Rules
    -----
    - **drawdown**: z-score > 2.0 over rolling window (min 3 entries).
    - **fill_rate**: below 0.90.
    - **slippage**: max slippage > 50 bps.

    Each metric has an independent 30-minute cooldown to suppress duplicate alerts.
    """

    _COOLDOWN_SECONDS: int = 1800  # 30 minutes
    _FILL_RATE_FLOOR: float = 0.90
    _SLIPPAGE_CEILING_BPS: float = 50.0
    _ZSCORE_THRESHOLD: float = 2.0
    _MIN_WINDOW_FOR_ZSCORE: int = 3

    def __init__(
        self,
        alerter: TelegramAlerter | None = None,
        window: int = 20,
    ) -> None:
        self._alerter = alerter
        self._window = window
        self._drawdown_history: deque[float] = deque(maxlen=window)
        self._last_alert: dict[str, float] = {}

    def check(self, metrics: CycleMetrics) -> list[str]:
        """Check metrics for anomalies. Returns list of triggered metric names."""
        triggered: list[str] = []
        if self._check_drawdown_zscore(metrics.drawdown_pct):
            triggered.append("drawdown")
        if self._check_fill_rate(metrics.fill_rate):
            triggered.append("fill_rate")
        if self._check_slippage(metrics.max_slippage_bps):
            triggered.append("slippage")
        return triggered

    # ── Internal checks ───────────────────────────────────────────────────────

    def _check_drawdown_zscore(self, drawdown_pct: float) -> bool:
        self._drawdown_history.append(drawdown_pct)
        if len(self._drawdown_history) < self._MIN_WINDOW_FOR_ZSCORE:
            return False
        mean = statistics.mean(self._drawdown_history)
        stdev = statistics.stdev(self._drawdown_history)
        if stdev == 0:
            return False
        z = (drawdown_pct - mean) / stdev
        if z > self._ZSCORE_THRESHOLD and self._is_cooled_down("drawdown"):
            self._fire_alert("drawdown", drawdown_pct, self._ZSCORE_THRESHOLD)
            return True
        return False

    def _check_fill_rate(self, fill_rate: float) -> bool:
        if fill_rate < self._FILL_RATE_FLOOR and self._is_cooled_down("fill_rate"):
            self._fire_alert("fill_rate", fill_rate, self._FILL_RATE_FLOOR)
            return True
        return False

    def _check_slippage(self, max_slippage_bps: float) -> bool:
        if max_slippage_bps > self._SLIPPAGE_CEILING_BPS and self._is_cooled_down("slippage"):
            self._fire_alert("slippage", max_slippage_bps, self._SLIPPAGE_CEILING_BPS)
            return True
        return False

    def _is_cooled_down(self, metric: str) -> bool:
        last = self._last_alert.get(metric, 0.0)
        return (time.monotonic() - last) >= self._COOLDOWN_SECONDS

    def _fire_alert(self, metric: str, value: float, threshold: float) -> None:
        self._last_alert[metric] = time.monotonic()
        msg = f"Sandbox anomaly: {metric} = {value:.2f} (threshold: {threshold:.2f})"
        _log.warning("sandbox_anomaly", metric=metric, value=value, threshold=threshold)
        if self._alerter is not None:
            from finalayze.api.alerts import AlertPriority  # noqa: PLC0415

            self._alerter.send_alert(msg, priority=AlertPriority.CRITICAL)
