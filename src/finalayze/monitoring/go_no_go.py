"""Go/No-Go gate evaluation for sandbox-to-production promotion.

Provides ``GoNoGoReporter`` which evaluates 8 criteria against sandbox metrics
and returns a ``GateReport`` with a PROCEED / DEFER / ABORT verdict.

Thresholds are loaded from ``config/gate_thresholds.yaml`` via ``GateThresholds.from_yaml``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path  # noqa: TC003  -- used at runtime in from_yaml
from typing import TYPE_CHECKING, Any

import structlog
import yaml

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

logger = structlog.get_logger(__name__)

__all__ = [
    "CriterionResult",
    "GateReport",
    "GateThresholds",
    "GateVerdict",
    "GoNoGoReporter",
    "ThresholdConfig",
]


# ---------------------------------------------------------------------------
# Schemas (frozen dataclasses, matching CycleMetrics pattern)
# ---------------------------------------------------------------------------


class GateVerdict(StrEnum):
    """Three-tier gate outcome."""

    PROCEED = "PROCEED"
    DEFER = "DEFER"
    ABORT = "ABORT"


@dataclass(frozen=True)
class CriterionResult:
    """Result of evaluating a single gate criterion."""

    name: str
    passed: bool
    actual: float
    threshold: float
    unit: str
    critical: bool


@dataclass(frozen=True)
class GateReport:
    """Full gate evaluation report."""

    verdict: GateVerdict
    criteria: list[CriterionResult]
    sandbox_days: int
    evaluated_at: datetime
    reason: str


@dataclass(frozen=True)
class ThresholdConfig:
    """Configuration for a single gate threshold."""

    threshold: float
    critical: bool
    source: str


@dataclass(frozen=True)
class GateThresholds:
    """Complete gate threshold configuration loaded from YAML."""

    min_sandbox_days: int
    uptime_pct: ThresholdConfig
    fill_rate_pct: ThresholdConfig
    max_drawdown_pct: ThresholdConfig
    min_trades_5d: ThresholdConfig
    signal_frequency_per_day: ThresholdConfig
    critical_errors_pct: ThresholdConfig
    max_slippage_bps: ThresholdConfig
    signal_divergence_pct: ThresholdConfig

    @classmethod
    def from_yaml(cls, path: Path) -> GateThresholds:
        """Load thresholds from gate_thresholds.yaml."""
        with path.open() as f:
            raw = yaml.safe_load(f)

        gate = raw["gate"]
        criteria = gate["criteria"]

        def _tc(name: str) -> ThresholdConfig:
            c = criteria[name]
            return ThresholdConfig(
                threshold=float(c["threshold"]),
                critical=bool(c["critical"]),
                source=str(c["source"]),
            )

        return cls(
            min_sandbox_days=int(gate["min_sandbox_days"]),
            uptime_pct=_tc("uptime_pct"),
            fill_rate_pct=_tc("fill_rate_pct"),
            max_drawdown_pct=_tc("max_drawdown_pct"),
            min_trades_5d=_tc("min_trades_5d"),
            signal_frequency_per_day=_tc("signal_frequency_per_day"),
            critical_errors_pct=_tc("critical_errors_pct"),
            max_slippage_bps=_tc("max_slippage_bps"),
            signal_divergence_pct=_tc("signal_divergence_pct"),
        )


# ---------------------------------------------------------------------------
# GoNoGoReporter
# ---------------------------------------------------------------------------


class GoNoGoReporter:
    """Evaluates 8 gate criteria against sandbox metrics.

    Usage::

        thresholds = GateThresholds.from_yaml(Path("config/gate_thresholds.yaml"))
        reporter = GoNoGoReporter(thresholds, market_id="moex")
        report = await reporter.evaluate(session)
    """

    def __init__(self, thresholds: GateThresholds, market_id: str = "moex") -> None:
        self._thresholds = thresholds
        self._market_id = market_id

    async def evaluate(self, session: AsyncSession) -> GateReport:
        """On-demand evaluation reading sandbox_metrics from DB."""
        rows = await self._load_recent_metrics(session)
        sandbox_days = self._compute_sandbox_days(rows)
        now = datetime.now(UTC)

        if sandbox_days < self._thresholds.min_sandbox_days:
            return GateReport(
                verdict=GateVerdict.DEFER,
                criteria=[],
                sandbox_days=sandbox_days,
                evaluated_at=now,
                reason=(
                    f"Insufficient data: {sandbox_days} days "
                    f"< {self._thresholds.min_sandbox_days} required"
                ),
            )

        criteria = [
            self._check_uptime(rows),
            self._check_fill_rate(rows),
            self._check_max_drawdown(rows),
            self._check_trade_count(rows),
            self._check_signal_frequency(rows, sandbox_days),
            self._check_critical_errors(rows),
            self._check_slippage(rows),
            self._check_signal_divergence(rows),
        ]

        critical_fails = [c for c in criteria if c.critical and not c.passed]
        non_critical_fails = [c for c in criteria if not c.critical and not c.passed]

        if critical_fails:
            verdict = GateVerdict.ABORT
            reason = f"Critical failures: {', '.join(c.name for c in critical_fails)}"
        elif non_critical_fails:
            verdict = GateVerdict.DEFER
            reason = f"Non-critical failures: {', '.join(c.name for c in non_critical_fails)}"
        else:
            verdict = GateVerdict.PROCEED
            reason = "All 8 criteria passed"

        return GateReport(
            verdict=verdict,
            criteria=criteria,
            sandbox_days=sandbox_days,
            evaluated_at=now,
            reason=reason,
        )

    # ------------------------------------------------------------------
    # Private check methods
    # ------------------------------------------------------------------

    def _check_uptime(self, rows: list[Any]) -> CriterionResult:
        """Check uptime: (max(uptime_cycles) / len(rows)) * 100 >= threshold."""
        total = len(rows)
        max_uptime = max((float(r.uptime_cycles) for r in rows), default=0.0)
        actual = (max_uptime / max(total, 1)) * 100.0
        t = self._thresholds.uptime_pct
        return CriterionResult(
            name="uptime_pct",
            passed=actual >= t.threshold,
            actual=actual,
            threshold=t.threshold,
            unit="%",
            critical=t.critical,
        )

    def _check_fill_rate(self, rows: list[Any]) -> CriterionResult:
        """Check average fill rate * 100 vs threshold."""
        fill_rates = [float(r.fill_rate) for r in rows if r.fill_rate is not None]
        avg_fill = (sum(fill_rates) / max(len(fill_rates), 1)) * 100.0 if fill_rates else 0.0
        t = self._thresholds.fill_rate_pct
        return CriterionResult(
            name="fill_rate_pct",
            passed=avg_fill >= t.threshold,
            actual=avg_fill,
            threshold=t.threshold,
            unit="%",
            critical=t.critical,
        )

    def _check_max_drawdown(self, rows: list[Any]) -> CriterionResult:
        """Check max(drawdown_pct) * 100 vs threshold (inverted: actual < threshold)."""
        drawdowns = [float(r.drawdown_pct) for r in rows if r.drawdown_pct is not None]
        max_dd = max(drawdowns, default=0.0) * 100.0
        t = self._thresholds.max_drawdown_pct
        return CriterionResult(
            name="max_drawdown_pct",
            passed=max_dd <= t.threshold,
            actual=max_dd,
            threshold=t.threshold,
            unit="%",
            critical=t.critical,
        )

    def _check_trade_count(self, rows: list[Any]) -> CriterionResult:
        """Check sum(trade_count) vs threshold."""
        total_trades = sum(int(r.trade_count) for r in rows)
        t = self._thresholds.min_trades_5d
        return CriterionResult(
            name="min_trades_5d",
            passed=total_trades >= t.threshold,
            actual=float(total_trades),
            threshold=t.threshold,
            unit="trades",
            critical=t.critical,
        )

    def _check_signal_frequency(self, rows: list[Any], days: int) -> CriterionResult:
        """Check sum(signals_generated) / max(days, 1) vs threshold."""
        total_signals = sum(int(r.signals_generated) for r in rows)
        freq = total_signals / max(days, 1)
        t = self._thresholds.signal_frequency_per_day
        return CriterionResult(
            name="signal_frequency_per_day",
            passed=freq >= t.threshold,
            actual=freq,
            threshold=t.threshold,
            unit="signals/day",
            critical=t.critical,
        )

    def _check_critical_errors(self, rows: list[Any]) -> CriterionResult:
        """Check sum(errors_caught) / max(len(rows), 1) * 100 vs threshold."""
        total_errors = sum(int(r.errors_caught) for r in rows)
        error_rate = (total_errors / max(len(rows), 1)) * 100.0
        t = self._thresholds.critical_errors_pct
        return CriterionResult(
            name="critical_errors_pct",
            passed=error_rate <= t.threshold,
            actual=error_rate,
            threshold=t.threshold,
            unit="%",
            critical=t.critical,
        )

    def _check_slippage(self, rows: list[Any]) -> CriterionResult:
        """Check max(max_slippage_bps) vs threshold."""
        slippages = [float(r.max_slippage_bps) for r in rows if r.max_slippage_bps is not None]
        max_slip = max(slippages, default=0.0)
        t = self._thresholds.max_slippage_bps
        return CriterionResult(
            name="max_slippage_bps",
            passed=max_slip <= t.threshold,
            actual=max_slip,
            threshold=t.threshold,
            unit="bps",
            critical=t.critical,
        )

    def _check_signal_divergence(self, rows: list[Any]) -> CriterionResult:  # noqa: ARG002
        """Placeholder: no backtest comparison data available yet."""
        t = self._thresholds.signal_divergence_pct
        return CriterionResult(
            name="signal_divergence_pct",
            passed=True,
            actual=0.0,
            threshold=t.threshold,
            unit="%",
            critical=t.critical,
        )

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    async def _load_recent_metrics(self, session: AsyncSession) -> list[Any]:
        """Query SandboxMetricRow WHERE market_id = self._market_id ORDER BY timestamp."""
        from sqlalchemy import select  # noqa: PLC0415

        from finalayze.core.models import SandboxMetricRow  # noqa: PLC0415

        stmt = (
            select(SandboxMetricRow)
            .where(SandboxMetricRow.market_id == self._market_id)
            .order_by(SandboxMetricRow.timestamp)
        )
        result = await session.execute(stmt)
        return list(result.scalars().all())

    @staticmethod
    def _compute_sandbox_days(rows: list[Any]) -> int:
        """Count distinct dates from row timestamps."""
        if not rows:
            return 0
        dates = {r.timestamp.date() for r in rows}
        return len(dates)
