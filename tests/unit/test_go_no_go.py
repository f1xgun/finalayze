"""Unit tests for GoNoGoReporter and gate evaluation logic."""

from __future__ import annotations

import textwrap
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from finalayze.monitoring.go_no_go import (
    CriterionResult,
    GateReport,
    GateThresholds,
    GateVerdict,
    GoNoGoReporter,
    ThresholdConfig,
)

# ---------------------------------------------------------------------------
# Helpers: mock SandboxMetricRow-like objects
# ---------------------------------------------------------------------------


@dataclass
class FakeMetricRow:
    """Minimal stand-in for SandboxMetricRow used in tests."""

    timestamp: datetime
    market_id: str = "moex"
    trade_count: int = 5
    pnl_rub: Decimal | None = Decimal("100.00")
    equity_rub: Decimal = Decimal("100000.00")
    fill_rate: Decimal | None = Decimal("0.9800")
    uptime_cycles: int = 100
    signals_generated: int = 10
    errors_caught: int = 0
    max_slippage_bps: Decimal | None = Decimal("5.00")
    avg_slippage_bps: Decimal | None = Decimal("3.00")
    drawdown_pct: Decimal | None = Decimal("0.0050")


def _make_rows(
    days: int = 5,
    *,
    base_date: datetime | None = None,
    rows_per_day: int = 10,
    **overrides: Any,
) -> list[FakeMetricRow]:
    """Create a list of fake metric rows spanning ``days`` trading days."""
    if base_date is None:
        base_date = datetime(2026, 3, 15, tzinfo=UTC)
    rows: list[FakeMetricRow] = []
    for d in range(days):
        for h in range(rows_per_day):
            ts = base_date + timedelta(days=d, hours=h)
            rows.append(FakeMetricRow(timestamp=ts, **overrides))
    return rows


def _default_thresholds() -> GateThresholds:
    """Return default thresholds matching the YAML config structure."""
    return GateThresholds(
        min_sandbox_days=5,
        uptime_pct=ThresholdConfig(threshold=99.0, critical=True, source="test"),
        fill_rate_pct=ThresholdConfig(threshold=95.0, critical=True, source="test"),
        max_drawdown_pct=ThresholdConfig(threshold=2.0, critical=True, source="test"),
        min_trades_5d=ThresholdConfig(threshold=5.0, critical=False, source="test"),
        signal_frequency_per_day=ThresholdConfig(threshold=1.0, critical=False, source="test"),
        critical_errors_pct=ThresholdConfig(threshold=1.0, critical=True, source="test"),
        max_slippage_bps=ThresholdConfig(threshold=50.0, critical=False, source="test"),
        signal_divergence_pct=ThresholdConfig(threshold=50.0, critical=False, source="test"),
    )


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------


class TestGateVerdict:
    """GateVerdict enum tests."""

    def test_verdict_values(self) -> None:
        assert GateVerdict.PROCEED == "PROCEED"
        assert GateVerdict.DEFER == "DEFER"
        assert GateVerdict.ABORT == "ABORT"

    def test_verdict_is_str_enum(self) -> None:
        assert isinstance(GateVerdict.PROCEED, str)


class TestCriterionResult:
    """CriterionResult dataclass tests."""

    def test_fields(self) -> None:
        cr = CriterionResult(
            name="uptime",
            passed=True,
            actual=99.5,
            threshold=99.0,
            unit="%",
            critical=True,
        )
        assert cr.name == "uptime"
        assert cr.passed is True
        assert cr.actual == 99.5
        assert cr.threshold == 99.0
        assert cr.unit == "%"
        assert cr.critical is True

    def test_frozen(self) -> None:
        cr = CriterionResult(
            name="x", passed=True, actual=1.0, threshold=1.0, unit="", critical=False
        )
        with pytest.raises(AttributeError):
            cr.name = "y"  # type: ignore[misc]


class TestGateReport:
    """GateReport dataclass tests."""

    def test_fields(self) -> None:
        now = datetime.now(UTC)
        report = GateReport(
            verdict=GateVerdict.PROCEED,
            criteria=[],
            sandbox_days=5,
            evaluated_at=now,
            reason="All good",
        )
        assert report.verdict == GateVerdict.PROCEED
        assert report.criteria == []
        assert report.sandbox_days == 5
        assert report.evaluated_at == now
        assert report.reason == "All good"


class TestGateThresholds:
    """GateThresholds loading tests."""

    def test_from_yaml(self, tmp_path: Path) -> None:
        yaml_content = textwrap.dedent("""\
            gate:
              min_sandbox_days: 5
              criteria:
                uptime_pct:
                  threshold: 99.0
                  critical: true
                  source: "engineering default"
                fill_rate_pct:
                  threshold: 95.0
                  critical: true
                  source: "engineering default"
                max_drawdown_pct:
                  threshold: 2.5
                  critical: true
                  source: "p90 of wf_max_drawdown from history.jsonl"
                min_trades_5d:
                  threshold: 10.0
                  critical: false
                  source: "p10 of trade_count from history.jsonl"
                signal_frequency_per_day:
                  threshold: 1.0
                  critical: false
                  source: "engineering default"
                critical_errors_pct:
                  threshold: 1.0
                  critical: true
                  source: "engineering default"
                max_slippage_bps:
                  threshold: 50.0
                  critical: false
                  source: "engineering default"
                signal_divergence_pct:
                  threshold: 50.0
                  critical: false
                  source: "engineering default"
        """)
        yaml_file = tmp_path / "gate_thresholds.yaml"
        yaml_file.write_text(yaml_content)

        thresholds = GateThresholds.from_yaml(yaml_file)
        assert thresholds.min_sandbox_days == 5
        assert thresholds.uptime_pct.threshold == 99.0
        assert thresholds.uptime_pct.critical is True
        assert thresholds.max_drawdown_pct.threshold == 2.5
        assert thresholds.min_trades_5d.critical is False


# ---------------------------------------------------------------------------
# GoNoGoReporter evaluation tests
# ---------------------------------------------------------------------------


class TestGoNoGoReporter:
    """GoNoGoReporter.evaluate tests."""

    @pytest.fixture
    def reporter(self) -> GoNoGoReporter:
        return GoNoGoReporter(thresholds=_default_thresholds(), market_id="moex")

    @pytest.fixture
    def mock_session(self) -> AsyncMock:
        return AsyncMock()

    @pytest.mark.asyncio
    async def test_defer_when_insufficient_data(
        self, reporter: GoNoGoReporter, mock_session: AsyncMock
    ) -> None:
        """DEFER when sandbox_days < min_sandbox_days (5)."""
        rows = _make_rows(days=3)
        with patch.object(reporter, "_load_recent_metrics", return_value=rows):
            report = await reporter.evaluate(mock_session)

        assert report.verdict == GateVerdict.DEFER
        assert report.sandbox_days == 3
        assert "Insufficient data" in report.reason

    @pytest.mark.asyncio
    async def test_proceed_when_all_pass(
        self, reporter: GoNoGoReporter, mock_session: AsyncMock
    ) -> None:
        """PROCEED when all 8 criteria pass."""
        rows = _make_rows(days=5, trade_count=5, fill_rate=Decimal("0.9800"))
        with patch.object(reporter, "_load_recent_metrics", return_value=rows):
            report = await reporter.evaluate(mock_session)

        assert report.verdict == GateVerdict.PROCEED
        assert len(report.criteria) == 8
        assert all(c.passed for c in report.criteria)

    @pytest.mark.asyncio
    async def test_abort_when_critical_fails_uptime(
        self, reporter: GoNoGoReporter, mock_session: AsyncMock
    ) -> None:
        """ABORT when critical criterion (uptime) fails."""
        # uptime_cycles=1 out of many rows = very low uptime
        rows = _make_rows(days=5, uptime_cycles=1)
        with patch.object(reporter, "_load_recent_metrics", return_value=rows):
            report = await reporter.evaluate(mock_session)

        assert report.verdict == GateVerdict.ABORT
        assert "Critical failures" in report.reason

    @pytest.mark.asyncio
    async def test_abort_when_critical_fails_fill_rate(
        self, reporter: GoNoGoReporter, mock_session: AsyncMock
    ) -> None:
        """ABORT when fill_rate is below threshold."""
        rows = _make_rows(days=5, fill_rate=Decimal("0.5000"))
        with patch.object(reporter, "_load_recent_metrics", return_value=rows):
            report = await reporter.evaluate(mock_session)

        assert report.verdict == GateVerdict.ABORT

    @pytest.mark.asyncio
    async def test_abort_when_max_drawdown_exceeds_threshold(
        self, reporter: GoNoGoReporter, mock_session: AsyncMock
    ) -> None:
        """ABORT when max drawdown exceeds threshold."""
        rows = _make_rows(days=5, drawdown_pct=Decimal("0.0500"))  # 5% > 2% threshold
        with patch.object(reporter, "_load_recent_metrics", return_value=rows):
            report = await reporter.evaluate(mock_session)

        assert report.verdict == GateVerdict.ABORT

    @pytest.mark.asyncio
    async def test_abort_when_critical_errors_too_high(
        self, reporter: GoNoGoReporter, mock_session: AsyncMock
    ) -> None:
        """ABORT when critical error rate exceeds threshold."""
        rows = _make_rows(days=5, errors_caught=10)  # 10 errors per row = high rate
        with patch.object(reporter, "_load_recent_metrics", return_value=rows):
            report = await reporter.evaluate(mock_session)

        assert report.verdict == GateVerdict.ABORT

    @pytest.mark.asyncio
    async def test_defer_when_non_critical_fails(
        self, reporter: GoNoGoReporter, mock_session: AsyncMock
    ) -> None:
        """DEFER when only non-critical criterion (trade_count) fails."""
        rows = _make_rows(days=5, trade_count=0)  # 0 trades < 5 threshold
        with patch.object(reporter, "_load_recent_metrics", return_value=rows):
            report = await reporter.evaluate(mock_session)

        assert report.verdict == GateVerdict.DEFER
        assert "Non-critical failures" in report.reason

    @pytest.mark.asyncio
    async def test_produces_exactly_8_criteria(
        self, reporter: GoNoGoReporter, mock_session: AsyncMock
    ) -> None:
        """Evaluate produces exactly 8 CriterionResult entries."""
        rows = _make_rows(days=5)
        with patch.object(reporter, "_load_recent_metrics", return_value=rows):
            report = await reporter.evaluate(mock_session)

        assert len(report.criteria) == 8

    @pytest.mark.asyncio
    async def test_check_uptime_computes_correctly(
        self, reporter: GoNoGoReporter, mock_session: AsyncMock
    ) -> None:
        """_check_uptime: (max(uptime_cycles) / len(rows)) * 100."""
        rows = _make_rows(days=5, uptime_cycles=50, rows_per_day=10)
        # max(uptime_cycles) = 50, len(rows) = 50 => 100%
        result = reporter._check_uptime(rows)
        assert result.passed is True
        assert result.actual == pytest.approx(100.0)

    @pytest.mark.asyncio
    async def test_check_max_drawdown_uses_max(
        self, reporter: GoNoGoReporter, mock_session: AsyncMock
    ) -> None:
        """_check_max_drawdown: max(drawdown_pct) over observation window."""
        rows = _make_rows(days=5, drawdown_pct=Decimal("0.0100"))  # 1% < 2% threshold
        # Override one row with higher drawdown
        rows[0] = FakeMetricRow(
            timestamp=rows[0].timestamp,
            drawdown_pct=Decimal("0.0150"),  # 1.5% still < 2%
        )
        result = reporter._check_max_drawdown(rows)
        assert result.passed is True
        assert result.actual == pytest.approx(1.5)  # max is 1.5%

    @pytest.mark.asyncio
    async def test_defer_empty_rows_after_min_days_check(
        self, reporter: GoNoGoReporter, mock_session: AsyncMock
    ) -> None:
        """DEFER with 0 days of data."""
        with patch.object(reporter, "_load_recent_metrics", return_value=[]):
            report = await reporter.evaluate(mock_session)

        assert report.verdict == GateVerdict.DEFER
        assert report.sandbox_days == 0

    @pytest.mark.asyncio
    async def test_signal_divergence_placeholder_passes(
        self, reporter: GoNoGoReporter, mock_session: AsyncMock
    ) -> None:
        """Signal divergence placeholder returns passed=True."""
        rows = _make_rows(days=5)
        result = reporter._check_signal_divergence(rows)
        assert result.passed is True
        assert result.actual == 0.0
