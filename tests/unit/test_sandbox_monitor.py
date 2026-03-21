"""Unit tests for SandboxMonitorService and related models."""

from __future__ import annotations

from dataclasses import fields
from datetime import UTC, datetime
from decimal import Decimal
from typing import ClassVar
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from finalayze.monitoring.sandbox_monitor import CycleMetrics, SandboxMonitorService


def _make_metrics(**overrides: object) -> CycleMetrics:
    """Create a CycleMetrics with sensible defaults, overridable per test."""
    defaults: dict[str, object] = {
        "timestamp": datetime(2026, 3, 21, 12, 0, 0, tzinfo=UTC),
        "trade_count": 5,
        "pnl_rub": Decimal("1000.00"),
        "equity_rub": Decimal("500000.00"),
        "fill_rate": 0.95,
        "uptime_cycles": 10,
        "signals_generated": 8,
        "errors_caught": 1,
        "max_slippage_bps": 12.5,
        "avg_slippage_bps": 5.0,
        "drawdown_pct": 0.02,
    }
    defaults.update(overrides)
    return CycleMetrics(**defaults)  # type: ignore[arg-type]


class TestCycleMetrics:
    """CycleMetrics dataclass field coverage."""

    EXPECTED_FIELDS: ClassVar[set[str]] = {
        "timestamp",
        "trade_count",
        "pnl_rub",
        "equity_rub",
        "fill_rate",
        "uptime_cycles",
        "signals_generated",
        "errors_caught",
        "max_slippage_bps",
        "avg_slippage_bps",
        "drawdown_pct",
    }

    def test_has_all_10_core_fields(self) -> None:
        field_names = {f.name for f in fields(CycleMetrics)}
        assert self.EXPECTED_FIELDS.issubset(field_names)

    def test_has_drawdown_pct(self) -> None:
        m = _make_metrics(drawdown_pct=0.05)
        assert m.drawdown_pct == pytest.approx(0.05)

    def test_frozen(self) -> None:
        m = _make_metrics()
        with pytest.raises(AttributeError):
            m.trade_count = 99  # type: ignore[misc]


class TestSandboxMonitorService:
    """SandboxMonitorService behaviour tests."""

    def test_record_slippage_appends(self) -> None:
        svc = SandboxMonitorService()
        svc.record_slippage(10.0)
        svc.record_slippage(20.0)
        assert svc.slippage_buffer == [10.0, 20.0]

    def test_cycle_count_starts_zero(self) -> None:
        svc = SandboxMonitorService()
        assert svc.cycle_count == 0

    def test_on_cycle_complete_increments_cycle_count(self) -> None:
        svc = SandboxMonitorService()
        svc._anomaly_detector = MagicMock()
        svc._anomaly_detector.check = MagicMock(return_value=[])
        with patch.object(svc, "_persist_metrics"):
            svc.on_cycle_complete(_make_metrics())
            svc.on_cycle_complete(_make_metrics())
        assert svc.cycle_count == 2

    def test_on_cycle_complete_clears_slippage_buffer(self) -> None:
        svc = SandboxMonitorService()
        svc._anomaly_detector = MagicMock()
        svc._anomaly_detector.check = MagicMock(return_value=[])
        svc.record_slippage(5.0)
        svc.record_slippage(15.0)
        with patch.object(svc, "_persist_metrics"):
            svc.on_cycle_complete(_make_metrics())
        assert svc.slippage_buffer == []

    def test_on_cycle_complete_calls_persist(self) -> None:
        svc = SandboxMonitorService()
        svc._anomaly_detector = MagicMock()
        svc._anomaly_detector.check = MagicMock(return_value=[])
        metrics = _make_metrics()
        with patch.object(svc, "_persist_metrics") as mock_persist:
            svc.on_cycle_complete(metrics)
        mock_persist.assert_called_once_with(metrics)

    def test_on_cycle_complete_calls_anomaly_check(self) -> None:
        svc = SandboxMonitorService()
        mock_detector = MagicMock()
        mock_detector.check = MagicMock(return_value=[])
        svc._anomaly_detector = mock_detector
        metrics = _make_metrics()
        with patch.object(svc, "_persist_metrics"):
            svc.on_cycle_complete(metrics)
        mock_detector.check.assert_called_once_with(metrics)

    def test_cycle_count_property(self) -> None:
        svc = SandboxMonitorService()
        assert svc.cycle_count == 0
        svc._cycle_count = 42
        assert svc.cycle_count == 42

    def test_slippage_buffer_property(self) -> None:
        svc = SandboxMonitorService()
        assert svc.slippage_buffer == []
        svc.record_slippage(7.5)
        assert svc.slippage_buffer == [7.5]


class TestSandboxMetricRow:
    """SandboxMetricRow ORM model tests."""

    def test_tablename(self) -> None:
        from finalayze.core.models import SandboxMetricRow

        assert SandboxMetricRow.__tablename__ == "sandbox_metrics"

    def test_composite_pk(self) -> None:
        from finalayze.core.models import SandboxMetricRow

        pk_cols = [c.name for c in SandboxMetricRow.__table__.primary_key.columns]
        assert "timestamp" in pk_cols
        assert "market_id" in pk_cols


class TestPersistMetricsAsync:
    """Test async persistence path with mocked DB."""

    @pytest.mark.asyncio
    async def test_persist_creates_row(self) -> None:
        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        mock_factory = MagicMock(return_value=mock_session)

        svc = SandboxMonitorService(market_id="moex")
        metrics = _make_metrics()

        with patch(
            "finalayze.core.db.get_async_session_factory",
            return_value=mock_factory,
        ):
            await svc._persist_metrics_async(metrics)

        mock_session.add.assert_called_once()
        mock_session.commit.assert_awaited_once()
