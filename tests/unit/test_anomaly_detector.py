"""Unit tests for AnomalyDetector z-score and threshold alerting."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from finalayze.monitoring.anomaly_detector import AnomalyDetector
from finalayze.monitoring.sandbox_monitor import CycleMetrics


def _make_metrics(**overrides: object) -> CycleMetrics:
    """Create a CycleMetrics with sensible defaults."""
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


class TestDrawdownZScore:
    """Drawdown z-score anomaly detection."""

    def test_zscore_skipped_when_window_too_small(self) -> None:
        """With < 3 entries, drawdown z-score check is skipped."""
        detector = AnomalyDetector()
        # Only 2 entries -- should not trigger
        result1 = detector.check(_make_metrics(drawdown_pct=0.01))
        result2 = detector.check(_make_metrics(drawdown_pct=0.01))
        assert "drawdown" not in result1
        assert "drawdown" not in result2

    def test_zscore_triggers_on_spike(self) -> None:
        """Drawdown z-score > 2.0 triggers alert when window >= 3."""
        detector = AnomalyDetector()
        # Build a stable baseline with enough entries
        for _ in range(10):
            detector.check(_make_metrics(drawdown_pct=0.02))
        # Now inject a large spike -- z-score should exceed 2.0
        result = detector.check(_make_metrics(drawdown_pct=0.50))
        assert "drawdown" in result

    def test_zscore_no_trigger_normal(self) -> None:
        """Stable drawdown values do not trigger."""
        detector = AnomalyDetector()
        for _ in range(5):
            result = detector.check(_make_metrics(drawdown_pct=0.02))
        assert "drawdown" not in result


class TestFillRate:
    """Fill rate threshold checks."""

    def test_low_fill_rate_triggers(self) -> None:
        detector = AnomalyDetector()
        result = detector.check(_make_metrics(fill_rate=0.85))
        assert "fill_rate" in result

    def test_normal_fill_rate_no_trigger(self) -> None:
        detector = AnomalyDetector()
        result = detector.check(_make_metrics(fill_rate=0.95))
        assert "fill_rate" not in result

    def test_boundary_fill_rate_no_trigger(self) -> None:
        detector = AnomalyDetector()
        result = detector.check(_make_metrics(fill_rate=0.90))
        assert "fill_rate" not in result


class TestSlippage:
    """Slippage threshold checks."""

    def test_high_slippage_triggers(self) -> None:
        detector = AnomalyDetector()
        result = detector.check(_make_metrics(max_slippage_bps=55.0))
        assert "slippage" in result

    def test_normal_slippage_no_trigger(self) -> None:
        detector = AnomalyDetector()
        result = detector.check(_make_metrics(max_slippage_bps=30.0))
        assert "slippage" not in result

    def test_boundary_slippage_no_trigger(self) -> None:
        detector = AnomalyDetector()
        result = detector.check(_make_metrics(max_slippage_bps=50.0))
        assert "slippage" not in result


class TestCooldown:
    """30-minute per-metric cooldown."""

    def test_cooldown_prevents_duplicate(self) -> None:
        """Same metric does not fire twice within 30 minutes."""
        detector = AnomalyDetector()
        result1 = detector.check(_make_metrics(fill_rate=0.80))
        assert "fill_rate" in result1
        # Second check within cooldown
        result2 = detector.check(_make_metrics(fill_rate=0.80))
        assert "fill_rate" not in result2

    def test_alert_fires_after_cooldown(self) -> None:
        """After cooldown expires, alert fires again."""
        detector = AnomalyDetector()
        result1 = detector.check(_make_metrics(fill_rate=0.80))
        assert "fill_rate" in result1

        # Simulate time passing beyond cooldown (1800s)
        with patch("finalayze.monitoring.anomaly_detector.time") as mock_time:
            # First call to monotonic was at some value; now return value + 1801
            mock_time.monotonic.return_value = detector._last_alert["fill_rate"] + 1801
            result2 = detector.check(_make_metrics(fill_rate=0.80))
        assert "fill_rate" in result2

    def test_independent_cooldowns(self) -> None:
        """Each metric has its own independent cooldown."""
        detector = AnomalyDetector()
        # Trigger both fill_rate and slippage
        result = detector.check(_make_metrics(fill_rate=0.80, max_slippage_bps=60.0))
        assert "fill_rate" in result
        assert "slippage" in result

        # Second check: both in cooldown
        result2 = detector.check(_make_metrics(fill_rate=0.80, max_slippage_bps=60.0))
        assert "fill_rate" not in result2
        assert "slippage" not in result2

        # Expire only fill_rate cooldown
        with patch("finalayze.monitoring.anomaly_detector.time") as mock_time:
            fill_rate_last = detector._last_alert["fill_rate"]
            mock_time.monotonic.return_value = fill_rate_last + 1801
            # slippage was set at roughly the same time, so it's also expired
            # To test independence, set slippage cooldown in the future
            detector._last_alert["slippage"] = mock_time.monotonic.return_value - 100
            result3 = detector.check(_make_metrics(fill_rate=0.80, max_slippage_bps=60.0))
        assert "fill_rate" in result3
        assert "slippage" not in result3


class TestNoAlerter:
    """Detector works with alerter=None (no-op)."""

    def test_no_crash_without_alerter(self) -> None:
        detector = AnomalyDetector(alerter=None)
        result = detector.check(_make_metrics(fill_rate=0.80, max_slippage_bps=60.0))
        assert "fill_rate" in result
        assert "slippage" in result

    def test_alert_with_alerter(self) -> None:
        mock_alerter = MagicMock()
        detector = AnomalyDetector(alerter=mock_alerter)
        detector.check(_make_metrics(fill_rate=0.80))
        mock_alerter.send_alert.assert_called_once()
