"""Tests for the validation report generator.

Verifies that generate_report() correctly reads CycleLogEntry data
and produces a markdown report with pass/fail assessment.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from finalayze.core.validation_logger import CycleLogEntry, ValidationLogger


def _make_entries(
    *,
    days: int = 5,
    cycles_per_day: int = 6,
    orders_per_cycle: int = 1,
    fills_per_cycle: int = 1,
    errors: int = 0,
    max_drawdown: float = 2.0,
    equity_start: float = 1_000_000.0,
) -> list[CycleLogEntry]:
    """Build sample CycleLogEntry list spanning ``days`` trading days."""
    entries: list[CycleLogEntry] = []
    base = datetime(2026, 3, 10, 10, 0, 0)
    equity = equity_start
    for d in range(days):
        day_ts = base + timedelta(days=d)
        for c in range(cycles_per_day):
            cycle_ts = day_ts + timedelta(hours=c)
            equity += 500  # slight growth
            dd = max_drawdown if (d == days - 1 and c == 0) else max_drawdown * 0.5
            entries.append(
                CycleLogEntry(
                    timestamp=cycle_ts,
                    cycle_type="equity",
                    duration_ms=1200,
                    instruments_processed=10,
                    signals_generated=2,
                    orders_submitted=orders_per_cycle,
                    orders_filled=fills_per_cycle,
                    errors_caught=errors if (d == 0 and c == 0) else 0,
                    equity_rub=equity,
                    drawdown_pct=dd,
                    circuit_breaker_level=0,
                )
            )
    return entries


class TestValidationReportPass:
    """Test that a healthy 5-day run produces PASS verdict."""

    def test_pass_verdict(self, tmp_path: Path) -> None:
        from scripts.generate_validation_report import generate_report

        log_path = tmp_path / "cycles.jsonl"
        output_path = tmp_path / "VALIDATION-REPORT.md"

        logger = ValidationLogger(log_path)
        for entry in _make_entries(days=5, orders_per_cycle=1, fills_per_cycle=1):
            logger.log_cycle(entry)

        result = generate_report(log_path, output_path)
        assert result is True
        content = output_path.read_text()
        assert "PASS" in content
        assert "FAIL" not in content or "FAIL" not in content.split("## Overall Verdict")[1]

    def test_report_contains_summary_sections(self, tmp_path: Path) -> None:
        from scripts.generate_validation_report import generate_report

        log_path = tmp_path / "cycles.jsonl"
        output_path = tmp_path / "VALIDATION-REPORT.md"

        logger = ValidationLogger(log_path)
        for entry in _make_entries(days=5):
            logger.log_cycle(entry)

        generate_report(log_path, output_path)
        content = output_path.read_text()
        assert "## Summary" in content
        assert "## Criteria Assessment" in content
        assert "## Per-Day Breakdown" in content
        assert "## Overall Verdict" in content


class TestValidationReportFailDays:
    """Test that fewer than 5 trading days produces FAIL."""

    def test_fewer_than_5_days_fail(self, tmp_path: Path) -> None:
        from scripts.generate_validation_report import generate_report

        log_path = tmp_path / "cycles.jsonl"
        output_path = tmp_path / "VALIDATION-REPORT.md"

        logger = ValidationLogger(log_path)
        for entry in _make_entries(days=3):
            logger.log_cycle(entry)

        result = generate_report(log_path, output_path)
        assert result is False
        content = output_path.read_text()
        assert "FAIL" in content


class TestValidationReportFailDrawdown:
    """Test that drawdown >5% produces FAIL."""

    def test_high_drawdown_fail(self, tmp_path: Path) -> None:
        from scripts.generate_validation_report import generate_report

        log_path = tmp_path / "cycles.jsonl"
        output_path = tmp_path / "VALIDATION-REPORT.md"

        logger = ValidationLogger(log_path)
        for entry in _make_entries(days=5, max_drawdown=7.5):
            logger.log_cycle(entry)

        result = generate_report(log_path, output_path)
        assert result is False
        content = output_path.read_text()
        assert "FAIL" in content


class TestValidationReportFailTrades:
    """Test that fewer than 10 trades produces FAIL."""

    def test_too_few_trades_fail(self, tmp_path: Path) -> None:
        from scripts.generate_validation_report import generate_report

        log_path = tmp_path / "cycles.jsonl"
        output_path = tmp_path / "VALIDATION-REPORT.md"

        logger = ValidationLogger(log_path)
        # 5 days, 1 cycle per day, 1 fill each = 5 total fills < 10
        for entry in _make_entries(days=5, cycles_per_day=1, fills_per_cycle=1):
            logger.log_cycle(entry)

        result = generate_report(log_path, output_path)
        assert result is False
        content = output_path.read_text()
        assert "FAIL" in content


class TestValidationReportFailErrors:
    """Test that critical errors produce FAIL."""

    def test_errors_fail(self, tmp_path: Path) -> None:
        from scripts.generate_validation_report import generate_report

        log_path = tmp_path / "cycles.jsonl"
        output_path = tmp_path / "VALIDATION-REPORT.md"

        logger = ValidationLogger(log_path)
        for entry in _make_entries(days=5, errors=3):
            logger.log_cycle(entry)

        result = generate_report(log_path, output_path)
        assert result is False
        content = output_path.read_text()
        assert "FAIL" in content


class TestValidationReportEmpty:
    """Test that empty data produces FAIL with 'no data'."""

    def test_empty_data_fail(self, tmp_path: Path) -> None:
        from scripts.generate_validation_report import generate_report

        log_path = tmp_path / "cycles.jsonl"
        output_path = tmp_path / "VALIDATION-REPORT.md"

        # Create empty file
        log_path.touch()

        result = generate_report(log_path, output_path)
        assert result is False
        content = output_path.read_text()
        assert "FAIL" in content
        assert "no data" in content.lower()

    def test_nonexistent_file_fail(self, tmp_path: Path) -> None:
        from scripts.generate_validation_report import generate_report

        log_path = tmp_path / "nonexistent.jsonl"
        output_path = tmp_path / "VALIDATION-REPORT.md"

        result = generate_report(log_path, output_path)
        assert result is False
        content = output_path.read_text()
        assert "FAIL" in content


class TestPerDayBreakdown:
    """Test that per-day breakdown table is correct."""

    def test_per_day_rows(self, tmp_path: Path) -> None:
        from scripts.generate_validation_report import generate_report

        log_path = tmp_path / "cycles.jsonl"
        output_path = tmp_path / "VALIDATION-REPORT.md"

        logger = ValidationLogger(log_path)
        for entry in _make_entries(days=5, cycles_per_day=4):
            logger.log_cycle(entry)

        generate_report(log_path, output_path)
        content = output_path.read_text()
        # Should have 5 day rows in the breakdown table
        assert "2026-03-10" in content
        assert "2026-03-14" in content
