"""Tests for ValidationLogger -- structured JSON cycle logging."""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from pathlib import Path

import pytest

from finalayze.core.validation_logger import CycleLogEntry, ValidationLogger


def _make_entry(**overrides: object) -> CycleLogEntry:
    """Create a CycleLogEntry with sensible defaults."""
    defaults: dict[str, object] = {
        "timestamp": datetime(2026, 3, 15, 10, 0, 0, tzinfo=UTC),
        "cycle_type": "equity",
        "duration_ms": 1234,
        "instruments_processed": 5,
        "signals_generated": 2,
        "orders_submitted": 1,
        "orders_filled": 1,
        "errors_caught": 0,
        "equity_rub": 1_000_000.0,
        "drawdown_pct": 0.01,
        "circuit_breaker_level": 0,
    }
    defaults.update(overrides)
    return CycleLogEntry(**defaults)  # type: ignore[arg-type]


class TestCycleLogEntry:
    """CycleLogEntry is a frozen dataclass with all required fields."""

    def test_fields_present(self) -> None:
        entry = _make_entry()
        assert entry.cycle_type == "equity"
        assert entry.duration_ms == 1234
        assert entry.instruments_processed == 5

    def test_frozen(self) -> None:
        entry = _make_entry()
        with pytest.raises(AttributeError):
            entry.cycle_type = "bond"  # type: ignore[misc]


class TestValidationLogger:
    """ValidationLogger writes/reads structured JSON cycle entries."""

    def test_log_cycle_creates_file(self, tmp_path: Path) -> None:
        log_path = tmp_path / "cycles.jsonl"
        logger = ValidationLogger(log_path=log_path)
        entry = _make_entry()
        logger.log_cycle(entry)
        assert log_path.exists()

    def test_log_cycle_appends_json_line(self, tmp_path: Path) -> None:
        log_path = tmp_path / "cycles.jsonl"
        logger = ValidationLogger(log_path=log_path)
        logger.log_cycle(_make_entry(cycle_type="equity"))
        logger.log_cycle(_make_entry(cycle_type="bond"))
        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 2
        first = json.loads(lines[0])
        assert first["cycle_type"] == "equity"
        second = json.loads(lines[1])
        assert second["cycle_type"] == "bond"

    def test_get_entries_round_trips(self, tmp_path: Path) -> None:
        log_path = tmp_path / "cycles.jsonl"
        logger = ValidationLogger(log_path=log_path)
        original = _make_entry(duration_ms=999, equity_rub=500_000.0)
        logger.log_cycle(original)
        entries = logger.get_entries()
        assert len(entries) == 1
        assert entries[0].duration_ms == 999
        assert entries[0].equity_rub == pytest.approx(500_000.0)

    def test_get_entries_empty_file(self, tmp_path: Path) -> None:
        log_path = tmp_path / "cycles.jsonl"
        logger = ValidationLogger(log_path=log_path)
        entries = logger.get_entries()
        assert entries == []

    def test_thread_safety_concurrent_writes(self, tmp_path: Path) -> None:
        log_path = tmp_path / "cycles.jsonl"
        logger = ValidationLogger(log_path=log_path)
        num_threads = 10
        entries_per_thread = 20
        barrier = threading.Barrier(num_threads)

        def writer() -> None:
            barrier.wait()
            for i in range(entries_per_thread):
                logger.log_cycle(_make_entry(duration_ms=i))

        threads = [threading.Thread(target=writer) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        entries = logger.get_entries()
        assert len(entries) == num_threads * entries_per_thread

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        log_path = tmp_path / "sub" / "dir" / "cycles.jsonl"
        logger = ValidationLogger(log_path=log_path)
        logger.log_cycle(_make_entry())
        assert log_path.exists()
