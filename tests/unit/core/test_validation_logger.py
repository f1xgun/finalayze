"""Tests for CycleLogEntry signal drop counter fields."""

from __future__ import annotations

from datetime import datetime, timezone

from finalayze.core.validation_logger import CycleLogEntry, ValidationLogger


def _make_entry(**overrides: object) -> CycleLogEntry:
    defaults = {
        "timestamp": datetime(2026, 4, 7, 12, 0, 0, tzinfo=timezone.utc),
        "cycle_type": "equity",
        "duration_ms": 1500,
        "instruments_processed": 10,
        "signals_generated": 3,
        "orders_submitted": 2,
        "orders_filled": 1,
        "errors_caught": 0,
        "equity_rub": 500_000.0,
        "drawdown_pct": 0.01,
        "circuit_breaker_level": 0,
    }
    defaults.update(overrides)
    return CycleLogEntry(**defaults)  # type: ignore[arg-type]


class TestCycleLogEntryDropCounters:
    """Verify the 3 new signal drop counter fields on CycleLogEntry."""

    def test_defaults_are_zero(self) -> None:
        entry = _make_entry()
        assert entry.signals_dropped_no_bars == 0
        assert entry.signals_dropped_below_threshold == 0
        assert entry.signals_dropped_pre_trade == 0

    def test_explicit_values(self) -> None:
        entry = _make_entry(
            signals_dropped_no_bars=5,
            signals_dropped_below_threshold=12,
            signals_dropped_pre_trade=3,
        )
        assert entry.signals_dropped_no_bars == 5
        assert entry.signals_dropped_below_threshold == 12
        assert entry.signals_dropped_pre_trade == 3

    def test_roundtrip_through_jsonl(self, tmp_path: object) -> None:
        """Write and read back an entry with drop counters via ValidationLogger."""
        from pathlib import Path  # noqa: PLC0415

        log_path = Path(str(tmp_path)) / "cycles.jsonl"
        logger = ValidationLogger(log_path=log_path)

        original = _make_entry(
            signals_dropped_no_bars=2,
            signals_dropped_below_threshold=7,
            signals_dropped_pre_trade=1,
        )
        logger.log_cycle(original)

        entries = logger.get_entries()
        assert len(entries) == 1
        restored = entries[0]
        assert restored.signals_dropped_no_bars == 2
        assert restored.signals_dropped_below_threshold == 7
        assert restored.signals_dropped_pre_trade == 1

    def test_backward_compat_missing_fields(self, tmp_path: object) -> None:
        """Old JSONL entries without drop fields should still parse (defaults to 0)."""
        import json
        from pathlib import Path  # noqa: PLC0415

        log_path = Path(str(tmp_path)) / "cycles.jsonl"
        # Write a legacy entry without the new fields
        legacy = {
            "timestamp": "2026-04-07T12:00:00+00:00",
            "cycle_type": "equity",
            "duration_ms": 1000,
            "instruments_processed": 5,
            "signals_generated": 2,
            "orders_submitted": 1,
            "orders_filled": 0,
            "errors_caught": 0,
            "equity_rub": 500000.0,
            "drawdown_pct": 0.01,
            "circuit_breaker_level": 0,
        }
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w") as f:
            f.write(json.dumps(legacy) + "\n")

        logger = ValidationLogger(log_path=log_path)
        entries = logger.get_entries()
        assert len(entries) == 1
        # New fields should default to 0
        assert entries[0].signals_dropped_no_bars == 0
        assert entries[0].signals_dropped_below_threshold == 0
        assert entries[0].signals_dropped_pre_trade == 0
