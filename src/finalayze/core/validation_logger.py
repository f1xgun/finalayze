"""Structured JSON cycle logger for sandbox validation.

Appends one JSON line per trading cycle for post-mortem analysis.
Thread-safe: uses threading.Lock for concurrent writes.

See docs/architecture/DEPENDENCY_LAYERS.md for layering rules.
"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class CycleLogEntry:
    """One entry per trading cycle (equity or bond)."""

    timestamp: datetime
    cycle_type: Literal["equity", "bond"]
    duration_ms: int
    instruments_processed: int
    signals_generated: int
    orders_submitted: int
    orders_filled: int
    errors_caught: int
    equity_rub: float
    drawdown_pct: float
    circuit_breaker_level: int


class ValidationLogger:
    """Append-only JSONL logger for cycle validation data.

    Thread-safe: concurrent calls to log_cycle() are serialized via lock.

    Args:
        log_path: Path to the JSONL file. Parent directories created automatically.
    """

    def __init__(self, log_path: Path | None = None) -> None:
        self._log_path = log_path or Path("results/validation/cycles.jsonl")
        self._lock = threading.Lock()

    def log_cycle(self, entry: CycleLogEntry) -> None:
        """Append a single JSON line for one cycle."""
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        data = asdict(entry)
        line = json.dumps(data, default=str) + "\n"
        with self._lock, self._log_path.open("a") as f:
            f.write(line)

    def get_entries(self) -> list[CycleLogEntry]:
        """Read all entries back from the JSONL file."""
        if not self._log_path.exists():
            return []
        entries: list[CycleLogEntry] = []
        with self._log_path.open() as f:
            for raw_line in f:
                stripped = raw_line.strip()
                if not stripped:
                    continue
                data = json.loads(stripped)
                # Parse timestamp back from ISO string
                data["timestamp"] = datetime.fromisoformat(data["timestamp"])
                entries.append(CycleLogEntry(**data))
        return entries
