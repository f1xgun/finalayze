"""Deterministic severity classifier (Phase 58 META-02).

SPEC §Requirement 2: pure-Python rules; LLM may NEVER raise the rule-derived
severity. ``classify(snapshot) -> Severity`` is the single source of truth.

Rules (SPEC line 33-37):
  FIX         ≡ ≥1 CRITICAL alert in last 30min OR drawdown_pct > 5.0
                OR equity_persist_failures >= 3
  INVESTIGATE ≡ (not FIX) AND (≥1 IMPORTANT alert in last 30min
                              OR drawdown_pct > 3.0)
  WATCH       ≡ (not FIX, not INVESTIGATE) AND (count(INFO) last hour >= 5
                                                OR ml_signal_error_rate > 0.01)
  HEALTHY     ≡ otherwise

D-03 graceful-degradation: when ALL critical fields are None (snapshot
unusable), classify returns HEALTHY (the runner records rationale
"snapshot_unusable" so dry-run row counts remain auditable).
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from finalayze.meta_agent.snapshot import Snapshot


class Severity(StrEnum):
    """Four-level severity ladder (SPEC §Requirement 2)."""

    HEALTHY = "HEALTHY"
    WATCH = "WATCH"
    INVESTIGATE = "INVESTIGATE"
    FIX = "FIX"


def classify(snapshot: Snapshot) -> Severity:
    """Return Severity for the given snapshot.

    Implementation lands in Task 58-01-05; this placeholder satisfies
    Task 58-01-02's import-only contract.
    """
    raise NotImplementedError("classify() implemented in Task 58-01-05")
