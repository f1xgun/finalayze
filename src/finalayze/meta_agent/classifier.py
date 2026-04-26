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

from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from finalayze.meta_agent.snapshot import AlertSummary, Snapshot

# SPEC-locked numeric thresholds (§Requirement 2 lines 33-37, §Constraints 117).
# Severity rule changes require a SPEC update — do NOT change these values
# without a corresponding 58-SPEC.md amendment.
_FIX_DRAWDOWN_THRESHOLD = 5.0
_INVESTIGATE_DRAWDOWN_THRESHOLD = 3.0
_FIX_PERSIST_FAILURE_THRESHOLD = 3
_WATCH_INFO_PER_HOUR_THRESHOLD = 5
_WATCH_ML_ERROR_RATE_THRESHOLD = 0.01

# Time windows for alert counts (SPEC line 34-36).
_FIX_ALERT_WINDOW = timedelta(minutes=30)
_WATCH_ALERT_WINDOW = timedelta(hours=1)


class Severity(StrEnum):
    """Four-level severity ladder (SPEC §Requirement 2)."""

    HEALTHY = "HEALTHY"
    WATCH = "WATCH"
    INVESTIGATE = "INVESTIGATE"
    FIX = "FIX"


def _alert_age(alert: AlertSummary, *, now: datetime) -> timedelta:
    """Return age of alert relative to ``now``. Tolerates ISO-with or
    without microseconds; returns a very-large delta on parse failure so
    the alert is excluded from any time-windowed count."""
    try:
        ts = datetime.fromisoformat(alert.timestamp)
    except ValueError:
        return timedelta(days=365)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    return now - ts


def _count_recent(
    alerts: list[AlertSummary],
    *,
    priority: str,
    window: timedelta,
    now: datetime,
) -> int:
    return sum(
        1
        for a in alerts
        if a.priority.upper() == priority and _alert_age(a, now=now) <= window
    )


def classify(snapshot: Snapshot) -> Severity:
    """Return Severity for the given snapshot (SPEC §Requirement 2).

    Rule cascade (FIX → INVESTIGATE → WATCH → HEALTHY) — first match wins.
    All thresholds module-level constants; LLM is NEVER an input here.

    D-03: when ALL critical fields are None (snapshot unusable), classify
    short-circuits to HEALTHY (the runner records rationale
    'snapshot_unusable' so dry-run row counts remain auditable).
    """
    # D-03 short-circuit: snapshot is entirely unusable.
    if (
        snapshot.alerts_last_hour is None
        and snapshot.drawdown_pct is None
        and snapshot.positions_summary is None
    ):
        return Severity.HEALTHY

    alerts = snapshot.alerts_last_hour or []
    now = snapshot.timestamp
    drawdown = snapshot.drawdown_pct
    persist_failures = snapshot.equity_persist_failures
    ml_err = snapshot.ml_signal_error_rate

    # FIX: ≥1 CRITICAL/30min OR DD>5.0 OR persist_failures>=3
    critical_30m = _count_recent(
        alerts, priority="CRITICAL", window=_FIX_ALERT_WINDOW, now=now,
    )
    if (
        critical_30m >= 1
        or (drawdown is not None and drawdown > _FIX_DRAWDOWN_THRESHOLD)
        or persist_failures >= _FIX_PERSIST_FAILURE_THRESHOLD
    ):
        return Severity.FIX

    # INVESTIGATE: ≥1 IMPORTANT/30min OR DD>3.0
    important_30m = _count_recent(
        alerts, priority="IMPORTANT", window=_FIX_ALERT_WINDOW, now=now,
    )
    if important_30m >= 1 or (
        drawdown is not None and drawdown > _INVESTIGATE_DRAWDOWN_THRESHOLD
    ):
        return Severity.INVESTIGATE

    # WATCH: ≥5 INFO/hour OR ml_err > 0.01
    info_1h = _count_recent(
        alerts, priority="INFO", window=_WATCH_ALERT_WINDOW, now=now,
    )
    if info_1h >= _WATCH_INFO_PER_HOUR_THRESHOLD or (
        ml_err is not None and ml_err > _WATCH_ML_ERROR_RATE_THRESHOLD
    ):
        return Severity.WATCH

    return Severity.HEALTHY


def summarise_with_llm(
    snapshot: Snapshot,
    llm_response: dict[str, object],
) -> tuple[str, str]:
    """Extract ``(summary, rationale)`` from the LLM response.

    SPEC §Requirement 2 LLM-rationale-only contract: the LLM may write
    free text but **may NEVER** raise the rule-derived severity. Any
    `severity` / `severity_override` field in the LLM response is
    intentionally ignored here.

    Returns ``(summary, rationale)`` strings; falls back to empty strings
    so the caller can persist a non-null row even when the LLM omits a
    field.
    """
    _ = snapshot  # snapshot is the rule input, not summarised here.
    summary_raw = llm_response.get("summary", "")
    rationale_raw = llm_response.get("rationale", "")
    summary = str(summary_raw) if summary_raw is not None else ""
    rationale = str(rationale_raw) if rationale_raw is not None else ""
    # severity_override is intentionally NOT consumed — LLM-rationale-only.
    return summary, rationale
