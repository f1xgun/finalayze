"""Tests for meta_agent.classifier (Phase 58 META-02).

Severity StrEnum + deterministic ``classify(snapshot) -> Severity``.
SPEC §Requirement 2: rules-driven, LLM may NEVER raise severity.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from enum import StrEnum

import pytest

# SPEC-locked thresholds (PLR2004 — module-level constants).
_FIX_DD = 5.0
_INVESTIGATE_DD = 3.0
_FIX_PERSIST_FAIL = 3
_WATCH_INFO_PER_HOUR = 5
_WATCH_ML_ERR = 0.01

# Boundary fixtures — values that sit just above / on / below thresholds.
_DD_FIX_ABOVE = 5.01
_DD_FIX_AT = 5.0
_DD_INVESTIGATE_ABOVE = 3.01
_DD_INVESTIGATE_AT = 3.0
_INFO_AT = 5
_INFO_BELOW = 4
_ML_ERR_ABOVE = 0.011
_ML_ERR_AT = 0.01

_NOW = datetime(2026, 4, 26, 12, 0, tzinfo=UTC)


def test_severity_enum_members() -> None:
    """Severity must expose four StrEnum members per SPEC §Requirement 2."""
    from finalayze.meta_agent.classifier import Severity

    assert issubclass(Severity, StrEnum)
    assert Severity.HEALTHY.value == "HEALTHY"
    assert Severity.WATCH.value == "WATCH"
    assert Severity.INVESTIGATE.value == "INVESTIGATE"
    assert Severity.FIX.value == "FIX"
    # Exactly four members — guards against accidental additions that would
    # silently bypass the SPEC-locked rule cascade.
    assert {m.value for m in Severity} == {"HEALTHY", "WATCH", "INVESTIGATE", "FIX"}


def _alert(priority: str, *, age_min: float = 1.0, atype: str = "anomaly_raw"):
    from finalayze.meta_agent.snapshot import AlertSummary

    return AlertSummary(
        id=f"00000000-0000-0000-0000-{int(age_min * 1000):012d}",
        timestamp=(_NOW - timedelta(minutes=age_min)).isoformat(),
        alert_type=atype,
        priority=priority,
        symbol=None,
        market_id=None,
        message="x",
        parent_id=None,
        delivery_status="sent",
    )


def _snap(
    *,
    alerts: list | None = None,
    drawdown: float | None = 0.0,
    persist_fail: int = 0,
    ml_err: float | None = None,
    has_positions: bool = True,
):
    from finalayze.meta_agent.snapshot import PositionsSummary, Snapshot

    positions = PositionsSummary(raw={"positions": []}) if has_positions else None
    return Snapshot(
        timestamp=_NOW,
        alerts_last_hour=alerts if alerts is not None else [],
        drawdown_pct=drawdown,
        equity_persist_failures=persist_fail,
        ml_signal_error_rate=ml_err,
        positions_summary=positions,
        raw={},
    )


# 12 boundary fixtures + the SPEC-mandated cases (SPEC line 38).
# (description, snapshot_kwargs, expected_severity)
_FIXTURES = [
    # 1. all-zero → HEALTHY
    ("all-zero", {}, "HEALTHY"),
    # 2. 1 CRITICAL alert in 30min → FIX
    ("1 CRITICAL/30min", {"alerts": [_alert("CRITICAL", age_min=10.0)]}, "FIX"),
    # 3. drawdown_pct=5.01 → FIX (above threshold)
    ("DD=5.01", {"drawdown": _DD_FIX_ABOVE}, "FIX"),
    # 4. equity_persist_failures=3 → FIX
    ("persist_fail=3", {"persist_fail": _FIX_PERSIST_FAIL}, "FIX"),
    # 5. 1 IMPORTANT alert in 30min, no FIX trigger → INVESTIGATE
    (
        "1 IMPORTANT/30min",
        {"alerts": [_alert("IMPORTANT", age_min=10.0)]},
        "INVESTIGATE",
    ),
    # 6. DD=3.01 → INVESTIGATE
    ("DD=3.01", {"drawdown": _DD_INVESTIGATE_ABOVE}, "INVESTIGATE"),
    # 7. ml_signal_error_rate=0.011 → WATCH
    ("ml_err=0.011", {"ml_err": _ML_ERR_ABOVE}, "WATCH"),
    # 8. count(INFO)=5 → WATCH
    (
        "INFO=5",
        {"alerts": [_alert("INFO", age_min=20.0) for _ in range(_INFO_AT)]},
        "WATCH",
    ),
    # 9. DD=5.0 → INVESTIGATE (boundary — strictly > 5.0 required for FIX)
    ("DD=5.0 boundary", {"drawdown": _DD_FIX_AT}, "INVESTIGATE"),
    # 10. DD=3.0 → HEALTHY (boundary — strictly > 3.0 required for INVESTIGATE)
    ("DD=3.0 boundary", {"drawdown": _DD_INVESTIGATE_AT}, "HEALTHY"),
    # 11. INFO=4 → HEALTHY (boundary — >=5 required for WATCH)
    (
        "INFO=4 boundary",
        {"alerts": [_alert("INFO", age_min=20.0) for _ in range(_INFO_BELOW)]},
        "HEALTHY",
    ),
    # 12. ml_err=0.01 → HEALTHY (boundary — strictly > 0.01 required for WATCH)
    ("ml_err=0.01 boundary", {"ml_err": _ML_ERR_AT}, "HEALTHY"),
]


@pytest.mark.parametrize(("desc", "kwargs", "expected"), _FIXTURES)
def test_classify_boundary_fixtures(desc: str, kwargs: dict, expected: str) -> None:
    """SPEC §Requirement 2 boundary fixtures (12 cases)."""
    from finalayze.meta_agent.classifier import Severity, classify

    snap = _snap(**kwargs)
    result = classify(snap)
    assert result == Severity(expected), f"{desc}: expected {expected}, got {result.value}"


def test_llm_severity_override_does_not_raise_classifier_verdict() -> None:
    """LLM-rationale-only contract (SPEC §Requirement 2 line 38).

    The classifier consumes ONLY the snapshot. Even when the LLM JSON
    provides a `severity_override='HEALTHY'`, a snapshot whose rule-derived
    severity is FIX (e.g. equity_persist_failures=3) MUST still classify
    as FIX. The override is irrelevant to ``classify()``.
    """
    from finalayze.meta_agent.classifier import Severity, classify

    snap = _snap(persist_fail=_FIX_PERSIST_FAIL)
    # LLM JSON with a severity_override field — passed nowhere because the
    # classifier accepts only the snapshot. The contract is enforced by the
    # function signature, not by an explicit ignore.
    llm_response = {"summary": "all good", "severity_override": "HEALTHY"}
    assert llm_response  # silence ruff F841 — the value is the test fixture
    assert classify(snap) is Severity.FIX


def test_classify_snapshot_unusable_short_circuits_to_healthy() -> None:
    """D-03: when ALL critical fields are None (snapshot unusable), classify
    returns HEALTHY (the runner records rationale 'snapshot_unusable')."""
    from finalayze.meta_agent.classifier import Severity, classify
    from finalayze.meta_agent.snapshot import Snapshot

    snap = Snapshot(
        timestamp=_NOW,
        alerts_last_hour=None,
        drawdown_pct=None,
        equity_persist_failures=0,
        ml_signal_error_rate=None,
        positions_summary=None,
        raw={},
    )
    assert classify(snap) is Severity.HEALTHY
