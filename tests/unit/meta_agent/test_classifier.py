"""Tests for meta_agent.classifier (Phase 58 META-02).

Severity StrEnum + deterministic ``classify(snapshot) -> Severity``.
SPEC §Requirement 2: rules-driven, LLM may NEVER raise severity.
"""

from __future__ import annotations

from enum import StrEnum


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
