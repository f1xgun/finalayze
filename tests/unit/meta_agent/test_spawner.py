"""Tests for meta_agent.spawner — read-only investigation subprocess (Phase 58-03).

Covers SPEC §Acceptance Criteria #10 + #11:
  - Exception classes (Task 58-03-01)
  - spawn_readonly happy path with monkeypatched subprocess (Task 58-03-04)
  - 300s timeout → SIGTERM → SIGKILL (Task 58-03-05)
  - Concurrent investigate → already_inflight (Task 58-03-06)

The CLI (`claude`) need NOT be on $PATH — tests monkeypatch
``asyncio.create_subprocess_exec`` so the spawner is exercised hermetically.
"""

from __future__ import annotations

import uuid

from finalayze.core.exceptions import FinalayzeError

# ── Module-level constants (PLR2004) ────────────────────────────────────────
_FAKE_DECISION_ID = uuid.UUID("deadbeef-0000-4000-8000-000000000001")


# ─────────────────────────────────────────────────────────────────────────────
# Task 58-03-01: MetaAgentDeniedPathError + MetaAgentSpawnCapExceededError
# ─────────────────────────────────────────────────────────────────────────────


def test_exceptions_inherit_finalayze_error_and_end_in_error() -> None:
    """SPEC AC #10 + #11 foundation: both meta-agent spawn exceptions
    inherit from FinalayzeError, end in 'Error' (N818), and raise with
    a context message.
    """
    from finalayze.meta_agent.exceptions import (
        MetaAgentDeniedPathError,
        MetaAgentSpawnCapExceededError,
    )

    # Subclass relationship.
    assert issubclass(MetaAgentDeniedPathError, FinalayzeError)
    assert issubclass(MetaAgentSpawnCapExceededError, FinalayzeError)

    # N818 — class names end in "Error".
    assert MetaAgentDeniedPathError.__name__.endswith("Error")
    assert MetaAgentSpawnCapExceededError.__name__.endswith("Error")

    # Raise + carry message.
    msg_denied = "denied path: src/finalayze/risk/manager.py"
    try:
        raise MetaAgentDeniedPathError(msg_denied)
    except MetaAgentDeniedPathError as exc:
        assert str(exc) == msg_denied

    msg_cap = "spawn cap exceeded for INVESTIGATE"
    try:
        raise MetaAgentSpawnCapExceededError(msg_cap)
    except MetaAgentSpawnCapExceededError as exc:
        assert str(exc) == msg_cap
