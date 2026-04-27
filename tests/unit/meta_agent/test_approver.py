"""Tests for meta_agent.approver.MetaAgentApprover (Phase 58-04 Tasks 05/06/07).

SPEC AC #12 + #17:
  - handle_approve flips status='approved' within 30 min, dispatches
    execute_fix_spawn (Task 05).
  - handle_approve on a 31-min-old row flips to status='expired'
    (Task 06).
  - expire_overdue_fix_decisions issues a single SQL UPDATE flipping
    'sent' → 'expired' for FIX rows older than 30 min (Task 07).

Tests use AsyncMock for persistence + executor; no real DB session.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

# Module-level constants (PLR2004).
_FAKE_DECISION_UUID = uuid.UUID("ab12cd34-0000-4000-8000-000000000001")
_FAKE_SHORT8 = "ab12cd34"
_WRONG_SHORT8 = "deadbeef"
_FAKE_TS = datetime(2026, 4, 26, 12, 0, tzinfo=UTC)
_APPROVE_TTL_MIN = 30
_NOW_FOR_TEST = datetime(2026, 4, 26, 12, 5, tzinfo=UTC)  # 5 min after _FAKE_TS
_NOW_AFTER_TTL = datetime(2026, 4, 26, 12, 31, tzinfo=UTC)  # 31 min after
_FAKE_CHAT_ID = "12345"


def _make_decision_row(
    *,
    severity: str = "FIX",
    status: str = "sent",
    created_at: datetime | None = None,
) -> Any:
    """Build a MetaAgentDecisionModel-shaped MagicMock for handler tests."""
    row = MagicMock()
    row.id = _FAKE_DECISION_UUID
    row.timestamp = _FAKE_TS
    row.severity = severity
    row.status = status
    row.created_at = created_at if created_at is not None else _FAKE_TS
    row.summary = "test summary"
    row.rationale = "test rationale"
    row.decision_metadata = None
    return row


# ─────────────────────────────────────────────────────────────────────────────
# Task 58-04-05: handle_approve happy path
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_handle_approve_within_ttl_flips_status_and_dispatches_spawn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SPEC AC #12: with a FIX-severity row in 'sent' state, created 5 min
    ago, calling handle_approve(short8, chat_id) MUST:
      1. Look up the row by short8 prefix.
      2. Flip status='approved' (via update_decision_status) exactly once.
      3. Dispatch execute_fix_spawn(decision) exactly once.
    """
    from finalayze.meta_agent.approver import MetaAgentApprover

    # Mock the lookup to return our fixture row (created 5 min before "now").
    fixture_row = _make_decision_row(
        severity="FIX",
        status="sent",
        created_at=_NOW_FOR_TEST - timedelta(minutes=5),
    )

    persistence = MagicMock()
    persistence.update_decision_status = MagicMock()
    executor = MagicMock()
    executor.execute_fix_spawn = AsyncMock()

    approver = MetaAgentApprover(
        executor=executor,
        persistence=persistence,
        approve_ttl_minutes=_APPROVE_TTL_MIN,
    )
    # Patch internal lookup to return the fixture.
    approver._lookup_by_short8 = AsyncMock(return_value=fixture_row)
    # Patch the "now" provider so age calculation is deterministic.
    monkeypatch.setattr(
        "finalayze.meta_agent.approver._now_utc",
        lambda: _NOW_FOR_TEST,
    )

    await approver.handle_approve(_FAKE_SHORT8, chat_id=_FAKE_CHAT_ID)

    # update_decision_status called exactly once with status='approved'.
    upd_calls = persistence.update_decision_status.call_args_list
    assert len(upd_calls) == 1, f"expected 1 update_decision_status call, got {upd_calls!r}"
    assert upd_calls[0].kwargs["status"] == "approved"
    assert upd_calls[0].kwargs["decision_id"] == _FAKE_DECISION_UUID
    assert upd_calls[0].kwargs["timestamp"] == _FAKE_TS

    # execute_fix_spawn dispatched exactly once with the decision.
    executor.execute_fix_spawn.assert_awaited_once_with(fixture_row)


@pytest.mark.asyncio
async def test_handle_approve_unknown_id_logs_and_skips() -> None:
    """SPEC AC #12: an unknown short8 must NOT change any DB state and
    MUST emit a structlog event 'meta_agent_approve_unknown_decision_id'.
    """
    import structlog

    from finalayze.meta_agent.approver import MetaAgentApprover

    persistence = MagicMock()
    persistence.update_decision_status = MagicMock()
    executor = MagicMock()
    executor.execute_fix_spawn = AsyncMock()

    approver = MetaAgentApprover(
        executor=executor,
        persistence=persistence,
        approve_ttl_minutes=_APPROVE_TTL_MIN,
    )
    # Lookup returns None — short8 doesn't match any row.
    approver._lookup_by_short8 = AsyncMock(return_value=None)

    with structlog.testing.capture_logs() as logs:
        await approver.handle_approve(_WRONG_SHORT8, chat_id=_FAKE_CHAT_ID)

    persistence.update_decision_status.assert_not_called()
    executor.execute_fix_spawn.assert_not_called()

    unknown_events = [
        log for log in logs if log.get("event") == "meta_agent_approve_unknown_decision_id"
    ]
    assert len(unknown_events) == 1, f"expected 1 unknown_decision_id event, got {unknown_events!r}"


@pytest.mark.asyncio
async def test_handle_approve_state_mismatch_skips() -> None:
    """SPEC AC #12: a row with severity!='FIX' or status!='sent' (e.g.
    already 'approved' / 'expired' / not a FIX decision) MUST NOT change
    DB state. Emits structlog event 'meta_agent_approve_state_mismatch'.
    """
    import structlog

    from finalayze.meta_agent.approver import MetaAgentApprover

    # Row is FIX but already approved — second /approve must no-op.
    fixture_row = _make_decision_row(severity="FIX", status="approved")

    persistence = MagicMock()
    persistence.update_decision_status = MagicMock()
    executor = MagicMock()
    executor.execute_fix_spawn = AsyncMock()

    approver = MetaAgentApprover(
        executor=executor,
        persistence=persistence,
        approve_ttl_minutes=_APPROVE_TTL_MIN,
    )
    approver._lookup_by_short8 = AsyncMock(return_value=fixture_row)

    with structlog.testing.capture_logs() as logs:
        await approver.handle_approve(_FAKE_SHORT8, chat_id=_FAKE_CHAT_ID)

    persistence.update_decision_status.assert_not_called()
    executor.execute_fix_spawn.assert_not_called()

    mismatch_events = [
        log for log in logs if log.get("event") == "meta_agent_approve_state_mismatch"
    ]
    assert len(mismatch_events) == 1, f"expected 1 state_mismatch event, got {mismatch_events!r}"


# ─────────────────────────────────────────────────────────────────────────────
# Task 58-04-06: handle_approve on 31-min-old row → status='expired'
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_handle_approve_expired_decision_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SPEC AC #12 + #17: a row created 31 min ago with status='sent' MUST
    flip to status='expired' (NOT 'approved'); execute_fix_spawn MUST NOT
    be called (TTL boundary).
    """
    from finalayze.meta_agent.approver import MetaAgentApprover

    fixture_row = _make_decision_row(
        severity="FIX",
        status="sent",
        created_at=_FAKE_TS,  # 31 min before _NOW_AFTER_TTL
    )

    persistence = MagicMock()
    persistence.update_decision_status = MagicMock()
    executor = MagicMock()
    executor.execute_fix_spawn = AsyncMock()

    approver = MetaAgentApprover(
        executor=executor,
        persistence=persistence,
        approve_ttl_minutes=_APPROVE_TTL_MIN,
    )
    approver._lookup_by_short8 = AsyncMock(return_value=fixture_row)
    monkeypatch.setattr(
        "finalayze.meta_agent.approver._now_utc",
        lambda: _NOW_AFTER_TTL,
    )

    await approver.handle_approve(_FAKE_SHORT8, chat_id=_FAKE_CHAT_ID)

    # update_decision_status called once with status='expired' — NOT 'approved'.
    upd_calls = persistence.update_decision_status.call_args_list
    assert len(upd_calls) == 1
    assert upd_calls[0].kwargs["status"] == "expired", (
        f"expired row must flip to 'expired', got {upd_calls[0].kwargs!r}"
    )

    # execute_fix_spawn NOT dispatched.
    executor.execute_fix_spawn.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
# Task 58-04-07: expire_overdue_fix_decisions sweep
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_expire_overdue_fix_decisions_runs_single_update(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SPEC AC #17 / D-14: expire_overdue_fix_decisions issues a single
    UPDATE statement filtering severity='FIX' AND status='sent' AND
    created_at <= NOW() - INTERVAL '30 minutes', flipping to 'expired'.
    Idempotent: a second call against the same fixture (zero rows
    matching) returns 0 affected. Emits structlog event
    'meta_agent_approve_sweep' with affected_count.
    """
    import structlog

    from finalayze.meta_agent.approver import MetaAgentApprover

    captured_stmts: list[Any] = []

    class _FakeResult:
        def __init__(self, rowcount: int) -> None:
            self.rowcount = rowcount

    class _FakeSession:
        def __init__(self, rowcount: int) -> None:
            self._rowcount = rowcount

        async def __aenter__(self) -> _FakeSession:
            return self

        async def __aexit__(self, *_a: Any) -> None:
            return None

        async def execute(self, stmt: Any) -> _FakeResult:
            captured_stmts.append(stmt)
            return _FakeResult(rowcount=self._rowcount)

        async def commit(self) -> None:
            return None

    persistence = MagicMock()
    # First call: 3 rows match. Second call: 0 rows match (idempotent).
    rowcounts = iter([3, 0])

    def _factory_factory() -> Any:
        rc = next(rowcounts)

        def _factory() -> _FakeSession:
            return _FakeSession(rowcount=rc)

        return _factory

    monkeypatch.setattr(
        persistence,
        "_get_bg_session_factory",
        _factory_factory,
    )

    executor = MagicMock()
    approver = MetaAgentApprover(
        executor=executor,
        persistence=persistence,
        approve_ttl_minutes=_APPROVE_TTL_MIN,
    )

    with structlog.testing.capture_logs() as logs:
        n1 = await approver.expire_overdue_fix_decisions()
        n2 = await approver.expire_overdue_fix_decisions()

    assert n1 == 3, f"first sweep should return rowcount=3, got {n1}"
    assert n2 == 0, f"second sweep (idempotent, no new rows) should return 0, got {n2}"

    # Two statements were issued (one per sweep), both UPDATE statements.
    assert len(captured_stmts) == 2, f"expected 2 UPDATE stmts, got {captured_stmts!r}"
    for stmt in captured_stmts:
        sql = str(stmt.compile(compile_kwargs={"literal_binds": False})).lower()
        assert "update" in sql and "agent_decisions" in sql, (
            f"expected UPDATE on agent_decisions, got SQL: {sql!r}"
        )
        assert "severity" in sql
        assert "status" in sql

    # Two structlog events.
    sweep_events = [log for log in logs if log.get("event") == "meta_agent_approve_sweep"]
    assert len(sweep_events) == 2, f"expected 2 sweep events, got {sweep_events!r}"
    assert sweep_events[0].get("affected_count") == 3
    assert sweep_events[1].get("affected_count") == 0
