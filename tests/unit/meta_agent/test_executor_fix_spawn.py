"""Tests for ActionExecutor.execute_fix_spawn (Phase 58-04 Task 09).

SPEC AC #12, #13, #14:
  - FIX-severity decision passed to executor.execute() sends a Telegram
    alert with the decision_id short8 + /approve instruction.
  - Once approved (handle_approve flips status), execute_fix_spawn:
    - Runs cap query (count of FIX rows today, UTC).
    - Loads the meta-agent-fix skill.
    - Runs validate_fix_prompt; on denied path → reject without spawning.
    - Creates the worktree.
    - Marks 'spawned'.
    - Awaits spawn_fix(prompt, ...).
    - Marks 'completed' (exit=0) or 'failed' (else).
  - cap=2 + 3rd FIX-approved → 'rejected', outcome='fix_spawn_cap_exceeded'.
  - Denied path in prompt → 'rejected', outcome='denied_path:...'.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

# Module-level constants (PLR2004).
_FAKE_DECISION_ID = uuid.UUID("ab12cd34-0000-4000-8000-000000000001")
_FAKE_TS = datetime(2026, 4, 26, 12, 0, tzinfo=UTC)
_FAKE_SHORT8 = "ab12cd34"
_FIX_CAP_2 = 2
_CAP_HIGH = 100
_FAKE_EXIT_OK = 0
_FAKE_EXIT_FAIL = 1


def _make_fix_decision() -> Any:
    """Build a FIX-severity MetaAgentDecisionModel-shaped MagicMock."""
    decision = MagicMock()
    decision.id = _FAKE_DECISION_ID
    decision.timestamp = _FAKE_TS
    decision.severity = "FIX"
    decision.summary = "fix needed"
    decision.rationale = "drawdown > 5%"
    decision.decision_metadata = None
    return decision


# ─────────────────────────────────────────────────────────────────────────────
# Task 58-04-09 (a): FIX-severity Telegram message contains short8 + /approve
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_execute_fix_severity_sends_telegram_with_short8(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SPEC AC #12: a FIX-severity decision passed to executor.execute()
    sends a Telegram alert whose body contains the decision_id short8
    AND the literal "/approve <short8>" instruction. Decision status
    flips to 'sent' (existing 58-02 contract).
    """
    from finalayze.meta_agent.classifier import Severity
    from finalayze.meta_agent.executor import ActionExecutor

    settings = MagicMock()
    settings.meta_agent_dry_run = False
    settings.meta_agent_max_telegram_alerts_per_day = _CAP_HIGH

    sent_messages: list[str] = []

    async def _fake_send(message: str, **_kwargs: Any) -> tuple[bool, uuid.UUID]:
        sent_messages.append(message)
        return True, uuid.uuid4()

    alerter = MagicMock()
    alerter._send = AsyncMock(side_effect=_fake_send)

    persistence = MagicMock()
    persistence.update_decision_status = MagicMock()

    # Stub the cap-query session so executor.execute() proceeds past gate.
    class _FakeRes:
        def scalar_one(self) -> int:
            return 0

    class _FakeSession:
        async def __aenter__(self) -> _FakeSession:
            return self

        async def __aexit__(self, *_a: Any) -> None:
            return None

        async def execute(self, _stmt: Any) -> _FakeRes:
            return _FakeRes()

    executor = ActionExecutor(
        settings=settings,
        alerter=alerter,
        persistence=persistence,
        session_factory=_FakeSession,
    )

    decision = _make_fix_decision()
    decision.severity = Severity.FIX.value

    await executor.execute(decision)

    # Telegram alert sent.
    assert len(sent_messages) == 1, f"expected 1 alert, got {sent_messages!r}"
    body = sent_messages[0]

    # Body contains the decision_id short8.
    assert _FAKE_SHORT8 in body, f"alert body must contain short8 {_FAKE_SHORT8!r}; got {body!r}"

    # Body contains the /approve instruction.
    assert "/approve" in body, f"alert body must contain '/approve'; got {body!r}"

    # Status flipped to 'sent' (existing 58-02 contract).
    upd_calls = persistence.update_decision_status.call_args_list
    assert any(call.kwargs.get("status") == "sent" for call in upd_calls), (
        f"expected status='sent' update, got {upd_calls!r}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Task 58-04-09 (b): execute_fix_spawn happy path — validate, worktree, spawn
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_execute_fix_spawn_after_approve_invokes_validate_and_spawn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SPEC AC #13: execute_fix_spawn(decision) runs the full pipeline:
      1. cap query passes.
      2. validate_fix_prompt invoked on the built prompt.
      3. create_fix_worktree invoked with short8.
      4. spawn_fix invoked with prompt + worktree cwd + paths.
      5. Decision status flipped to 'completed' (exit=0).
    """
    from pathlib import Path

    from finalayze.meta_agent.executor import ActionExecutor
    from finalayze.meta_agent.spawner import SpawnOutcome

    settings = MagicMock()
    settings.meta_agent_dry_run = False
    settings.meta_agent_max_fix_spawns_per_day = _CAP_HIGH

    persistence = MagicMock()
    persistence.update_decision_status = MagicMock()
    persistence._spawn_count_today_async = AsyncMock(return_value=0)

    executor = ActionExecutor(
        settings=settings,
        alerter=MagicMock(),
        persistence=persistence,
    )

    validate_calls: list[str] = []

    def _fake_validate(prompt: str, *, denied_paths: list[str]) -> None:  # noqa: ARG001
        validate_calls.append(prompt)

    worktree_calls: list[str] = []
    fake_worktree = Path(f".worktrees/meta-agent-fix-{_FAKE_SHORT8}")

    def _fake_create_worktree(short8: str, **_kwargs: Any) -> Path:
        worktree_calls.append(short8)
        return fake_worktree

    spawn_calls: list[dict[str, Any]] = []
    fake_outcome = SpawnOutcome(
        exit_code=_FAKE_EXIT_OK,
        stdout='{"type":"result","is_error":false}\n',
        stderr="",
        timed_out=False,
        killed_by_killswitch=False,
    )

    async def _fake_spawn_fix(prompt: str, **kwargs: Any) -> SpawnOutcome:
        spawn_calls.append({"prompt": prompt, **kwargs})
        return fake_outcome

    monkeypatch.setattr(
        "finalayze.meta_agent.executor.validate_fix_prompt",
        _fake_validate,
    )
    monkeypatch.setattr(
        "finalayze.meta_agent.executor.create_fix_worktree",
        _fake_create_worktree,
    )
    monkeypatch.setattr(
        "finalayze.meta_agent.executor.spawn_fix",
        _fake_spawn_fix,
    )

    decision = _make_fix_decision()

    await executor.execute_fix_spawn(decision)

    # Cap query.
    persistence._spawn_count_today_async.assert_awaited_once_with("FIX")

    # validate_fix_prompt invoked.
    assert len(validate_calls) == 1, f"expected 1 validate call, got {validate_calls!r}"

    # create_fix_worktree called with the short8.
    assert worktree_calls == [_FAKE_SHORT8], (
        f"expected worktree for {_FAKE_SHORT8}, got {worktree_calls!r}"
    )

    # spawn_fix invoked with worktree cwd.
    assert len(spawn_calls) == 1
    assert spawn_calls[0]["decision_id"] == _FAKE_DECISION_ID
    assert spawn_calls[0]["cwd"] == fake_worktree

    # Status transitions: 'spawned' then 'completed'.
    upd_calls = persistence.update_decision_status.call_args_list
    statuses = [c.kwargs.get("status") for c in upd_calls]
    assert "spawned" in statuses, f"expected 'spawned' transition, got {statuses!r}"
    assert "completed" in statuses, f"expected 'completed' transition, got {statuses!r}"


# ─────────────────────────────────────────────────────────────────────────────
# Task 58-04-09 (c): cap=2 third → rejected, fix_spawn_cap_exceeded
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_execute_fix_spawn_third_in_day_is_rejected_with_cap_exceeded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SPEC AC #14: with meta_agent_max_fix_spawns_per_day=2 and the cap
    query returning 2, execute_fix_spawn does NOT call spawn_fix; status
    flipped to 'rejected' with outcome='fix_spawn_cap_exceeded'.
    """
    import structlog

    from finalayze.meta_agent.executor import ActionExecutor

    settings = MagicMock()
    settings.meta_agent_dry_run = False
    settings.meta_agent_max_fix_spawns_per_day = _FIX_CAP_2

    persistence = MagicMock()
    persistence.update_decision_status = MagicMock()
    persistence._spawn_count_today_async = AsyncMock(return_value=_FIX_CAP_2)

    spawn_called: list[Any] = []

    async def _fake_spawn_fix(*_args: Any, **_kwargs: Any) -> Any:
        spawn_called.append("called")
        msg = "spawn_fix should NOT be called when cap is hit"
        raise AssertionError(msg)

    monkeypatch.setattr(
        "finalayze.meta_agent.executor.spawn_fix",
        _fake_spawn_fix,
    )

    executor = ActionExecutor(
        settings=settings,
        alerter=MagicMock(),
        persistence=persistence,
    )

    decision = _make_fix_decision()

    with structlog.testing.capture_logs() as logs:
        await executor.execute_fix_spawn(decision)

    # spawn_fix NEVER called.
    assert spawn_called == [], f"spawn_fix must not run when cap is hit, got {spawn_called!r}"

    # Status='rejected' with outcome='fix_spawn_cap_exceeded'.
    upd_calls = persistence.update_decision_status.call_args_list
    assert len(upd_calls) == 1
    rejected_call = upd_calls[0]
    assert rejected_call.kwargs.get("status") == "rejected"
    assert rejected_call.kwargs.get("outcome") == "fix_spawn_cap_exceeded"

    # Structlog event.
    cap_events = [log for log in logs if log.get("event") == "meta_agent_fix_spawn_cap_exceeded"]
    assert len(cap_events) == 1, f"expected 1 fix_spawn_cap_exceeded event, got {cap_events!r}"


# ─────────────────────────────────────────────────────────────────────────────
# Task 58-04-09 (d): denied path in prompt → 'rejected', outcome='denied_path:...'
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_execute_fix_spawn_denied_path_marks_decision_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SPEC AC #13: when validate_fix_prompt raises MetaAgentDeniedPathError,
    executor flips status='rejected' with outcome containing 'denied_path:'
    and the offending path. spawn_fix is NEVER invoked.
    """
    from finalayze.meta_agent.exceptions import MetaAgentDeniedPathError
    from finalayze.meta_agent.executor import ActionExecutor

    settings = MagicMock()
    settings.meta_agent_dry_run = False
    settings.meta_agent_max_fix_spawns_per_day = _CAP_HIGH

    persistence = MagicMock()
    persistence.update_decision_status = MagicMock()
    persistence._spawn_count_today_async = AsyncMock(return_value=0)

    def _fake_validate(_prompt: str, *, denied_paths: list[str]) -> None:  # noqa: ARG001
        msg = "Fix prompt references denied path: 'src/finalayze/risk/'"
        raise MetaAgentDeniedPathError(msg)

    spawn_called: list[Any] = []

    async def _fake_spawn_fix(*_args: Any, **_kwargs: Any) -> Any:
        spawn_called.append("called")
        msg = "spawn_fix must not run when validator rejects"
        raise AssertionError(msg)

    monkeypatch.setattr(
        "finalayze.meta_agent.executor.validate_fix_prompt",
        _fake_validate,
    )
    monkeypatch.setattr(
        "finalayze.meta_agent.executor.spawn_fix",
        _fake_spawn_fix,
    )

    executor = ActionExecutor(
        settings=settings,
        alerter=MagicMock(),
        persistence=persistence,
    )

    decision = _make_fix_decision()

    await executor.execute_fix_spawn(decision)

    # spawn_fix never called.
    assert spawn_called == [], f"spawn_fix must not run after denied_path, got {spawn_called!r}"

    # Status='rejected' with outcome containing 'denied_path:'.
    upd_calls = persistence.update_decision_status.call_args_list
    statuses = [c.kwargs.get("status") for c in upd_calls]
    assert "rejected" in statuses, f"expected 'rejected' status, got {statuses!r}"
    rejected_call = next(c for c in upd_calls if c.kwargs.get("status") == "rejected")
    outcome = rejected_call.kwargs.get("outcome", "")
    assert outcome.startswith("denied_path:"), (
        f"outcome must start with 'denied_path:', got {outcome!r}"
    )
    assert "src/finalayze/risk/" in outcome, (
        f"outcome must include the denied path, got {outcome!r}"
    )
