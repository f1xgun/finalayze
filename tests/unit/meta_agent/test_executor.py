"""Tests for meta_agent.executor.ActionExecutor (Phase 58-02, META-05).

Covers SPEC §Acceptance Criteria #8 + #9:
  - Persist envelope helpers (persist_decision, update_decision_status).
  - Dry-run short-circuit on the FIRST line of execute().
  - HEALTHY → no Telegram (severity-below-threshold gate).
  - WATCH/INVESTIGATE/FIX → send Telegram, stamp metadata, status='sent'.
  - Daily cap enforcement (UTC-day boundary).
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

# Module-level constants (PLR2004).
_FAKE_ALERT_UUID = uuid.UUID("f00dbeef-1234-4abc-8def-0123456789ab")
_FAKE_DECISION_ID = uuid.UUID("deadbeef-0000-4000-8000-000000000001")
_FAKE_TS = datetime(2026, 4, 26, 12, 0, tzinfo=UTC)
_FAKE_COUNT = 7
_CAP_HIGH = 100
_CAP_TWO = 2
_NUM_THIRD = 3


# ─────────────────────────────────────────────────────────────────────────────
# Task 58-02-01: persist_decision + update_decision_status fire-and-forget
# ─────────────────────────────────────────────────────────────────────────────


def test_persistence_helpers_use_fire_and_forget_envelope() -> None:
    """SPEC AC #8 foundation: TradingPersistence.persist_decision and
    update_decision_status exist, never raise when _db_url is None, and
    log db_persist_skipped (PERSIST-05 envelope, mirrors persist_alert).
    """
    import structlog

    from finalayze.orchestration.db_persistence import TradingPersistence

    persistence = TradingPersistence(db_url=None, async_loop=None)

    # Capture structlog events. When _db_url is None, both helpers must
    # log db_persist_skipped and return without raising.
    with structlog.testing.capture_logs() as logs:
        persistence.persist_decision(
            decision_id=_FAKE_DECISION_ID,
            timestamp=_FAKE_TS,
            severity="HEALTHY",
            summary="s",
            rationale="r",
            actions=[],
            dry_run=True,
            decision_metadata=None,
            parent_decision_id=None,
            status="queued",
        )
        persistence.update_decision_status(
            decision_id=_FAKE_DECISION_ID,
            timestamp=_FAKE_TS,
            status="sent",
            outcome=None,
        )

    skipped = [
        log
        for log in logs
        if log.get("event") == "db_persist_skipped"
        and log.get("table") == "agent_decisions"
    ]
    assert len(skipped) >= 2, (
        f"expected >=2 db_persist_skipped events for agent_decisions, got {logs!r}"
    )

    # AND: when a session factory is mocked (db_url set), persist_decision
    # enqueues a MetaAgentDecisionModel insert via the same envelope.
    persistence_with_db = TradingPersistence(db_url=None, async_loop=None)
    # Patch _persist_to_db to inspect the table arg without spinning up a real session.
    captured: dict[str, Any] = {}

    def _capture(coro: Any, *, table: str, **ctx: Any) -> None:
        captured["table"] = table
        captured["ctx"] = ctx
        # Close the coroutine so the test does not leak a "never awaited" warning.
        coro.close()

    persistence_with_db._persist_to_db = _capture  # type: ignore[method-assign]
    persistence_with_db.persist_decision(
        decision_id=_FAKE_DECISION_ID,
        timestamp=_FAKE_TS,
        severity="HEALTHY",
        summary="s",
        rationale="r",
        actions=[],
        dry_run=True,
        decision_metadata=None,
        parent_decision_id=None,
        status="queued",
    )
    assert captured["table"] == "agent_decisions"
    assert captured["ctx"]["severity_key"] == "HEALTHY"
    assert captured["ctx"]["decision_id_key"] == str(_FAKE_DECISION_ID)


# ─────────────────────────────────────────────────────────────────────────────
# Task 58-02-01b: update_decision_status accepts metadata_patch kwarg
#                 (JSONB SELECT-then-merge-then-UPDATE)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_update_decision_status_metadata_patch_merges_into_decision_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SPEC AC #8 dependency for Task 58-02-05.

    With an existing row whose decision_metadata = {'trace_id': 'abc'},
    calling update_decision_status(metadata_patch={'telegram_alert_id': '...'})
    must SELECT current metadata, merge {**current, **patch}, and issue an
    UPDATE whose .values() includes decision_metadata={'trace_id': 'abc',
    'telegram_alert_id': '...'} (existing keys preserved, patch keys override).

    Calling with metadata_patch=None must NOT touch decision_metadata.
    Calling with metadata_patch={} must NOT touch decision_metadata.
    """
    from finalayze.orchestration.db_persistence import TradingPersistence

    persistence = TradingPersistence(db_url=None, async_loop=None)

    # ── case 1: metadata_patch supplies a new key ──────────────────────────
    captured_select_calls: list[Any] = []
    captured_update_values: list[dict[str, Any]] = []

    class _FakeScalarRes:
        def __init__(self, value: Any) -> None:
            self._value = value

        def scalar_one_or_none(self) -> Any:
            return self._value

    class _FakeSession:
        def __init__(self, current_meta: Any) -> None:
            self._current_meta = current_meta

        async def __aenter__(self) -> _FakeSession:
            return self

        async def __aexit__(self, *a: Any) -> None:
            return None

        async def execute(self, stmt: Any) -> Any:
            # Distinguish SELECT vs UPDATE by inspecting the compiled SQL.
            stmt_str = str(stmt).lower()
            if stmt_str.startswith("select"):
                captured_select_calls.append(stmt)
                return _FakeScalarRes(self._current_meta)
            # UPDATE — record the .values() dict on the compiled statement.
            captured_update_values.append(dict(stmt.compile().params))
            return MagicMock()

        async def commit(self) -> None:
            return None

    def _make_factory(current_meta: Any) -> Any:
        def _factory() -> _FakeSession:
            return _FakeSession(current_meta)

        return _factory

    monkeypatch.setattr(
        persistence,
        "_get_bg_session_factory",
        lambda: _make_factory({"trace_id": "abc"}),
    )

    # Direct call to the async helper (bypasses _persist_to_db so we can await).
    fake_alert_id = str(_FAKE_ALERT_UUID)
    await persistence._update_decision_status_async(
        decision_id=_FAKE_DECISION_ID,
        timestamp=_FAKE_TS,
        status="sent",
        outcome=None,
        metadata_patch={"telegram_alert_id": fake_alert_id},
    )

    # SELECT was issued for the merge.
    assert len(captured_select_calls) == 1, "expected one SELECT for current metadata"
    # UPDATE values include merged decision_metadata.
    assert len(captured_update_values) == 1
    update_values = captured_update_values[0]
    assert update_values["status"] == "sent"
    # Column is named "metadata" at the DB level (decision_metadata is the
    # Python attr; SQLAlchemy reserved-word workaround per AP-3).
    merged = update_values["metadata"]
    assert merged == {"trace_id": "abc", "telegram_alert_id": fake_alert_id}, (
        f"expected deep-merge, got {merged!r}"
    )

    # ── case 2: metadata_patch=None → no SELECT, no decision_metadata in UPDATE ──
    captured_select_calls.clear()
    captured_update_values.clear()
    monkeypatch.setattr(
        persistence,
        "_get_bg_session_factory",
        lambda: _make_factory({"trace_id": "abc"}),
    )
    await persistence._update_decision_status_async(
        decision_id=_FAKE_DECISION_ID,
        timestamp=_FAKE_TS,
        status="failed",
        outcome="boom",
        metadata_patch=None,
    )
    assert len(captured_select_calls) == 0, "metadata_patch=None must NOT issue SELECT"
    assert len(captured_update_values) == 1
    assert "metadata" not in captured_update_values[0]
    assert captured_update_values[0]["status"] == "failed"
    assert captured_update_values[0]["outcome"] == "boom"

    # ── case 3: metadata_patch={} → no SELECT, no decision_metadata in UPDATE ──
    captured_select_calls.clear()
    captured_update_values.clear()
    monkeypatch.setattr(
        persistence,
        "_get_bg_session_factory",
        lambda: _make_factory({"trace_id": "abc"}),
    )
    await persistence._update_decision_status_async(
        decision_id=_FAKE_DECISION_ID,
        timestamp=_FAKE_TS,
        status="queued_capped",
        outcome=None,
        metadata_patch={},
    )
    assert len(captured_select_calls) == 0, "metadata_patch={} must NOT issue SELECT"
    assert len(captured_update_values) == 1
    assert "metadata" not in captured_update_values[0]


# ─────────────────────────────────────────────────────────────────────────────
# Task 58-02-02: ActionExecutor dry-run short-circuit (FIRST line of execute())
# ─────────────────────────────────────────────────────────────────────────────


def _make_decision(severity: str = "FIX") -> Any:
    """Build a MetaAgentDecisionModel-shaped object for executor tests."""
    decision = MagicMock()
    decision.id = _FAKE_DECISION_ID
    decision.timestamp = _FAKE_TS
    decision.severity = severity
    decision.summary = "test summary"
    decision.rationale = "test rationale"
    decision.decision_metadata = None
    return decision


@pytest.mark.asyncio
async def test_execute_dry_run_short_circuits_first_line() -> None:
    """SPEC AC #7 + #8: with meta_agent_dry_run=True, execute() must
    short-circuit on its FIRST line. No Telegram send, no persistence
    update, no session open. Returns ExecutionResult(skipped=True,
    reason='dry_run', telegram_alert_id=None) and emits structlog event
    'meta_agent_executor_dry_run_skipped' exactly once. (PATTERNS AP-10.)
    """
    import structlog

    from finalayze.meta_agent.executor import ActionExecutor, ExecutionResult

    settings = MagicMock()
    settings.meta_agent_dry_run = True

    alerter = MagicMock()
    alerter._send = AsyncMock()
    persistence = MagicMock()
    persistence.update_decision_status = MagicMock()

    executor = ActionExecutor(
        settings=settings,
        alerter=alerter,
        persistence=persistence,
    )

    decision = _make_decision(severity="FIX")

    with structlog.testing.capture_logs() as logs:
        result = await executor.execute(decision)

    assert isinstance(result, ExecutionResult)
    assert result.skipped is True
    assert result.reason == "dry_run"
    assert result.telegram_alert_id is None

    # Zero side effects.
    alerter._send.assert_not_called()
    persistence.update_decision_status.assert_not_called()

    dryrun_events = [
        log for log in logs if log.get("event") == "meta_agent_executor_dry_run_skipped"
    ]
    assert len(dryrun_events) == 1, (
        f"expected exactly 1 dry_run_skipped event, got {dryrun_events!r}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Task 58-02-03: HEALTHY severity → no Telegram, no persistence touch
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_execute_healthy_severity_does_not_send_telegram() -> None:
    """SPEC §Requirement 5: only WATCH/INVESTIGATE/FIX trigger Telegram.
    With dry_run=False and severity=HEALTHY, executor returns
    ExecutionResult(skipped=True, reason='severity_below_threshold',
    telegram_alert_id=None); zero alerter._send and zero
    persistence.update_decision_status calls.
    """
    from finalayze.meta_agent.classifier import Severity
    from finalayze.meta_agent.executor import ActionExecutor, ExecutionResult

    settings = MagicMock()
    settings.meta_agent_dry_run = False

    alerter = MagicMock()
    alerter._send = AsyncMock()
    persistence = MagicMock()
    persistence.update_decision_status = MagicMock()

    executor = ActionExecutor(
        settings=settings,
        alerter=alerter,
        persistence=persistence,
    )
    decision = _make_decision(severity=Severity.HEALTHY.value)

    result = await executor.execute(decision)

    assert isinstance(result, ExecutionResult)
    assert result.skipped is True
    assert result.reason == "severity_below_threshold"
    assert result.telegram_alert_id is None

    alerter._send.assert_not_called()
    persistence.update_decision_status.assert_not_called()
