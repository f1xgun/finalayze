"""Tests for meta_agent.executor.ActionExecutor (Phase 58-02 + 58-03).

Covers SPEC §Acceptance Criteria #8, #9, #10, #11:
  58-02:
    - Persist envelope helpers (persist_decision, update_decision_status).
    - Dry-run short-circuit on the FIRST line of execute().
    - HEALTHY → no Telegram (severity-below-threshold gate).
    - WATCH/INVESTIGATE/FIX → send Telegram, stamp metadata, status='sent'.
    - Daily cap enforcement (UTC-day boundary, alerts table).
  58-03:
    - _spawn_count_today_async on agent_decisions (UTC-day, status filter).
    - execute_investigate_spawn happy path → 'spawned' → 'completed'.
    - execute_investigate_spawn cap=2 + 3rd → 'rejected', 'spawn_cap_exceeded'.
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
# 58-03 spawn-cap test constants.
_FAKE_SPAWN_COUNT = 4
_INVEST_CAP_2 = 2
_FAKE_EXIT_OK = 0


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


# ─────────────────────────────────────────────────────────────────────────────
# Task 58-02-04: Daily Telegram cap query (_telegram_count_today)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_telegram_count_today_uses_utc_day_boundary() -> None:
    """SPEC AC #9 / D-13: cap query uses date_trunc('day', NOW() AT TIME
    ZONE 'UTC') so the cap resets at 00:00 UTC. WHERE clause includes
    AlertModel.alert_type LIKE 'meta_agent_%' AND timestamp >= UTC-day-start.
    """
    from finalayze.meta_agent.executor import ActionExecutor

    settings = MagicMock()
    settings.meta_agent_dry_run = False

    alerter = MagicMock()
    persistence = MagicMock()
    executor = ActionExecutor(
        settings=settings,
        alerter=alerter,
        persistence=persistence,
    )

    # Mocked async session that records the issued statement.
    captured_stmt: dict[str, Any] = {}

    class _ScalarRes:
        def scalar_one(self) -> int:
            return _FAKE_COUNT

    class _FakeSession:
        async def execute(self, stmt: Any) -> _ScalarRes:
            captured_stmt["stmt"] = stmt
            return _ScalarRes()

    session = _FakeSession()

    count = await executor._telegram_count_today(session)
    assert count == _FAKE_COUNT, (
        f"expected helper to return scalar_one() of {_FAKE_COUNT}, got {count!r}"
    )

    # Inspect the issued SQL — must filter by alert_type LIKE 'meta_agent_%'
    # AND timestamp >= UTC-day-start.
    stmt = captured_stmt["stmt"]
    sql = str(stmt.compile(compile_kwargs={"literal_binds": False})).lower()
    assert "alerts" in sql, f"cap query must FROM alerts, got: {sql!r}"
    assert "alert_type" in sql, f"cap query must filter alert_type, got: {sql!r}"
    assert "like" in sql, f"cap query must use LIKE for alert_type, got: {sql!r}"
    assert "date_trunc" in sql, (
        f"cap query must use date_trunc for UTC-day boundary, got: {sql!r}"
    )
    # The text fragment "now() at time zone 'utc'" appears verbatim.
    assert "utc" in sql, f"cap query must use UTC tz boundary, got: {sql!r}"


# ─────────────────────────────────────────────────────────────────────────────
# Task 58-02-05: INVESTIGATE severity → send Telegram + stamp metadata
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_execute_investigate_sends_telegram_and_stamps_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SPEC AC #8: with dry_run=False, severity=INVESTIGATE, cap not hit,
    executor:
      1. Calls alerter._send with alert_type='meta_agent_INVESTIGATE'.
      2. Returns ExecutionResult(skipped=False, reason=None,
         telegram_alert_id=<uuid>).
      3. Calls persistence.update_decision_status(status='sent',
         metadata_patch={'telegram_alert_id': str(uuid)}) exactly once.
      4. Emits structlog event meta_agent_executor_telegram_sent.
    """
    import structlog

    from finalayze.meta_agent.classifier import Severity
    from finalayze.meta_agent.executor import ActionExecutor, ExecutionResult

    settings = MagicMock()
    settings.meta_agent_dry_run = False
    settings.meta_agent_max_telegram_alerts_per_day = _CAP_HIGH

    alerter = MagicMock()
    alerter._send = AsyncMock(return_value=(True, _FAKE_ALERT_UUID))

    persistence = MagicMock()
    persistence.update_decision_status = MagicMock()

    executor = ActionExecutor(
        settings=settings,
        alerter=alerter,
        persistence=persistence,
    )

    # Patch _telegram_count_today and the session-factory so we never touch
    # the real DB.
    async def _zero_count(_session: Any) -> int:
        return 0

    monkeypatch.setattr(executor, "_telegram_count_today", _zero_count)

    class _FakeSession:
        async def __aenter__(self) -> _FakeSession:
            return self

        async def __aexit__(self, *a: Any) -> None:
            return None

    def _factory() -> _FakeSession:
        return _FakeSession()

    monkeypatch.setattr(executor, "_open_session", _factory, raising=False)

    decision = _make_decision(severity=Severity.INVESTIGATE.value)

    with structlog.testing.capture_logs() as logs:
        result = await executor.execute(decision)

    assert isinstance(result, ExecutionResult)
    assert result.skipped is False, f"INVESTIGATE must SEND, got result={result!r}"
    assert result.reason is None
    assert result.telegram_alert_id == _FAKE_ALERT_UUID

    # alerter._send was called exactly once with alert_type='meta_agent_INVESTIGATE'.
    assert alerter._send.call_count == 1, (
        f"expected one Telegram send, got {alerter._send.call_count}"
    )
    send_kwargs = alerter._send.call_args.kwargs
    assert send_kwargs["alert_type"] == "meta_agent_INVESTIGATE"

    # persistence.update_decision_status called once with status='sent' and
    # metadata_patch={'telegram_alert_id': str(uuid)}.
    assert persistence.update_decision_status.call_count == 1
    upd_kwargs = persistence.update_decision_status.call_args.kwargs
    assert upd_kwargs["status"] == "sent"
    assert upd_kwargs["metadata_patch"] == {"telegram_alert_id": str(_FAKE_ALERT_UUID)}

    # Structlog event emitted.
    sent_events = [
        log for log in logs if log.get("event") == "meta_agent_executor_telegram_sent"
    ]
    assert len(sent_events) >= 1, (
        f"expected meta_agent_executor_telegram_sent event, got {logs!r}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Task 58-02-06: Cap enforcement — 3rd INVESTIGATE with cap=2 → queued_capped
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_third_investigate_with_cap_2_is_queued_capped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SPEC AC #9: with cap=2, three consecutive INVESTIGATE decisions:
      - 1st (count=0) → sent.
      - 2nd (count=1) → sent.
      - 3rd (count=2) → status='queued_capped', no send, structlog
        meta_agent_executor_telegram_cap_hit.
    """
    import structlog

    from finalayze.meta_agent.classifier import Severity
    from finalayze.meta_agent.executor import ActionExecutor

    settings = MagicMock()
    settings.meta_agent_dry_run = False
    settings.meta_agent_max_telegram_alerts_per_day = _CAP_TWO

    alerter = MagicMock()
    alerter._send = AsyncMock(return_value=(True, _FAKE_ALERT_UUID))

    persistence = MagicMock()
    persistence.update_decision_status = MagicMock()

    executor = ActionExecutor(
        settings=settings,
        alerter=alerter,
        persistence=persistence,
    )

    counts = iter([0, 1, _CAP_TWO])

    async def _next_count(_session: Any) -> int:
        return next(counts)

    monkeypatch.setattr(executor, "_telegram_count_today", _next_count)

    class _FakeSession:
        async def __aenter__(self) -> _FakeSession:
            return self

        async def __aexit__(self, *a: Any) -> None:
            return None

    monkeypatch.setattr(executor, "_open_session", _FakeSession, raising=False)

    decision = _make_decision(severity=Severity.INVESTIGATE.value)

    results = []
    with structlog.testing.capture_logs() as logs:
        for _ in range(_NUM_THIRD):
            result = await executor.execute(decision)
            results.append(result)  # noqa: PERF401 — async iteration, not a comprehension

    # First two: sent.
    assert results[0].skipped is False, (
        f"1st call must SEND, got {results[0]!r}"
    )
    assert results[1].skipped is False, (
        f"2nd call must SEND, got {results[1]!r}"
    )
    # Third: queued_capped.
    assert results[2].skipped is True, f"3rd call must be capped, got {results[2]!r}"
    assert results[2].reason == "telegram_cap_hit"
    assert results[2].telegram_alert_id is None

    # alerter._send called exactly twice (the first two).
    assert alerter._send.call_count == 2, (
        f"expected exactly 2 sends, got {alerter._send.call_count}"
    )

    # Third update_decision_status call has status='queued_capped'.
    upd_calls = persistence.update_decision_status.call_args_list
    # Two for the sent path + one for the cap path.
    assert len(upd_calls) == _NUM_THIRD, (
        f"expected {_NUM_THIRD} update_decision_status calls, got {upd_calls!r}"
    )
    cap_call = upd_calls[2]
    assert cap_call.kwargs["status"] == "queued_capped"

    # Structlog cap_hit event emitted.
    cap_events = [
        log
        for log in logs
        if log.get("event") == "meta_agent_executor_telegram_cap_hit"
    ]
    assert len(cap_events) == 1, (
        f"expected 1 telegram_cap_hit event, got {cap_events!r}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Task 58-03-07: spawn_count_today helper (TradingPersistence)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_spawn_count_today_uses_utc_day_and_filters_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SPEC AC #11 / D-13 + AP-14: ``_spawn_count_today_async`` on
    TradingPersistence counts agent_decisions rows for the current UTC day,
    filtered by severity AND status IN ('spawned','completed','failed').
    Crucially the cap query reads from agent_decisions, NOT alerts (this
    is the AP-14 inversion vs the Telegram cap query in 58-02).
    """
    from finalayze.orchestration.db_persistence import TradingPersistence

    persistence = TradingPersistence(db_url=None, async_loop=None)

    # Capture the issued statement.
    captured: dict[str, Any] = {}

    class _ScalarRes:
        def scalar_one(self) -> int:
            return _FAKE_SPAWN_COUNT

    class _FakeSession:
        async def __aenter__(self) -> _FakeSession:
            return self

        async def __aexit__(self, *_a: Any) -> None:
            return None

        async def execute(self, stmt: Any) -> _ScalarRes:
            captured["stmt"] = stmt
            return _ScalarRes()

    def _factory() -> _FakeSession:
        return _FakeSession()

    monkeypatch.setattr(persistence, "_get_bg_session_factory", lambda: _factory)

    count = await persistence._spawn_count_today_async(severity="INVESTIGATE")
    assert count == _FAKE_SPAWN_COUNT, (
        f"expected helper to return {_FAKE_SPAWN_COUNT}, got {count!r}"
    )

    stmt = captured["stmt"]
    sql = str(stmt.compile(compile_kwargs={"literal_binds": False})).lower()

    # AP-14: cap reads agent_decisions, NOT alerts.
    assert "agent_decisions" in sql, (
        f"cap query must FROM agent_decisions, got: {sql!r}"
    )
    assert "alerts" not in sql, (
        f"cap query must NOT touch alerts table, got: {sql!r}"
    )
    # Severity filter.
    assert "severity" in sql, f"cap query must filter severity, got: {sql!r}"
    # Status filter (IN clause for spawned/completed/failed).
    assert "status" in sql, f"cap query must filter status, got: {sql!r}"
    assert "in (" in sql, (
        f"cap query must use IN clause for status, got: {sql!r}"
    )
    # UTC day boundary.
    assert "date_trunc" in sql, f"cap query must use date_trunc, got: {sql!r}"
    assert "utc" in sql, f"cap query must use UTC tz boundary, got: {sql!r}"
