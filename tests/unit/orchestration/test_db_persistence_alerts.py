"""Tests for persist_alert + update_alert_status fire-and-forget helpers.

Phase 57-01, ALRT-03. Validates the PERSIST-05 envelope on the new
`TradingPersistence.persist_alert` and `TradingPersistence.update_alert_status`
methods used by the Phase 57-02 alerter write hook.

These tests follow the pattern established by `TestPersistStopSnapshots` in
tests/unit/core/test_db_persistence.py — they use mocker to spy on
`_persist_to_db` / `_run_async` rather than spinning up a real DB. Live DB
behaviour is exercised by tests/integration/migrations/test_009_alerts.py
once the migration runs against TimescaleDB.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest


def _make_persistence(db_url: str | None = "postgresql+asyncpg://test/db"):  # type: ignore[no-untyped-def]
    from finalayze.orchestration.db_persistence import TradingPersistence

    loop = asyncio.new_event_loop()
    return TradingPersistence(db_url=db_url, async_loop=loop)


def test_persist_alert_inserts_row(mocker: pytest.FixtureRequest) -> None:
    """persist_alert calls _persist_to_db with table='alerts' and the row coro."""
    persistence = _make_persistence()
    captured: dict[str, object] = {}

    def _fake_persist(coro, *, table, **ctx):  # type: ignore[no-untyped-def]
        captured["table"] = table
        captured.update(ctx)
        coro.close()  # prevent "coroutine was never awaited" warning

    mocker.patch.object(persistence, "_persist_to_db", side_effect=_fake_persist)  # type: ignore[attr-defined]

    alert_id = uuid.uuid4()
    now = datetime.now(UTC)
    persistence.persist_alert(
        alert_id=alert_id,
        timestamp=now,
        alert_type="signal",
        priority="INFO",
        message="BUY SBER conf=0.72",
        symbol="SBER",
        market_id="moex",
    )
    assert captured["table"] == "alerts"
    # alert_type forwarded under 'alert_type_key' to avoid collision with the
    # structlog reserved 'event' key precedent (mirrors event_kind in
    # persist_stop_snapshots).
    assert captured["alert_type_key"] == "signal"
    assert captured["symbol"] == "SBER"


def test_persist_alert_fire_and_forget_on_db_failure(
    mocker: pytest.FixtureRequest,
) -> None:
    """persist_alert MUST NOT raise on _persist_to_db failure; counter increments."""
    persistence = _make_persistence()

    # Force the underlying _run_async to raise — exercises the real PERSIST-05
    # envelope inside _persist_to_db (logs db_persist_failed + increments counter).
    # Close the coroutine to suppress 'coroutine was never awaited' RuntimeWarning.
    def _raise(coro, *args, **kwargs):  # type: ignore[no-untyped-def]
        if asyncio.iscoroutine(coro):
            coro.close()
        msg = "db down"
        raise RuntimeError(msg)

    mocker.patch.object(  # type: ignore[attr-defined]
        persistence, "_run_async", side_effect=_raise
    )
    mock_metric = mocker.patch("finalayze.api.metrics.db_write_failures")  # type: ignore[attr-defined]

    # No exception escapes
    result = persistence.persist_alert(
        alert_id=uuid.uuid4(),
        timestamp=datetime.now(UTC),
        alert_type="signal",
        priority="INFO",
        message="text",
        symbol="SBER",
    )
    assert result is None  # void return; method must not raise
    mock_metric.labels.assert_called_with(table="alerts")
    mock_metric.labels.return_value.inc.assert_called_once()


def test_update_alert_status_updates_existing_row(
    mocker: pytest.FixtureRequest,
) -> None:
    """update_alert_status routes through _persist_to_db with status_update op."""
    persistence = _make_persistence()
    captured: dict[str, object] = {}

    def _fake_persist(coro, *, table, **ctx):  # type: ignore[no-untyped-def]
        captured["table"] = table
        captured.update(ctx)
        coro.close()

    mocker.patch.object(persistence, "_persist_to_db", side_effect=_fake_persist)  # type: ignore[attr-defined]

    alert_id = uuid.uuid4()
    now = datetime.now(UTC)
    persistence.update_alert_status(alert_id, now, "sent")

    assert captured["table"] == "alerts"
    assert captured["op"] == "status_update"
    assert captured["status"] == "sent"


def test_update_alert_status_fire_and_forget(
    mocker: pytest.FixtureRequest,
) -> None:
    """update_alert_status MUST NOT raise on DB failure (PERSIST-05)."""
    persistence = _make_persistence()

    def _raise(coro, *args, **kwargs):  # type: ignore[no-untyped-def]
        if asyncio.iscoroutine(coro):
            coro.close()
        msg = "db down"
        raise RuntimeError(msg)

    mocker.patch.object(  # type: ignore[attr-defined]
        persistence, "_run_async", side_effect=_raise
    )
    mocker.patch("finalayze.api.metrics.db_write_failures")  # type: ignore[attr-defined]

    result = persistence.update_alert_status(uuid.uuid4(), datetime.now(UTC), "failed")
    assert result is None  # void return; method must not raise


def test_persist_alert_with_parent_id_threaded(
    mocker: pytest.FixtureRequest,
) -> None:
    """persist_alert accepts parent_id (anomaly raw + LLM follow-up pair, D-04)."""
    persistence = _make_persistence()
    captured_calls: list[dict[str, object]] = []

    def _fake_persist(coro, *, table, **ctx):  # type: ignore[no-untyped-def]
        captured_calls.append({"table": table, **ctx})
        coro.close()

    mocker.patch.object(persistence, "_persist_to_db", side_effect=_fake_persist)  # type: ignore[attr-defined]

    parent_id = uuid.uuid4()
    child_id = uuid.uuid4()
    now = datetime.now(UTC)
    # Raw row first
    persistence.persist_alert(
        alert_id=parent_id,
        timestamp=now,
        alert_type="anomaly_raw",
        priority="CRITICAL",
        message="raw anomaly text",
        symbol="GAZP",
    )
    # LLM follow-up with parent_id pointing back to raw
    persistence.persist_alert(
        alert_id=child_id,
        timestamp=now,
        alert_type="anomaly_llm",
        priority="INFO",
        message="LLM interpretation text",
        symbol="GAZP",
        parent_id=parent_id,
    )
    expected_call_count = 2
    assert len(captured_calls) == expected_call_count
    assert captured_calls[0]["alert_type_key"] == "anomaly_raw"
    assert captured_calls[1]["alert_type_key"] == "anomaly_llm"


def test_persist_alert_async_helper_returns_coroutine() -> None:
    """_persist_alert_async returns an awaitable coroutine (cleanup pattern)."""
    persistence = _make_persistence()
    coro = persistence._persist_alert_async(
        alert_id=uuid.uuid4(),
        timestamp=datetime.now(UTC),
        alert_type="signal",
        priority="INFO",
        symbol="SBER",
        market_id="moex",
        message="text",
        parent_id=None,
        delivery_status="queued",
        alert_metadata=None,
    )
    assert asyncio.iscoroutine(coro)
    coro.close()


def test_update_alert_status_async_helper_returns_coroutine() -> None:
    """_update_alert_status_async returns an awaitable coroutine."""
    persistence = _make_persistence()
    coro = persistence._update_alert_status_async(
        alert_id=uuid.uuid4(),
        timestamp=datetime.now(UTC),
        delivery_status="sent",
    )
    assert asyncio.iscoroutine(coro)
    coro.close()
