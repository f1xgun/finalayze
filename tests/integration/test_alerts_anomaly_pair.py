"""Integration tests for anomaly raw + LLM follow-up parent_id threading.

Phase 57-04 Task 3 (D-04 + Pitfall 2):
- The TradingLoop anomaly orchestration must capture the alert_id of the
  raw alert (alert_type='anomaly_raw') and thread it into the async LLM
  enrichment (alert_type='anomaly_llm') so the persisted child row's
  ``parent_id`` FK is populated at insert time.
- The persistence layer (`TradingPersistence._persist_alert_async`) must
  retry once on FK violation (parent commit race), then degrade to
  ``parent_id=NULL`` rather than dropping the row (revision M5 — flat
  try/except ladder, NOT a loop with continue, so the happy path writes
  exactly once).

The persistence-side tests do not require a real DB — they patch the
session factory + AlertModel constructor so we can simulate IntegrityError
on demand without spinning up TimescaleDB.
"""

from __future__ import annotations

import asyncio
import inspect
import uuid
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.exc import IntegrityError

from finalayze.api.alerts import TelegramAlerter
from finalayze.orchestration.db_persistence import TradingPersistence

# ---------- Helpers --------------------------------------------------------------

_SYMBOL = "SBER"
_MARKET_ID = "ru_blue_chips"


def _make_anomaly() -> SimpleNamespace:
    """Build a duck-typed AnomalyResult."""
    return SimpleNamespace(
        symbol=_SYMBOL,
        market_id=_MARKET_ID,
        price_move_pct=15.0,
        sigma=5.0,
        volume_ratio=1.2,
        anomaly_type="price",
    )


def _make_tl() -> Any:
    """Construct an uninitialised TradingLoop just enough to exercise its methods."""
    from finalayze.core.trading_loop import TradingLoop  # noqa: PLC0415

    return object.__new__(TradingLoop)


# ---------- Tests for the orchestration / threading -----------------------------


@pytest.mark.asyncio
async def test_raw_alert_passes_anomaly_raw_type() -> None:
    """Raw anomaly send fires with alert_type='anomaly_raw' and parent_id=None."""
    raw_uuid = uuid.uuid4()
    alerter = MagicMock(spec=TelegramAlerter)
    alerter._send = AsyncMock(return_value=(True, raw_uuid))

    tl = _make_tl()
    tl._alerter = alerter
    tl._llm_client = None  # raw-only path

    anomaly = _make_anomaly()
    raw_text = f"ANOMALY {_SYMBOL}: {anomaly.price_move_pct:+.1f}%"
    raw_id = await tl._handle_anomaly_async(_SYMBOL, _MARKET_ID, anomaly, raw_text)

    assert raw_id == raw_uuid
    alerter._send.assert_awaited()
    kwargs = alerter._send.await_args.kwargs
    assert kwargs.get("alert_type") == "anomaly_raw", kwargs
    assert kwargs.get("parent_id") is None, kwargs


@pytest.mark.asyncio
async def test_enrich_receives_parent_id() -> None:
    """The orchestration scheduler passes parent_id=<raw_id> to _enrich_anomaly_async."""
    raw_uuid = uuid.uuid4()
    alerter = MagicMock(spec=TelegramAlerter)
    alerter._send = AsyncMock(return_value=(True, raw_uuid))

    tl = _make_tl()
    tl._alerter = alerter
    tl._llm_client = MagicMock()

    captured: dict[str, Any] = {}

    async def _capture_enrich(
        symbol: str,
        market_id: str,
        anomaly: object,
        *,
        parent_id: uuid.UUID | None = None,
    ) -> None:
        captured["symbol"] = symbol
        captured["market_id"] = market_id
        captured["anomaly"] = anomaly
        captured["parent_id"] = parent_id

    tl._enrich_anomaly_async = _capture_enrich  # type: ignore[method-assign]

    anomaly = _make_anomaly()
    raw_text = "ANOMALY raw"
    await tl._handle_anomaly_async(_SYMBOL, _MARKET_ID, anomaly, raw_text)

    # Yield once so the create_task'd coroutine actually runs.
    await asyncio.sleep(0)

    assert captured.get("parent_id") == raw_uuid, (
        f"Expected parent_id={raw_uuid}, captured: {captured}"
    )


@pytest.mark.asyncio
async def test_enrich_passes_parent_id_to_send() -> None:
    """_enrich_anomaly_async forwards parent_id into _send(alert_type='anomaly_llm')."""
    parent_uuid = uuid.uuid4()
    alerter = MagicMock(spec=TelegramAlerter)
    alerter._send = AsyncMock(return_value=(True, uuid.uuid4()))

    llm_client = AsyncMock()
    llm_client.complete = AsyncMock(return_value="LLM explanation")

    tl = _make_tl()
    tl._alerter = alerter
    tl._llm_client = llm_client

    anomaly = _make_anomaly()
    await tl._enrich_anomaly_async(
        _SYMBOL, _MARKET_ID, anomaly, parent_id=parent_uuid,
    )

    alerter._send.assert_awaited_once()
    kwargs = alerter._send.await_args.kwargs
    assert kwargs.get("alert_type") == "anomaly_llm", kwargs
    assert kwargs.get("parent_id") == parent_uuid, kwargs


# ---------- Tests for the FK retry / degradation in persistence ----------------


def _make_persistence() -> TradingPersistence:
    """Build a TradingPersistence with a non-None db_url (no real DB needed).

    The session factory will be replaced per-test, so the URL is irrelevant
    apart from satisfying the ``self._db_url is None`` short-circuit guard.
    """
    persistence = TradingPersistence.__new__(TradingPersistence)
    persistence._db_url = "postgresql+asyncpg://fake/db"  # type: ignore[attr-defined]
    persistence._async_loop = None  # type: ignore[attr-defined]
    persistence._settings = None  # type: ignore[attr-defined]
    persistence._bg_session_factory = None  # type: ignore[attr-defined]
    return persistence


def _patch_session_factory(persistence: TradingPersistence, behaviours: list[str]) -> list[str]:
    """Patch _get_bg_session_factory so each session.commit() is driven by ``behaviours``.

    Each entry is one of:
      * "ok"  — commit succeeds.
      * "fk_error" — commit raises IntegrityError.
    Returns a recorder list of the operations actually attempted.
    """
    recorder: list[str] = []
    behaviour_iter = iter(behaviours)

    class _FakeSession:
        def __init__(self) -> None:
            self._behaviour = next(behaviour_iter)

        async def __aenter__(self) -> _FakeSession:
            return self

        async def __aexit__(self, *args: object) -> None:
            return None

        def add(self, _row: object) -> None:
            recorder.append("add")

        async def commit(self) -> None:
            recorder.append(f"commit:{self._behaviour}")
            if self._behaviour == "fk_error":
                raise IntegrityError("INSERT", {}, Exception("fk violation"))

    def _factory_callable() -> _FakeSession:
        return _FakeSession()

    persistence._get_bg_session_factory = lambda: _factory_callable  # type: ignore[method-assign]
    return recorder


@pytest.mark.asyncio
async def test_fk_violation_retry_succeeds_on_second_attempt() -> None:
    """First commit raises FK violation; sleep+retry succeeds with same parent_id."""
    persistence = _make_persistence()
    recorder = _patch_session_factory(
        persistence,
        ["fk_error", "ok"],  # 1st attempt fails, retry succeeds
    )

    parent_id = uuid.uuid4()
    await persistence._persist_alert_async(
        alert_id=uuid.uuid4(),
        timestamp=datetime.now(UTC),
        alert_type="anomaly_llm",
        priority="INFO",
        symbol=_SYMBOL,
        market_id=_MARKET_ID,
        message="LLM follow-up",
        parent_id=parent_id,
        delivery_status="queued",
        alert_metadata=None,
    )

    # Two write attempts; the second succeeded with the original parent_id.
    commit_events = [r for r in recorder if r.startswith("commit:")]
    assert commit_events == ["commit:fk_error", "commit:ok"], recorder


@pytest.mark.asyncio
async def test_fk_violation_falls_back_to_null_after_retry_exhausted() -> None:
    """Both attempts raise FK violation; the third write degrades parent_id to NULL."""
    persistence = _make_persistence()
    recorder = _patch_session_factory(
        persistence,
        ["fk_error", "fk_error", "ok"],  # both retries fail; NULL fallback succeeds
    )

    parent_id = uuid.uuid4()
    await persistence._persist_alert_async(
        alert_id=uuid.uuid4(),
        timestamp=datetime.now(UTC),
        alert_type="anomaly_llm",
        priority="INFO",
        symbol=_SYMBOL,
        market_id=_MARKET_ID,
        message="LLM follow-up",
        parent_id=parent_id,
        delivery_status="queued",
        alert_metadata=None,
    )

    commit_events = [r for r in recorder if r.startswith("commit:")]
    # Three commits total: try1 (fk), retry (fk), final NULL fallback (ok).
    assert commit_events == ["commit:fk_error", "commit:fk_error", "commit:ok"], recorder


@pytest.mark.asyncio
async def test_happy_path_writes_exactly_once() -> None:
    """Revision M5: when no FK error, _do_write is invoked exactly once."""
    persistence = _make_persistence()
    recorder = _patch_session_factory(persistence, ["ok"])

    await persistence._persist_alert_async(
        alert_id=uuid.uuid4(),
        timestamp=datetime.now(UTC),
        alert_type="signal",
        priority="INFO",
        symbol=_SYMBOL,
        market_id=_MARKET_ID,
        message="signal alert",
        parent_id=uuid.uuid4(),  # parent_id present BUT no FK error
        delivery_status="queued",
        alert_metadata=None,
    )

    commit_events = [r for r in recorder if r.startswith("commit:")]
    add_events = [r for r in recorder if r == "add"]
    assert commit_events == ["commit:ok"], (
        f"Happy path must commit exactly once, got: {recorder}"
    )
    assert len(add_events) == 1, f"Happy path must add exactly one row, got: {recorder}"


def test_persist_alert_async_source_uses_do_write_helper() -> None:
    """Revision M5 source-presence guard: inner _do_write helper exists, no loop fall-through."""
    src = inspect.getsource(TradingPersistence._persist_alert_async)
    assert "_do_write" in src, (
        "Revision M5: _persist_alert_async must use the inner _do_write helper"
    )
    assert "for attempt in" not in src, (
        "Revision M5: the old loop-with-continue pattern must be gone "
        "(double-write risk eliminated)"
    )
