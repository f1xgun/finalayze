"""Integration tests for anomaly raw + LLM follow-up parent_id threading.

Phase 57-04 Task 3 (D-04):
- The TradingLoop anomaly orchestration must capture the alert_id of the
  raw alert (alert_type='anomaly_raw') and thread it into the async LLM
  enrichment (alert_type='anomaly_llm') so the persisted child row's
  ``parent_id`` is populated at insert time.

Phase 57 UAT gap-closure: parent_id has NO database-level FK to alerts(id).
TimescaleDB hypertables forbid the UNIQUE (id) constraint that a self-FK
would require, so integrity is managed at the application layer (raw
alerts always persist before LLM follow-ups send). The FK retry ladder
that the original Plan 04 introduced is gone; ``_persist_alert_async``
now writes exactly once.

The persistence-side test does not require a real DB — it patches the
session factory.
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
    from finalayze.orchestration.trading_loop import TradingLoop  # noqa: PLC0415

    return object.__new__(TradingLoop)


# ---------- Tests for the orchestration / threading -----------------------------


@pytest.mark.asyncio
async def test_raw_alert_passes_anomaly_raw_type() -> None:
    """Raw anomaly send fires with alert_type='anomaly_raw' and parent_id=None."""
    raw_uuid = uuid.uuid4()
    alerter = MagicMock(spec=TelegramAlerter)
    alerter.send_async = AsyncMock(return_value=(True, raw_uuid))

    tl = _make_tl()
    tl._alerter = alerter
    tl._llm_client = None  # raw-only path

    anomaly = _make_anomaly()
    raw_text = f"ANOMALY {_SYMBOL}: {anomaly.price_move_pct:+.1f}%"
    raw_id = await tl._handle_anomaly_async(_SYMBOL, _MARKET_ID, anomaly, raw_text)

    assert raw_id == raw_uuid
    alerter.send_async.assert_awaited()
    kwargs = alerter.send_async.await_args.kwargs
    assert kwargs.get("alert_type") == "anomaly_raw", kwargs
    assert kwargs.get("parent_id") is None, kwargs


@pytest.mark.asyncio
async def test_enrich_receives_parent_id() -> None:
    """The orchestration scheduler passes parent_id=<raw_id> to handler.enrich."""
    raw_uuid = uuid.uuid4()
    alerter = MagicMock(spec=TelegramAlerter)
    alerter.send_async = AsyncMock(return_value=(True, raw_uuid))

    tl = _make_tl()
    tl._alerter = alerter
    tl._llm_client = MagicMock()

    # Construct the anomaly handler (which is normally done in __init__)
    from finalayze.orchestration.anomaly_handler import AnomalyHandler  # noqa: PLC0415

    tl._anomaly_handler = AnomalyHandler(
        alerter,
        lambda: tl._llm_client,
    )

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

    # Patch the handler's enrich method (the actual orchestration point)
    tl._anomaly_handler.enrich = _capture_enrich  # type: ignore[method-assign]

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
    alerter.send_async = AsyncMock(return_value=(True, uuid.uuid4()))

    llm_client = AsyncMock()
    llm_client.complete = AsyncMock(return_value="LLM explanation")

    tl = _make_tl()
    tl._alerter = alerter
    tl._llm_client = llm_client

    anomaly = _make_anomaly()
    await tl._enrich_anomaly_async(
        _SYMBOL,
        _MARKET_ID,
        anomaly,
        parent_id=parent_uuid,
    )

    alerter.send_async.assert_awaited_once()
    kwargs = alerter.send_async.await_args.kwargs
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


def _patch_session_factory(persistence: TradingPersistence) -> list[str]:
    """Patch _get_bg_session_factory with a single-shot fake session.

    Records every ``add`` and ``commit`` call so tests can assert exactly
    one write happens per ``_persist_alert_async`` invocation.
    """
    recorder: list[str] = []

    class _FakeSession:
        async def __aenter__(self) -> _FakeSession:
            return self

        async def __aexit__(self, *args: object) -> None:
            return None

        def add(self, _row: object) -> None:
            recorder.append("add")

        async def commit(self) -> None:
            recorder.append("commit")

    def _factory_callable() -> _FakeSession:
        return _FakeSession()

    persistence._get_bg_session_factory = lambda: _factory_callable  # type: ignore[method-assign]
    return recorder


@pytest.mark.asyncio
async def test_persist_alert_async_writes_exactly_once() -> None:
    """parent_id has no DB FK — single write per call, no retry ladder."""
    persistence = _make_persistence()
    recorder = _patch_session_factory(persistence)

    await persistence._persist_alert_async(
        alert_id=uuid.uuid4(),
        timestamp=datetime.now(UTC),
        alert_type="anomaly_llm",
        priority="INFO",
        symbol=_SYMBOL,
        market_id=_MARKET_ID,
        message="LLM follow-up",
        parent_id=uuid.uuid4(),
        delivery_status="queued",
        alert_metadata=None,
    )

    assert recorder == ["add", "commit"], (
        f"_persist_alert_async must write exactly once; got: {recorder}"
    )


def test_persist_alert_async_source_has_no_fk_retry_ladder() -> None:
    """parent_id integrity is app-managed — no FK retry ladder remains."""
    src = inspect.getsource(TradingPersistence._persist_alert_async)
    assert "IntegrityError" not in src, (
        "FK retry ladder removed — TimescaleDB hypertable forbids the UNIQUE "
        "constraint that a self-FK would require, so parent_id is a plain "
        "nullable UUID and FK retry is no longer needed"
    )
    assert "for attempt in" not in src, "No retry loop — single write per call"
    assert "asyncio.sleep" not in src, "No retry sleep — single write per call"
