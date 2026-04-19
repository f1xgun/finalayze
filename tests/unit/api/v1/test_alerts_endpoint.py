"""Unit tests for GET /api/v1/alerts (ALRT-03 D-16/D-18).

Validates the new alerts list endpoint:
  - Pagination via ?page=N&page_size=N (default 50, max 200, D-18).
  - Filters: alert_type (multi), symbol, priority, since/until (D-16).
  - Ordered by timestamp DESC.
  - Includes parent_id for threaded anomaly pairs.
  - Graceful DB-failure degradation: returns ``{alerts: [], total: 0}``
    rather than 500 (mirrors /portfolio/history hybrid pattern from
    Phase 56-03).

Tests mock ``finalayze.core.db.get_async_session_factory`` (source module,
NOT the API alias) since the handler does a function-local
``from finalayze.core.db import get_async_session_factory`` (Phase 56-03
lesson). Row stand-ins use SimpleNamespace to avoid SQLAlchemy ``bool()``
on clauses (raises ``TypeError: Boolean value of this clause is not
defined``).
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from fastapi.testclient import TestClient

from finalayze.main import create_app

# ---------- Helpers --------------------------------------------------------------

_PRIORITY_CRITICAL = "CRITICAL"
_PRIORITY_INFO = "INFO"
_TOTAL_75 = 75
_PAGE_50 = 50
_PAGE_25 = 25
_OVERSIZED_PAGE = 300
_HTTP_VALIDATION_ERROR = 422


def _auth() -> dict[str, str]:
    from config.settings import Settings

    return {"X-API-Key": Settings().api_key}


def _make_row(
    *,
    timestamp: datetime,
    alert_type: str = "signal",
    priority: str = _PRIORITY_INFO,
    symbol: str | None = "SBER",
    market_id: str | None = "moex",
    message: str = "test alert",
    parent_id: uuid.UUID | None = None,
    delivery_status: str = "sent",
    alert_id: uuid.UUID | None = None,
) -> SimpleNamespace:
    """Duck-typed AlertModel row.

    The handler accesses .id, .timestamp, .alert_type, .priority, .symbol,
    .market_id, .message, .parent_id, .delivery_status. Using
    SimpleNamespace avoids SQLAlchemy mapper machinery — rows never enter a
    session, so attribute setters via ``__set__`` don't apply.
    """
    return SimpleNamespace(
        id=alert_id or uuid.uuid4(),
        timestamp=timestamp,
        alert_type=alert_type,
        priority=priority,
        symbol=symbol,
        market_id=market_id,
        message=message,
        parent_id=parent_id,
        delivery_status=delivery_status,
    )


def _patch_session_factory(rows: list[SimpleNamespace]) -> Any:
    """Patch ``finalayze.core.db.get_async_session_factory`` for the handler.

    The fake session inspects the SQL text to differentiate the count query
    from the row query, applies in-memory filters that mirror what the real
    SQLAlchemy ``where`` clauses would have done, then returns a Result
    that exposes ``.scalars().all()`` (rows) or ``.scalar()`` (count).
    """

    class _Result:
        def __init__(self, items: list[Any], scalar_value: int | None = None) -> None:
            self._items = items
            self._scalar = scalar_value

        def scalars(self) -> _Result:
            return self

        def all(self) -> list[Any]:
            return list(self._items)

        def scalar(self) -> int | None:
            return self._scalar

    class _Session:
        async def __aenter__(self) -> _Session:
            return self

        async def __aexit__(self, *args: object) -> None:
            return None

        async def execute(self, stmt: Any) -> _Result:
            sql = str(stmt).lower()

            # Apply in-memory filters from the test-set state
            filtered = list(rows)
            if _Session._alert_type_filter is not None:
                filtered = [
                    r for r in filtered if r.alert_type in _Session._alert_type_filter
                ]
            if _Session._symbol_filter is not None:
                filtered = [r for r in filtered if r.symbol == _Session._symbol_filter]
            if _Session._priority_filter is not None:
                filtered = [r for r in filtered if r.priority == _Session._priority_filter]
            if _Session._since_filter is not None:
                filtered = [r for r in filtered if r.timestamp >= _Session._since_filter]
            if _Session._until_filter is not None:
                filtered = [r for r in filtered if r.timestamp <= _Session._until_filter]

            # Determine count query vs row query
            if "count" in sql:
                return _Result([], scalar_value=len(filtered))

            # Row query: order desc by timestamp, apply limit + offset
            filtered.sort(key=lambda r: r.timestamp, reverse=True)
            limit = _Session._limit if _Session._limit is not None else len(filtered)
            offset = _Session._offset if _Session._offset is not None else 0
            paged = filtered[offset : offset + limit]
            return _Result(paged)

    _Session._alert_type_filter = None  # type: ignore[attr-defined]
    _Session._symbol_filter = None  # type: ignore[attr-defined]
    _Session._priority_filter = None  # type: ignore[attr-defined]
    _Session._since_filter = None  # type: ignore[attr-defined]
    _Session._until_filter = None  # type: ignore[attr-defined]
    _Session._limit = None  # type: ignore[attr-defined]
    _Session._offset = None  # type: ignore[attr-defined]

    def _factory_callable() -> _Session:
        return _Session()

    def _get_factory() -> Any:
        return _factory_callable

    patcher = patch(
        "finalayze.core.db.get_async_session_factory",
        side_effect=_get_factory,
    )
    return patcher, _Session


def _patch_failing_session() -> Any:
    """Patch session factory so .execute() raises — exercises graceful degradation."""

    class _BadSession:
        async def __aenter__(self) -> _BadSession:
            return self

        async def __aexit__(self, *args: object) -> None:
            return None

        async def execute(self, stmt: Any) -> Any:
            raise RuntimeError("DB down")

    def _factory_callable() -> _BadSession:
        return _BadSession()

    def _get_factory() -> Any:
        return _factory_callable

    return patch(
        "finalayze.core.db.get_async_session_factory",
        side_effect=_get_factory,
    )


# ---------- Tests ----------------------------------------------------------------


def test_list_alerts_returns_paginated_response() -> None:
    """Page 1 with default page_size=50 returns 50 alerts of 75 total."""
    now = datetime.now(UTC)
    rows = [
        _make_row(timestamp=now - timedelta(minutes=i)) for i in range(_TOTAL_75)
    ]

    patcher, sess = _patch_session_factory(rows)
    sess._limit = _PAGE_50  # type: ignore[attr-defined]
    sess._offset = 0  # type: ignore[attr-defined]

    with patcher:
        resp = TestClient(create_app()).get(
            "/api/v1/alerts?page=1&page_size=50",
            headers=_auth(),
        )

    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert len(data["alerts"]) == _PAGE_50
    assert data["total"] == _TOTAL_75
    assert data["page"] == 1
    assert data["page_size"] == _PAGE_50


def test_list_alerts_page_2_returns_remaining() -> None:
    """Page 2 with page_size=50 returns the remaining 25 alerts."""
    now = datetime.now(UTC)
    rows = [
        _make_row(timestamp=now - timedelta(minutes=i)) for i in range(_TOTAL_75)
    ]

    patcher, sess = _patch_session_factory(rows)
    sess._limit = _PAGE_50  # type: ignore[attr-defined]
    sess._offset = _PAGE_50  # type: ignore[attr-defined]

    with patcher:
        resp = TestClient(create_app()).get(
            "/api/v1/alerts?page=2&page_size=50",
            headers=_auth(),
        )

    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert len(data["alerts"]) == _PAGE_25
    assert data["page"] == 2


def test_list_alerts_filters_by_alert_type() -> None:
    """?alert_type=stop_loss&alert_type=signal narrows result set."""
    now = datetime.now(UTC)
    rows = [
        _make_row(timestamp=now - timedelta(minutes=0), alert_type="stop_loss"),
        _make_row(timestamp=now - timedelta(minutes=1), alert_type="signal"),
        _make_row(timestamp=now - timedelta(minutes=2), alert_type="anomaly_raw"),
    ]

    patcher, sess = _patch_session_factory(rows)
    sess._alert_type_filter = ["stop_loss", "signal"]  # type: ignore[attr-defined]

    with patcher:
        resp = TestClient(create_app()).get(
            "/api/v1/alerts?alert_type=stop_loss&alert_type=signal",
            headers=_auth(),
        )

    assert resp.status_code == 200, resp.text
    types = sorted(a["alert_type"] for a in resp.json()["alerts"])
    assert types == ["signal", "stop_loss"]


def test_list_alerts_filters_by_symbol() -> None:
    """?symbol=SBER returns only SBER rows."""
    now = datetime.now(UTC)
    rows = [
        _make_row(timestamp=now - timedelta(minutes=0), symbol="SBER"),
        _make_row(timestamp=now - timedelta(minutes=1), symbol="GAZP"),
        _make_row(timestamp=now - timedelta(minutes=2), symbol="SBER"),
    ]

    patcher, sess = _patch_session_factory(rows)
    sess._symbol_filter = "SBER"  # type: ignore[attr-defined]

    with patcher:
        resp = TestClient(create_app()).get(
            "/api/v1/alerts?symbol=SBER",
            headers=_auth(),
        )

    assert resp.status_code == 200, resp.text
    symbols = {a["symbol"] for a in resp.json()["alerts"]}
    assert symbols == {"SBER"}


def test_list_alerts_filters_by_priority() -> None:
    """?priority=CRITICAL filters to AlertPriority.CRITICAL.name (revision Mi5)."""
    now = datetime.now(UTC)
    rows = [
        _make_row(timestamp=now - timedelta(minutes=0), priority=_PRIORITY_CRITICAL),
        _make_row(timestamp=now - timedelta(minutes=1), priority=_PRIORITY_INFO),
        _make_row(timestamp=now - timedelta(minutes=2), priority=_PRIORITY_CRITICAL),
    ]

    patcher, sess = _patch_session_factory(rows)
    sess._priority_filter = _PRIORITY_CRITICAL  # type: ignore[attr-defined]

    with patcher:
        resp = TestClient(create_app()).get(
            f"/api/v1/alerts?priority={_PRIORITY_CRITICAL}",
            headers=_auth(),
        )

    assert resp.status_code == 200, resp.text
    priorities = {a["priority"] for a in resp.json()["alerts"]}
    assert priorities == {_PRIORITY_CRITICAL}


def test_list_alerts_filters_by_date_range() -> None:
    """?since=...&until=... narrows to a time window."""
    now = datetime.now(UTC).replace(microsecond=0)
    rows = [
        _make_row(timestamp=now - timedelta(hours=h)) for h in (0, 6, 12, 18, 24)
    ]
    since = now - timedelta(hours=18)
    until = now - timedelta(hours=6)

    patcher, sess = _patch_session_factory(rows)
    sess._since_filter = since  # type: ignore[attr-defined]
    sess._until_filter = until  # type: ignore[attr-defined]

    with patcher:
        resp = TestClient(create_app()).get(
            f"/api/v1/alerts?since={since.isoformat()}&until={until.isoformat()}",
            headers=_auth(),
        )

    assert resp.status_code == 200, resp.text
    timestamps = [datetime.fromisoformat(a["timestamp"]) for a in resp.json()["alerts"]]
    for ts in timestamps:
        assert since <= ts <= until


def test_list_alerts_orders_desc() -> None:
    """Rows must be ordered by timestamp DESC (newest first)."""
    now = datetime.now(UTC)
    # Insert in random order but expect DESC return
    rows = [
        _make_row(timestamp=now - timedelta(hours=h)) for h in (5, 1, 10, 3, 0)
    ]

    patcher, _ = _patch_session_factory(rows)
    with patcher:
        resp = TestClient(create_app()).get("/api/v1/alerts", headers=_auth())

    assert resp.status_code == 200, resp.text
    timestamps = [datetime.fromisoformat(a["timestamp"]) for a in resp.json()["alerts"]]
    assert timestamps == sorted(timestamps, reverse=True), (
        f"Expected DESC order, got: {timestamps}"
    )


def test_list_alerts_page_size_validation() -> None:
    """page_size=300 (> _MAX_PAGE_SIZE=200) returns 422 Unprocessable Entity."""
    resp = TestClient(create_app()).get(
        f"/api/v1/alerts?page_size={_OVERSIZED_PAGE}",
        headers=_auth(),
    )
    assert resp.status_code == _HTTP_VALIDATION_ERROR, resp.text


def test_list_alerts_includes_parent_id() -> None:
    """parent_id is preserved in the response (anomaly threading prerequisite)."""
    now = datetime.now(UTC)
    parent_uuid = uuid.uuid4()
    child_uuid = uuid.uuid4()
    rows = [
        _make_row(
            timestamp=now,
            alert_id=child_uuid,
            alert_type="anomaly_llm",
            parent_id=parent_uuid,
        ),
    ]

    patcher, _ = _patch_session_factory(rows)
    with patcher:
        resp = TestClient(create_app()).get("/api/v1/alerts", headers=_auth())

    assert resp.status_code == 200, resp.text
    alerts = resp.json()["alerts"]
    assert len(alerts) == 1
    assert alerts[0]["parent_id"] == str(parent_uuid)


def test_list_alerts_db_error_returns_empty() -> None:
    """When the session.execute raises, handler returns ``{alerts:[], total:0}`` not 500."""
    with _patch_failing_session():
        resp = TestClient(create_app()).get("/api/v1/alerts", headers=_auth())

    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["alerts"] == []
    assert data["total"] == 0
