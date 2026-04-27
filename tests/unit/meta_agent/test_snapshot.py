"""Tests for meta_agent.snapshot (Phase 58 META-01).

D-01: REST self-call via httpx + X-API-Key.
D-03: REST collector tolerates partial endpoint failure (None on a field).

Mocks the FastAPI surface with respx (already a project test dep).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import httpx
import pytest
import respx
import structlog
from structlog.testing import capture_logs

# Module-level constants (PLR2004).
_BASE = "http://127.0.0.1:8000"
_KEY = "test-api-key"
_NOW = datetime(2026, 4, 26, 12, 0, tzinfo=UTC)
_SINCE = (_NOW - timedelta(hours=1)).isoformat()
_DRAWDOWN_FIXTURE = 1.2
_PERSIST_FAILURES_FIXTURE = 0
_HTTP_INTERNAL_ERROR = 500


@pytest.fixture
def client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url=_BASE,
        headers={"X-API-Key": _KEY},
        timeout=10.0,
    )


@pytest.mark.asyncio
@respx.mock
async def test_build_snapshot_happy_path_populates_all_fields(
    client: httpx.AsyncClient,
) -> None:
    """All three REST endpoints respond 200 — Snapshot fields all populate."""
    from finalayze.meta_agent.snapshot import Snapshot, build_snapshot

    respx.get(f"{_BASE}/api/v1/alerts").mock(
        return_value=httpx.Response(
            200,
            json={
                "alerts": [
                    {
                        "id": "00000000-0000-0000-0000-000000000001",
                        "timestamp": _NOW.isoformat(),
                        "alert_type": "anomaly_raw",
                        "priority": "INFO",
                        "symbol": "SBER",
                        "market_id": "moex",
                        "message": "noise",
                        "parent_id": None,
                        "delivery_status": "sent",
                    },
                ],
                "total": 1,
                "page": 1,
                "page_size": 50,
            },
        ),
    )
    respx.get(f"{_BASE}/api/v1/portfolio/performance").mock(
        return_value=httpx.Response(
            200,
            json={"equity": 100000.0, "drawdown_pct": _DRAWDOWN_FIXTURE},
        ),
    )
    respx.get(f"{_BASE}/api/v1/positions").mock(
        return_value=httpx.Response(200, json={"positions": []}),
    )

    snap = await build_snapshot(client, now=_NOW)
    assert isinstance(snap, Snapshot)
    assert snap.timestamp == _NOW
    assert snap.alerts_last_hour is not None
    assert len(snap.alerts_last_hour) == 1
    assert snap.drawdown_pct == pytest.approx(_DRAWDOWN_FIXTURE)
    assert snap.equity_persist_failures == _PERSIST_FAILURES_FIXTURE
    assert snap.ml_signal_error_rate is None
    assert snap.positions_summary is not None

    # frozen=True invariant — Pydantic raises ValidationError on mutation.
    with pytest.raises((TypeError, ValueError)):
        snap.timestamp = _NOW + timedelta(hours=1)  # type: ignore[misc]

    await client.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_build_snapshot_alerts_endpoint_500_sets_field_none(
    client: httpx.AsyncClient,
) -> None:
    """D-03: A single endpoint returning 5xx yields None on the corresponding
    Snapshot field. The other two endpoints still populate. A structlog
    ``meta_agent_snapshot_partial`` event is emitted with endpoint='alerts'.
    """
    from finalayze.meta_agent.snapshot import build_snapshot

    respx.get(f"{_BASE}/api/v1/alerts").mock(
        return_value=httpx.Response(500, json={"detail": "DB timeout"}),
    )
    respx.get(f"{_BASE}/api/v1/portfolio/performance").mock(
        return_value=httpx.Response(
            200,
            json={"equity": 100000.0, "drawdown_pct": _DRAWDOWN_FIXTURE},
        ),
    )
    respx.get(f"{_BASE}/api/v1/positions").mock(
        return_value=httpx.Response(200, json={"positions": []}),
    )

    with capture_logs() as logs:
        snap = await build_snapshot(client, now=_NOW)

    assert snap.alerts_last_hour is None  # graceful: no raise
    assert snap.drawdown_pct == pytest.approx(_DRAWDOWN_FIXTURE)
    assert snap.positions_summary is not None

    # Find the partial-failure log event for the alerts endpoint.
    partial_events = [
        log
        for log in logs
        if log.get("event") == "meta_agent_snapshot_partial" and log.get("endpoint") == "alerts"
    ]
    assert partial_events, f"Expected meta_agent_snapshot_partial(endpoint='alerts'); got {logs!r}"
    assert partial_events[0].get("status") == _HTTP_INTERNAL_ERROR

    await client.aclose()
