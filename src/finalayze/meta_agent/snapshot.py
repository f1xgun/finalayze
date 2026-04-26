"""System-health snapshot collector (Phase 58 META-01).

D-01: REST self-call via httpx + X-API-Key. The collector treats the
running FastAPI process as its own client — same auth surface as an
external operator, so any future REST contract change is caught by the
same contract tests.

D-03: Tolerates partial endpoint failure. A single endpoint returning
5xx (or raising at the transport layer) yields ``None`` on the
corresponding ``Snapshot`` field plus a structlog ``meta_agent_snapshot_partial``
event. The classifier short-circuits to HEALTHY when ALL critical fields
are None (snapshot unusable).

PATTERNS row "snapshot.py" — analog: ``dashboard/api_client.py:13-36``
(auth-injection); ``markets/fx_service.py:31`` (persistent async client);
``tests/unit/test_dashboard_api_client.py:25-33`` (respx mock).
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

import structlog
from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    import httpx

_log = structlog.get_logger()

# Endpoint paths (D-01) — fixed contract; do NOT parameterise.
_ALERTS_PATH = "/api/v1/alerts"
_PERFORMANCE_PATH = "/api/v1/portfolio/performance"
_POSITIONS_PATH = "/api/v1/positions"

# 5xx threshold (D-03 partial-failure trigger).
_HTTP_SERVER_ERROR_FLOOR = 500


class AlertSummary(BaseModel):
    """One alert row from /api/v1/alerts (Phase 57-04 envelope subset).

    Frozen — snapshot is an immutable evidence record.
    """

    model_config = ConfigDict(frozen=True, extra="ignore")
    id: str
    timestamp: str
    alert_type: str
    priority: str
    symbol: str | None = None
    market_id: str | None = None
    message: str
    parent_id: str | None = None
    delivery_status: str


class PositionsSummary(BaseModel):
    """Aggregated positions snapshot. Stays loose (no per-position decomp)
    until Phase 58-02 extends the classifier with position-level rules.
    """

    model_config = ConfigDict(frozen=True, extra="allow")
    raw: dict[str, Any]


class Snapshot(BaseModel):
    """One cycle's worth of system-health evidence.

    SPEC §Requirement 1 line 28: ``Snapshot(timestamp, alerts_last_hour,
    drawdown_pct, equity_persist_failures, ml_signal_error_rate,
    positions_summary, raw)``.

    All fields are populated when their source endpoint responds 200; on
    5xx or transport error the field is set to ``None`` (D-03) and the
    classifier short-circuits if ALL critical fields are unusable.
    """

    model_config = ConfigDict(frozen=True)
    timestamp: datetime
    alerts_last_hour: list[AlertSummary] | None
    drawdown_pct: float | None
    equity_persist_failures: int = 0
    ml_signal_error_rate: float | None = None
    positions_summary: PositionsSummary | None
    raw: dict[str, Any] = {}


async def _fetch_one(
    client: httpx.AsyncClient,
    *,
    path: str,
    params: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Fetch a single endpoint. Return parsed JSON or None on failure (D-03).

    Logs ``meta_agent_snapshot_partial`` for any 5xx / transport error so the
    runner has an audit trail of endpoint health.
    """
    try:
        resp = await client.get(path, params=params)
    except Exception as exc:  # noqa: BLE001 — partial-failure envelope (D-03)
        _log.warning(
            "meta_agent_snapshot_partial",
            endpoint=_endpoint_label(path),
            status=None,
            reason=str(exc.__class__.__name__),
        )
        return None
    if resp.status_code >= _HTTP_SERVER_ERROR_FLOOR:
        _log.warning(
            "meta_agent_snapshot_partial",
            endpoint=_endpoint_label(path),
            status=resp.status_code,
        )
        return None
    try:
        body = resp.json()
    except Exception:  # noqa: BLE001
        _log.warning(
            "meta_agent_snapshot_partial",
            endpoint=_endpoint_label(path),
            status=resp.status_code,
            reason="json_decode",
        )
        return None
    if not isinstance(body, dict):
        return None
    return body


def _endpoint_label(path: str) -> str:
    """Map URL path to a stable structlog label."""
    if path == _ALERTS_PATH:
        return "alerts"
    if path == _PERFORMANCE_PATH:
        return "performance"
    if path == _POSITIONS_PATH:
        return "positions"
    return path


async def build_snapshot(
    client: httpx.AsyncClient,
    *,
    now: datetime,
) -> Snapshot:
    """Fan out three GET calls, assemble a frozen Snapshot.

    Per D-01: caller constructs ``httpx.AsyncClient(base_url=..., headers=
    {"X-API-Key": ...})`` so this function stays auth-agnostic and
    test-friendly. Per D-03: partial endpoint failure does NOT raise; the
    failing field is set to None.
    """
    since = (now - timedelta(hours=1)).isoformat()
    alerts_body, perf_body, pos_body = await asyncio.gather(
        _fetch_one(client, path=_ALERTS_PATH, params={"since": since}),
        _fetch_one(client, path=_PERFORMANCE_PATH, params={"days": 1}),
        _fetch_one(client, path=_POSITIONS_PATH),
        return_exceptions=False,
    )

    alerts: list[AlertSummary] | None
    if alerts_body is None:
        alerts = None
    else:
        alerts = [AlertSummary.model_validate(a) for a in alerts_body.get("alerts", [])]

    drawdown: float | None
    if perf_body is None:
        drawdown = None
    else:
        dd = perf_body.get("drawdown_pct")
        drawdown = float(dd) if dd is not None else None

    positions: PositionsSummary | None
    if pos_body is None:
        positions = None
    else:
        positions = PositionsSummary(raw=pos_body)

    raw_payload: dict[str, Any] = {
        "alerts": alerts_body,
        "performance": perf_body,
        "positions": pos_body,
    }

    return Snapshot(
        timestamp=now,
        alerts_last_hour=alerts,
        drawdown_pct=drawdown,
        equity_persist_failures=0,  # populated in 58-02 (no metric source yet)
        ml_signal_error_rate=None,  # populated in 58-02
        positions_summary=positions,
        raw=raw_payload,
    )
