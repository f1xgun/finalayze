"""Tests for /health/production and /kill REST endpoints.

Validates:
  - GET /api/v1/health/production returns per-component JSON
  - GET /api/v1/health/production returns 200/503 based on health
  - POST /api/v1/kill triggers KillSwitch and returns result
  - POST /api/v1/kill returns 503 when not configured
  - GET /api/v1/health/production returns 503 when not configured
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# Import the router -- we'll patch module-level state
from finalayze.api.v1 import system

_HEALTH_URL = "/api/v1/health/production"
_KILL_URL = "/api/v1/kill"


@dataclass(frozen=True)
class _FakeHealthResult:
    broker_ok: bool = True
    feed_fresh: bool = True
    loop_alive: bool = True
    timestamp: datetime = datetime(2026, 3, 22, tzinfo=UTC)
    details: dict[str, str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.details is None:
            object.__setattr__(self, "details", {"broker": "ok", "feed": "fresh", "loop": "alive"})


@dataclass(frozen=True)
class _FakeKillResult:
    orders_cancelled: int = 3
    scheduler_stopped: bool = True
    breakers_escalated: int = 2
    alert_sent: bool = True
    elapsed_seconds: float = 1.5


def _make_app() -> FastAPI:
    """Create a minimal FastAPI app with the system router."""
    app = FastAPI()
    app.include_router(system.router, prefix="/api/v1")
    return app


class TestHealthProductionEndpoint:
    """Tests for GET /api/v1/health/production."""

    def test_returns_json_with_expected_fields(self) -> None:
        """Test 1: Returns JSON with broker_ok, feed_fresh, loop_alive, overall, timestamp."""
        monitor = MagicMock()
        monitor.check_now.return_value = _FakeHealthResult()

        with patch.object(system, "_health_monitor", monitor):
            client = TestClient(_make_app())
            resp = client.get(_HEALTH_URL)

        assert resp.status_code == 200
        data = resp.json()
        assert "broker_ok" in data
        assert "feed_fresh" in data
        assert "loop_alive" in data
        assert "overall" in data
        assert "timestamp" in data
        assert data["overall"] is True

    def test_returns_503_when_unhealthy(self) -> None:
        """Test 2: Returns 503 when any check fails."""
        monitor = MagicMock()
        monitor.check_now.return_value = _FakeHealthResult(broker_ok=False)

        with patch.object(system, "_health_monitor", monitor):
            client = TestClient(_make_app())
            resp = client.get(_HEALTH_URL)

        assert resp.status_code == 503
        data = resp.json()
        # HTTPException wraps in "detail"
        detail = data.get("detail", data)
        assert detail["overall"] is False

    def test_returns_503_when_not_configured(self) -> None:
        """Test 5: Returns 503 if health monitor not configured."""
        with patch.object(system, "_health_monitor", None):
            client = TestClient(_make_app())
            resp = client.get(_HEALTH_URL)

        assert resp.status_code == 503
        data = resp.json()
        assert "not configured" in data["detail"].lower()


class TestKillEndpoint:
    """Tests for POST /api/v1/kill."""

    def test_kill_returns_result_json(self) -> None:
        """Test 3: POST /kill returns KillSwitchResult JSON."""
        ks = MagicMock()
        ks.activate.return_value = _FakeKillResult()

        with patch.object(system, "_kill_switch", ks):
            client = TestClient(_make_app())
            resp = client.post(_KILL_URL)

        assert resp.status_code == 200
        data = resp.json()
        assert data["orders_cancelled"] == 3
        assert data["scheduler_stopped"] is True
        assert data["breakers_escalated"] == 2
        ks.activate.assert_called_once()

    def test_kill_returns_503_when_not_configured(self) -> None:
        """Test 4: POST /kill returns 503 if kill switch not configured."""
        with patch.object(system, "_kill_switch", None):
            client = TestClient(_make_app())
            resp = client.post(_KILL_URL)

        assert resp.status_code == 503
        data = resp.json()
        assert "not configured" in data["detail"].lower()
