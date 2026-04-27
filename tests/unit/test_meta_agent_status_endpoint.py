"""Tests for the GET /api/v1/meta-agent/status endpoint and the
``meta_agent_*`` Settings cluster (Phase 58-01, META-04 / META-08 surface).

Initial test (Task 58-01-01) covers only the Settings field defaults.
The status-endpoint tests live further down (Task 58-01-11) and assume
the FastAPI router exists.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# Module-level constants (PLR2004 — no magic numbers in tests).
_DEFAULT_INTERVAL_MIN = 30
_DEFAULT_TG_CAP = 12
_DEFAULT_SPAWN_CAP = 10
_DEFAULT_FIX_CAP = 2
_HTTP_OK = 200
_HTTP_UNAUTHORIZED = 401
_SOME_DT = datetime(2026, 4, 26, 12, 0, tzinfo=UTC)
_API_KEY = "test-api-key"


def test_settings_exposes_meta_agent_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """A freshly constructed ``Settings()`` exposes all six meta_agent_* defaults
    (SPEC §Constraints line 117) — safety defaults: enabled=False, dry_run=True.

    The local .env may override LLM_PROVIDER with the operator's headless
    sentinel (`claude_code_headless`) which is not in the project's Literal —
    explicitly set a valid value via monkeypatch so this test exercises only
    the meta_agent_* surface.
    """
    monkeypatch.setenv("FINALAYZE_LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("FINALAYZE_MODE", "debug")

    from config.settings import Settings

    s = Settings()
    assert s.meta_agent_enabled is False
    assert s.meta_agent_dry_run is True
    assert s.meta_agent_interval_minutes == _DEFAULT_INTERVAL_MIN
    assert s.meta_agent_max_telegram_alerts_per_day == _DEFAULT_TG_CAP
    assert s.meta_agent_max_spawns_per_day == _DEFAULT_SPAWN_CAP
    assert s.meta_agent_max_fix_spawns_per_day == _DEFAULT_FIX_CAP


# ── Status endpoint tests (Task 58-01-11) ──────────────────────────────────────


@pytest.fixture
def app(monkeypatch: pytest.MonkeyPatch) -> FastAPI:
    """Build a FastAPI app with only the meta_agent router mounted, plus
    api_key_auth wired against ``_API_KEY``."""
    monkeypatch.setenv("FINALAYZE_API_KEY", _API_KEY)
    monkeypatch.setenv("FINALAYZE_LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("FINALAYZE_MODE", "debug")

    from config.settings import get_settings

    get_settings.cache_clear()

    from finalayze.api.v1.meta_agent import router as meta_agent_router

    app = FastAPI()
    app.include_router(meta_agent_router, prefix="/api/v1")
    yield app
    get_settings.cache_clear()


def _wired_runner(*, last_run_ts: datetime | None = None) -> MagicMock:
    runner = MagicMock()
    runner._last_run_ts = last_run_ts
    runner.status_snapshot.return_value = {
        "enabled": False,
        "dry_run": True,
        "last_run_ts": last_run_ts,
        "scheduler_active": False,
        "inflight_spawns": {"investigate": 0, "fix": 0},
    }
    return runner


def test_status_endpoint_returns_five_field_envelope(app: FastAPI) -> None:
    """SPEC §AC #16: GET /api/v1/meta-agent/status returns the five-field
    envelope when wired with a runner."""
    from finalayze.api.v1.meta_agent import set_runner

    set_runner(_wired_runner())
    client = TestClient(app)
    resp = client.get("/api/v1/meta-agent/status", headers={"X-API-Key": _API_KEY})
    assert resp.status_code == _HTTP_OK
    body = resp.json()
    assert body == {
        "enabled": False,
        "dry_run": True,
        "last_run_ts": None,
        "scheduler_active": False,
        "inflight_spawns": {"investigate": 0, "fix": 0},
    }
    set_runner(None)


def test_status_endpoint_requires_api_key(app: FastAPI) -> None:
    """No X-API-Key → 401 (api_key_auth dependency)."""
    from finalayze.api.v1.meta_agent import set_runner

    set_runner(_wired_runner())
    client = TestClient(app)
    resp = client.get("/api/v1/meta-agent/status")
    assert resp.status_code == _HTTP_UNAUTHORIZED
    set_runner(None)


def test_status_endpoint_reflects_last_run_ts(app: FastAPI) -> None:
    """When runner._last_run_ts is set, the status endpoint surfaces the ISO ts."""
    from finalayze.api.v1.meta_agent import set_runner

    set_runner(_wired_runner(last_run_ts=_SOME_DT))
    client = TestClient(app)
    resp = client.get("/api/v1/meta-agent/status", headers={"X-API-Key": _API_KEY})
    assert resp.status_code == _HTTP_OK
    body = resp.json()
    # Pydantic serialises tz-aware datetime as ISO with +00:00.
    assert body["last_run_ts"] is not None
    assert body["last_run_ts"].startswith("2026-04-26T12:00:00")
    set_runner(None)
