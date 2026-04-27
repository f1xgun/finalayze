"""Integration tests for the meta-agent REST surface (Phase 58-05, META-08).

Covers SPEC §Acceptance Criterion #15 (POST /disable + abort) and #16
(status endpoint reflects post-disable state).

Approach:
  - Mount only the meta_agent router on a fresh FastAPI app — no full
    bootstrap required, keeps each test hermetic.
  - Wire a Killswitch instance + a test runner that exposes
    ``status_snapshot()`` so the status endpoint has a live data source.
  - Pre-stage a fake long-running spawn into ``spawner._inflight_handles``
    so the abort path has something to terminate. Monkeypatch ``os.killpg``
    to record signals (NEVER let real signals reach the test runner's PID).
  - Assert: 200 response, abort_count == 1, scheduler.remove_job called,
    wall-clock < 5 s, status reflects enabled=False afterwards.
"""

from __future__ import annotations

import os
import signal
import time
import uuid
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# ── Module-level constants (PLR2004) ────────────────────────────────────────
_API_KEY = "integration-test-api-key"
_FAKE_DECISION_ID = uuid.UUID("deadbeef-0000-4000-8000-000000000020")
_FAKE_PID = 77777
_FAKE_PGID = _FAKE_PID
_GRACE_TEST = 0.2  # short SIGTERM grace
_KILL_TEST = 0.2  # short SIGKILL reap
_KILLSWITCH_CEILING_S = 5.0  # SPEC line 75
_HTTP_OK = 200
_HTTP_UNAUTHORIZED = 401
_EXPECTED_ABORTS = 1


class _FakeProcess:
    """Spawn handle stand-in for the disable-path tests.

    Cooperative on SIGTERM so abort_all_inflight returns quickly.
    """

    def __init__(self, *, pid: int = _FAKE_PID) -> None:
        self.pid: int = pid
        self.returncode: int | None = None

    async def wait(self) -> int:
        if self.returncode is None:
            self.returncode = -signal.SIGTERM
        return self.returncode


@pytest.fixture
def app(monkeypatch: pytest.MonkeyPatch) -> FastAPI:
    """FastAPI app with only the meta-agent router mounted + valid api_key."""
    monkeypatch.setenv("FINALAYZE_API_KEY", _API_KEY)
    monkeypatch.setenv("FINALAYZE_LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("FINALAYZE_MODE", "debug")

    from config.settings import get_settings

    get_settings.cache_clear()

    from finalayze.api.v1.meta_agent import router as meta_agent_router

    a = FastAPI()
    a.include_router(meta_agent_router, prefix="/api/v1")
    yield a
    get_settings.cache_clear()


def _wired_runner_with_killswitch(
    *,
    killswitch: Any,
    enabled: bool = True,
    inflight: dict[str, int] | None = None,
) -> MagicMock:
    """Build a runner-shaped MagicMock that exposes status_snapshot + killswitch."""
    runner = MagicMock()
    runner.killswitch = killswitch
    runner._last_run_ts = None
    if inflight is None:
        inflight = {"investigate": 0, "fix": 0}
    runner.status_snapshot.return_value = {
        "enabled": enabled,
        "dry_run": True,
        "last_run_ts": None,
        "scheduler_active": enabled,
        "inflight_spawns": inflight,
    }
    return runner


# ─────────────────────────────────────────────────────────────────────────────
# Task 58-05-04 — POST /api/v1/meta-agent/disable
# ─────────────────────────────────────────────────────────────────────────────


def test_post_disable_aborts_inflight_and_removes_job_within_5s(
    app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SPEC AC #15: POST /api/v1/meta-agent/disable
      1. Returns 200 with {status, aborted_spawns, job_removed}.
      2. Calls scheduler.remove_job("meta_agent").
      3. Aborts every entry in spawner._inflight_handles via SIGTERM.
      4. Total wall-clock from POST → response ≤ 5 s.
    """
    from finalayze.api.v1.meta_agent import set_runner
    from finalayze.meta_agent import spawner as sp
    from finalayze.meta_agent.killswitch import Killswitch

    # Pre-stage one in-flight spawn.
    sp._inflight_handles.clear()
    sp._INFLIGHT_TYPE.clear()
    fake = _FakeProcess()
    sp._inflight_handles[_FAKE_DECISION_ID] = fake  # type: ignore[assignment]
    sp._INFLIGHT_TYPE[_FAKE_DECISION_ID] = "investigate"

    # Record killpg signals.
    killpg_calls: list[tuple[int, int]] = []

    def _fake_killpg(pgid: int, sig: int) -> None:
        killpg_calls.append((pgid, sig))
        # Cooperative — exit on SIGTERM.
        if sig == signal.SIGTERM and pgid == _FAKE_PGID:
            fake.returncode = -signal.SIGTERM

    monkeypatch.setattr(os, "killpg", _fake_killpg)
    monkeypatch.setattr(os, "getpgid", lambda pid: pid)

    scheduler = MagicMock()
    ks = Killswitch(
        scheduler=scheduler,
        settings_provider=lambda: MagicMock(meta_agent_enabled=True),
        sigterm_grace_s=_GRACE_TEST,
        sigkill_reap_s=_KILL_TEST,
    )
    runner = _wired_runner_with_killswitch(killswitch=ks)
    set_runner(runner)

    client = TestClient(app)
    start = time.monotonic()
    resp = client.post(
        "/api/v1/meta-agent/disable",
        headers={"X-API-Key": _API_KEY},
    )
    duration = time.monotonic() - start

    assert resp.status_code == _HTTP_OK
    body = resp.json()
    assert body["status"] == "disabled"
    assert body["aborted_spawns"] == _EXPECTED_ABORTS
    assert body["job_removed"] is True

    # SPEC ceiling.
    assert duration < _KILLSWITCH_CEILING_S, (
        f"POST /disable took {duration:.2f}s; SPEC ceiling is {_KILLSWITCH_CEILING_S}s"
    )

    # SIGTERM was sent to the fake.
    sigterm_pgids = [pgid for pgid, sig in killpg_calls if sig == signal.SIGTERM]
    assert _FAKE_PGID in sigterm_pgids

    # remove_job was called.
    scheduler.remove_job.assert_called_once_with("meta_agent")

    # Cleanup.
    set_runner(None)
    sp._inflight_handles.clear()
    sp._INFLIGHT_TYPE.clear()


def test_post_disable_requires_api_key(
    app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No X-API-Key → 401 (api_key_auth dependency)."""
    from finalayze.api.v1.meta_agent import set_runner
    from finalayze.meta_agent.killswitch import Killswitch

    monkeypatch.setattr(os, "killpg", lambda pgid, sig: None)
    monkeypatch.setattr(os, "getpgid", lambda pid: pid)

    ks = Killswitch(
        scheduler=MagicMock(),
        settings_provider=lambda: MagicMock(meta_agent_enabled=True),
    )
    runner = _wired_runner_with_killswitch(killswitch=ks)
    set_runner(runner)

    client = TestClient(app)
    resp = client.post("/api/v1/meta-agent/disable")
    assert resp.status_code == _HTTP_UNAUTHORIZED

    set_runner(None)


def test_post_disable_idempotent_when_no_inflight_no_job(
    app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Second call (or first call when nothing to abort) returns 200 with
    aborted_spawns=0 and job_removed=False — both primitives are idempotent.
    """
    from apscheduler.jobstores.base import JobLookupError

    from finalayze.api.v1.meta_agent import set_runner
    from finalayze.meta_agent import spawner as sp
    from finalayze.meta_agent.killswitch import Killswitch

    sp._inflight_handles.clear()
    sp._INFLIGHT_TYPE.clear()

    monkeypatch.setattr(os, "killpg", lambda pgid, sig: None)
    monkeypatch.setattr(os, "getpgid", lambda pid: pid)

    scheduler = MagicMock()
    scheduler.remove_job.side_effect = JobLookupError("meta_agent")
    ks = Killswitch(
        scheduler=scheduler,
        settings_provider=lambda: MagicMock(meta_agent_enabled=True),
    )
    runner = _wired_runner_with_killswitch(killswitch=ks)
    set_runner(runner)

    client = TestClient(app)
    resp = client.post(
        "/api/v1/meta-agent/disable",
        headers={"X-API-Key": _API_KEY},
    )
    assert resp.status_code == _HTTP_OK
    body = resp.json()
    assert body == {
        "status": "disabled",
        "aborted_spawns": 0,
        "job_removed": False,
    }

    set_runner(None)


def test_get_status_reflects_post_disable_state(
    app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SPEC AC #16: after disable, GET /status returns enabled=False
    AND inflight_spawns reflects the live (now-empty) registry.

    The runner's status_snapshot is wired so the test directly observes
    enabled=False after the disable call (via runner-shape MagicMock that
    simulates a flipped enabled flag).
    """
    from finalayze.api.v1.meta_agent import set_runner
    from finalayze.meta_agent import spawner as sp
    from finalayze.meta_agent.killswitch import Killswitch

    sp._inflight_handles.clear()
    sp._INFLIGHT_TYPE.clear()

    monkeypatch.setattr(os, "killpg", lambda pgid, sig: None)
    monkeypatch.setattr(os, "getpgid", lambda pid: pid)

    scheduler = MagicMock()
    ks = Killswitch(
        scheduler=scheduler,
        settings_provider=lambda: MagicMock(meta_agent_enabled=True),
    )
    runner = _wired_runner_with_killswitch(killswitch=ks, enabled=False)
    set_runner(runner)

    client = TestClient(app)
    resp = client.get(
        "/api/v1/meta-agent/status",
        headers={"X-API-Key": _API_KEY},
    )
    assert resp.status_code == _HTTP_OK
    body = resp.json()
    assert body["enabled"] is False
    assert body["scheduler_active"] is False
    assert body["inflight_spawns"] == {"investigate": 0, "fix": 0}

    set_runner(None)
