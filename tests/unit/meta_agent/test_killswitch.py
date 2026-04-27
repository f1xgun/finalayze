"""Tests for meta_agent.killswitch — Phase 58-05 (META-08, SPEC AC #15).

The killswitch contract:
  - Two trigger paths: REST POST /disable (Plan 58-05 Task 04) and the
    env-var poller (Task 03). Both call the same primitives.
  - ``abort_all_inflight()`` iterates ``meta_agent.spawner._inflight_handles``
    and signals each subprocess via the SIGTERM → 3 s grace → SIGKILL
    sequence already implemented in ``spawner._terminate_process_group``
    (RESEARCH §3.2). Total wall-clock budget: ≤ 5 s (SPEC line 75).
  - ``remove_job()`` calls ``scheduler.remove_job("meta_agent")``;
    idempotent on ``JobLookupError``.
  - ``start()/stop()`` manage the env-var poller task (Task 03).

These tests monkeypatch ``os.killpg`` so that no real signals reach any
PIDs (a fake PID could collide with the test runner's PID and crash the
suite).
"""

from __future__ import annotations

import asyncio
import os
import signal
import time
import uuid
from typing import Any
from unittest.mock import MagicMock

import pytest

# ── Module-level constants (PLR2004 — no magic numbers in tests) ───────────
_FAKE_DECISION_ID = uuid.UUID("deadbeef-0000-4000-8000-000000000010")
_FAKE_DECISION_ID_2 = uuid.UUID("deadbeef-0000-4000-8000-000000000011")
_FAKE_PID = 88888
_FAKE_PID_2 = 88889
_FAKE_PGID = _FAKE_PID
_FAKE_PGID_2 = _FAKE_PID_2
_KILLSWITCH_CEILING_S = 5.0  # SPEC line 75 — wall-clock ceiling for abort
_GRACE_TEST = 0.2  # short grace window for tests
_KILL_TEST = 0.2  # short SIGKILL reap window for tests


class _FakeProcess:
    """Stand-in for ``asyncio.subprocess.Process`` for killswitch tests.

    ``ignore_sigterm=True`` makes ``wait()`` block until ``returncode`` is
    set externally (e.g. by a fake SIGKILL handler). Mirrors the
    ``_FakeProcess`` from ``test_spawner.py`` but trimmed to what the
    killswitch needs.
    """

    def __init__(
        self,
        *,
        pid: int = _FAKE_PID,
        ignore_sigterm: bool = False,
    ) -> None:
        self.pid: int = pid
        self.returncode: int | None = None
        self._ignore_sigterm = ignore_sigterm

    async def wait(self) -> int:
        if self._ignore_sigterm:
            while self.returncode is None:
                await asyncio.sleep(0.01)
            return self.returncode
        if self.returncode is None:
            self.returncode = 0
        return self.returncode


# ─────────────────────────────────────────────────────────────────────────────
# Task 58-05-01 — Killswitch.abort_all_inflight() primitive
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_abort_all_inflight_terminates_via_killpg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SPEC AC #15: ``abort_all_inflight()`` iterates the in-flight registry
    and signals every entry via ``os.killpg(pgid, SIGTERM)``. With a
    fake process whose ``wait()`` exits immediately on SIGTERM, no
    SIGKILL is sent. With ``ignore_sigterm=True``, the spawner falls
    through to ``os.killpg(pgid, SIGKILL)``. Total wall-clock duration
    of the call ≤ 5 s.
    """
    from finalayze.meta_agent import spawner as sp
    from finalayze.meta_agent.killswitch import Killswitch

    # Pre-condition: registry empty.
    sp._inflight_handles.clear()

    # Two fake spawns: one obeys SIGTERM (cooperative), one ignores.
    cooperative = _FakeProcess(pid=_FAKE_PID, ignore_sigterm=False)
    stubborn = _FakeProcess(pid=_FAKE_PID_2, ignore_sigterm=True)
    sp._inflight_handles[_FAKE_DECISION_ID] = cooperative  # type: ignore[assignment]
    sp._inflight_handles[_FAKE_DECISION_ID_2] = stubborn  # type: ignore[assignment]

    # Record killpg calls. The handler simulates SIGKILL by setting
    # returncode so the stubborn fake's wait() can return.
    killpg_calls: list[tuple[int, int]] = []

    def _fake_killpg(pgid: int, sig: int) -> None:
        killpg_calls.append((pgid, sig))
        if sig == signal.SIGKILL:
            if pgid == _FAKE_PGID:
                cooperative.returncode = -signal.SIGKILL
            if pgid == _FAKE_PGID_2:
                stubborn.returncode = -signal.SIGKILL
        elif sig == signal.SIGTERM:
            # Cooperative process honors SIGTERM by exiting.
            if pgid == _FAKE_PGID:
                cooperative.returncode = -signal.SIGTERM

    def _fake_getpgid(pid: int) -> int:
        return pid  # start_new_session=True → pgid == pid

    monkeypatch.setattr(os, "killpg", _fake_killpg)
    monkeypatch.setattr(os, "getpgid", _fake_getpgid)

    scheduler = MagicMock()
    settings_provider = MagicMock(return_value=MagicMock(meta_agent_enabled=True))

    ks = Killswitch(
        scheduler=scheduler,
        settings_provider=settings_provider,
        sigterm_grace_s=_GRACE_TEST,
        sigkill_reap_s=_KILL_TEST,
    )

    start = time.monotonic()
    count = await ks.abort_all_inflight()
    duration = time.monotonic() - start

    # Aborted both registered spawns.
    assert count == 2

    # Wall-clock under SPEC ceiling.
    assert duration < _KILLSWITCH_CEILING_S, (
        f"abort_all_inflight took {duration:.2f}s; SPEC ceiling is "
        f"{_KILLSWITCH_CEILING_S}s"
    )

    # Both SIGTERM calls happened.
    sigterm_pgids = [pgid for pgid, sig in killpg_calls if sig == signal.SIGTERM]
    assert _FAKE_PGID in sigterm_pgids
    assert _FAKE_PGID_2 in sigterm_pgids

    # SIGKILL ONLY for the stubborn fake (cooperative obeyed SIGTERM).
    sigkill_pgids = [pgid for pgid, sig in killpg_calls if sig == signal.SIGKILL]
    assert _FAKE_PGID_2 in sigkill_pgids
    assert _FAKE_PGID not in sigkill_pgids

    # Cleanup.
    sp._inflight_handles.clear()


@pytest.mark.asyncio
async def test_abort_all_inflight_returns_zero_when_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No in-flight spawns → returns 0 with no killpg calls."""
    from finalayze.meta_agent import spawner as sp
    from finalayze.meta_agent.killswitch import Killswitch

    sp._inflight_handles.clear()

    killpg_calls: list[tuple[int, int]] = []
    monkeypatch.setattr(
        os,
        "killpg",
        lambda pgid, sig: killpg_calls.append((pgid, sig)),
    )

    ks = Killswitch(
        scheduler=MagicMock(),
        settings_provider=MagicMock(),
        sigterm_grace_s=_GRACE_TEST,
        sigkill_reap_s=_KILL_TEST,
    )
    count = await ks.abort_all_inflight()
    assert count == 0
    assert killpg_calls == []


# ─────────────────────────────────────────────────────────────────────────────
# Task 58-05-02 — Killswitch.remove_job() calls scheduler.remove_job("meta_agent")
# ─────────────────────────────────────────────────────────────────────────────


def test_remove_job_calls_scheduler() -> None:
    """SPEC AC #15: ``remove_job()`` calls
    ``scheduler.remove_job("meta_agent")`` exactly once and returns True
    on success.
    """
    from finalayze.meta_agent.killswitch import Killswitch

    scheduler = MagicMock()
    ks = Killswitch(
        scheduler=scheduler,
        settings_provider=MagicMock(),
    )
    result = ks.remove_job()
    assert result is True
    scheduler.remove_job.assert_called_once_with("meta_agent")


def test_remove_job_idempotent_on_missing_job() -> None:
    """SPEC AC #15: a second ``remove_job()`` call (or any call when the
    job is not registered) returns False without raising — the killswitch
    must be safe to invoke twice (REST + env-var paths can race).
    """
    from apscheduler.jobstores.base import JobLookupError

    from finalayze.meta_agent.killswitch import Killswitch

    scheduler = MagicMock()
    scheduler.remove_job.side_effect = JobLookupError("meta_agent")
    ks = Killswitch(
        scheduler=scheduler,
        settings_provider=MagicMock(),
    )
    result = ks.remove_job()
    assert result is False
    scheduler.remove_job.assert_called_once_with("meta_agent")


# ─────────────────────────────────────────────────────────────────────────────
# Task 58-05-03 — Killswitch._watch_env() 1 s poller (env-var → abort within 5 s)
# ─────────────────────────────────────────────────────────────────────────────


_POLL_WAIT_S = 4.0  # seconds — give the 1 s poller time to detect the flip + abort


@pytest.mark.asyncio
async def test_env_var_flip_aborts_inflight_within_5s(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SPEC AC #15 (env-var path): the 1 s poller detects
    ``meta_agent_enabled=False`` within ≤ 1 s, then triggers
    ``abort_all_inflight()`` + ``remove_job()``. Total wall-clock from
    flip to abort completion ≤ 5 s.

    Test scaffolding:
      - Pre-stage one fake INFLIGHT spawn whose ``wait()`` returns on
        SIGTERM (cooperative).
      - Stub ``config.settings.get_settings`` to return successive
        ``meta_agent_enabled`` values: True, True, False, False, ...
        The poller starts → reads True → sleeps 1 s → reads True → sleeps
        1 s → reads False → fires abort path. The flip happens at the
        third invocation (~2 s after start), so total wall-clock ≤
        2 s + grace + reap ≈ < 5 s.
      - Cleanup: ``await ks.stop()`` cancels the poller cleanly so the
        test loop can exit.
    """
    from config import settings as cfg_settings_module
    from finalayze.meta_agent import spawner as sp
    from finalayze.meta_agent.killswitch import Killswitch

    # Pre-condition: registry has one fake spawn.
    sp._inflight_handles.clear()
    cooperative = _FakeProcess(pid=_FAKE_PID, ignore_sigterm=False)
    sp._inflight_handles[_FAKE_DECISION_ID] = cooperative  # type: ignore[assignment]

    # killpg recorder.
    killpg_calls: list[tuple[int, int]] = []

    def _fake_killpg(pgid: int, sig: int) -> None:
        killpg_calls.append((pgid, sig))
        if sig == signal.SIGTERM and pgid == _FAKE_PGID:
            cooperative.returncode = -signal.SIGTERM

    def _fake_getpgid(pid: int) -> int:
        return pid

    monkeypatch.setattr(os, "killpg", _fake_killpg)
    monkeypatch.setattr(os, "getpgid", _fake_getpgid)

    # Stub get_settings: returns a Settings whose meta_agent_enabled flips
    # to False on the third invocation.
    sequence_lock = asyncio.Lock()
    state = {"calls": 0}

    def _fake_get_settings() -> Any:
        state["calls"] += 1
        # First call returns True; second onward return False.
        enabled = state["calls"] <= 1
        s = MagicMock()
        s.meta_agent_enabled = enabled
        return s

    # cache_clear is required by the Killswitch poller to invalidate
    # any lru_cache wrap on get_settings. Patch both the function and
    # its cache_clear attribute.
    _fake_get_settings.cache_clear = lambda: None  # type: ignore[attr-defined]
    monkeypatch.setattr(cfg_settings_module, "get_settings", _fake_get_settings)

    # First reading via settings_provider returns True; subsequent reads
    # via the get_settings poll path will flip to False.
    initial_settings = MagicMock()
    initial_settings.meta_agent_enabled = True

    scheduler = MagicMock()
    ks = Killswitch(
        scheduler=scheduler,
        settings_provider=lambda: initial_settings,
        sigterm_grace_s=_GRACE_TEST,
        sigkill_reap_s=_KILL_TEST,
    )

    start = time.monotonic()
    await ks.start()
    assert ks._poller_task is not None  # poller running

    # Wait long enough for the poller to detect the flip (1 s × 2 sleeps
    # + abort budget). Cooperative SIGTERM exits immediately.
    async with sequence_lock:
        await asyncio.sleep(_POLL_WAIT_S)
    duration = time.monotonic() - start

    # Stop the poller cleanly.
    await ks.stop()

    # SPEC ceiling.
    assert duration < _KILLSWITCH_CEILING_S + _POLL_WAIT_S, (
        f"poller took {duration:.2f}s; should be under "
        f"{_KILLSWITCH_CEILING_S + _POLL_WAIT_S}s"
    )

    # Abort triggered: SIGTERM was sent to the fake.
    sigterm_pgids = [pgid for pgid, sig in killpg_calls if sig == signal.SIGTERM]
    assert _FAKE_PGID in sigterm_pgids, (
        f"poller did not abort the in-flight spawn; killpg_calls={killpg_calls}"
    )

    # remove_job called.
    scheduler.remove_job.assert_called_with("meta_agent")

    # Cleanup.
    sp._inflight_handles.clear()


@pytest.mark.asyncio
async def test_stop_cancels_poller_when_started() -> None:
    """``stop()`` cancels the poller task cleanly even when no flip has
    occurred. After stop, the poller task is in a finished state.
    """
    from finalayze.meta_agent.killswitch import Killswitch

    initial_settings = MagicMock()
    initial_settings.meta_agent_enabled = True
    ks = Killswitch(
        scheduler=MagicMock(),
        settings_provider=lambda: initial_settings,
    )
    await ks.start()
    assert ks._poller_task is not None
    await ks.stop()
    assert ks._poller_task is None or ks._poller_task.done()
