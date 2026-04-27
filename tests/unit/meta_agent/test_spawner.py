"""Tests for meta_agent.spawner — read-only investigation subprocess (Phase 58-03)
and FIX-spawn pipeline (Phase 58-04).

Covers SPEC §Acceptance Criteria #10 + #11 + #13:
  - Exception classes (Task 58-03-01)
  - spawn_readonly happy path with monkeypatched subprocess (Task 58-03-04)
  - 300s timeout → SIGTERM → SIGKILL (Task 58-03-05)
  - Concurrent investigate → already_inflight (Task 58-03-06)
  - spawn_fix argv uses Edit + worktree cwd (Task 58-04-04)
  - Concurrent fix → already_inflight via _FIX_LOCK (Task 58-04-04)

The CLI (`claude`) need NOT be on $PATH — tests monkeypatch
``asyncio.create_subprocess_exec`` so the spawner is exercised hermetically.
"""

from __future__ import annotations

import asyncio
import os
import signal
import time
import uuid
from typing import Any

import pytest

from finalayze.core.exceptions import FinalayzeError

# ── Module-level constants (PLR2004) ────────────────────────────────────────
_FAKE_DECISION_ID = uuid.UUID("deadbeef-0000-4000-8000-000000000001")
_FAKE_DECISION_ID_2 = uuid.UUID("deadbeef-0000-4000-8000-000000000002")
_FAKE_PID = 99999
_FAKE_PGID = _FAKE_PID  # start_new_session=True → pgid == pid
_TIMEOUT_SHORT = 2  # seconds, for timeout test
_LONG_SLEEP = 600  # seconds, fake "infinite" subprocess
_GRACE_TEST = 0.2  # seconds — short SIGTERM grace for test
_KILL_TEST = 0.2  # seconds — short SIGKILL reap for test
_EXIT_OK = 0
_EXIT_TIMEOUT = -1


# ── Fake subprocess infrastructure ──────────────────────────────────────────


class _FakeProcess:
    """Stand-in for ``asyncio.subprocess.Process`` — records what the spawner does.

    ``ignore_sigterm=True`` makes ``wait()`` block forever (or until
    ``returncode`` is set externally, e.g. by a fake SIGKILL handler).
    Useful for testing SIGTERM→SIGKILL escalation: the fake represents a
    process that doesn't honor SIGTERM, forcing the spawner into the
    SIGKILL phase.
    """

    def __init__(
        self,
        *,
        stdout: bytes = b"",
        stderr: bytes = b"",
        exit_code: int = 0,
        sleep_before_exit: float = 0.0,
        ignore_sigterm: bool = False,
    ) -> None:
        self.pid: int = _FAKE_PID
        self.returncode: int | None = None
        self._stdout = stdout
        self._stderr = stderr
        self._exit_code = exit_code
        self._sleep = sleep_before_exit
        self._ignore_sigterm = ignore_sigterm
        self.terminate_called = False
        self.kill_called = False

    async def communicate(self) -> tuple[bytes, bytes]:
        if self._sleep > 0:
            await asyncio.sleep(self._sleep)
        self.returncode = self._exit_code
        return self._stdout, self._stderr

    async def wait(self) -> int:
        # Spin until returncode is set — by either communicate() (happy path)
        # or by an external SIGKILL handler that mutates returncode.
        if self._ignore_sigterm:
            while self.returncode is None:
                await asyncio.sleep(0.01)
            return self.returncode
        if self.returncode is None:
            self.returncode = self._exit_code
        return self.returncode

    def terminate(self) -> None:
        self.terminate_called = True

    def kill(self) -> None:
        self.kill_called = True
        self.returncode = -signal.SIGKILL

    # Streams — used only when the spawner drains after termination.
    @property
    def stdout(self) -> Any:
        return _FakeStream(self._stdout)

    @property
    def stderr(self) -> Any:
        return _FakeStream(self._stderr)


class _FakeStream:
    def __init__(self, data: bytes) -> None:
        self._data = data

    async def read(self) -> bytes:
        return self._data


# ─────────────────────────────────────────────────────────────────────────────
# Task 58-03-01: MetaAgentDeniedPathError + MetaAgentSpawnCapExceededError
# ─────────────────────────────────────────────────────────────────────────────


def test_exceptions_inherit_finalayze_error_and_end_in_error() -> None:
    """SPEC AC #10 + #11 foundation: both meta-agent spawn exceptions
    inherit from FinalayzeError, end in 'Error' (N818), and raise with
    a context message.
    """
    from finalayze.meta_agent.exceptions import (
        MetaAgentDeniedPathError,
        MetaAgentSpawnCapExceededError,
    )

    # Subclass relationship.
    assert issubclass(MetaAgentDeniedPathError, FinalayzeError)
    assert issubclass(MetaAgentSpawnCapExceededError, FinalayzeError)

    # N818 — class names end in "Error".
    assert MetaAgentDeniedPathError.__name__.endswith("Error")
    assert MetaAgentSpawnCapExceededError.__name__.endswith("Error")

    # Raise + carry message.
    msg_denied = "denied path: src/finalayze/risk/manager.py"
    with pytest.raises(MetaAgentDeniedPathError) as exc_info:
        raise MetaAgentDeniedPathError(msg_denied)
    assert str(exc_info.value) == msg_denied

    msg_cap = "spawn cap exceeded for INVESTIGATE"
    with pytest.raises(MetaAgentSpawnCapExceededError) as exc_info:
        raise MetaAgentSpawnCapExceededError(msg_cap)
    assert str(exc_info.value) == msg_cap


# ─────────────────────────────────────────────────────────────────────────────
# Task 58-03-04: spawn_readonly happy path
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_spawn_readonly_happy_path_captures_stdout_and_exit_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SPEC AC #10: spawn_readonly invokes ``claude`` with the SPEC-locked
    flag set, captures stdout/stderr/exit_code into a SpawnOutcome,
    strips ANTHROPIC_API_KEY from env (subscription auth), and uses
    ``start_new_session=True`` so the process group is killable.
    """
    from finalayze.meta_agent.spawner import (
        SpawnOutcome,
        spawn_readonly,
    )

    captured: dict[str, Any] = {}

    fake_stdout = b'{"type":"assistant","content":"hello"}\n{"type":"result","is_error":false}\n'
    fake_proc = _FakeProcess(stdout=fake_stdout, stderr=b"", exit_code=_EXIT_OK)

    async def _fake_create(*args: Any, **kwargs: Any) -> _FakeProcess:
        captured["args"] = args
        captured["kwargs"] = kwargs
        return fake_proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_create)
    # Pre-stage an env var that should be stripped.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "should-be-stripped")

    outcome = await spawn_readonly("test prompt", decision_id=_FAKE_DECISION_ID)

    # Outcome shape.
    assert isinstance(outcome, SpawnOutcome)
    assert outcome.exit_code == _EXIT_OK
    assert outcome.timed_out is False
    assert outcome.killed_by_killswitch is False
    # stdout truncated to UTF-8 string; the JSON-newline content is preserved.
    assert "hello" in outcome.stdout
    assert "is_error" in outcome.stdout
    assert outcome.stderr == ""

    # Argv inspection — SPEC-locked flags.
    args = captured["args"]
    assert args[0] == "claude"
    assert args[1] == "-p"
    assert args[2] == "test prompt"
    # Required flags appear (order-independent membership check).
    assert "--allowedTools" in args
    invest_flag_idx = args.index("--allowedTools")
    assert args[invest_flag_idx + 1] == "Read,Grep,Bash"
    assert "--max-turns" in args
    mt_idx = args.index("--max-turns")
    assert args[mt_idx + 1] == "20"

    # Subprocess control kwargs.
    kwargs = captured["kwargs"]
    assert kwargs.get("start_new_session") is True
    # Env stripped of ANTHROPIC_API_KEY.
    env = kwargs.get("env")
    assert env is not None, "spawner must explicitly pass env (so it can strip key)"
    assert "ANTHROPIC_API_KEY" not in env
    # stdin DEVNULL so child can't block reading a parent stdin.
    assert kwargs.get("stdin") == asyncio.subprocess.DEVNULL


@pytest.mark.asyncio
async def test_spawn_readonly_strips_inflight_handle_after_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SPEC AC #10 / D-07: the in-flight handle registry is populated
    while the spawn runs and cleaned in the ``finally`` block. After a
    successful spawn, the registry is empty so the next investigate can
    take the lock.
    """
    from finalayze.meta_agent import spawner as sp

    fake_proc = _FakeProcess(stdout=b"ok\n", exit_code=_EXIT_OK)

    async def _fake_create(*_args: Any, **_kwargs: Any) -> _FakeProcess:
        return fake_proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_create)

    await sp.spawn_readonly("hello", decision_id=_FAKE_DECISION_ID)

    assert _FAKE_DECISION_ID not in sp._inflight_handles, (
        "spawn_readonly must clear _inflight_handles[decision_id] in finally"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Task 58-03-05: 300s timeout terminates process group (SIGTERM → SIGKILL)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_spawn_readonly_timeout_terminates_process_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SPEC AC #10: when ``proc.communicate()`` exceeds ``timeout_s``,
    spawn_readonly:
      1. Calls ``os.killpg(pgid, SIGTERM)``.
      2. Waits up to ``sigterm_grace_s`` for the proc to exit.
      3. On second timeout, calls ``os.killpg(pgid, SIGKILL)``.
      4. Returns SpawnOutcome(timed_out=True, killed_by_killswitch=False).
      5. Emits structlog event ``meta_agent_spawn_timeout``.

    Test parameters use shortened grace + kill windows so the wall-clock
    bound stays under a second.
    """
    import structlog

    from finalayze.meta_agent.spawner import spawn_readonly

    # FakeProcess never finishes communicate() within the test timeout, AND
    # ignores SIGTERM (forcing the spawner into the SIGKILL escalation path).
    fake_proc = _FakeProcess(
        stdout=b"partial output\n",
        stderr=b"",
        sleep_before_exit=_LONG_SLEEP,
        ignore_sigterm=True,
    )

    async def _fake_create(*_args: Any, **_kwargs: Any) -> _FakeProcess:
        return fake_proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_create)

    # Record killpg signals + arguments.
    killpg_calls: list[tuple[int, int]] = []

    def _fake_killpg(pgid: int, sig: int) -> None:
        killpg_calls.append((pgid, sig))
        # Simulate SIGKILL effect — set returncode so wait() returns.
        if sig == signal.SIGKILL:
            fake_proc.returncode = -signal.SIGKILL

    def _fake_getpgid(_pid: int) -> int:
        return _FAKE_PGID

    monkeypatch.setattr(os, "killpg", _fake_killpg)
    monkeypatch.setattr(os, "getpgid", _fake_getpgid)

    start = time.monotonic()
    with structlog.testing.capture_logs() as logs:
        outcome = await spawn_readonly(
            "blocking prompt",
            decision_id=_FAKE_DECISION_ID,
            timeout_s=_TIMEOUT_SHORT,
            sigterm_grace_s=_GRACE_TEST,
            sigkill_reap_s=_KILL_TEST,
        )
    elapsed = time.monotonic() - start

    # Wall-clock: timeout (~2s) + grace (0.2s) + kill (0.2s) ≈ 2.4s.
    # Allow generous headroom for slow CI machines.
    max_wall_s = _TIMEOUT_SHORT + _GRACE_TEST + _KILL_TEST + 1.0
    assert elapsed < max_wall_s, f"timeout path took {elapsed:.2f}s; expected < {max_wall_s:.2f}s"

    # Outcome shape.
    assert outcome.timed_out is True, f"expected timed_out=True, got {outcome!r}"
    assert outcome.killed_by_killswitch is False
    # exit_code falls back to -SIGKILL (set by our fake) or -1 if proc never reaped.

    # SIGTERM then SIGKILL on the pgid.
    sig_seq = [sig for (_pgid, sig) in killpg_calls]
    assert signal.SIGTERM in sig_seq, f"expected SIGTERM call, got {killpg_calls!r}"
    assert signal.SIGKILL in sig_seq, f"expected SIGKILL call, got {killpg_calls!r}"
    # Order: SIGTERM first, then SIGKILL.
    sigterm_idx = sig_seq.index(signal.SIGTERM)
    sigkill_idx = sig_seq.index(signal.SIGKILL)
    assert sigterm_idx < sigkill_idx, f"SIGTERM must precede SIGKILL; sequence={sig_seq!r}"
    # All calls used the same pgid (process-group control).
    pgids = {pgid for (pgid, _sig) in killpg_calls}
    assert pgids == {_FAKE_PGID}, f"expected single pgid {_FAKE_PGID}, got {pgids}"

    # Structlog event emitted.
    timeout_events = [log for log in logs if log.get("event") == "meta_agent_spawn_timeout"]
    assert len(timeout_events) == 1, (
        f"expected 1 meta_agent_spawn_timeout event, got {timeout_events!r}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Task 58-03-06: _INVESTIGATE_LOCK — concurrent attempt → 'already_inflight'
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_concurrent_investigate_spawns_rejected_with_already_inflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SPEC §Boundaries (line 105): at most one in-flight investigate spawn
    per spawn-type. The second concurrent caller observes
    ``_INVESTIGATE_LOCK.locked() is True`` WITHOUT taking the lock and
    returns SpawnOutcome(stderr='already_inflight'). Structlog event
    ``meta_agent_spawn_already_inflight`` is emitted by the rejected caller.

    The first spawn must still complete normally (lock released cleanly).
    """
    import structlog

    from finalayze.meta_agent import spawner as sp
    from finalayze.meta_agent.spawner import spawn_readonly

    # First spawn: slow but eventually exits.
    slow_proc = _FakeProcess(
        stdout=b"slow result\n",
        stderr=b"",
        sleep_before_exit=0.5,  # Short enough for the test, long enough for race.
    )
    fast_proc = _FakeProcess(stdout=b"fast result\n", exit_code=_EXIT_OK)

    procs = iter([slow_proc, fast_proc])

    async def _fake_create(*_args: Any, **_kwargs: Any) -> _FakeProcess:
        return next(procs)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_create)

    # Verify lock is initially free.
    assert sp._INVESTIGATE_LOCK.locked() is False, "lock must start unlocked"

    # Schedule the first spawn as a background task; it holds the lock for
    # ~0.5s. Then immediately attempt a second spawn — it must observe
    # the lock as held and return 'already_inflight'.
    first_task = asyncio.create_task(
        spawn_readonly("first prompt", decision_id=_FAKE_DECISION_ID),
    )
    # Yield briefly so the first task acquires the lock.
    await asyncio.sleep(0.05)

    # Pre-condition: the lock IS held by the first task.
    assert sp._INVESTIGATE_LOCK.locked() is True, "first spawn must have acquired the lock by now"

    with structlog.testing.capture_logs() as logs:
        second_outcome = await spawn_readonly(
            "second prompt",
            decision_id=_FAKE_DECISION_ID_2,
        )

    # Second outcome shape — rejected without taking the lock.
    assert second_outcome.exit_code == _EXIT_TIMEOUT, (
        f"expected synthetic exit_code={_EXIT_TIMEOUT}, got {second_outcome!r}"
    )
    assert second_outcome.stderr == "already_inflight"
    assert second_outcome.stdout == ""
    assert second_outcome.timed_out is False
    assert second_outcome.killed_by_killswitch is False

    # Structlog event from the rejected caller.
    rejected_events = [
        log for log in logs if log.get("event") == "meta_agent_spawn_already_inflight"
    ]
    assert len(rejected_events) == 1, f"expected 1 already_inflight event, got {rejected_events!r}"
    assert rejected_events[0].get("decision_id_key") == str(_FAKE_DECISION_ID_2)

    # The first spawn must still complete cleanly and release the lock.
    first_outcome = await first_task
    assert first_outcome.exit_code == _EXIT_OK, (
        f"first spawn should have exited 0, got {first_outcome!r}"
    )
    assert sp._INVESTIGATE_LOCK.locked() is False, (
        "lock must be released after first spawn completes"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Task 58-04-04: spawn_fix() reuses spawner infra with FIX argv + _FIX_LOCK
# ─────────────────────────────────────────────────────────────────────────────


_FIX_MAX_TURNS_STR = "40"
_WORKTREE_CWD = "/tmp/.worktrees/meta-agent-fix-abc"  # noqa: S108
_FIX_ALLOWED_PATHS = ["src/finalayze/strategies/presets/", "config/segments.py"]
_FIX_DENIED_PATHS = [
    "src/finalayze/risk/",
    "src/finalayze/execution/",
    "src/finalayze/core/",
]


@pytest.mark.asyncio
async def test_spawn_fix_argv_uses_edit_and_worktree_cwd(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SPEC AC #13 (Task 58-04-04): ``spawn_fix(prompt, decision_id, cwd,
    allowed_paths, denied_paths, timeout_s)`` invokes ``claude`` with:
      - ``--allowedTools "Read,Grep,Edit,Bash"`` (Edit included for FIX)
      - ``--add-dir <worktree>`` so the spawned CLI has filesystem access
        to the worktree
      - ``--max-turns 40`` (FIX ceiling, not the 20 from investigate)
      - ``cwd=<worktree>`` so any relative paths resolve inside the worktree
      - ``start_new_session=True`` (process-group killability)
    """
    from pathlib import Path

    from finalayze.meta_agent.spawner import SpawnOutcome, spawn_fix

    captured: dict[str, Any] = {}
    fake_proc = _FakeProcess(
        stdout=b'{"type":"result","is_error":false}\n',
        exit_code=_EXIT_OK,
    )

    async def _fake_create(*args: Any, **kwargs: Any) -> _FakeProcess:
        captured["args"] = args
        captured["kwargs"] = kwargs
        return fake_proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_create)

    outcome = await spawn_fix(
        "fix prompt body",
        decision_id=_FAKE_DECISION_ID,
        cwd=Path(_WORKTREE_CWD),
        allowed_paths=_FIX_ALLOWED_PATHS,
        denied_paths=_FIX_DENIED_PATHS,
        timeout_s=600,
    )

    # Outcome shape — happy path.
    assert isinstance(outcome, SpawnOutcome)
    assert outcome.exit_code == _EXIT_OK
    assert outcome.timed_out is False
    assert outcome.killed_by_killswitch is False

    # Argv inspection — FIX-specific flag set.
    args = captured["args"]
    assert args[0] == "claude"
    assert args[1] == "-p"
    assert args[2] == "fix prompt body"

    # --allowedTools includes Edit (FIX-specific vs investigate's
    # "Read,Grep,Bash").
    assert "--allowedTools" in args
    at_idx = args.index("--allowedTools")
    assert args[at_idx + 1] == "Read,Grep,Edit,Bash", (
        f"expected FIX allowedTools='Read,Grep,Edit,Bash', got {args[at_idx + 1]!r}"
    )

    # --add-dir points at the worktree.
    assert "--add-dir" in args
    ad_idx = args.index("--add-dir")
    assert args[ad_idx + 1] == _WORKTREE_CWD, (
        f"expected --add-dir={_WORKTREE_CWD}, got {args[ad_idx + 1]!r}"
    )

    # --max-turns 40 (FIX ceiling).
    assert "--max-turns" in args
    mt_idx = args.index("--max-turns")
    assert args[mt_idx + 1] == _FIX_MAX_TURNS_STR, (
        f"expected --max-turns=40 for FIX, got {args[mt_idx + 1]!r}"
    )

    # Subprocess control kwargs.
    kwargs = captured["kwargs"]
    assert kwargs.get("start_new_session") is True
    assert kwargs.get("cwd") == _WORKTREE_CWD, (
        f"subprocess cwd must be the worktree path; got {kwargs.get('cwd')!r}"
    )
    assert kwargs.get("stdin") == asyncio.subprocess.DEVNULL


@pytest.mark.asyncio
async def test_concurrent_fix_spawns_rejected_with_already_inflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SPEC §Boundaries (line 105): at most one in-flight FIX spawn at a
    time. The second concurrent caller observes ``_FIX_LOCK.locked() is
    True`` WITHOUT taking the lock and returns
    SpawnOutcome(stderr='already_inflight').

    NOTE: ``_FIX_LOCK`` is a SEPARATE lock from ``_INVESTIGATE_LOCK`` —
    a FIX spawn does NOT block investigate spawns and vice versa
    (D-07 — locks per spawn-type).
    """
    import structlog
    from pathlib import Path

    from finalayze.meta_agent import spawner as sp
    from finalayze.meta_agent.spawner import spawn_fix

    slow_proc = _FakeProcess(
        stdout=b"slow fix result\n",
        stderr=b"",
        sleep_before_exit=0.5,
    )
    fast_proc = _FakeProcess(stdout=b"fast fix result\n", exit_code=_EXIT_OK)

    procs = iter([slow_proc, fast_proc])

    async def _fake_create(*_args: Any, **_kwargs: Any) -> _FakeProcess:
        return next(procs)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_create)

    # Verify lock is initially free.
    assert sp._FIX_LOCK.locked() is False, "FIX lock must start unlocked"

    first_task = asyncio.create_task(
        spawn_fix(
            "first fix prompt",
            decision_id=_FAKE_DECISION_ID,
            cwd=Path(_WORKTREE_CWD),
            allowed_paths=_FIX_ALLOWED_PATHS,
            denied_paths=_FIX_DENIED_PATHS,
        ),
    )
    # Yield briefly so the first task acquires the lock.
    await asyncio.sleep(0.05)

    # Pre-condition.
    assert sp._FIX_LOCK.locked() is True, "first fix spawn must have acquired _FIX_LOCK"

    with structlog.testing.capture_logs() as logs:
        second_outcome = await spawn_fix(
            "second fix prompt",
            decision_id=_FAKE_DECISION_ID_2,
            cwd=Path(_WORKTREE_CWD),
            allowed_paths=_FIX_ALLOWED_PATHS,
            denied_paths=_FIX_DENIED_PATHS,
        )

    # Second outcome: rejected, no subprocess spawned.
    assert second_outcome.exit_code == _EXIT_TIMEOUT, (
        f"expected synthetic exit_code={_EXIT_TIMEOUT}, got {second_outcome!r}"
    )
    assert second_outcome.stderr == "already_inflight"
    assert second_outcome.stdout == ""

    # Structlog event from the rejected caller (spawn_type='fix').
    rejected_events = [
        log for log in logs if log.get("event") == "meta_agent_spawn_already_inflight"
    ]
    assert len(rejected_events) == 1, (
        f"expected 1 already_inflight event for fix, got {rejected_events!r}"
    )
    assert rejected_events[0].get("spawn_type") == "fix"

    # First spawn must still complete and release the lock.
    first_outcome = await first_task
    assert first_outcome.exit_code == _EXIT_OK, (
        f"first fix spawn should have exited 0, got {first_outcome!r}"
    )
    assert sp._FIX_LOCK.locked() is False, "FIX lock must be released after first spawn completes"
