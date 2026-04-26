"""Tests for meta_agent.spawner — read-only investigation subprocess (Phase 58-03).

Covers SPEC §Acceptance Criteria #10 + #11:
  - Exception classes (Task 58-03-01)
  - spawn_readonly happy path with monkeypatched subprocess (Task 58-03-04)
  - 300s timeout → SIGTERM → SIGKILL (Task 58-03-05)
  - Concurrent investigate → already_inflight (Task 58-03-06)

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
    """Stand-in for ``asyncio.subprocess.Process`` — records what the spawner does."""

    def __init__(
        self,
        *,
        stdout: bytes = b"",
        stderr: bytes = b"",
        exit_code: int = 0,
        sleep_before_exit: float = 0.0,
    ) -> None:
        self.pid: int = _FAKE_PID
        self.returncode: int | None = None
        self._stdout = stdout
        self._stderr = stderr
        self._exit_code = exit_code
        self._sleep = sleep_before_exit
        self.terminate_called = False
        self.kill_called = False

    async def communicate(self) -> tuple[bytes, bytes]:
        if self._sleep > 0:
            await asyncio.sleep(self._sleep)
        self.returncode = self._exit_code
        return self._stdout, self._stderr

    async def wait(self) -> int:
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
    try:
        raise MetaAgentDeniedPathError(msg_denied)
    except MetaAgentDeniedPathError as exc:
        assert str(exc) == msg_denied

    msg_cap = "spawn cap exceeded for INVESTIGATE"
    try:
        raise MetaAgentSpawnCapExceededError(msg_cap)
    except MetaAgentSpawnCapExceededError as exc:
        assert str(exc) == msg_cap


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

    fake_stdout = (
        b'{"type":"assistant","content":"hello"}\n'
        b'{"type":"result","is_error":false}\n'
    )
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
