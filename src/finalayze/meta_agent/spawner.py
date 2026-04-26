"""Subprocess spawner for meta-agent investigation runs (Phase 58-03).

This module spawns the local ``claude`` CLI binary via raw
``asyncio.create_subprocess_exec`` (NOT ``claude-agent-sdk`` — see SPEC
deltas D-04/D-05 + RESEARCH §2.7). The CLI inherits the operator's
Max-subscription auth from ``~/.claude/.credentials.json`` because we
explicitly strip ``ANTHROPIC_API_KEY`` from the spawn env (precedence
rule #6 in code.claude.com/docs/en/authentication).

Process-group control: ``start_new_session=True`` makes the spawn the
leader of a new process group whose pgid equals its pid. The killswitch
(Plan 58-05) iterates ``_inflight_handles`` and signals the entire group
via ``os.killpg(pgid, SIGTERM)`` → 3s wait → ``os.killpg(pgid, SIGKILL)``.
On macOS this is the only way to reliably reach the CLI's own subprocess
descendants.

Concurrency (D-07): one in-flight investigate spawn at a time, guarded
by a module-level ``_INVESTIGATE_LOCK``. A second concurrent attempt
returns ``SpawnOutcome(stderr='already_inflight')`` WITHOUT taking the
lock (non-blocking ``Lock.locked()`` check). FIX-spawn (Plan 58-04) gets
its own ``_FIX_LOCK`` declared here so 58-04 only adds the entry-point.

Output capture (D-06): stdout/stderr are bytes; we truncate at 64 KiB
and decode with ``errors='replace'``. Final wire shape is plain text
appended to ``decision.outcome``; the executor (Plan 58-08) wraps it.

This module imports stdlib only (asyncio, os, signal) — Layer 6.
"""

from __future__ import annotations

import asyncio
import os
import signal
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from uuid import UUID

_log = structlog.get_logger()

# ── Module-level constants (PLR2004; magic-number-free in source) ────────────
_OUTCOME_MAX_BYTES = 64 * 1024  # D-06 — 64 KiB cap on captured streams
_DEFAULT_TIMEOUT_S = 300  # SPEC §Requirement 6 — 300s investigate timeout
_DEFAULT_MAX_TURNS = "20"  # SPEC §Requirement 6 / AC #10 — 20-turn ceiling
_SIGTERM_GRACE_S = 3.0  # killswitch SIGTERM grace before SIGKILL (D-05)
_SIGKILL_REAP_S = 1.0  # bounded SIGKILL reap (kernel guarantees this)
_DRAIN_TIMEOUT_S = 1.0  # post-termination stream drain budget
_TRUNCATION_MARKER = "\n[truncated_at=64KiB]"
_TIMEOUT_OUTCOME_RC = -1  # synthetic returncode for timed-out spawns

# ── Concurrency guards (D-07) ───────────────────────────────────────────────
# Per-spawn-type locks. INVESTIGATE owned by 58-03; FIX declared here so
# Plan 58-04 only adds the entry-point that takes it. Both expose
# ``Lock.locked()`` for non-blocking try-acquire.
_INVESTIGATE_LOCK: asyncio.Lock = asyncio.Lock()
_FIX_LOCK: asyncio.Lock = asyncio.Lock()

# Shared in-flight registry. Killswitch (58-05) iterates this dict and calls
# ``os.killpg(os.getpgid(proc.pid), signal.SIGTERM)`` on each entry. Keyed by
# decision_id so the executor can correlate failed spawns with their row.
_inflight_handles: dict[UUID, asyncio.subprocess.Process] = {}


@dataclass(frozen=True)
class SpawnOutcome:
    """Result of one ``spawn_readonly`` invocation.

    Frozen so callers cannot mutate post-return. The executor (Plan 58-03
    Task 08) builds the persisted ``decision.outcome`` string from these
    fields:

      - ``exit_code``: child process return code, or ``-1`` if timed out
        / killed before exit.
      - ``stdout`` / ``stderr``: UTF-8 strings, truncated at 64 KiB
        (D-06). Truncation appends ``\\n[truncated_at=64KiB]``.
      - ``timed_out``: True when the 300s wait_for tripped.
      - ``killed_by_killswitch``: True when ``CancelledError`` propagated
        from the killswitch (Plan 58-05 surfaces this).
    """

    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool
    killed_by_killswitch: bool


def _truncate(b: bytes, limit: int = _OUTCOME_MAX_BYTES) -> str:
    """Decode bytes as UTF-8 (lossy) and truncate at ``limit`` with marker."""
    if len(b) <= limit:
        return b.decode("utf-8", errors="replace")
    return b[:limit].decode("utf-8", errors="replace") + _TRUNCATION_MARKER


async def _drain(stream: asyncio.StreamReader | None) -> bytes:
    """Bounded read of a stream after the process has been terminated."""
    if stream is None:
        return b""
    try:
        return await asyncio.wait_for(stream.read(), timeout=_DRAIN_TIMEOUT_S)
    except (TimeoutError, asyncio.TimeoutError):
        return b""


async def _terminate_process_group(
    proc: asyncio.subprocess.Process,
    *,
    grace_s: float = _SIGTERM_GRACE_S,
    kill_s: float = _SIGKILL_REAP_S,
) -> None:
    """SIGTERM the whole process group → wait → SIGKILL on timeout (D-05).

    Total wall-clock bound: ``grace_s + kill_s`` (~4s with defaults), well
    under SPEC's 5s killswitch ceiling. Tests inject shorter values via the
    kwargs.
    """
    if proc.returncode is not None:
        return  # Already exited; nothing to signal.

    try:
        pgid = os.getpgid(proc.pid)
    except (ProcessLookupError, PermissionError, OSError):
        # Process disappeared between the check and the lookup; treat as
        # already-dead. Real exit code surfaces via proc.returncode.
        return

    # SIGTERM phase.
    try:
        os.killpg(pgid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return

    try:
        await asyncio.wait_for(proc.wait(), timeout=grace_s)
        return  # Cooperative shutdown succeeded.
    except (TimeoutError, asyncio.TimeoutError):
        pass

    # SIGKILL phase — unblockable; kernel guarantees fast reap.
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        return

    try:
        await asyncio.wait_for(proc.wait(), timeout=kill_s)
    except (TimeoutError, asyncio.TimeoutError):
        # Should be unreachable on a sane kernel — but never raise from
        # the killswitch path.
        _log.warning(
            "meta_agent_spawn_kill_reap_timeout",
            pid=proc.pid,
            pgid=pgid,
        )


def _build_argv(
    prompt: str,
    *,
    cwd: Path | None = None,
    allowed_tools: str = "Read,Grep,Bash",
    max_turns: str = _DEFAULT_MAX_TURNS,
) -> list[str]:
    """Build the ``claude -p`` argv (RESEARCH §3.1, §3.3).

    ``--allowedTools`` is camelCase per the CLI's flag parser. We pair
    ``--output-format stream-json`` with ``--verbose`` (required by the
    streaming mode).
    """
    argv: list[str] = [
        "claude",
        "-p", prompt,
        "--output-format", "stream-json",
        "--verbose",
        "--allowedTools", allowed_tools,
        "--max-turns", max_turns,
    ]
    if cwd is not None:
        argv.extend(["--add-dir", str(cwd)])
    return argv


def _strip_anthropic_api_key(env: dict[str, str]) -> dict[str, str]:
    """Return a copy of ``env`` with ``ANTHROPIC_API_KEY`` removed.

    The CLI's auth precedence (rule #6 in code.claude.com/docs) prefers
    ``ANTHROPIC_API_KEY`` when set. We unset it so the CLI falls back to
    the operator's Max subscription stored in ``~/.claude/.credentials.json``.
    """
    return {k: v for k, v in env.items() if k != "ANTHROPIC_API_KEY"}


async def spawn_readonly(  # noqa: PLR0913 — kwargs are config knobs, not hidden state
    prompt: str,
    *,
    decision_id: UUID,
    cwd: Path | None = None,
    timeout_s: int = _DEFAULT_TIMEOUT_S,
    output_max_bytes: int = _OUTCOME_MAX_BYTES,
    sigterm_grace_s: float = _SIGTERM_GRACE_S,
    sigkill_reap_s: float = _SIGKILL_REAP_S,
) -> SpawnOutcome:
    """Spawn the ``claude`` CLI in read-only mode for one investigation.

    Behaviour:
      1. If ``_INVESTIGATE_LOCK`` is held by another task, return
         ``SpawnOutcome(stderr='already_inflight')`` immediately WITHOUT
         taking the lock. (D-07 — non-blocking try-acquire pattern.)
      2. Take the lock. Build the argv (``--allowedTools Read,Grep,Bash
         --max-turns 20 --output-format stream-json --verbose``). Strip
         ``ANTHROPIC_API_KEY`` from env so the CLI uses the operator's
         Max subscription. Spawn with ``start_new_session=True`` for
         process-group killability.
      3. Register the handle in ``_inflight_handles[decision_id]``.
      4. ``await asyncio.wait_for(proc.communicate(), timeout=timeout_s)``.
      5. On TimeoutError, terminate the process group (SIGTERM(3s)→SIGKILL).
      6. On CancelledError (killswitch), terminate and re-raise.
      7. Truncate stdout/stderr to ``output_max_bytes`` and return.

    The function NEVER raises (except CancelledError, which it propagates
    after termination). All other failure modes surface via the
    ``SpawnOutcome`` fields.
    """
    # D-07 non-blocking try-acquire. Second concurrent investigate exits
    # without taking the lock. We do NOT use ``async with`` here because
    # we need the early-return semantics.
    if _INVESTIGATE_LOCK.locked():
        _log.warning(
            "meta_agent_spawn_already_inflight",
            decision_id_key=str(decision_id),
            spawn_type="investigate",
        )
        return SpawnOutcome(
            exit_code=_TIMEOUT_OUTCOME_RC,
            stdout="",
            stderr="already_inflight",
            timed_out=False,
            killed_by_killswitch=False,
        )

    async with _INVESTIGATE_LOCK:
        argv = _build_argv(prompt, cwd=cwd)
        env = _strip_anthropic_api_key(dict(os.environ))

        _log.info(
            "meta_agent_spawn_started",
            decision_id_key=str(decision_id),
            spawn_type="investigate",
            timeout_s=timeout_s,
        )

        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            cwd=str(cwd) if cwd is not None else None,
            start_new_session=True,
        )

        _inflight_handles[decision_id] = proc

        timed_out = False
        killed_by_killswitch = False
        stdout_bytes = b""
        stderr_bytes = b""

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(),
                timeout=timeout_s,
            )
        except (TimeoutError, asyncio.TimeoutError):
            timed_out = True
            _log.warning(
                "meta_agent_spawn_timeout",
                decision_id_key=str(decision_id),
                spawn_type="investigate",
                timeout_s=timeout_s,
            )
            await _terminate_process_group(
                proc, grace_s=sigterm_grace_s, kill_s=sigkill_reap_s,
            )
            stdout_bytes = await _drain(proc.stdout)
            stderr_bytes = await _drain(proc.stderr)
        except asyncio.CancelledError:
            killed_by_killswitch = True
            _log.warning(
                "meta_agent_spawn_cancelled",
                decision_id_key=str(decision_id),
                spawn_type="investigate",
            )
            await _terminate_process_group(
                proc, grace_s=sigterm_grace_s, kill_s=sigkill_reap_s,
            )
            raise
        finally:
            _inflight_handles.pop(decision_id, None)

        exit_code = proc.returncode if proc.returncode is not None else _TIMEOUT_OUTCOME_RC

        if not timed_out:
            _log.info(
                "meta_agent_spawn_completed",
                decision_id_key=str(decision_id),
                spawn_type="investigate",
                exit_code=exit_code,
                stdout_bytes=len(stdout_bytes),
                stderr_bytes=len(stderr_bytes),
            )

        return SpawnOutcome(
            exit_code=exit_code,
            stdout=_truncate(stdout_bytes, output_max_bytes),
            stderr=_truncate(stderr_bytes, output_max_bytes),
            timed_out=timed_out,
            killed_by_killswitch=killed_by_killswitch,
        )
