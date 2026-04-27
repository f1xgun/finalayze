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
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from pathlib import Path
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
    except TimeoutError:
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
    except TimeoutError:
        pass

    # SIGKILL phase — unblockable; kernel guarantees fast reap.
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        return

    try:
        await asyncio.wait_for(proc.wait(), timeout=kill_s)
    except TimeoutError:
        # Should be unreachable on a sane kernel — but never raise from
        # the killswitch path.
        _log.warning(
            "meta_agent_spawn_kill_reap_timeout",
            pid=proc.pid,
            pgid=pgid,
        )


# 58-04: FIX-spawn turn ceiling — distinct from the 20-turn investigate cap.
_FIX_MAX_TURNS = "40"
_FIX_TIMEOUT_S = 600  # SPEC §Requirement 7 — 600s fix-spawn timeout


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
        "-p",
        prompt,
        "--output-format",
        "stream-json",
        "--verbose",
        "--allowedTools",
        allowed_tools,
        "--max-turns",
        max_turns,
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


async def _run_claude_subprocess(
    argv: list[str],
    *,
    decision_id: UUID,
    spawn_type: str,
    cwd: Path | None,
    timeout_s: int,
    output_max_bytes: int,
    sigterm_grace_s: float,
    sigkill_reap_s: float,
) -> SpawnOutcome:
    """Shared subprocess body for ``spawn_readonly`` and ``spawn_fix``.

    Both public spawners build their own ``argv`` (with the appropriate
    ``--allowedTools``, ``--max-turns``, ``--add-dir`` flags) and acquire
    their own lock BEFORE calling this helper. This function:

      1. Strips ``ANTHROPIC_API_KEY`` from env (subscription-auth).
      2. Spawns the subprocess with ``start_new_session=True`` (process-
         group killability).
      3. Registers the handle in the shared ``_inflight_handles`` registry
         (consumed by 58-05's killswitch).
      4. ``await asyncio.wait_for(proc.communicate(), timeout=timeout_s)``.
      5. On TimeoutError, terminates the process group (SIGTERM(3s)→SIGKILL).
      6. On CancelledError (killswitch), terminates and re-raises.
      7. Truncates stdout/stderr to ``output_max_bytes`` and returns.

    Per AP-1: raw subprocess.exec + ``os.killpg`` on the pgid (NOT
    ``proc.terminate()``). The process group escape hatch is what makes
    the killswitch deterministic on macOS.
    """
    env = _strip_anthropic_api_key(dict(os.environ))

    _log.info(
        "meta_agent_spawn_started",
        decision_id_key=str(decision_id),
        spawn_type=spawn_type,
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
    except TimeoutError:
        timed_out = True
        _log.warning(
            "meta_agent_spawn_timeout",
            decision_id_key=str(decision_id),
            spawn_type=spawn_type,
            timeout_s=timeout_s,
        )
        await _terminate_process_group(
            proc,
            grace_s=sigterm_grace_s,
            kill_s=sigkill_reap_s,
        )
        stdout_bytes = await _drain(proc.stdout)
        stderr_bytes = await _drain(proc.stderr)
    except asyncio.CancelledError:
        killed_by_killswitch = True
        _log.warning(
            "meta_agent_spawn_cancelled",
            decision_id_key=str(decision_id),
            spawn_type=spawn_type,
        )
        await _terminate_process_group(
            proc,
            grace_s=sigterm_grace_s,
            kill_s=sigkill_reap_s,
        )
        raise
    finally:
        _inflight_handles.pop(decision_id, None)

    exit_code = proc.returncode if proc.returncode is not None else _TIMEOUT_OUTCOME_RC

    if not timed_out:
        _log.info(
            "meta_agent_spawn_completed",
            decision_id_key=str(decision_id),
            spawn_type=spawn_type,
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


async def spawn_readonly(
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
        return await _run_claude_subprocess(
            argv,
            decision_id=decision_id,
            spawn_type="investigate",
            cwd=cwd,
            timeout_s=timeout_s,
            output_max_bytes=output_max_bytes,
            sigterm_grace_s=sigterm_grace_s,
            sigkill_reap_s=sigkill_reap_s,
        )


async def spawn_fix(
    prompt: str,
    *,
    decision_id: UUID,
    cwd: Path,
    allowed_paths: list[str],  # noqa: ARG001 — reserved for future per-path injection
    denied_paths: list[str],  # noqa: ARG001 — denied set already enforced via path_validator
    timeout_s: int = _FIX_TIMEOUT_S,
    output_max_bytes: int = _OUTCOME_MAX_BYTES,
    sigterm_grace_s: float = _SIGTERM_GRACE_S,
    sigkill_reap_s: float = _SIGKILL_REAP_S,
) -> SpawnOutcome:
    """Spawn the ``claude`` CLI in FIX mode (Edit allowed, worktree cwd).

    SPEC §Requirement 7 + AC #13. Mirrors ``spawn_readonly`` but:
      - ``--allowedTools "Read,Grep,Edit,Bash"`` (Edit included).
      - ``--add-dir <worktree>`` ties the CLI's filesystem reach to the
        worktree.
      - ``--max-turns 40`` (FIX ceiling, vs investigate's 20).
      - ``cwd=<worktree>`` so relative paths resolve inside the worktree.
      - Acquires ``_FIX_LOCK`` (separate from ``_INVESTIGATE_LOCK`` —
        FIX and INVESTIGATE can run concurrently per D-07).

    The ``allowed_paths`` and ``denied_paths`` kwargs are accepted for
    API symmetry with the skill schema; the actual enforcement happens
    in the executor's pre-spawn ``validate_fix_prompt`` call (Plan 58-04
    Task 09) AND in the fix-skill's filesystem boundary (the spawned CLI
    refuses Edit operations outside its allowed tool list). The kwargs
    are reserved for future per-path injection (e.g. when the CLI grows
    a flag for path allow-listing directly).

    Returns the same ``SpawnOutcome`` shape as ``spawn_readonly``.
    Concurrent FIX spawn → ``SpawnOutcome(stderr='already_inflight')``.
    Never raises except CancelledError (killswitch propagates).
    """
    if _FIX_LOCK.locked():
        _log.warning(
            "meta_agent_spawn_already_inflight",
            decision_id_key=str(decision_id),
            spawn_type="fix",
        )
        return SpawnOutcome(
            exit_code=_TIMEOUT_OUTCOME_RC,
            stdout="",
            stderr="already_inflight",
            timed_out=False,
            killed_by_killswitch=False,
        )

    async with _FIX_LOCK:
        argv = _build_argv(
            prompt,
            cwd=cwd,
            allowed_tools="Read,Grep,Edit,Bash",
            max_turns=_FIX_MAX_TURNS,
        )
        return await _run_claude_subprocess(
            argv,
            decision_id=decision_id,
            spawn_type="fix",
            cwd=cwd,
            timeout_s=timeout_s,
            output_max_bytes=output_max_bytes,
            sigterm_grace_s=sigterm_grace_s,
            sigkill_reap_s=sigkill_reap_s,
        )
