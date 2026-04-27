"""Git worktree creation for FIX-severity meta-agent runs (Phase 58-04).

SPEC §Requirement 7 + AC #13 — every FIX spawn runs inside a fresh git
worktree under ``.worktrees/meta-agent-fix-<id8>`` branched from HEAD.
The spawned ``claude -p`` is constrained to this directory via ``--add-dir``,
giving the path-allow-list enforcement a hard filesystem boundary on top of
the in-prompt validator (``path_validator.validate_fix_prompt``).

Per D-16, the worktree is OPERATOR-MANAGED: the spawner does NOT auto-delete
after the spawn completes. The operator inspects, tests, and manually opens
a PR (via ``gh pr create``) before running ``git worktree remove
.worktrees/meta-agent-fix-<id8>`` to clean up.

Subprocess invocation is synchronous (worktree creation is fast) — RESEARCH
Open Q #6 + plan body. Failures wrap into ``MetaAgentWorktreeError``.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import structlog

from finalayze.meta_agent.exceptions import MetaAgentWorktreeError

_log = structlog.get_logger()

# Module-level constants (PLR2004 / no magic strings). The two-element
# layout below is locked: planner cannot rename without SPEC change.
_WORKTREE_ROOT = Path(".worktrees")
_BRANCH_PREFIX = "meta-agent-fix-"


def create_fix_worktree(short8: str, *, base: str = "HEAD") -> Path:
    """Create a fresh git worktree at ``.worktrees/meta-agent-fix-<short8>``.

    Invokes ``git worktree add <target> -b <branch> <base>`` synchronously.
    On success returns the worktree ``Path`` (relative to project root).
    On failure (``CalledProcessError`` from git, e.g. branch already exists),
    raises ``MetaAgentWorktreeError`` carrying the git stderr in its message
    so the executor can stamp ``decision.outcome``.

    Args:
        short8: First 8 hex chars of the decision UUID. Used both as the
            worktree directory suffix AND the branch name suffix so the
            two coordinates always match.
        base: Git ref to branch from. Defaults to ``HEAD``; tests / future
            use cases may override (e.g. ``origin/main``).

    Returns:
        ``Path`` to the new worktree directory (relative). The caller (the
        executor) passes this Path as ``cwd`` and ``--add-dir`` to
        ``spawn_fix``.

    Raises:
        MetaAgentWorktreeError: ``git worktree add`` failed. The git
            stderr is included in the exception message for operator
            audit.
    """
    target = _WORKTREE_ROOT / f"{_BRANCH_PREFIX}{short8}"
    branch = f"{_BRANCH_PREFIX}{short8}"
    argv = ["git", "worktree", "add", str(target), "-b", branch, base]

    try:
        result = subprocess.run(  # noqa: S603 — argv built from internal ids
            argv,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        _log.warning(
            "meta_agent_worktree_create_failed",
            short8=short8,
            target=str(target),
            branch=branch,
            returncode=exc.returncode,
            stderr=exc.stderr,
        )
        msg = (
            f"git worktree add failed (rc={exc.returncode}) for "
            f"{target}: {exc.stderr.strip()}"
        )
        raise MetaAgentWorktreeError(msg) from exc

    _log.info(
        "meta_agent_worktree_created",
        short8=short8,
        target=str(target),
        branch=branch,
        stdout=result.stdout.strip(),
    )
    return target
