"""Tests for meta_agent.worktree.create_fix_worktree (Phase 58-04 Task 03).

SPEC AC #13 — fix-spawn pipeline creates a fresh
``.worktrees/meta-agent-fix-<id8>`` worktree branched from HEAD before
spawning ``claude -p``. The worktree is operator-managed (D-16) — the
spawner does NOT auto-delete.

Tests monkeypatch ``subprocess.run`` so no real worktree is created
(would pollute the working tree). The CalledProcessError path raises
``MetaAgentWorktreeError``.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

# Module-level constants (PLR2004).
_FAKE_SHORT8 = "abcd1234"
_EXPECTED_PATH = Path(".worktrees") / f"meta-agent-fix-{_FAKE_SHORT8}"
_EXPECTED_BRANCH = f"meta-agent-fix-{_FAKE_SHORT8}"


def test_create_fix_worktree_invokes_git_with_correct_args(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SPEC AC #13: ``create_fix_worktree(short8)`` invokes
    ``git worktree add .worktrees/meta-agent-fix-<short8> -b
    meta-agent-fix-<short8> HEAD`` via ``subprocess.run`` with
    ``check=True, capture_output=True, text=True`` and returns the worktree
    Path.
    """
    from finalayze.meta_agent.worktree import create_fix_worktree

    captured: dict[str, Any] = {}

    def _fake_run(*args: Any, **kwargs: Any) -> Any:
        captured["args"] = args
        captured["kwargs"] = kwargs
        # Return a fake CompletedProcess shape — the function ignores its result.
        return subprocess.CompletedProcess(
            args=args[0] if args else [],
            returncode=0,
            stdout="Preparing worktree (new branch)\n",
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", _fake_run)

    path = create_fix_worktree(_FAKE_SHORT8)

    # Returned path matches the locked layout.
    assert path == _EXPECTED_PATH, f"expected {_EXPECTED_PATH!r}, got {path!r}"

    # subprocess.run called with the SPEC argv.
    args = captured["args"]
    argv = args[0]
    assert argv == [
        "git",
        "worktree",
        "add",
        str(_EXPECTED_PATH),
        "-b",
        _EXPECTED_BRANCH,
        "HEAD",
    ], f"unexpected argv: {argv!r}"

    kwargs = captured["kwargs"]
    assert kwargs.get("check") is True, "must pass check=True"
    assert kwargs.get("capture_output") is True, "must capture stdout/stderr"
    assert kwargs.get("text") is True, "must use text mode"


def test_create_fix_worktree_raises_worktree_error_on_git_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SPEC AC #13: when ``git worktree add`` fails (e.g. branch already
    exists from a prior crashed cycle), the helper raises
    ``MetaAgentWorktreeError`` carrying the git stderr in its message.
    The executor (Plan 58-04-09) catches this and marks the decision
    'failed' with outcome='worktree_create_failed:<stderr>'.
    """
    from finalayze.meta_agent.exceptions import MetaAgentWorktreeError
    from finalayze.meta_agent.worktree import create_fix_worktree

    def _fake_run_fails(*args: Any, **kwargs: Any) -> Any:  # noqa: ARG001
        raise subprocess.CalledProcessError(
            returncode=128,
            cmd=args[0] if args else [],
            output="",
            stderr="fatal: 'meta-agent-fix-abcd1234' is already used by worktree\n",
        )

    monkeypatch.setattr(subprocess, "run", _fake_run_fails)

    with pytest.raises(MetaAgentWorktreeError) as exc_info:
        create_fix_worktree(_FAKE_SHORT8)

    assert "already used by worktree" in str(exc_info.value), (
        f"expected git stderr in exception message, got {exc_info.value!r}"
    )


def test_create_fix_worktree_accepts_custom_base(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ``base`` kwarg defaults to ``HEAD`` but can be overridden for
    tests / future use cases.
    """
    from finalayze.meta_agent.worktree import create_fix_worktree

    captured: dict[str, Any] = {}

    def _fake_run(*args: Any, **kwargs: Any) -> Any:  # noqa: ARG001
        captured["args"] = args
        return subprocess.CompletedProcess(
            args=args[0] if args else [], returncode=0, stdout="", stderr=""
        )

    monkeypatch.setattr(subprocess, "run", _fake_run)

    create_fix_worktree(_FAKE_SHORT8, base="origin/main")
    argv = captured["args"][0]
    assert argv[-1] == "origin/main", f"base kwarg must be the last argv arg; got {argv!r}"
