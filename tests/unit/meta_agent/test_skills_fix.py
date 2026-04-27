"""Structural tests for the meta-agent-fix skill (Phase 58-04 Task 02).

D-11 + AP-13 — assert structural invariants only:
  - ``denied_paths`` includes the three locked protected directories
    (``src/finalayze/risk/``, ``src/finalayze/execution/``,
    ``src/finalayze/core/``).
  - ``allowed_paths`` is exactly the locked allow-list.
  - ``allowed_tools`` includes ``Edit``.
  - ``disallowed_tools`` includes ``Bash(claude)`` (AP-8 — no recursive spawns)
    and ``Write``.
  - ``max_turns == 40`` (FIX-skill ceiling).
"""

from __future__ import annotations

from pathlib import Path

# Module-level constants (PLR2004).
_FIX_MAX_TURNS = 40
_REPO_ROOT = Path(__file__).resolve().parents[3]
_FIX_SKILL = _REPO_ROOT / ".claude" / "skills" / "meta-agent-fix" / "SKILL.md"

_LOCKED_DENIED_PATHS = [
    "src/finalayze/risk/",
    "src/finalayze/execution/",
    "src/finalayze/core/",
]
_LOCKED_ALLOWED_PATHS = sorted(
    [
        "src/finalayze/strategies/presets/",
        "config/segments.py",
    ],
)


def test_fix_skill_exists_and_parses() -> None:
    """SPEC §Boundaries: a skill package ships at
    .claude/skills/meta-agent-fix/SKILL.md with the FIX-spawn directives.
    """
    from finalayze.meta_agent.skill_loader import load_skill

    assert _FIX_SKILL.exists(), f"Missing fix skill at {_FIX_SKILL}"

    spec = load_skill(_FIX_SKILL)
    assert spec.name == "meta-agent-fix"
    assert spec.max_turns == _FIX_MAX_TURNS


def test_fix_skill_denies_protected_paths() -> None:
    """AP-13 + SPEC AC #13: ``denied_paths`` MUST include all three
    protected directories. The pre-spawn validator pulls this list and
    enforces it on the prompt.
    """
    from finalayze.meta_agent.skill_loader import load_skill

    spec = load_skill(_FIX_SKILL)
    for required in _LOCKED_DENIED_PATHS:
        assert required in spec.denied_paths, (
            f"fix skill denied_paths must include {required!r}; got {spec.denied_paths!r}"
        )


def test_fix_skill_allows_only_presets_and_segments() -> None:
    """SPEC AC #13: ``allowed_paths`` is exactly the locked two-element
    set: ``src/finalayze/strategies/presets/`` and ``config/segments.py``.
    """
    from finalayze.meta_agent.skill_loader import load_skill

    spec = load_skill(_FIX_SKILL)
    assert sorted(spec.allowed_paths) == _LOCKED_ALLOWED_PATHS, (
        f"fix skill allowed_paths must be exactly {_LOCKED_ALLOWED_PATHS!r}; "
        f"got {sorted(spec.allowed_paths)!r}"
    )


def test_fix_skill_allowed_tools_include_edit() -> None:
    """SPEC AC #13 — fix skill must allow Edit (it is the action that
    produces the proposed remediation). Read, Grep, Bash also required so
    the spawned claude can inspect + run validation tests.
    """
    from finalayze.meta_agent.skill_loader import load_skill

    spec = load_skill(_FIX_SKILL)
    assert "Edit" in spec.allowed_tools, (
        f"fix skill must allow Edit; got allowed_tools={spec.allowed_tools!r}"
    )
    assert "Read" in spec.allowed_tools
    assert "Grep" in spec.allowed_tools
    assert "Bash" in spec.allowed_tools


def test_fix_skill_disallows_recursive_claude_spawns() -> None:
    """AP-8 — the spawned claude subprocess must NOT spawn another claude
    subprocess. ``Bash(claude)`` is in disallowed_tools. Write is also
    denied — fix skill uses Edit (in-place patches) only, never blanket
    Write.
    """
    from finalayze.meta_agent.skill_loader import load_skill

    spec = load_skill(_FIX_SKILL)
    assert "Bash(claude)" in spec.disallowed_tools, (
        f"fix skill must deny Bash(claude) (AP-8 no recursive spawns); "
        f"got disallowed_tools={spec.disallowed_tools!r}"
    )
    assert "Write" in spec.disallowed_tools, (
        f"fix skill must deny Write (Edit-only invariant); "
        f"got disallowed_tools={spec.disallowed_tools!r}"
    )


def test_fix_skill_body_is_non_empty() -> None:
    """The system prompt body must be non-empty. Operators iterate freely
    on this text; the test only locks "exists" not "contains XYZ" (D-11).
    """
    from finalayze.meta_agent.skill_loader import load_skill

    spec = load_skill(_FIX_SKILL)
    assert len(spec.system_prompt.strip()) > 0
