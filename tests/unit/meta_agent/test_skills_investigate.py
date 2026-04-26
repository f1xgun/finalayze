"""Structural tests for the meta-agent-investigate skill (Phase 58-03 Task 03).

D-11: assert structural invariants only. Iteration on prompt body must NOT
break tests; only schema changes do (D-08/D-09 nested ``finalayze_spawner``
key + AP-8 no-recursive-spawn).
"""

from __future__ import annotations

from pathlib import Path

# Module-level constants (PLR2004).
_INVEST_MAX_TURNS = 20
_REPO_ROOT = Path(__file__).resolve().parents[3]
_INVESTIGATE_SKILL = (
    _REPO_ROOT / ".claude" / "skills" / "meta-agent-investigate" / "SKILL.md"
)


def test_investigate_skill_exists_and_parses() -> None:
    """SPEC §Boundaries: a skill package ships at
    .claude/skills/meta-agent-investigate/SKILL.md with the expected
    spawner directives.
    """
    from finalayze.meta_agent.skill_loader import load_skill

    assert _INVESTIGATE_SKILL.exists(), (
        f"Missing investigate skill at {_INVESTIGATE_SKILL}"
    )

    spec = load_skill(_INVESTIGATE_SKILL)
    assert spec.name == "meta-agent-investigate"
    assert spec.allowed_tools == ["Read", "Grep", "Bash"]
    assert spec.max_turns == _INVEST_MAX_TURNS
    # Investigate skill is read-only — no path lists (those are FIX-only,
    # owned by Plan 58-04).
    assert spec.allowed_paths == []
    assert spec.denied_paths == []


def test_investigate_skill_disallows_recursive_claude_spawns() -> None:
    """AP-8 — the spawned claude subprocess must NOT spawn another claude
    subprocess. ``Bash(claude)`` is in the disallowed_tools list. Edit and
    Write are also denied (read-only skill).
    """
    from finalayze.meta_agent.skill_loader import load_skill

    spec = load_skill(_INVESTIGATE_SKILL)
    assert "Bash(claude)" in spec.disallowed_tools, (
        f"investigate skill must deny Bash(claude) (AP-8 no recursive spawns); "
        f"got disallowed_tools={spec.disallowed_tools!r}"
    )
    assert "Edit" in spec.disallowed_tools, "read-only skill must deny Edit"
    assert "Write" in spec.disallowed_tools, "read-only skill must deny Write"


def test_investigate_skill_body_is_non_empty() -> None:
    """The system prompt body must be non-empty (operators iterate on this
    text freely; this test only locks "exists", not "contains XYZ").
    """
    from finalayze.meta_agent.skill_loader import load_skill

    spec = load_skill(_INVESTIGATE_SKILL)
    assert len(spec.system_prompt.strip()) > 0
