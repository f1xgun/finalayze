"""Tests for meta_agent.skill_loader — YAML front-matter parser (Phase 58-03).

D-08/D-09/D-11: skills are markdown files with a YAML front-matter block
under the ``finalayze_spawner`` key. The spawner loads them at startup;
iteration on prompt body alone must NOT break tests — only structural
changes do.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# Module-level constants (PLR2004).
_DEFAULT_MAX_TURNS = 20


def _write_skill(path: Path, body: str) -> Path:
    """Write a complete SKILL.md (front-matter + body) to ``path`` and return."""
    path.write_text(body, encoding="utf-8")
    return path


def test_load_skill_parses_minimum_fields(tmp_path: Path) -> None:
    """SkillSpec is populated from YAML front-matter + body. The
    ``finalayze_spawner`` nested block carries the spawner-specific
    directives (allowed_tools, disallowed_tools, max_turns, paths).
    """
    from finalayze.meta_agent.skill_loader import SkillSpec, load_skill

    body = """---
name: test-skill
description: A test skill description.
finalayze_spawner:
  allowed_tools:
    - Read
    - Grep
    - Bash
  disallowed_tools:
    - Edit
    - Write
    - "Bash(claude)"
  max_turns: 20
  permission_mode: bypassPermissions
---

# Test Skill Body

This is the system prompt body.
"""
    path = _write_skill(tmp_path / "SKILL.md", body)
    spec = load_skill(path)

    assert isinstance(spec, SkillSpec)
    assert spec.name == "test-skill"
    assert spec.description == "A test skill description."
    assert spec.allowed_tools == ["Read", "Grep", "Bash"]
    assert spec.disallowed_tools == ["Edit", "Write", "Bash(claude)"]
    assert spec.max_turns == _DEFAULT_MAX_TURNS
    assert spec.permission_mode == "bypassPermissions"
    assert spec.allowed_paths == []
    assert spec.denied_paths == []
    # Body stripped, no leading/trailing whitespace, but content preserved.
    assert spec.system_prompt.startswith("# Test Skill Body")
    assert "system prompt body" in spec.system_prompt


def test_load_skill_raises_on_missing_front_matter(tmp_path: Path) -> None:
    """Skill files MUST start with ``---\\n``. Missing front-matter raises
    ValueError with 'missing YAML front-matter' in the message.
    """
    from finalayze.meta_agent.skill_loader import load_skill

    body = "# No front-matter\n\nJust body content.\n"
    path = _write_skill(tmp_path / "SKILL.md", body)

    with pytest.raises(ValueError, match="missing YAML front-matter"):
        load_skill(path)


def test_load_skill_supports_allowed_and_denied_paths(tmp_path: Path) -> None:
    """For FIX-style skills (used in 58-04), the front-matter may carry
    ``allowed_paths`` and ``denied_paths`` lists which the pre-spawn
    validator consumes. The investigate skill leaves these empty.
    """
    from finalayze.meta_agent.skill_loader import load_skill

    body = """---
name: fix-style-skill
description: Skill with path lists.
finalayze_spawner:
  allowed_tools: [Read, Grep, Edit, Bash]
  disallowed_tools: ["Bash(claude)"]
  max_turns: 40
  permission_mode: bypassPermissions
  allowed_paths:
    - src/finalayze/strategies/presets/
    - config/segments.py
  denied_paths:
    - src/finalayze/risk/
    - src/finalayze/execution/
    - src/finalayze/core/
---

# Body
"""
    path = _write_skill(tmp_path / "SKILL.md", body)
    spec = load_skill(path)

    assert spec.allowed_paths == [
        "src/finalayze/strategies/presets/",
        "config/segments.py",
    ]
    assert spec.denied_paths == [
        "src/finalayze/risk/",
        "src/finalayze/execution/",
        "src/finalayze/core/",
    ]
