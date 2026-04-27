"""Skill front-matter loader for the meta-agent spawner (Phase 58-03).

Skills live at ``.claude/skills/meta-agent-{investigate,fix}/SKILL.md`` and
ship a YAML front-matter block under the ``finalayze_spawner`` namespace
(D-08/D-09 — namespaced to avoid collision with the official Skills schema).

The spawner reads SKILL.md once at startup, threads ``system_prompt`` (body)
into ``claude -p`` via ``--append-system-prompt``, and uses
``allowed_tools`` / ``disallowed_tools`` to build the CLI flag set.

D-11: structural invariants are tested separately; iteration on the prompt
body alone must NOT break tests — only schema changes do.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from pathlib import Path

# Module-level defaults — keep numeric/string constants out of the dataclass
# definition so callers can override per-skill from the YAML.
_DEFAULT_MAX_TURNS = 20
_DEFAULT_PERMISSION_MODE = "bypassPermissions"

# Front-matter delimiter used by skill files. Must match the canonical
# Markdown front-matter convention used by Anthropic skills + Jekyll/etc.
_FRONT_MATTER_DELIM = "---\n"


@dataclass(frozen=True)
class SkillSpec:
    """Parsed skill specification.

    Frozen so callers cannot mutate post-load. Empty list defaults are
    safe because dataclasses gives each instance its own list (via
    ``field(default_factory=list)``).
    """

    name: str
    description: str
    system_prompt: str
    allowed_tools: list[str] = field(default_factory=list)
    disallowed_tools: list[str] = field(default_factory=list)
    max_turns: int = _DEFAULT_MAX_TURNS
    permission_mode: str = _DEFAULT_PERMISSION_MODE
    # FIX-skill only — investigate skill leaves these empty.
    allowed_paths: list[str] = field(default_factory=list)
    denied_paths: list[str] = field(default_factory=list)


def load_skill(path: Path) -> SkillSpec:
    """Parse a SKILL.md file into a ``SkillSpec``.

    Raises ``ValueError`` if the file does not begin with the YAML
    front-matter delimiter (``---\\n``). Raises ``FileNotFoundError`` if
    the path does not exist (propagated from ``Path.read_text``).

    The ``finalayze_spawner`` nested key carries the spawner-specific
    directives (D-09). Top-level ``name`` + ``description`` are kept at the
    top level for compatibility with the existing skill format.
    """
    text = path.read_text(encoding="utf-8")
    if not text.startswith(_FRONT_MATTER_DELIM):
        msg = f"{path}: missing YAML front-matter (expected leading '---\\n')"
        raise ValueError(msg)

    # Split into [empty, front_matter, body]. ``maxsplit=2`` so any '---' in
    # the body itself is preserved.
    _empty, fm, body = text.split(_FRONT_MATTER_DELIM, 2)
    meta: dict[str, Any] = yaml.safe_load(fm) or {}
    spawner: dict[str, Any] = meta.get("finalayze_spawner") or {}

    return SkillSpec(
        name=meta["name"],
        description=meta["description"],
        system_prompt=body.strip(),
        allowed_tools=list(spawner.get("allowed_tools") or []),
        disallowed_tools=list(spawner.get("disallowed_tools") or []),
        max_turns=int(spawner.get("max_turns", _DEFAULT_MAX_TURNS)),
        permission_mode=str(
            spawner.get("permission_mode", _DEFAULT_PERMISSION_MODE),
        ),
        allowed_paths=list(spawner.get("allowed_paths") or []),
        denied_paths=list(spawner.get("denied_paths") or []),
    )
