"""Pre-spawn path validator for FIX-severity meta-agent runs (Phase 58-04).

SPEC §Requirement 7 + AC #13 — the FIX-spawn pipeline must reject any prompt
that references a denied path BEFORE the worktree is created and BEFORE
``claude -p`` is spawned. The validator performs a substring scan
(case-sensitive) against the prompt; any hit raises
``MetaAgentDeniedPathError``.

Denied paths (locked in SPEC line 65 + AP-13):
  - ``src/finalayze/risk/``
  - ``src/finalayze/execution/``
  - ``src/finalayze/core/``

The validator is invoked from ``ActionExecutor.execute_fix_spawn`` and from
the fix-skill's structural-invariant tests. Substring scan suffices for the
threat model: the prompt is built by the executor (NOT by free-form operator
input), so adversarial obfuscation is out of scope (SPEC line 396).
"""

from __future__ import annotations

from finalayze.meta_agent.exceptions import MetaAgentDeniedPathError


def validate_fix_prompt(prompt: str, *, denied_paths: list[str]) -> None:
    """Reject ``prompt`` if it contains any of the ``denied_paths`` substrings.

    Case-sensitive substring scan. Returns ``None`` if no denied path is
    referenced; raises ``MetaAgentDeniedPathError`` with the offending path
    in the message if any match is found.

    Args:
        prompt: The FIX-spawn user-turn prompt about to be passed to
            ``claude -p``.
        denied_paths: List of substrings that must NOT appear anywhere in
            ``prompt``. Typically populated from the fix skill's
            ``denied_paths`` YAML key (see ``SkillSpec.denied_paths``).

    Raises:
        MetaAgentDeniedPathError: When at least one ``denied_paths`` entry
            appears as a substring of ``prompt``. The exception message
            names the first offending path so the operator can audit.
    """
    for denied in denied_paths:
        if denied in prompt:
            msg = f"Fix prompt references denied path: {denied!r}"
            raise MetaAgentDeniedPathError(msg)
