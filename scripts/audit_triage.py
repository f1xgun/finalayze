"""Risk-class triage for the autonomous audit loop (safety core).

Decides whether a proposed fix (a set of changed file paths) may be AUTO-MERGED
unattended, or must be ESCALATED to a human. The policy is **default-risky**
(fail-safe): a change is SAFE only when EVERY changed path is in the narrow safe
class; anything else -- and anything unrecognised -- is RISKY.

Safe class (auto-merge on green CI):
  - documentation: any ``*.md`` (incl. AGENTS.md/README) and everything under ``docs/``
  - tests: everything under ``tests/``
  - dependency lockfile refresh: ``uv.lock`` alone

Everything else is RISKY and routed to a human PR + Telegram escalation -- in
particular anything under ``src/finalayze/`` (strategy / risk / ML / execution /
core money math), ``config/``, ``alembic/`` (migrations), ``scripts/``,
``docker/``, ``.github/`` (CI gates), and ``pyproject.toml`` (tool config + deps).

This module is pure stdlib (no project imports) so it sits below every layer and
can be unit-tested in isolation. It NEVER places orders and has no side effects
beyond reading argv when run as a CLI.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from enum import StrEnum


class RiskClass(StrEnum):
    """Whether a change may be auto-merged unattended."""

    SAFE = "safe"
    RISKY = "risky"


@dataclass(frozen=True)
class TriageVerdict:
    """Result of classifying a set of changed paths."""

    risk_class: RiskClass
    reason: str
    offending_path: str | None = None


# Prefixes whose entire subtree is auto-mergeable.
_SAFE_PREFIXES: tuple[str, ...] = ("docs/", "tests/")
# Exact paths that are auto-mergeable on their own (dependency lock refresh).
_SAFE_EXACT: frozenset[str] = frozenset({"uv.lock"})
# Suffixes that are auto-mergeable anywhere in the tree (documentation).
_SAFE_SUFFIXES: tuple[str, ...] = (".md",)


def _normalize(path: str) -> str:
    return path.strip().lstrip("./").replace("\\", "/")


def _is_safe_path(path: str) -> bool:
    """True only if ``path`` is unambiguously in the safe (auto-merge) class."""
    n = _normalize(path)
    if not n:
        return False
    if n in _SAFE_EXACT:
        return True
    if n.endswith(_SAFE_SUFFIXES):
        return True
    return n.startswith(_SAFE_PREFIXES)


def classify_change(changed_paths: list[str]) -> TriageVerdict:
    """Classify a set of changed paths; default-risky.

    SAFE only when there is at least one change AND every changed path is in the
    safe class. An empty change set is RISKY (nothing to auto-merge -- never a
    silent no-op). The first non-safe path is reported so the escalation message
    can name exactly what tripped the gate.
    """
    paths = [p for p in changed_paths if _normalize(p)]
    if not paths:
        return TriageVerdict(RiskClass.RISKY, "no changed files to classify")
    for p in paths:
        if not _is_safe_path(p):
            return TriageVerdict(
                RiskClass.RISKY,
                f"{_normalize(p)} is outside the safe class (docs/tests/uv.lock) -> human review",
                _normalize(p),
            )
    return TriageVerdict(
        RiskClass.SAFE,
        f"all {len(paths)} changed path(s) are docs/tests/lockfile -> auto-merge eligible",
    )


def main(argv: list[str]) -> int:
    """CLI: ``audit_triage.py <path> [<path> ...]`` -> prints verdict, exits 0=safe, 2=risky."""
    verdict = classify_change(argv)
    print(f"{verdict.risk_class.value.upper()}: {verdict.reason}")
    return 0 if verdict.risk_class is RiskClass.SAFE else 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
