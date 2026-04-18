#!/usr/bin/env python3
"""Validate the AGENTS.md Harness-Engineering graph against .agents/manifest.jsonl.

Checks:
1. Coverage — every AGENTS.md on disk has exactly one manifest entry (and vice-versa).
2. Parent/child integrity — referenced paths exist in the manifest.
3. Reachability — exactly one root node; every node is reachable from it.
4. Cross-link resolution — every markdown link in every AGENTS.md file resolves.

Exits non-zero on the first category with errors, printing all findings.
Run: ``python scripts/graph_check.py``.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ROOT / ".agents" / "manifest.jsonl"

# Skip trees that are not part of the live graph.
EXCLUDE_DIR_PARTS: frozenset[str] = frozenset(
    {".claude", ".worktrees", ".venv", ".agents", ".git", "node_modules"}
)

LINK_RE = re.compile(r"\]\(([^)]+\.md)\)")


def _is_excluded(path: Path) -> bool:
    return any(part in EXCLUDE_DIR_PARTS for part in path.parts)


def discover_agents_md(root: Path) -> set[str]:
    """Return repo-relative paths of every live AGENTS.md file."""
    found: set[str] = set()
    for p in root.rglob("AGENTS.md"):
        rel = p.relative_to(root)
        if _is_excluded(rel):
            continue
        found.add(str(rel))
    return found


def load_manifest() -> list[dict[str, Any]]:
    if not MANIFEST_PATH.exists():
        raise SystemExit(f"graph_check: manifest not found at {MANIFEST_PATH}")
    nodes: list[dict[str, Any]] = []
    for i, line in enumerate(MANIFEST_PATH.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            nodes.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"graph_check: manifest line {i}: {exc}") from exc
    return nodes


def check_coverage(on_disk: set[str], nodes: list[dict[str, Any]]) -> list[str]:
    manifest_paths = {n["path"] for n in nodes}
    errors: list[str] = []
    errors.extend(f"on disk but not in manifest: {p}" for p in sorted(on_disk - manifest_paths))
    errors.extend(f"in manifest but not on disk: {p}" for p in sorted(manifest_paths - on_disk))
    return errors


def check_parent_child(nodes: list[dict[str, Any]]) -> list[str]:
    paths = {n["path"] for n in nodes}
    by_path: dict[str, dict[str, Any]] = {n["path"]: n for n in nodes}
    errors: list[str] = []
    for node in nodes:
        parent = node.get("parent")
        if parent is not None and parent not in paths:
            errors.append(f"{node['path']}: parent '{parent}' not in manifest")
        for child in node.get("children", []):
            if child not in paths:
                errors.append(f"{node['path']}: child '{child}' not in manifest")
                continue
            if by_path[child].get("parent") != node["path"]:
                errors.append(f"{node['path']}: child '{child}' does not point back as parent")
    return errors


def check_reachability(nodes: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    roots = [n for n in nodes if n.get("parent") is None]
    if len(roots) != 1:
        errors.append(f"expected exactly 1 root node, found {len(roots)}")
        return errors
    by_path: dict[str, dict[str, Any]] = {n["path"]: n for n in nodes}
    seen: set[str] = set()
    stack: list[str] = [roots[0]["path"]]
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        stack.extend(by_path[current].get("children", []))
    unreachable = {n["path"] for n in nodes} - seen
    errors.extend(f"unreachable from root: {p}" for p in sorted(unreachable))
    return errors


def check_links(on_disk: set[str]) -> list[str]:
    errors: list[str] = []
    for rel in sorted(on_disk):
        path = ROOT / rel
        text = path.read_text()
        for match in LINK_RE.finditer(text):
            target = match.group(1)
            if target.startswith("http://") or target.startswith("https://"):
                continue
            target_no_anchor = target.split("#", 1)[0]
            if not target_no_anchor:
                continue
            resolved = (path.parent / target_no_anchor).resolve()
            if not resolved.exists():
                errors.append(f"{rel}: broken link -> {target}")
    return errors


def main() -> int:
    on_disk = discover_agents_md(ROOT)
    nodes = load_manifest()

    errors: list[str] = []
    errors.extend(check_coverage(on_disk, nodes))
    errors.extend(check_parent_child(nodes))
    errors.extend(check_reachability(nodes))
    errors.extend(check_links(on_disk))

    if errors:
        for msg in errors:
            print(f"graph_check: {msg}", file=sys.stderr)
        print(f"graph_check: FAILED ({len(errors)} error(s))", file=sys.stderr)
        return 1

    print(f"graph_check: OK ({len(nodes)} nodes, {len(on_disk)} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
