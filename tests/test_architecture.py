"""Mechanical dependency-layer enforcement (audit 2026-06-28).

DEPENDENCY_LAYERS.md promised the 0->6 "imports flow downward only" invariant was
enforced, but no check existed (graph_check.py only validates the AGENTS.md graph
vs the manifest, not actual imports). This test closes that gap.

It is a RATCHET: a small, explicit baseline of pre-existing module-level upward
imports is allowed (all four are documented backward-compat ``sys.modules`` shims
in ``core/`` that re-export modules which physically moved to a higher layer), and
ANY new module-level upward import fails the test. Only true runtime imports count
-- imports inside ``if TYPE_CHECKING:`` blocks and function-local (deferred) imports
are excluded, because those are the sanctioned way to reference a higher layer
without a runtime dependency.

Layer map source of truth: ``.agents/manifest.jsonl`` (per-package ``layer``).
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src" / "finalayze"

# Pre-existing, sanctioned upward imports: backward-compat sys.modules shims in
# core/ that alias modules which moved to a higher layer. (source relpath, target pkg).
_BASELINE: frozenset[tuple[str, str]] = frozenset(
    {
        ("core/alerts.py", "api"),
        ("core/telegram_bot.py", "api"),
        ("core/bond_cycle.py", "orchestration"),
        ("core/trading_loop.py", "orchestration"),
    }
)

_CONFIG_PKG = "__config__"


def _layer_map() -> dict[str, int]:
    """Package -> layer from the manifest (the project's source of truth)."""
    mapping: dict[str, int] = {}
    for line in (_REPO_ROOT / ".agents" / "manifest.jsonl").read_text().splitlines():
        rec = json.loads(line)
        layer = rec.get("layer")
        path = rec.get("path", "")
        if layer is None:
            continue
        if path.startswith("src/finalayze/") and path.endswith("/AGENTS.md"):
            mapping[path[len("src/finalayze/") : -len("/AGENTS.md")]] = layer
        elif path == "config/AGENTS.md":
            mapping[_CONFIG_PKG] = layer
    return mapping


def _target_pkg(module: str) -> str | None:
    if module.startswith("finalayze."):
        return module.split(".")[1]
    if module == "config" or module.startswith("config."):
        return _CONFIG_PKG
    return None


def _module_level_imports(tree: ast.Module) -> list[str]:
    """Top-level imports only -- skip TYPE_CHECKING blocks and function-local imports."""
    targets: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            targets.extend(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            targets.append(node.module)
    return targets


def _current_violations() -> set[tuple[str, str]]:
    layer = _layer_map()
    violations: set[tuple[str, str]] = set()
    for py in _SRC.rglob("*.py"):
        rel = py.relative_to(_SRC)
        if len(rel.parts) < 2:  # noqa: PLR2004 — top-level finalayze/*.py has no package layer
            continue
        src_pkg = rel.parts[0]
        if src_pkg not in layer:
            continue
        src_layer = layer[src_pkg]
        for mod in _module_level_imports(ast.parse(py.read_text())):
            tgt = _target_pkg(mod)
            if tgt is None or tgt not in layer or tgt == src_pkg:
                continue
            if layer[tgt] > src_layer:
                violations.add((py.relative_to(_SRC).as_posix(), tgt))
    return violations


def test_no_new_upward_layer_imports() -> None:
    new = _current_violations() - _BASELINE
    assert not new, (
        "New module-level UPWARD imports violate the 0->6 dependency-layer invariant "
        f"(DEPENDENCY_LAYERS.md). Move the dependency down, defer it (function-local or "
        f"TYPE_CHECKING import), or invert it. Offending (file, target_package): {sorted(new)}"
    )


def test_baseline_shims_still_exist() -> None:
    # Keep the allow-list honest: if a baselined shim is deleted/fixed, drop it here.
    stale = {src for src, _ in _BASELINE if not (_SRC / src).exists()}
    assert not stale, f"Baselined files no longer exist; remove from _BASELINE: {sorted(stale)}"
