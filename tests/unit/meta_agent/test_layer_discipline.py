"""Layer-6 import-graph discipline test (Phase 58-05, SPEC AC #19).

SPEC §Constraints (line 112):
  > Layer: Module sits at Layer 6 (over orchestration). May import from
  > core/, api/, orchestration/. Must not be imported by any module at
  > Layer ≤ 5.

This test enforces the rule by walking every Python file under
``src/finalayze/{core,config,markets,data,risk,execution,strategies,
ml,backtest,analysis,monitoring,orchestration}/`` and asserting that
NO ``import`` / ``from ... import`` statement targets
``finalayze.meta_agent.*``.

Allowed importers (Layer 6+):
  - ``src/finalayze/api/``
  - ``src/finalayze/dashboard/``
  - ``src/finalayze/meta_agent/`` (intra-module)
  - ``src/finalayze/bootstrap.py`` / ``src/finalayze/main.py`` (wiring)

The well-known SPEC-allowed exception is ``orchestration/trading_loop.py``
which lazy-imports ``finalayze.meta_agent.scheduler.register_meta_agent_job``
inside ``TradingLoop.start()`` — this is a runtime late import (NOT a
module-level import) used only when ``settings.meta_agent_enabled=True``.
The test allows this single late import via an allow-list (the
``register_meta_agent_job`` symbol) so the wiring path stays valid.

Implementation: pure-Python ``ast`` walk; runs in CI without depending
on the ``ast-index`` binary.
"""

from __future__ import annotations

import ast
from pathlib import Path

# ── Module-level constants (PLR2004; magic-number-free in source) ────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_SRC_ROOT = _PROJECT_ROOT / "src" / "finalayze"

# Layer ≤ 5 directories per docs/architecture/DEPENDENCY_LAYERS.md.
# meta_agent / api / dashboard live at Layer 6 and are excluded.
_LAYER_LE_5_DIRS = (
    "core",
    "config",
    "markets",
    "data",
    "risk",
    "execution",
    "strategies",
    "ml",
    "backtest",
    "analysis",
    "monitoring",
    "orchestration",
)

# Allow-list: SPEC-allowed late-binding imports for known wiring points.
# Format: {(relative_file_path, imported_name) -> rationale}.
# trading_loop.py imports register_meta_agent_job inside start() to wire
# the cron job — this is the single SPEC-allowed Layer-5 importer per
# the Phase 58 PLAN body (Hand-off integration point) and CONTEXT D-12.
_ALLOWED_LATE_IMPORTS: set[tuple[str, str]] = {
    ("orchestration/trading_loop.py", "finalayze.meta_agent.scheduler"),
}

_META_AGENT_PREFIX = "finalayze.meta_agent"


def _iter_py_files() -> list[Path]:
    """Walk every .py file under the Layer-≤-5 directories."""
    files: list[Path] = []
    for layer_dir in _LAYER_LE_5_DIRS:
        root = _SRC_ROOT / layer_dir
        if not root.exists():
            continue
        files.extend(root.rglob("*.py"))
    return files


def _imports_in_file(path: Path) -> list[tuple[str, int]]:
    """Return [(imported_module, line_no), ...] for any meta_agent imports.

    Catches both:
      - ``import finalayze.meta_agent.X``
      - ``from finalayze.meta_agent[.X] import Y``
    """
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
    except (SyntaxError, UnicodeDecodeError):
        return []

    hits: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            hits.extend(
                (alias.name, node.lineno)
                for alias in node.names
                if alias.name == _META_AGENT_PREFIX
                or alias.name.startswith(_META_AGENT_PREFIX + ".")
            )
        elif isinstance(node, ast.ImportFrom):
            if node.module is None:
                continue
            if node.module == _META_AGENT_PREFIX or node.module.startswith(
                _META_AGENT_PREFIX + ".",
            ):
                hits.append((node.module, node.lineno))
    return hits


def test_no_layer_5_or_below_imports_meta_agent() -> None:
    """SPEC AC #19: no module at Layer ≤ 5 imports ``finalayze.meta_agent.*``.

    Allowed exception: ``orchestration/trading_loop.py`` may late-import
    ``finalayze.meta_agent.scheduler.register_meta_agent_job`` inside
    ``TradingLoop.start()`` — this is the SPEC-allowed wiring point per
    CONTEXT D-12 (Plan 58-05 PLAN body Hand-off Integration Points).
    """
    violations: list[str] = []

    for path in _iter_py_files():
        rel = path.relative_to(_SRC_ROOT).as_posix()
        for imported, lineno in _imports_in_file(path):
            if (rel, imported) in _ALLOWED_LATE_IMPORTS:
                continue
            violations.append(
                f"  {rel}:{lineno} imports {imported}",
            )

    assert not violations, (
        "Layer ≤ 5 modules must NOT import finalayze.meta_agent.* "
        "(SPEC AC #19). Violations:\n" + "\n".join(violations)
    )


def test_meta_agent_module_exists_and_is_importable() -> None:
    """Sanity check: the meta_agent module exists and the discipline test
    is meaningful (not silently passing because there is nothing to scan).
    """
    import finalayze.meta_agent  # noqa: F401

    assert (_SRC_ROOT / "meta_agent" / "__init__.py").exists()


def test_layer_discipline_scan_visits_orchestration_and_core() -> None:
    """Sanity check: the scanner actually visits the Layer ≤ 5 directories
    we expect (otherwise a misconfigured _LAYER_LE_5_DIRS would silently
    pass the discipline test).
    """
    files = _iter_py_files()
    suffixes = {p.relative_to(_SRC_ROOT).as_posix() for p in files}
    assert any(s.startswith("orchestration/") for s in suffixes), (
        "scan must visit orchestration/ — otherwise the discipline test "
        "would not catch a trading_loop.py violation."
    )
    assert any(s.startswith("core/") for s in suffixes), (
        "scan must visit core/ — otherwise the discipline test would not catch a core/ violation."
    )
