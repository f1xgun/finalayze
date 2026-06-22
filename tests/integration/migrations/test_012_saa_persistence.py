"""Schema tests for Alembic migration 012_saa_persistence.py (Phase 77).

Static-analysis only: parses the migration file with ``ast`` and greps
the source for required SQL. No live DB connection — runs as a fast unit
test despite living under tests/integration/migrations/.

Mirrors tests/integration/migrations/test_010_agent_decisions.py line-for-line.
"""

from __future__ import annotations

import ast
from pathlib import Path

_MIGRATION_PATH = Path("alembic/versions/012_saa_persistence.py")


def _read_migration_source() -> str:
    return _MIGRATION_PATH.read_text(encoding="utf-8")


def _extract_module_assignment(source: str, name: str) -> object:
    """Return the literal value of a module-level assignment."""
    tree = ast.parse(source)
    for node in tree.body:
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == name
            and node.value is not None
        ):
            return ast.literal_eval(node.value)
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return ast.literal_eval(node.value)
    msg = f"module-level assignment {name!r} not found"
    raise AssertionError(msg)


def test_migration_revision_id_is_012() -> None:
    src = _read_migration_source()
    assert _extract_module_assignment(src, "revision") == "012"


def test_migration_down_revision_is_011() -> None:
    src = _read_migration_source()
    assert _extract_module_assignment(src, "down_revision") == "011"


def test_migration_upgrade_creates_saa_portfolios_table() -> None:
    src = _read_migration_source()
    assert "saa_portfolios" in src


def test_migration_upgrade_creates_deposit_tranches_table() -> None:
    src = _read_migration_source()
    assert "deposit_tranches" in src


def test_migration_uses_foreign_key_to_saa_portfolios() -> None:
    src = _read_migration_source()
    assert 'sa.ForeignKey("saa_portfolios.id"' in src


def test_migration_downgrade_drops_both_tables() -> None:
    src = _read_migration_source()
    tree = ast.parse(src)
    downgrade_fn = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "downgrade"
    )
    downgrade_src = ast.unparse(downgrade_fn)
    assert "drop_table" in downgrade_src
    assert "deposit_tranches" in downgrade_src
    assert "saa_portfolios" in downgrade_src
