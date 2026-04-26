"""Schema tests for Alembic migration 010_agent_decisions.py (Phase 58-01).

Static-analysis only: parses the migration file with ``ast`` and greps
the source for required SQL. No live DB connection — runs as a fast unit
test despite living under tests/integration/migrations/.

Mirrors tests/integration/migrations/test_009_alerts.py line-for-line.
"""

from __future__ import annotations

import ast
from pathlib import Path

_MIGRATION_PATH = Path("alembic/versions/010_agent_decisions.py")


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


def test_migration_revision_id_is_010() -> None:
    src = _read_migration_source()
    assert _extract_module_assignment(src, "revision") == "010"


def test_migration_down_revision_is_009() -> None:
    src = _read_migration_source()
    assert _extract_module_assignment(src, "down_revision") == "009"


def test_migration_upgrade_creates_hypertable() -> None:
    src = _read_migration_source()
    assert "create_hypertable('agent_decisions', 'timestamp'" in src
    assert "if_not_exists => TRUE" in src
    assert "migrate_data => TRUE" in src


def test_migration_uses_create_table_if_not_exists() -> None:
    src = _read_migration_source()
    assert "CREATE TABLE IF NOT EXISTS agent_decisions" in src


def test_migration_parent_decision_id_is_plain_uuid_without_fk() -> None:
    """parent_decision_id is nullable UUID without a database self-FK.

    TimescaleDB hypertables forbid the UNIQUE (id) constraint that a
    self-FK would require — same constraint as alerts.parent_id (see
    migration 009 docstring lines 38-49 and test_009_alerts.py).
    """
    src = _read_migration_source()
    assert "parent_decision_id UUID" in src
    assert "FOREIGN KEY (parent_decision_id)" not in src
    assert "UNIQUE (id)" not in src


def test_migration_renames_metadata_column_at_orm_level() -> None:
    """The DB column is the bare ``metadata``; the ORM Python attr is
    ``decision_metadata`` (rename done in core/models.py)."""
    src = _read_migration_source()
    assert "metadata JSONB" in src


def test_migration_has_compression_and_retention() -> None:
    src = _read_migration_source()
    assert "add_compression_policy('agent_decisions'" in src
    assert "add_retention_policy('agent_decisions'" in src
    assert "compress_segmentby = 'severity'" in src


def test_migration_downgrade_drops_table() -> None:
    src = _read_migration_source()
    tree = ast.parse(src)
    downgrade_fn = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "downgrade"
    )
    downgrade_src = ast.unparse(downgrade_fn)
    assert "DROP TABLE IF EXISTS agent_decisions" in downgrade_src
