"""Schema tests for Alembic migration 009_alerts.py (Phase 57-01, ALRT-03).

These tests are static-analysis only: they parse the migration file with `ast`
and grep the source for required SQL. No live DB connection required, so they
run as fast unit tests despite living under tests/integration/migrations/.

Live `alembic upgrade head` against a TimescaleDB instance is exercised by
tests/integration/test_alembic_upgrade.py — that suite picks up 009 once the
file lands and `upgrade head` is run against the integration DB.
"""

from __future__ import annotations

import ast
from pathlib import Path

_MIGRATION_PATH = Path("alembic/versions/009_alerts.py")


def _read_migration_source() -> str:
    return _MIGRATION_PATH.read_text(encoding="utf-8")


def _extract_module_assignment(source: str, name: str) -> object:
    """Parse the migration with `ast` and return the value of a module-level
    assignment (revision / down_revision / branch_labels / depends_on).
    """
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


def test_migration_revision_id_is_009() -> None:
    src = _read_migration_source()
    assert _extract_module_assignment(src, "revision") == "009"


def test_migration_down_revision_is_008() -> None:
    src = _read_migration_source()
    assert _extract_module_assignment(src, "down_revision") == "008"


def test_migration_upgrade_creates_hypertable() -> None:
    src = _read_migration_source()
    assert "create_hypertable('alerts', 'timestamp'" in src
    assert "if_not_exists => TRUE" in src
    assert "migrate_data => TRUE" in src


def test_migration_uses_create_table_if_not_exists() -> None:
    src = _read_migration_source()
    assert "CREATE TABLE IF NOT EXISTS alerts" in src


def test_migration_parent_id_is_plain_uuid_without_fk() -> None:
    """parent_id is a nullable UUID without a database-level self-FK.

    Phase 57 UAT gap-closure: TimescaleDB hypertables forbid UNIQUE
    constraints that exclude the partition column (timestamp), so a
    self-FK on alerts(id) is impractical — the FK requires UNIQUE on
    `id` alone, which conflicts with the hypertable rule. parent_id
    integrity is managed at the application layer instead (raw alerts
    persist before LLM follow-ups thread parent_id).

    Migration 009 must therefore declare parent_id as a plain nullable
    UUID with NO FOREIGN KEY clause and NO UNIQUE (id) constraint.
    """
    src = _read_migration_source()
    assert "parent_id UUID" in src, "parent_id must remain a nullable UUID column"
    assert "FOREIGN KEY (parent_id)" not in src, (
        "parent_id must NOT carry a database FK — TimescaleDB hypertable "
        "constraint conflict; integrity is managed at the application layer"
    )
    assert "UNIQUE (id)" not in src, (
        "alerts must NOT declare UNIQUE (id) — TimescaleDB rejects UNIQUE "
        "constraints that don't include the partition column (timestamp)"
    )


def test_migration_has_compression_and_retention() -> None:
    src = _read_migration_source()
    assert "add_compression_policy('alerts'" in src
    assert "add_retention_policy('alerts'" in src


def test_migration_downgrade_drops_table() -> None:
    """downgrade() function must contain DROP TABLE IF EXISTS alerts."""
    src = _read_migration_source()
    tree = ast.parse(src)
    downgrade_fn = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "downgrade"
    )
    downgrade_src = ast.unparse(downgrade_fn)
    assert "DROP TABLE IF EXISTS alerts" in downgrade_src
