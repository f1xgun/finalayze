"""Schema test for Alembic migration 014_rebalance_reason_text.py (Phase 84).

Static-analysis only (ast + source grep); no live DB. Mirrors test_013_rebalance_persistence.py.
"""

from __future__ import annotations

import ast
from pathlib import Path

_MIGRATION_PATH = Path("alembic/versions/014_rebalance_reason_text.py")


def _read() -> str:
    return _MIGRATION_PATH.read_text(encoding="utf-8")


def _extract(source: str, name: str) -> object:
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


def test_revision_is_014() -> None:
    assert _extract(_read(), "revision") == "014"


def test_down_revision_is_013() -> None:
    assert _extract(_read(), "down_revision") == "013"


def test_upgrade_alters_reason_to_text() -> None:
    src = _read()
    assert "alter_column" in src
    assert '"saa_rebalance_orders"' in src
    assert '"reason"' in src
    assert "sa.Text()" in src
