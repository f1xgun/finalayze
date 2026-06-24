"""Schema tests for Alembic migration 013_rebalance_persistence.py (Phase 82).

Static-analysis only: parses the migration with ``ast`` and greps the source for required SQL. No
live DB connection. Mirrors test_012_saa_persistence.py.
"""

from __future__ import annotations

import ast
from pathlib import Path

_MIGRATION_PATH = Path("alembic/versions/013_rebalance_persistence.py")


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


def test_revision_is_013() -> None:
    assert _extract(_read(), "revision") == "013"


def test_down_revision_is_012() -> None:
    assert _extract(_read(), "down_revision") == "012"


def test_creates_runs_table() -> None:
    assert "saa_rebalance_runs" in _read()


def test_creates_orders_table() -> None:
    assert "saa_rebalance_orders" in _read()


def test_orders_fk_cascades_to_runs() -> None:
    assert 'sa.ForeignKey("saa_rebalance_runs.id", ondelete="CASCADE")' in _read()


def test_runs_fk_restricts_to_portfolios() -> None:
    assert 'sa.ForeignKey("saa_portfolios.id", ondelete="RESTRICT")' in _read()


def test_migration_has_expected_column_types() -> None:
    """Lock the migration's column types byte-for-byte (L-04 parity; AH-02)."""
    src = _read()
    # saa_rebalance_runs
    assert "sa.Numeric(20, 2)" in src  # budget_rub
    assert "sa.Numeric(8, 4)" in src  # fill_rate
    assert "sa.Date()" in src  # as_of
    assert "sa.DateTime(timezone=True)" in src  # created_at
    # saa_rebalance_orders
    assert "sa.Numeric(28, 8)" in src  # requested_qty / filled_qty
    assert "sa.String(40)" in src  # symbol
    assert "sa.String(64)" in src  # client_order_id
    assert "sa.Text()" in src  # reason -- unbounded (CR-CORR-01)


def test_downgrade_drops_both_tables_child_first() -> None:
    src = _read()
    tree = ast.parse(src)
    downgrade_fn = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "downgrade"
    )
    downgrade_src = ast.unparse(downgrade_fn)
    assert "drop_table" in downgrade_src
    assert "saa_rebalance_orders" in downgrade_src
    assert "saa_rebalance_runs" in downgrade_src
    # child (orders) dropped before parent (runs)
    assert downgrade_src.index("saa_rebalance_orders") < downgrade_src.index("saa_rebalance_runs")
