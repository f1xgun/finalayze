"""Unit tests for migration 011 + FundamentalSnapshotModel ORM (Phase 59, FUND-01).

Verifies the bootstrap-safe hypertable migration and that the ORM column set
matches the migration SQL byte-for-byte (shape parity, RESEARCH Pitfall 3).
These tests do not require a live DB — they inspect the module source + ORM.
"""

from __future__ import annotations

import importlib
import inspect

from finalayze.core.models import FundamentalSnapshotModel

_MIGRATION_MODULE = "alembic.versions.011_fundamental_snapshots"

_EXPECTED_COLUMNS = {
    "as_of",
    "symbol",
    "pe_ratio",
    "ev_ebitda",
    "revenue_ttm",
    "net_margin",
    "roe",
    "eps_ttm",
    "dividend_yield",
    "market_cap",
    "currency",
}


def _migration_source() -> str:
    module = importlib.import_module(_MIGRATION_MODULE)
    return inspect.getsource(module)


class TestMigration011Revision:
    """Revision chaining."""

    def test_revision_and_down_revision(self) -> None:
        module = importlib.import_module(_MIGRATION_MODULE)
        assert module.revision == "011"
        assert module.down_revision == "010"


class TestMigration011BootstrapSafe:
    """Idempotent DDL on a create_all()-bootstrapped DB."""

    def test_upgrade_sql_is_bootstrap_safe(self) -> None:
        src = _migration_source()
        assert "CREATE TABLE IF NOT EXISTS fundamental_snapshots" in src
        assert "create_hypertable" in src
        assert "if_not_exists => TRUE" in src
        assert "migrate_data => TRUE" in src

    def test_primary_key_carries_partition_column(self) -> None:
        src = _migration_source()
        assert "PRIMARY KEY (as_of, symbol)" in src


class TestShapeParity:
    """ORM column set must match the migration's column set exactly."""

    def test_tablename(self) -> None:
        assert FundamentalSnapshotModel.__tablename__ == "fundamental_snapshots"

    def test_orm_columns_match_expected(self) -> None:
        orm_columns = set(FundamentalSnapshotModel.__table__.columns.keys())
        assert orm_columns == _EXPECTED_COLUMNS

    def test_composite_primary_key(self) -> None:
        pk_columns = {col.name for col in FundamentalSnapshotModel.__table__.primary_key}
        assert pk_columns == {"as_of", "symbol"}
