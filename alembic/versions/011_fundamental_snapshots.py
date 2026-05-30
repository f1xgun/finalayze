"""011 fundamental snapshots hypertable

Revision ID: 011
Revises: 010
Create Date: 2026-05-30

Creates the ``fundamental_snapshots`` TimescaleDB hypertable used by Phase 59
(FUND-01) to store point-in-time fundamental snapshots per symbol. Each row
carries an ``as_of`` (publication/fetch) date so look-ahead control downstream
is a simple ``WHERE as_of <= :D ORDER BY as_of DESC LIMIT 1`` read.

KEY PATTERN (mirrors migrations 009b / 010): bootstrap-safe
``CREATE TABLE IF NOT EXISTS`` plus
``create_hypertable(if_not_exists => TRUE, migrate_data => TRUE)``. The table is
also declared by the SQLAlchemy ORM at
``src/finalayze/core/models.py`` (``FundamentalSnapshotModel``) and may already
exist — and may already carry rows — in environments bootstrapped via
``Base.metadata.create_all()``. Without ``IF NOT EXISTS`` the migration would
raise ``relation "fundamental_snapshots" already exists``; without
``migrate_data => TRUE`` ``create_hypertable`` would raise "table is not empty"
(Pitfall 3 in 59-RESEARCH.md).

Column SQL types MUST match the FundamentalSnapshotModel ORM Numeric precisions
byte-for-byte (shape parity).
"""

from __future__ import annotations

from alembic import op

revision: str = "011"
down_revision: str | None = "010"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    # IF NOT EXISTS — table may already exist in environments bootstrapped via
    # Base.metadata.create_all() (Pitfall 3 in 59-RESEARCH.md).
    # Shape MUST match FundamentalSnapshotModel ORM in core/models.py exactly.
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS fundamental_snapshots (
            as_of TIMESTAMP WITH TIME ZONE NOT NULL,
            symbol VARCHAR(30) NOT NULL,
            pe_ratio NUMERIC(14, 4),
            ev_ebitda NUMERIC(14, 4),
            revenue_ttm NUMERIC(20, 2),
            net_margin NUMERIC(10, 6),
            roe NUMERIC(10, 6),
            eps_ttm NUMERIC(14, 4),
            dividend_yield NUMERIC(10, 6),
            market_cap NUMERIC(20, 2),
            currency VARCHAR(3),
            PRIMARY KEY (as_of, symbol)
        )
        """
    )
    # Hypertable conversion — idempotent via if_not_exists => TRUE.
    # migrate_data => TRUE handles the case where Base.metadata.create_all()
    # already created the table AND populated rows; without this, create_hypertable
    # raises "table is not empty" (TimescaleDB FeatureNotSupported).
    op.execute(
        "SELECT create_hypertable('fundamental_snapshots', 'as_of', "
        "if_not_exists => TRUE, migrate_data => TRUE)"
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_fundamental_snapshots_symbol_asof
        ON fundamental_snapshots (symbol, as_of DESC)
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_fundamental_snapshots_symbol_asof")
    op.execute("DROP TABLE IF EXISTS fundamental_snapshots")
