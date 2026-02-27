"""003 portfolio snapshots

Revision ID: 003
Revises: 002
Create Date: 2026-02-27
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "003"
down_revision = "002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "portfolio_snapshots",
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("market_id", sa.String(10), nullable=False),
        sa.Column("equity", sa.Numeric(14, 4), nullable=True),
        sa.Column("cash", sa.Numeric(14, 4), nullable=True),
        sa.Column("daily_pnl", sa.Numeric(14, 4), nullable=True),
        sa.Column("drawdown_pct", sa.Float(), nullable=True),
        sa.Column("mode", sa.String(10), nullable=True),
        sa.PrimaryKeyConstraint("timestamp", "market_id"),
    )
    op.execute(
        "SELECT create_hypertable('portfolio_snapshots', 'timestamp', if_not_exists => TRUE)"
    )


def downgrade() -> None:
    op.drop_table("portfolio_snapshots")
