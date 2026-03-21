"""005 sandbox metrics

Revision ID: 005
Revises: 004
Create Date: 2026-03-21
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "005"
down_revision: str | None = "004"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.create_table(
        "sandbox_metrics",
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("market_id", sa.String(10), nullable=False),
        sa.Column("trade_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("pnl_rub", sa.Numeric(14, 4), nullable=True),
        sa.Column("equity_rub", sa.Numeric(14, 4), nullable=False),
        sa.Column("fill_rate", sa.Numeric(5, 4), nullable=True),
        sa.Column("uptime_cycles", sa.Integer, nullable=False, server_default="0"),
        sa.Column("signals_generated", sa.Integer, nullable=False, server_default="0"),
        sa.Column("errors_caught", sa.Integer, nullable=False, server_default="0"),
        sa.Column("max_slippage_bps", sa.Numeric(8, 2), nullable=True),
        sa.Column("avg_slippage_bps", sa.Numeric(8, 2), nullable=True),
        sa.Column("drawdown_pct", sa.Numeric(7, 4), nullable=True),
        sa.PrimaryKeyConstraint("timestamp", "market_id"),
    )
    op.execute("SELECT create_hypertable('sandbox_metrics', 'timestamp', if_not_exists => TRUE)")


def downgrade() -> None:
    op.drop_table("sandbox_metrics")
