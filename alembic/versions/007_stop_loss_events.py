"""007 stop-loss events

Revision ID: 007
Revises: 006
Create Date: 2026-04-18
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "007"
down_revision: str | None = "006"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.create_table(
        "stop_loss_events",
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("symbol", sa.String(30), nullable=False),
        sa.Column("market_id", sa.String(20), nullable=False),
        sa.Column("event_type", sa.String(20), nullable=False),
        sa.Column("entry_price", sa.Numeric(14, 4), nullable=True),
        sa.Column("current_stop", sa.Numeric(14, 4), nullable=True),
        sa.Column("highest_price", sa.Numeric(14, 4), nullable=True),
        sa.Column("atr_value", sa.Numeric(14, 4), nullable=True),
        sa.Column("activation_atr", sa.Numeric(6, 4), nullable=True),
        sa.Column("trail_atr", sa.Numeric(6, 4), nullable=True),
        sa.Column("trail_activated", sa.Boolean, nullable=True),
        sa.Column("current_price", sa.Numeric(14, 4), nullable=True),
        sa.PrimaryKeyConstraint("timestamp", "symbol", "market_id"),
    )
    op.execute("SELECT create_hypertable('stop_loss_events', 'timestamp', if_not_exists => TRUE)")
    op.create_index(
        "ix_stop_loss_events_symbol_ts",
        "stop_loss_events",
        ["symbol", sa.text("timestamp DESC")],
    )
    op.execute(
        "SELECT add_retention_policy('stop_loss_events', INTERVAL '365 days', "
        "if_not_exists => TRUE)"
    )
    op.execute(
        "ALTER TABLE stop_loss_events SET ("
        "  timescaledb.compress, "
        "  timescaledb.compress_segmentby = 'symbol, market_id'"
        ")"
    )
    op.execute(
        "SELECT add_compression_policy('stop_loss_events', INTERVAL '7 days', "
        "if_not_exists => TRUE)"
    )


def downgrade() -> None:
    op.execute("SELECT remove_compression_policy('stop_loss_events', if_exists => TRUE)")
    op.execute("SELECT remove_retention_policy('stop_loss_events', if_exists => TRUE)")
    op.drop_index("ix_stop_loss_events_symbol_ts", table_name="stop_loss_events")
    op.drop_table("stop_loss_events")
