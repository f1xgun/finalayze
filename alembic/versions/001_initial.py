"""Initial migration: create all tables.

Revision ID: 001
Revises:
Create Date: 2026-02-22
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

if TYPE_CHECKING:
    from collections.abc import Sequence

# revision identifiers, used by Alembic.
revision: str = "001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create markets, segments, instruments, candles, signals, orders tables."""
    # --- markets ---
    op.create_table(
        "markets",
        sa.Column("id", sa.String(10), primary_key=True),
        sa.Column("name", sa.String(50), nullable=False),
        sa.Column("currency", sa.String(3), nullable=False),
        sa.Column("timezone", sa.String(30), nullable=False),
        sa.Column("open_time", sa.Time(), nullable=False),
        sa.Column("close_time", sa.Time(), nullable=False),
    )

    # --- segments ---
    op.create_table(
        "segments",
        sa.Column("id", sa.String(30), primary_key=True),
        sa.Column(
            "market_id",
            sa.String(10),
            sa.ForeignKey("markets.id"),
            nullable=False,
        ),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("active_strategies", postgresql.ARRAY(sa.Text()), nullable=True),
        sa.Column(
            "strategy_params",
            postgresql.JSONB(),
            nullable=True,
            server_default="{}",
        ),
        sa.Column("ml_model_id", sa.String(50), nullable=True),
        sa.Column(
            "max_allocation_pct",
            sa.Numeric(5, 4),
            nullable=False,
            server_default="0.30",
        ),
        sa.Column("news_languages", postgresql.ARRAY(sa.Text()), nullable=True),
    )

    # --- instruments ---
    op.create_table(
        "instruments",
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column(
            "market_id",
            sa.String(10),
            sa.ForeignKey("markets.id"),
            nullable=False,
        ),
        sa.Column(
            "segment_id",
            sa.String(30),
            sa.ForeignKey("segments.id"),
            nullable=True,
        ),
        sa.Column("name", sa.String(200), nullable=True),
        sa.Column("figi", sa.String(20), nullable=True),
        sa.Column("instrument_type", sa.String(20), nullable=True),
        sa.Column("currency", sa.String(3), nullable=True),
        sa.Column("lot_size", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default="true"),
        sa.PrimaryKeyConstraint("symbol", "market_id"),
    )

    # --- candles ---
    op.create_table(
        "candles",
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("market_id", sa.String(10), sa.ForeignKey("markets.id"), nullable=False),
        sa.Column("timeframe", sa.String(5), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("open", sa.Numeric(12, 4), nullable=False),
        sa.Column("high", sa.Numeric(12, 4), nullable=False),
        sa.Column("low", sa.Numeric(12, 4), nullable=False),
        sa.Column("close", sa.Numeric(12, 4), nullable=False),
        sa.Column("volume", sa.BigInteger(), nullable=False),
        sa.Column("source", sa.String(20), nullable=True),
        sa.PrimaryKeyConstraint("symbol", "market_id", "timeframe", "timestamp"),
    )

    # Convert candles to TimescaleDB hypertable
    op.execute("SELECT create_hypertable('candles', 'timestamp', migrate_data => true)")

    # --- signals ---
    op.create_table(
        "signals",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
        ),
        sa.Column("strategy_name", sa.String(50), nullable=False),
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("market_id", sa.String(10), nullable=False),
        sa.Column("segment_id", sa.String(30), nullable=False),
        sa.Column("direction", sa.String(4), nullable=False),
        sa.Column("confidence", sa.Numeric(5, 4), nullable=False),
        sa.Column("features", postgresql.JSONB(), nullable=True),
        sa.Column("reasoning", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("mode", sa.String(10), nullable=True),
    )

    # --- orders ---
    op.create_table(
        "orders",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
        ),
        sa.Column(
            "signal_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("signals.id"),
            nullable=True,
        ),
        sa.Column("broker", sa.String(20), nullable=False),
        sa.Column("broker_order_id", sa.String(100), nullable=True),
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("market_id", sa.String(10), nullable=False),
        sa.Column("side", sa.String(4), nullable=False),
        sa.Column("order_type", sa.String(20), nullable=False),
        sa.Column("quantity", sa.Numeric(12, 4), nullable=False),
        sa.Column("limit_price", sa.Numeric(12, 4), nullable=True),
        sa.Column("stop_price", sa.Numeric(12, 4), nullable=True),
        sa.Column("currency", sa.String(3), nullable=True),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column(
            "filled_quantity",
            sa.Numeric(12, 4),
            nullable=False,
            server_default="0",
        ),
        sa.Column("filled_avg_price", sa.Numeric(12, 4), nullable=True),
        sa.Column("submitted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("filled_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("risk_checks", postgresql.JSONB(), nullable=True),
        sa.Column("mode", sa.String(10), nullable=True),
    )


def downgrade() -> None:
    """Drop all tables in reverse order."""
    op.drop_table("orders")
    op.drop_table("signals")
    op.drop_table("candles")
    op.drop_table("instruments")
    op.drop_table("segments")
    op.drop_table("markets")
