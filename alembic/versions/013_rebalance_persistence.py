"""013 rebalance persistence -- saa_rebalance_runs + saa_rebalance_orders tables.

Revision ID: 013
Revises: 012
Create Date: 2026-06-24

Phase 82 (v11.2): persist each rebalance run + its per-leg order outcomes as an audit trail.
  1. saa_rebalance_runs: one row per real (submit) run -- plan id, mode, budget, reconciliation
     rollup (status + fill_rate).
  2. saa_rebalance_orders: one row per AUTO leg -- requested/filled qty, status, client_order_id.

Both PLAIN tables. Types MUST match the ORM byte-for-byte (shape parity):
  - money: Numeric(20, 2) (budget_rub); fill_rate: Numeric(8, 4)
  - quantities: Numeric(28, 8) (requested_qty/filled_qty)
  - timestamps: DateTime(timezone=True); as_of: Date (no tz)
  - FK saa_rebalance_runs.portfolio_id -> saa_portfolios.id ON DELETE RESTRICT
  - FK saa_rebalance_orders.run_id -> saa_rebalance_runs.id ON DELETE CASCADE
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "013"
down_revision: str = "012"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    """Create saa_rebalance_runs and saa_rebalance_orders tables."""
    # --- saa_rebalance_runs ---
    op.create_table(
        "saa_rebalance_runs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "portfolio_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("saa_portfolios.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("plan_id", sa.String(120), nullable=False),
        sa.Column("as_of", sa.Date(), nullable=False),
        sa.Column("mode", sa.String(12), nullable=False),
        sa.Column("budget_rub", sa.Numeric(20, 2), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("fill_rate", sa.Numeric(8, 4), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_saa_rebalance_runs_portfolio_created",
        "saa_rebalance_runs",
        ["portfolio_id", "created_at"],
    )

    # --- saa_rebalance_orders ---
    op.create_table(
        "saa_rebalance_orders",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "run_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("saa_rebalance_runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("asset_class", sa.String(12), nullable=False),
        sa.Column("symbol", sa.String(40), nullable=False),
        sa.Column("side", sa.String(4), nullable=False),
        sa.Column("requested_qty", sa.Numeric(28, 8), nullable=False),
        sa.Column("filled_qty", sa.Numeric(28, 8), nullable=False, server_default="0"),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("client_order_id", sa.String(64), nullable=False),
        sa.Column("reason", sa.String(255), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_saa_rebalance_orders_run_id", "saa_rebalance_orders", ["run_id"])


def downgrade() -> None:
    """Drop saa_rebalance_orders and saa_rebalance_runs tables (reverse FK order)."""
    op.drop_table("saa_rebalance_orders")
    op.drop_table("saa_rebalance_runs")
