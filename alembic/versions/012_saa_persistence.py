"""012 SAA persistence layer — saa_portfolios + deposit_tranches tables.

Revision ID: 012
Revises: 011
Create Date: 2026-06-23

Phase 77 (v11.2 W2): persist the two pieces of genuinely-mutable, non-reconstructable
user state so the SAA portfolio is reloadable:
  1. saa_portfolios: portfolio identity + risk choice (low-cardinality plain table).
  2. deposit_tranches: one mutable row per ladder rung, mirroring DepositTranche 1:1.

Both tables are PLAIN (NOT hypertables, NOT TimescaleDB). The migration creates
them symmetrically and downgrades by dropping in reverse FK order.

Types MUST match the ORM models byte-for-byte (shape parity):
  - money columns: Numeric(20, 2) for principal/accrued_net/accrued_gross/budget_rub
  - annual_rate: Numeric(8, 4) (e.g., 0.0420 for 4.20%)
  - date columns: Date (no timezone)
  - timestamps: DateTime(timezone=True)
  - risk_profile / bank_id: String
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "012"
down_revision: str = "011"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    """Create saa_portfolios and deposit_tranches tables."""
    # --- saa_portfolios ---
    op.create_table(
        "saa_portfolios",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("risk_profile", sa.String(12), nullable=False),
        sa.Column("budget_rub", sa.Numeric(20, 2), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default="true"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("deposit_accumulators", postgresql.JSONB(), nullable=True),
    )
    op.create_index("ix_saa_portfolios_is_active", "saa_portfolios", ["is_active"])

    # --- deposit_tranches ---
    op.create_table(
        "deposit_tranches",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "portfolio_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("saa_portfolios.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("principal", sa.Numeric(20, 2), nullable=False),
        sa.Column("term_months", sa.Integer(), nullable=False),
        sa.Column("annual_rate", sa.Numeric(8, 4), nullable=False),
        sa.Column("open_date", sa.Date(), nullable=False),
        sa.Column("maturity_date", sa.Date(), nullable=False),
        sa.Column("accrued_net", sa.Numeric(20, 2), nullable=False, server_default="0"),
        sa.Column("accrued_gross", sa.Numeric(20, 2), nullable=False, server_default="0"),
        sa.Column("broken", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("bank_id", sa.String(50), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_deposit_tranches_portfolio_id", "deposit_tranches", ["portfolio_id"])


def downgrade() -> None:
    """Drop deposit_tranches and saa_portfolios tables (reverse FK order)."""
    op.drop_table("deposit_tranches")
    op.drop_table("saa_portfolios")
