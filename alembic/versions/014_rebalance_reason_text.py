"""014 widen saa_rebalance_orders.reason to Text.

Revision ID: 014
Revises: 013
Create Date: 2026-06-24

CR-CORR-01 follow-up: migration 013 originally created ``reason`` as VARCHAR(255). A broker
rejection reason (an arbitrary gRPC/exchange error string) can exceed that, and on Postgres an
overflow RAISES (StringDataRightTruncation) and rolls back the WHOLE single-transaction audit
persist -- losing the run record, precisely in the FAILED-leg case audit exists for (surfaced by the
sandbox cert). The ORM already declares ``Text``; editing 013 in place did NOT fix a DB that already
applied the VARCHAR(255) form, so this forward migration ALTERs it. Idempotent: a no-op on a DB
whose 013 already created ``reason`` as Text.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "014"
down_revision: str = "013"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    """Widen saa_rebalance_orders.reason from VARCHAR(255) to Text (unbounded)."""
    op.alter_column(
        "saa_rebalance_orders",
        "reason",
        type_=sa.Text(),
        existing_type=sa.String(255),
        existing_nullable=True,
    )


def downgrade() -> None:
    """Narrow saa_rebalance_orders.reason back to VARCHAR(255) (may fail on long values)."""
    op.alter_column(
        "saa_rebalance_orders",
        "reason",
        type_=sa.String(255),
        existing_type=sa.Text(),
        existing_nullable=True,
    )
