"""008 add signal_price to signals

Revision ID: 008
Revises: 007
Create Date: 2026-04-18
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "008"
down_revision: str | None = "007"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.add_column(
        "signals",
        sa.Column("signal_price", sa.Numeric(12, 4), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("signals", "signal_price")
