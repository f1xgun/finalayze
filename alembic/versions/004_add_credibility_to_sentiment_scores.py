"""004 add credibility to sentiment_scores

Revision ID: 004
Revises: 003
Create Date: 2026-04-14
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "004"
down_revision: str | None = "003"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.add_column(
        "sentiment_scores",
        sa.Column("credibility", sa.Numeric(5, 4), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("sentiment_scores", "credibility")
