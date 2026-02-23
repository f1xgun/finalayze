"""Add news_articles and sentiment_scores tables.

Revision ID: 002
Revises: 001
Create Date: 2026-02-23
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

if TYPE_CHECKING:
    from collections.abc import Sequence

revision: str = "002"
down_revision: str | None = "001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create news_articles and sentiment_scores tables."""
    op.create_table(
        "news_articles",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("source", sa.String(50), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column("content", sa.Text(), nullable=True),
        sa.Column("url", sa.Text(), nullable=True),
        sa.Column("language", sa.String(5), nullable=False, server_default="en"),
        sa.Column("published_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column(
            "symbols",
            postgresql.ARRAY(sa.String(20)),
            nullable=False,
            server_default="{}",
        ),
        sa.Column(
            "affected_segments",
            postgresql.ARRAY(sa.String(30)),
            nullable=False,
            server_default="{}",
        ),
        sa.Column("scope", sa.String(20), nullable=True),
        sa.Column("category", sa.String(30), nullable=True),
        sa.Column("raw_sentiment", sa.Numeric(5, 4), nullable=True),
        sa.Column("credibility_score", sa.Numeric(5, 4), nullable=True),
        sa.Column("llm_analysis", postgresql.JSONB(), nullable=True),
        sa.Column("is_processed", sa.Boolean(), nullable=False, server_default="false"),
    )

    op.create_table(
        "sentiment_scores",
        sa.Column("symbol", sa.String(20), nullable=False, primary_key=True),
        sa.Column("market_id", sa.String(10), nullable=False, primary_key=True),
        sa.Column(
            "timestamp",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            primary_key=True,
        ),
        sa.Column("news_sentiment", sa.Numeric(5, 4), nullable=True),
        sa.Column("social_sentiment", sa.Numeric(5, 4), nullable=True),
        sa.Column("composite_sentiment", sa.Numeric(5, 4), nullable=True),
        sa.Column("confidence", sa.Numeric(5, 4), nullable=True),
    )


def downgrade() -> None:
    """Drop news tables."""
    op.drop_table("sentiment_scores")
    op.drop_table("news_articles")
