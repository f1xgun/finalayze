"""006 sentiment ml continuous aggregate

Convert sentiment_scores to a TimescaleDB hypertable and create the
sentiment_7d_avg continuous aggregate with an hourly auto-refresh policy.

Revision ID: 006
Revises: 005
Create Date: 2026-04-15
"""

from __future__ import annotations

from alembic import op

revision: str = "006"
down_revision: str | None = "005"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    """Convert sentiment_scores to hypertable, create cagg, add refresh policy."""
    # Step 1: Convert sentiment_scores to a TimescaleDB hypertable.
    # migrate_data => TRUE preserves any existing rows.
    # if_not_exists => TRUE provides idempotent safety.
    op.execute(
        "SELECT create_hypertable('sentiment_scores', 'timestamp', "
        "migrate_data => TRUE, if_not_exists => TRUE)"
    )

    # Step 2: Create daily-bucket continuous aggregate over the hypertable.
    # Uses composite_sentiment (weighted combination of news + social).
    # WITH NO DATA defers initial materialization to the first policy execution.
    op.execute("""
        CREATE MATERIALIZED VIEW sentiment_7d_avg
        WITH (timescaledb.continuous) AS
        SELECT
            symbol,
            market_id,
            time_bucket(INTERVAL '1 day', "timestamp") AS bucket,
            AVG(composite_sentiment)::numeric(5,4)      AS avg_score,
            COUNT(*)                                     AS article_count
        FROM sentiment_scores
        GROUP BY symbol, market_id, time_bucket(INTERVAL '1 day', "timestamp")
        WITH NO DATA
    """)

    # Step 3: Add refresh policy — refresh last 30 days once per hour.
    # end_offset = '1 day' excludes the incomplete current bucket.
    op.execute("""
        SELECT add_continuous_aggregate_policy(
            'sentiment_7d_avg',
            start_offset    => INTERVAL '30 days',
            end_offset      => INTERVAL '1 day',
            schedule_interval => INTERVAL '1 hour',
            if_not_exists   => TRUE
        )
    """)


def downgrade() -> None:
    """Drop the continuous aggregate view.

    Must drop the dependent cagg before any hypertable cleanup.
    Hypertable-to-regular-table conversion is not natively supported;
    for dev/test environments with no data, this suffices.
    """
    op.execute("DROP MATERIALIZED VIEW IF EXISTS sentiment_7d_avg")
