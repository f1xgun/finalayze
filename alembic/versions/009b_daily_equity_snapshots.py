"""009b daily equity snapshots

Revision ID: 009b
Revises: 009
Create Date: 2026-04-19

Creates the `daily_equity_snapshots` TimescaleDB hypertable used by
Phase 56's per-cycle equity writer (`DailyReportingService`) and the
`/portfolio/performance` analytics endpoint.

KEY DEVIATION from migration 007: this migration uses raw
`CREATE TABLE IF NOT EXISTS` SQL (rather than `op.create_table`)
because the table is also declared by the SQLAlchemy ORM at
`src/finalayze/core/models.py:325-333` (DailyEquitySnapshot) and may
already exist in environments bootstrapped via
`Base.metadata.create_all()`. Without `IF NOT EXISTS`, the migration
would raise `relation "daily_equity_snapshots" already exists` in
those environments (Pitfall 3 in 56-RESEARCH.md).
"""

from __future__ import annotations

from alembic import op

revision: str = "009b"
down_revision: str | None = "009"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    # IF NOT EXISTS — table may already exist in environments bootstrapped via
    # Base.metadata.create_all() (Pitfall 3 in 56-RESEARCH.md).
    # Shape MUST match DailyEquitySnapshot ORM at core/models.py:325-333 exactly.
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS daily_equity_snapshots (
            timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
            market_id VARCHAR(20) NOT NULL,
            equity NUMERIC(14, 4) NOT NULL,
            currency VARCHAR(3) NOT NULL DEFAULT 'USD',
            PRIMARY KEY (timestamp, market_id)
        )
        """
    )
    # Hypertable conversion — idempotent via if_not_exists => TRUE (mirrors 007 line 36).
    # migrate_data => TRUE handles the case where Base.metadata.create_all() already
    # created the table AND the daily_reset writer populated it with rows; without this,
    # create_hypertable raises "table is not empty" (TimescaleDB FeatureNotSupported).
    # See Pitfall 3 in 56-RESEARCH.md (extended sub-case discovered during execution).
    op.execute(
        "SELECT create_hypertable('daily_equity_snapshots', 'timestamp', "
        "if_not_exists => TRUE, migrate_data => TRUE)"
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_daily_equity_snapshots_market_ts
        ON daily_equity_snapshots (market_id, timestamp DESC)
        """
    )
    # Compression policy after 30 days (Claude's discretion per CONTEXT D-04).
    # ~60-100 rows/day/market * 30 days = ~3000 rows/market uncompressed window.
    op.execute(
        "ALTER TABLE daily_equity_snapshots SET ("
        "  timescaledb.compress, "
        "  timescaledb.compress_segmentby = 'market_id'"
        ")"
    )
    op.execute(
        "SELECT add_compression_policy('daily_equity_snapshots', INTERVAL '30 days', "
        "if_not_exists => TRUE)"
    )
    # 365-day retention — matches stop_loss_events policy (alembic 007 line 43).
    op.execute(
        "SELECT add_retention_policy('daily_equity_snapshots', INTERVAL '365 days', "
        "if_not_exists => TRUE)"
    )


def downgrade() -> None:
    op.execute("SELECT remove_retention_policy('daily_equity_snapshots', if_exists => TRUE)")
    op.execute("SELECT remove_compression_policy('daily_equity_snapshots', if_exists => TRUE)")
    op.execute("DROP INDEX IF EXISTS ix_daily_equity_snapshots_market_ts")
    op.execute("DROP TABLE IF EXISTS daily_equity_snapshots")
