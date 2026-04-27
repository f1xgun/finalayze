"""009 alerts table

Revision ID: 009
Revises: 008
Create Date: 2026-04-19

Creates the `alerts` TimescaleDB hypertable used by Phase 57's
TelegramAlerter write hook (`_send` / `_send_sync`), the
`/api/v1/alerts` paginated endpoint, and the `/alerts` Streamlit page.

KEY PATTERN: raw `CREATE TABLE IF NOT EXISTS` plus
`create_hypertable(if_not_exists => TRUE, migrate_data => TRUE)` mirroring
migration 008 (per the 2026-04-19 lesson on `daily_equity_snapshots`):
the table is also declared by the SQLAlchemy ORM at
`src/finalayze/core/models.py` (AlertModel) and may already exist + carry
rows in environments bootstrapped via `Base.metadata.create_all()` before
this migration runs.

The `parent_id` self-FK with `ON DELETE SET NULL` supports the anomaly
raw + LLM follow-up two-row schema (D-04). On orphaned-child FK race
(Pitfall 2 in 57-RESEARCH.md), the child row still persists with
`parent_id=NULL` rather than being dropped.
"""

from __future__ import annotations

from alembic import op

revision: str = "009"
down_revision: str | None = "008"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    # IF NOT EXISTS — defensive against Base.metadata.create_all() bootstrap
    # (per 2026-04-19 lesson on migration 008 daily_equity_snapshots).
    # parent_id is a plain nullable UUID — NO database-level FK to alerts(id).
    # Two TimescaleDB constraints make a real self-FK impractical:
    #   1. Hypertables forbid UNIQUE constraints that don't include the
    #      partition column (timestamp), so we can't put UNIQUE on `id`.
    #   2. Without UNIQUE on `id`, Postgres can't accept FOREIGN KEY
    #      (parent_id) REFERENCES alerts(id).
    # parent_id integrity is managed at the application layer: the anomaly
    # path threads parent_id only after the raw alert has been persisted
    # (TelegramAlerter._send returns the alert_id only AFTER successful
    # write). On any insert failure the persist envelope swallows + logs;
    # alerts is an audit log with 365-day retention, so dangling parent_id
    # references are cosmetic — the dashboard /alerts page tolerates orphans.
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS alerts (
            id UUID NOT NULL DEFAULT gen_random_uuid(),
            timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
            alert_type VARCHAR(30) NOT NULL,
            priority VARCHAR(10) NOT NULL,
            symbol VARCHAR(30),
            market_id VARCHAR(20),
            message TEXT NOT NULL,
            parent_id UUID,
            delivery_status VARCHAR(10) NOT NULL DEFAULT 'queued',
            metadata JSONB,
            PRIMARY KEY (timestamp, id)
        )
        """
    )
    op.execute(
        "SELECT create_hypertable('alerts', 'timestamp', "
        "if_not_exists => TRUE, migrate_data => TRUE)"
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_alerts_type_ts
        ON alerts (alert_type, timestamp DESC)
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_alerts_symbol_ts
        ON alerts (symbol, timestamp DESC) WHERE symbol IS NOT NULL
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_alerts_parent
        ON alerts (parent_id) WHERE parent_id IS NOT NULL
        """
    )
    op.execute(
        "ALTER TABLE alerts SET ("
        "  timescaledb.compress, "
        "  timescaledb.compress_segmentby = 'alert_type'"
        ")"
    )
    op.execute("SELECT add_compression_policy('alerts', INTERVAL '30 days', if_not_exists => TRUE)")
    op.execute("SELECT add_retention_policy('alerts', INTERVAL '365 days', if_not_exists => TRUE)")


def downgrade() -> None:
    op.execute("SELECT remove_retention_policy('alerts', if_exists => TRUE)")
    op.execute("SELECT remove_compression_policy('alerts', if_exists => TRUE)")
    op.execute("DROP INDEX IF EXISTS ix_alerts_parent")
    op.execute("DROP INDEX IF EXISTS ix_alerts_symbol_ts")
    op.execute("DROP INDEX IF EXISTS ix_alerts_type_ts")
    op.execute("DROP TABLE IF EXISTS alerts")
