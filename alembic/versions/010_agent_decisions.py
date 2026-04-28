"""010 agent_decisions hypertable

Revision ID: 010
Revises: 009b
Create Date: 2026-04-26

Creates the ``agent_decisions`` TimescaleDB hypertable used by Phase 58
meta-agent (META-03). Mirrors migration 009 (alerts) ergonomics:

  - Composite PK ``(timestamp, id)`` for hypertable partitioning.
  - ``parent_decision_id UUID`` is a plain nullable column without a
    database-level self-FK. TimescaleDB hypertables forbid uniqueness
    constraints over only the id column (the unique must include the
    partition column ``timestamp``); the self-FK is therefore impractical
    and integrity is managed at the application layer (the runner only
    threads ``parent_decision_id`` after the parent row is persisted via
    the fire-and-forget envelope).
  - The ``metadata`` column is intentionally bare; the ORM Python attr is
    renamed to ``decision_metadata`` (SQLAlchemy ``DeclarativeBase``
    reserves the ``metadata`` attribute). See
    ``finalayze.core.models.MetaAgentDecisionModel``.

KEY PATTERN: bootstrap-safe ``CREATE TABLE IF NOT EXISTS`` plus
``create_hypertable(if_not_exists => TRUE, migrate_data => TRUE)``
mirroring migrations 008 and 009. The table may already exist (and may
already carry rows) in environments bootstrapped via
``Base.metadata.create_all()`` before this migration runs.

Indices (SPEC §Requirement 3):
  - ``(severity, timestamp DESC)`` — primary read pattern for status / dashboard.
  - ``(status, timestamp DESC) WHERE status IN ('queued','sent')`` — partial,
    accelerates the approve-expiry sweep that runs at the start of each tick.
  - ``(parent_decision_id) WHERE parent_decision_id IS NOT NULL`` — partial,
    mirrors alerts.parent_id index pattern.

Compression after 30 days segmented by ``severity``; retention 365 days.
"""

from __future__ import annotations

from alembic import op

revision: str = "010"
down_revision: str | None = "009b"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS agent_decisions (
            id UUID NOT NULL DEFAULT gen_random_uuid(),
            timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
            severity VARCHAR(15) NOT NULL,
            summary TEXT NOT NULL,
            rationale TEXT NOT NULL,
            actions JSONB NOT NULL DEFAULT '[]'::jsonb,
            outcome TEXT,
            dry_run BOOLEAN NOT NULL DEFAULT TRUE,
            metadata JSONB,
            parent_decision_id UUID,
            status VARCHAR(15) NOT NULL DEFAULT 'queued',
            created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
            PRIMARY KEY (timestamp, id)
        )
        """
    )
    op.execute(
        "SELECT create_hypertable('agent_decisions', 'timestamp', "
        "if_not_exists => TRUE, migrate_data => TRUE)"
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_agent_decisions_severity_ts
        ON agent_decisions (severity, timestamp DESC)
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_agent_decisions_status_ts
        ON agent_decisions (status, timestamp DESC)
        WHERE status IN ('queued', 'sent')
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_agent_decisions_parent
        ON agent_decisions (parent_decision_id)
        WHERE parent_decision_id IS NOT NULL
        """
    )
    op.execute(
        "ALTER TABLE agent_decisions SET ("
        "  timescaledb.compress, "
        "  timescaledb.compress_segmentby = 'severity'"
        ")"
    )
    op.execute(
        "SELECT add_compression_policy('agent_decisions', INTERVAL '30 days', "
        "if_not_exists => TRUE)"
    )
    op.execute(
        "SELECT add_retention_policy('agent_decisions', INTERVAL '365 days', if_not_exists => TRUE)"
    )


def downgrade() -> None:
    op.execute("SELECT remove_retention_policy('agent_decisions', if_exists => TRUE)")
    op.execute("SELECT remove_compression_policy('agent_decisions', if_exists => TRUE)")
    op.execute("DROP INDEX IF EXISTS ix_agent_decisions_parent")
    op.execute("DROP INDEX IF EXISTS ix_agent_decisions_status_ts")
    op.execute("DROP INDEX IF EXISTS ix_agent_decisions_severity_ts")
    op.execute("DROP TABLE IF EXISTS agent_decisions")
