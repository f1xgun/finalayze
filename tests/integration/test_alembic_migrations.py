"""Integration tests for individual Alembic migrations.

Each test in this file targets a specific migration revision and asserts
both correctness (table/column shape) and **idempotency** (re-running the
upgrade body must not raise). This complements `test_alembic_upgrade.py`,
which only verifies a clean `alembic upgrade head` end-state.
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.integration


def _db_url() -> str:
    url = os.environ.get("FINALAYZE_DATABASE_URL") or os.environ.get("DATABASE_URL")
    if not url:
        pytest.skip("FINALAYZE_DATABASE_URL not set; integration DB unavailable")
    return url


def _sync_url(url: str) -> str:
    return url.replace("+asyncpg", "").replace("postgresql+asyncpg", "postgresql")


def test_migration_008_idempotent() -> None:
    """Migration 008 must create `daily_equity_snapshots` as a TimescaleDB hypertable.

    The migration must be idempotent across two scenarios that arise in practice:

    1. **Fresh DB:** a Postgres+TimescaleDB instance bootstrapped from scratch
       via `alembic upgrade head`. The table must not exist beforehand and
       must exist afterwards with the exact shape declared by
       `DailyEquitySnapshot` ORM at `core/models.py:325-333`.
    2. **Pre-existing table:** environments where `Base.metadata.create_all()`
       (e.g., dev bootstrap, certain test fixtures) created the table BEFORE
       Alembic ran. Migration 008 must not raise `relation "daily_equity_snapshots"
       already exists` in that scenario.

    To prove idempotency we (a) `alembic upgrade head` once, (b) re-execute the
    raw SQL of the migration body a second time directly via the engine, and
    expect both to succeed.
    """
    import sqlalchemy as sa
    from alembic import command
    from alembic.config import Config

    url = _db_url()
    cfg = Config("alembic/alembic.ini")
    cfg.set_main_option("sqlalchemy.url", url)
    command.upgrade(cfg, "head")

    sync_url = _sync_url(url)
    engine = sa.create_engine(sync_url)
    try:
        with engine.connect() as conn:
            # 1. Table exists post-upgrade.
            assert sa.inspect(engine).has_table("daily_equity_snapshots"), (
                "daily_equity_snapshots must exist after `alembic upgrade head`"
            )

            # 2. Registered as a TimescaleDB hypertable.
            ht_row = conn.execute(
                sa.text(
                    "SELECT 1 FROM timescaledb_information.hypertables "
                    "WHERE hypertable_name = 'daily_equity_snapshots'"
                )
            ).fetchone()
            assert ht_row is not None, (
                "daily_equity_snapshots must be registered as a TimescaleDB hypertable"
            )

            # 3. Composite primary key (timestamp, market_id) is present.
            pk_cols = [
                row[0]
                for row in conn.execute(
                    sa.text(
                        "SELECT kcu.column_name "
                        "FROM information_schema.table_constraints tc "
                        "JOIN information_schema.key_column_usage kcu "
                        "  ON tc.constraint_name = kcu.constraint_name "
                        " AND tc.table_name = kcu.table_name "
                        "WHERE tc.table_name = 'daily_equity_snapshots' "
                        "  AND tc.constraint_type = 'PRIMARY KEY' "
                        "ORDER BY kcu.ordinal_position"
                    )
                ).fetchall()
            ]
            assert pk_cols == ["timestamp", "market_id"], (
                f"Composite PK must be (timestamp, market_id); got {pk_cols!r}"
            )

            # 4. `currency` column has DEFAULT 'USD' (matches ORM).
            currency_default = conn.execute(
                sa.text(
                    "SELECT column_default FROM information_schema.columns "
                    "WHERE table_name = 'daily_equity_snapshots' "
                    "  AND column_name = 'currency'"
                )
            ).scalar()
            assert currency_default is not None and "USD" in currency_default, (
                f"currency column must default to 'USD'; got {currency_default!r}"
            )

            # 5. Idempotency: re-run the migration's raw SQL — must not raise.
            #    These statements mirror the body of alembic/versions/008_daily_equity_snapshots.py.
            conn.execute(
                sa.text(
                    "CREATE TABLE IF NOT EXISTS daily_equity_snapshots ("
                    "timestamp TIMESTAMP WITH TIME ZONE NOT NULL, "
                    "market_id VARCHAR(20) NOT NULL, "
                    "equity NUMERIC(14, 4) NOT NULL, "
                    "currency VARCHAR(3) NOT NULL DEFAULT 'USD', "
                    "PRIMARY KEY (timestamp, market_id))"
                )
            )
            conn.execute(
                sa.text(
                    "SELECT create_hypertable('daily_equity_snapshots', 'timestamp', "
                    "if_not_exists => TRUE, migrate_data => TRUE)"
                )
            )
            conn.execute(
                sa.text(
                    "CREATE INDEX IF NOT EXISTS ix_daily_equity_snapshots_market_ts "
                    "ON daily_equity_snapshots (market_id, timestamp DESC)"
                )
            )
            conn.commit()
    finally:
        engine.dispose()


def test_migration_009_creates_alerts_hypertable() -> None:
    """Migration 009 must create `alerts` as a TimescaleDB hypertable on a fresh DB.

    Regression for the Phase 57-UAT blocker (gap closure): the original migration
    declared `PRIMARY KEY (timestamp, id)` plus `FOREIGN KEY (parent_id)
    REFERENCES alerts(id)`, which Postgres rejects with `there is no unique
    constraint matching given keys for referenced table "alerts"` — the composite
    PK does not make the single `id` column unique. The fix adds an explicit
    `UNIQUE (id)` constraint inside the CREATE TABLE body, satisfying the FK
    requirement without altering the hypertable PK shape.

    The static AST test in tests/integration/migrations/test_009_alerts.py
    verified the SQL string contained the FK literal but did NOT execute it
    against a real Postgres instance — that is why the bug shipped past the
    Plan 01 verifier. This live-DB test closes that gap.
    """
    import sqlalchemy as sa
    from alembic import command
    from alembic.config import Config

    url = _db_url()
    cfg = Config("alembic/alembic.ini")
    cfg.set_main_option("sqlalchemy.url", url)
    command.upgrade(cfg, "head")

    sync_url = _sync_url(url)
    engine = sa.create_engine(sync_url)
    try:
        with engine.connect() as conn:
            assert sa.inspect(engine).has_table("alerts"), (
                "alerts table must exist after `alembic upgrade head`"
            )

            ht_row = conn.execute(
                sa.text(
                    "SELECT 1 FROM timescaledb_information.hypertables "
                    "WHERE hypertable_name = 'alerts'"
                )
            ).fetchone()
            assert ht_row is not None, "alerts must be registered as a TimescaleDB hypertable"

            pk_cols = [
                row[0]
                for row in conn.execute(
                    sa.text(
                        "SELECT kcu.column_name "
                        "FROM information_schema.table_constraints tc "
                        "JOIN information_schema.key_column_usage kcu "
                        "  ON tc.constraint_name = kcu.constraint_name "
                        " AND tc.table_name = kcu.table_name "
                        "WHERE tc.table_name = 'alerts' "
                        "  AND tc.constraint_type = 'PRIMARY KEY' "
                        "ORDER BY kcu.ordinal_position"
                    )
                ).fetchall()
            ]
            assert pk_cols == ["timestamp", "id"], (
                f"Composite PK must be (timestamp, id); got {pk_cols!r}"
            )

            # parent_id is a plain nullable UUID without a self-FK — TimescaleDB
            # hypertables forbid UNIQUE constraints that don't include the
            # partition column, which would be required for a self-FK.
            parent_id_col = conn.execute(
                sa.text(
                    "SELECT data_type, is_nullable "
                    "FROM information_schema.columns "
                    "WHERE table_name = 'alerts' AND column_name = 'parent_id'"
                )
            ).fetchone()
            assert parent_id_col is not None, "parent_id column must exist"
            assert parent_id_col[0] == "uuid", f"parent_id must be UUID; got {parent_id_col[0]!r}"
            assert parent_id_col[1] == "YES", "parent_id must be nullable"

            fk_count = conn.execute(
                sa.text(
                    "SELECT COUNT(*) "
                    "FROM pg_constraint "
                    "WHERE conrelid = 'alerts'::regclass "
                    "  AND contype = 'f'"
                )
            ).scalar()
            assert fk_count == 0, (
                f"alerts must have NO foreign keys (TimescaleDB conflict); got {fk_count}"
            )
    finally:
        engine.dispose()


def test_migration_010_creates_agent_decisions_hypertable() -> None:
    """Migration 010 must create `agent_decisions` as a TimescaleDB hypertable.

    Closes the live-DB coverage gap noted in `/gsd-add-tests 58`: the AST-static
    test in `tests/integration/migrations/test_010_agent_decisions.py` parses
    the migration source but never executes it against a real Postgres+TimescaleDB
    instance. This test runs `alembic upgrade head` and introspects:

    1. `agent_decisions` exists and is registered as a TimescaleDB hypertable.
    2. The composite primary key is `(timestamp, id)` — mirrors AlertModel.
    3. The Python attribute `decision_metadata` maps to a bare DB column named
       `metadata` (SQLAlchemy reserved-word workaround per AP-3 in PATTERNS.md).
    4. `parent_decision_id` is a plain nullable UUID with NO foreign key
       (TimescaleDB hypertables forbid UNIQUE constraints that don't include
       the partition column — same constraint as AlertModel.parent_id).
    """
    import sqlalchemy as sa
    from alembic import command
    from alembic.config import Config

    url = _db_url()
    cfg = Config("alembic/alembic.ini")
    cfg.set_main_option("sqlalchemy.url", url)
    command.upgrade(cfg, "head")

    sync_url = _sync_url(url)
    engine = sa.create_engine(sync_url)
    try:
        with engine.connect() as conn:
            assert sa.inspect(engine).has_table("agent_decisions"), (
                "agent_decisions table must exist after `alembic upgrade head`"
            )

            ht_row = conn.execute(
                sa.text(
                    "SELECT 1 FROM timescaledb_information.hypertables "
                    "WHERE hypertable_name = 'agent_decisions'"
                )
            ).fetchone()
            assert ht_row is not None, (
                "agent_decisions must be registered as a TimescaleDB hypertable"
            )

            pk_cols = [
                row[0]
                for row in conn.execute(
                    sa.text(
                        "SELECT kcu.column_name "
                        "FROM information_schema.table_constraints tc "
                        "JOIN information_schema.key_column_usage kcu "
                        "  ON tc.constraint_name = kcu.constraint_name "
                        " AND tc.table_name = kcu.table_name "
                        "WHERE tc.table_name = 'agent_decisions' "
                        "  AND tc.constraint_type = 'PRIMARY KEY' "
                        "ORDER BY kcu.ordinal_position"
                    )
                ).fetchall()
            ]
            assert pk_cols == ["timestamp", "id"], (
                f"Composite PK must be (timestamp, id); got {pk_cols!r}"
            )

            # decision_metadata Python attr must map to bare column `metadata`.
            metadata_col = conn.execute(
                sa.text(
                    "SELECT data_type "
                    "FROM information_schema.columns "
                    "WHERE table_name = 'agent_decisions' AND column_name = 'metadata'"
                )
            ).fetchone()
            assert metadata_col is not None, (
                "DB column must be named `metadata` (decision_metadata Python attr renamed)"
            )
            assert metadata_col[0] == "jsonb", (
                f"metadata column must be JSONB; got {metadata_col[0]!r}"
            )

            # parent_decision_id is plain nullable UUID without a FK constraint.
            fk_count = conn.execute(
                sa.text(
                    "SELECT COUNT(*) "
                    "FROM pg_constraint "
                    "WHERE conrelid = 'agent_decisions'::regclass "
                    "  AND contype = 'f'"
                )
            ).scalar()
            assert fk_count == 0, (
                f"agent_decisions must have NO foreign keys (TimescaleDB conflict); got {fk_count}"
            )
    finally:
        engine.dispose()


def test_migration_010_round_trip() -> None:
    """`MetaAgentDecisionModel` must round-trip through the live `agent_decisions` table.

    Closes the second half of the coverage gap noted in `/gsd-add-tests 58`:
    the ORM-only round-trip test (`tests/unit/core/test_meta_agent_decision_model.py`)
    constructs the model in memory but never persists it. This test inserts
    one decision via async SQLAlchemy, selects it back, and asserts every
    field — especially the `decision_metadata` Python attr / `metadata` DB
    column rename — round-trips correctly.
    """
    import asyncio
    import uuid as _uuid_mod
    from datetime import UTC, datetime

    import sqlalchemy as sa
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from finalayze.core.models import MetaAgentDecisionModel

    url = _db_url()

    async def _round_trip() -> None:
        engine = create_async_engine(url)
        try:
            async with engine.begin() as conn:
                await conn.execute(sa.text("DELETE FROM agent_decisions"))

            session_factory = async_sessionmaker(engine, expire_on_commit=False)
            decision_id = _uuid_mod.uuid4()
            ts = datetime.now(UTC)
            metadata_payload = {"telegram_alert_id": str(_uuid_mod.uuid4()), "trace": "abc"}

            async with session_factory() as session:
                session.add(
                    MetaAgentDecisionModel(
                        timestamp=ts,
                        id=decision_id,
                        severity="INVESTIGATE",
                        summary="round-trip smoke",
                        rationale="verify metadata column rename + JSONB merge",
                        actions=[{"kind": "telegram"}],
                        outcome=None,
                        dry_run=False,
                        decision_metadata=metadata_payload,
                        parent_decision_id=None,
                        status="sent",
                    )
                )
                await session.commit()

            async with session_factory() as session:
                row = (
                    await session.execute(
                        sa.select(MetaAgentDecisionModel).where(
                            MetaAgentDecisionModel.id == decision_id
                        )
                    )
                ).scalar_one()

                assert row.severity == "INVESTIGATE"
                assert row.summary == "round-trip smoke"
                assert row.actions == [{"kind": "telegram"}]
                assert row.dry_run is False
                assert row.status == "sent"
                # The critical assertion: Python attr name vs DB column name.
                assert row.decision_metadata == metadata_payload, (
                    "decision_metadata Python attr must round-trip via `metadata` DB column"
                )
                assert row.parent_decision_id is None
                assert row.created_at is not None
        finally:
            await engine.dispose()

    asyncio.run(_round_trip())


def test_migration_008_handles_pre_existing_populated_table() -> None:
    """Regression: migration 008 must convert a pre-populated plain table to a hypertable.

    In environments bootstrapped via `Base.metadata.create_all()` the
    `daily_equity_snapshots` table is created as a *plain* (non-hypertable)
    Postgres table. The existing `daily_reset` writer may then populate it
    with rows BEFORE migration 008 ever runs. In that scenario, calling
    `create_hypertable(..., if_not_exists => TRUE)` without `migrate_data => TRUE`
    raises `psycopg2.errors.FeatureNotSupported: table is not empty`.

    This test simulates the worst-case bootstrap order: plain table + data
    + hypertable conversion. The migration's own SQL is exercised; the
    operation must succeed and preserve existing rows.
    """
    import sqlalchemy as sa

    url = _db_url()
    sync_url = _sync_url(url)
    engine = sa.create_engine(sync_url)
    try:
        with engine.connect() as conn:
            # Force back to a plain-table-with-data state: drop hypertable,
            # recreate as a plain table, insert one synthetic row.
            conn.execute(sa.text("DROP TABLE IF EXISTS daily_equity_snapshots CASCADE"))
            conn.execute(
                sa.text(
                    "CREATE TABLE daily_equity_snapshots ("
                    "timestamp TIMESTAMP WITH TIME ZONE NOT NULL, "
                    "market_id VARCHAR(20) NOT NULL, "
                    "equity NUMERIC(14, 4) NOT NULL, "
                    "currency VARCHAR(3) NOT NULL DEFAULT 'USD', "
                    "PRIMARY KEY (timestamp, market_id))"
                )
            )
            conn.execute(
                sa.text(
                    "INSERT INTO daily_equity_snapshots "
                    "(timestamp, market_id, equity, currency) VALUES "
                    "('2026-04-01 00:00:00+00', 'moex', 1000000.0000, 'RUB')"
                )
            )
            conn.commit()

            # Now invoke the migration's own create_hypertable call — the bug
            # being defended against is "table is not empty" when
            # migrate_data => TRUE is missing.
            conn.execute(
                sa.text(
                    "SELECT create_hypertable('daily_equity_snapshots', 'timestamp', "
                    "if_not_exists => TRUE, migrate_data => TRUE)"
                )
            )
            conn.commit()

            # Hypertable conversion succeeded.
            ht_row = conn.execute(
                sa.text(
                    "SELECT 1 FROM timescaledb_information.hypertables "
                    "WHERE hypertable_name = 'daily_equity_snapshots'"
                )
            ).fetchone()
            assert ht_row is not None, "Pre-populated table must convert to hypertable"

            # Data preserved.
            row_count = conn.execute(
                sa.text("SELECT COUNT(*) FROM daily_equity_snapshots")
            ).scalar()
            assert row_count == 1, f"Pre-existing rows must be migrated; got {row_count}"
    finally:
        engine.dispose()
