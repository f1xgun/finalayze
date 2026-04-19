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
                    "if_not_exists => TRUE)"
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
