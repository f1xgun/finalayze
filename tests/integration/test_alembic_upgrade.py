from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.integration


def _db_url() -> str:
    url = os.environ.get("FINALAYZE_DATABASE_URL") or os.environ.get("DATABASE_URL")
    if not url:
        pytest.skip("FINALAYZE_DATABASE_URL not set; integration DB unavailable")
    return url


def test_007_stop_loss_events_applies() -> None:
    """After `alembic upgrade head`, the stop_loss_events hypertable must exist with policies."""
    import sqlalchemy as sa
    from alembic import command
    from alembic.config import Config

    url = _db_url()
    # Run migrations (idempotent via if_not_exists guards)
    cfg = Config("alembic/alembic.ini")
    cfg.set_main_option("sqlalchemy.url", url)
    command.upgrade(cfg, "head")

    # Connect synchronously to TimescaleDB
    sync_url = url.replace("+asyncpg", "").replace("postgresql+asyncpg", "postgresql")
    engine = sa.create_engine(sync_url)
    with engine.connect() as conn:
        # Table exists
        res = conn.execute(
            sa.text(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_name = 'stop_loss_events'"
            )
        ).fetchone()
        assert res is not None, "stop_loss_events table must exist after upgrade"

        # Hypertable registered
        ht = conn.execute(
            sa.text(
                "SELECT hypertable_name FROM timescaledb_information.hypertables "
                "WHERE hypertable_name = 'stop_loss_events'"
            )
        ).fetchone()
        assert ht is not None, "stop_loss_events must be registered as a hypertable"

        # Retention policy attached
        jobs = conn.execute(
            sa.text(
                "SELECT job_id FROM timescaledb_information.jobs "
                "WHERE hypertable_name = 'stop_loss_events' "
                "  AND proc_name = 'policy_retention'"
            )
        ).fetchall()
        assert len(jobs) >= 1, "Retention policy must be attached"

        # Compression policy attached
        cjobs = conn.execute(
            sa.text(
                "SELECT job_id FROM timescaledb_information.jobs "
                "WHERE hypertable_name = 'stop_loss_events' "
                "  AND proc_name = 'policy_compression'"
            )
        ).fetchall()
        assert len(cjobs) >= 1, "Compression policy must be attached"


def test_008_signal_price_column_applies() -> None:
    """After upgrade head, signals.signal_price must exist, be numeric(12,4) nullable."""
    import sqlalchemy as sa
    from alembic import command
    from alembic.config import Config

    url = _db_url()
    cfg = Config("alembic/alembic.ini")
    cfg.set_main_option("sqlalchemy.url", url)
    command.upgrade(cfg, "head")

    sync_url = url.replace("+asyncpg", "").replace("postgresql+asyncpg", "postgresql")
    engine = sa.create_engine(sync_url)
    with engine.connect() as conn:
        row = conn.execute(
            sa.text(
                "SELECT column_name, data_type, is_nullable, numeric_precision, numeric_scale "
                "FROM information_schema.columns "
                "WHERE table_name = 'signals' AND column_name = 'signal_price'"
            )
        ).fetchone()
        assert row is not None, "signal_price column must exist after upgrade"
        assert row.data_type == "numeric"
        assert row.is_nullable == "YES"
        assert row.numeric_precision == 12
        assert row.numeric_scale == 4


def test_008_signal_price_column_downgrades() -> None:
    """After downgrade -1 from rev 008, signals.signal_price must be absent."""
    import sqlalchemy as sa
    from alembic import command
    from alembic.config import Config

    url = _db_url()
    cfg = Config("alembic/alembic.ini")
    cfg.set_main_option("sqlalchemy.url", url)
    command.upgrade(cfg, "008")
    command.downgrade(cfg, "007")

    sync_url = url.replace("+asyncpg", "").replace("postgresql+asyncpg", "postgresql")
    engine = sa.create_engine(sync_url)
    with engine.connect() as conn:
        row = conn.execute(
            sa.text(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'signals' AND column_name = 'signal_price'"
            )
        ).fetchone()
        assert row is None, "signal_price column must be removed after downgrade"
    # Re-upgrade so later tests see the column
    command.upgrade(cfg, "head")
