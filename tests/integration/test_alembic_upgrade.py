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
