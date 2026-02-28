"""Alembic environment configuration.

Uses a synchronous psycopg2 engine for migrations even though the application
uses asyncpg at runtime.  Alembic's DDL runner is synchronous; using asyncpg
here would cause ``asyncpg.connect()`` to be called outside an event loop.

The ``sqlalchemy.url`` in alembic.ini should use ``postgresql+asyncpg://``.
This module converts it to ``postgresql+psycopg2://`` automatically.
"""

from __future__ import annotations

from logging.config import fileConfig

from alembic import context
from sqlalchemy import create_engine, pool

from finalayze.core.models import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _get_sync_url() -> str:
    """Return a psycopg2-compatible URL derived from the configured asyncpg URL."""
    url = config.get_main_option("sqlalchemy.url") or ""
    # Convert asyncpg driver to psycopg2 for synchronous Alembic migrations
    return url.replace("postgresql+asyncpg://", "postgresql+psycopg2://")


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode (generates SQL without a live connection)."""
    sync_url = _get_sync_url()
    context.configure(
        url=sync_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode using a synchronous psycopg2 engine."""
    sync_url = _get_sync_url()
    connectable = create_engine(sync_url, poolclass=pool.NullPool)

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
