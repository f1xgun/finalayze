"""Async database engine and session factory (Layer 2).

Provides a module-level engine and session factory built from ``config.settings``.
Use ``get_db()`` as a FastAPI dependency to obtain a scoped ``AsyncSession``.

Example::

    from finalayze.core.db import get_db

    @router.get("/items")
    async def list_items(session: AsyncSession = Depends(get_db)):
        ...
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from sqlalchemy.ext.asyncio import AsyncEngine

__all__ = [
    "AsyncSession",
    "async_sessionmaker",
    "create_async_engine",
    "get_async_session_factory",
    "get_db",
    "reset_engine",
]

# Module-level cache for engine and session factory.
# Keyed by database_url so that different URLs (e.g. in tests) get distinct pools.
_engine_cache: dict[str, AsyncEngine] = {}
_factory_cache: dict[str, async_sessionmaker[AsyncSession]] = {}


def get_async_session_factory() -> async_sessionmaker[AsyncSession]:
    """Return a cached ``async_sessionmaker`` for the current settings.

    The factory and its underlying engine are created lazily on first call
    and cached by ``database_url``.  Subsequent calls with the same URL
    return the same factory, avoiding the connection-pool leak that occurred
    when a new engine was created on every invocation.
    """
    from config.settings import get_settings  # noqa: PLC0415

    settings = get_settings()
    url = settings.database_url

    if url not in _factory_cache:
        engine = create_async_engine(url, echo=False, pool_pre_ping=True)
        _engine_cache[url] = engine
        _factory_cache[url] = async_sessionmaker(
            engine, class_=AsyncSession, expire_on_commit=False
        )

    return _factory_cache[url]


def reset_engine() -> None:
    """Clear cached engines and factories.

    Intended for test teardown so that each test can inject a fresh
    ``database_url`` via env-var overrides without hitting a stale cache.
    """
    _engine_cache.clear()
    _factory_cache.clear()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency that yields a scoped ``AsyncSession``.

    Commits on success and rolls back on exception, then closes the session.

    Usage::

        from fastapi import Depends
        from finalayze.core.db import get_db

        async def endpoint(session: AsyncSession = Depends(get_db)):
            ...
    """
    factory = get_async_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
