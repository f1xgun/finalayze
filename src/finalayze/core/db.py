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
    "dispose_engines",
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
        engine = create_async_engine(
            url,
            echo=False,
            pool_pre_ping=True,
            pool_size=settings.db_pool_size,
            max_overflow=settings.db_max_overflow,
            pool_timeout=settings.db_pool_timeout,
            pool_recycle=settings.db_pool_recycle,
        )
        _engine_cache[url] = engine
        _factory_cache[url] = async_sessionmaker(
            engine, class_=AsyncSession, expire_on_commit=False
        )

    return _factory_cache[url]


# Backward-compatible alias used by dashboard endpoints (portfolio, trades, signals).
async_session_factory = get_async_session_factory


def reset_engine() -> None:
    """Clear cached engines and factories WITHOUT disposing (sync test teardown).

    Intended for test teardown so that each test can inject a fresh
    ``database_url`` via env-var overrides without hitting a stale cache.

    WARNING: this drops Python references but does NOT close the underlying
    asyncpg pools. In a long-lived process always prefer the async
    :func:`dispose_engines` so connections are returned to the server; clearing
    alone orphans the pool (audit 2026-06-28, HIGH connection leak).
    """
    _engine_cache.clear()
    _factory_cache.clear()


async def dispose_engines() -> None:
    """Dispose every cached engine's connection pool, then clear the caches.

    Closes the underlying asyncpg connections (unlike :func:`reset_engine`) so
    they are returned to PostgreSQL. Call from the FastAPI lifespan shutdown.
    Each ``dispose()`` is best-effort so one bad engine cannot block the rest.
    """
    for url, engine in list(_engine_cache.items()):
        try:
            await engine.dispose()
        except Exception:
            from structlog import get_logger  # noqa: PLC0415

            get_logger().debug("engine_dispose_failed", url=url, exc_info=True)
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
