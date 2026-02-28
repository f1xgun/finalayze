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

__all__ = [
    "AsyncSession",
    "async_sessionmaker",
    "create_async_engine",
    "get_async_session_factory",
    "get_db",
]


def get_async_session_factory() -> async_sessionmaker[AsyncSession]:
    """Create and return an ``async_sessionmaker`` using current settings.

    The factory is constructed lazily so that ``database_url`` is read at
    call time rather than at module import time — enabling env-var injection
    in tests.
    """
    from config.settings import get_settings  # noqa: PLC0415

    settings = get_settings()
    engine = create_async_engine(settings.database_url, echo=False, pool_pre_ping=True)
    return async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


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
