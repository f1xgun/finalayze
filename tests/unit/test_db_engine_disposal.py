"""Regression tests for the async-engine connection leak (audit 2026-06-28, HIGH).

TradingPersistence kept one SQLAlchemy async engine per event-loop id but never
disposed them, so every loop recreation (stop/start, meta-agent on uvicorn vs
background loop) orphaned a 5-7 connection pool -> PostgreSQL "too many clients"
after weeks of uptime. core/db.py.reset_engine() likewise only .clear()'d the
cache without disposing. These tests pin the disposal contract.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from finalayze.core import db as dbmod
from finalayze.orchestration.db_persistence import TradingPersistence


async def test_persistence_dispose_all_disposes_engines_and_clears(monkeypatch) -> None:
    disposed: list[object] = []

    def fake_create_async_engine(*_args: object, **_kwargs: object) -> MagicMock:
        engine = MagicMock(name="engine")
        engine.dispose = AsyncMock(side_effect=lambda: disposed.append(engine))
        return engine

    monkeypatch.setattr("sqlalchemy.ext.asyncio.create_async_engine", fake_create_async_engine)

    persistence = TradingPersistence(db_url="postgresql+asyncpg://u:p@h/db", async_loop=None)
    # Creating the factory on the running loop must now ALSO track the engine.
    persistence._get_bg_session_factory()
    assert len(persistence._bg_engines) == 1
    assert len(persistence._bg_session_factories) == 1

    await persistence.dispose_all()

    assert len(disposed) == 1, "engine.dispose() must be awaited"
    assert persistence._bg_engines == {}
    assert persistence._bg_session_factories == {}


async def test_persistence_dispose_all_is_idempotent_and_safe_when_empty() -> None:
    persistence = TradingPersistence(db_url=None, async_loop=None)
    # No engines created yet; must not raise.
    await persistence.dispose_all()
    assert persistence._bg_engines == {}


async def test_core_db_dispose_engines_disposes_and_clears(monkeypatch) -> None:
    disposed: list[int] = []
    engine = MagicMock(name="app-engine")
    engine.dispose = AsyncMock(side_effect=lambda: disposed.append(1))

    dbmod._engine_cache.clear()
    dbmod._factory_cache.clear()
    dbmod._engine_cache["url-1"] = engine
    dbmod._factory_cache["url-1"] = MagicMock()

    await dbmod.dispose_engines()

    assert disposed == [1]
    assert dbmod._engine_cache == {}
    assert dbmod._factory_cache == {}
