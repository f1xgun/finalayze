"""Shared test fixtures for Finalayze."""

from __future__ import annotations

import pytest
from config.modes import WorkMode
from config.settings import Settings


@pytest.fixture
def settings() -> Settings:
    """Create test settings with debug mode."""
    return Settings(
        mode=WorkMode.DEBUG,
        database_url="postgresql+asyncpg://test:test@localhost:5432/test",
        redis_url="redis://localhost:6379/1",
    )
