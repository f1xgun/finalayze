"""Shared test fixtures for Finalayze."""

from __future__ import annotations

# torch must be imported before lightgbm to prevent OpenMP thread-pool conflicts
# that cause segmentation faults when both libraries are used together.
import torch  # noqa: F401  # isort: skip
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
