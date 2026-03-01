"""Tests for 6D.4: Explicit DB connection pool sizing."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from finalayze.core.db import get_async_session_factory, reset_engine


def test_engine_created_with_pool_params() -> None:
    """Verify engine is created with pool_size, max_overflow, pool_timeout, pool_recycle."""
    reset_engine()

    mock_settings = MagicMock()
    mock_settings.database_url = "postgresql+asyncpg://test:test@localhost/test"
    mock_settings.db_pool_size = 15
    mock_settings.db_max_overflow = 7
    mock_settings.db_pool_timeout = 20
    mock_settings.db_pool_recycle = 900

    with (
        patch("config.settings.get_settings", return_value=mock_settings),
        patch("finalayze.core.db.create_async_engine") as mock_create,
    ):
        mock_create.return_value = MagicMock()
        get_async_session_factory()

        mock_create.assert_called_once_with(
            "postgresql+asyncpg://test:test@localhost/test",
            echo=False,
            pool_pre_ping=True,
            pool_size=15,
            max_overflow=7,
            pool_timeout=20,
            pool_recycle=900,
        )

    reset_engine()
