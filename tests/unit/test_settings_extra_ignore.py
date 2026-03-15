"""Tests that Settings ignores non-FINALAYZE_ env vars from .env.

Verifies that Docker Compose variables (POSTGRES_USER, POSTGRES_DB, etc.)
do not cause pydantic ValidationError when present in the environment.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _clear_settings_cache() -> None:
    """Clear the lru_cache so each test gets a fresh Settings instance."""
    from config.settings import get_settings

    get_settings.cache_clear()
    yield  # type: ignore[misc]
    get_settings.cache_clear()


class TestSettingsExtraIgnore:
    """Settings model with extra='ignore' silently drops unknown fields."""

    def test_non_prefixed_env_vars_ignored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Docker Compose vars like POSTGRES_USER should not raise."""
        monkeypatch.setenv("POSTGRES_USER", "finalayze")
        monkeypatch.setenv("POSTGRES_PASSWORD", "secret123")
        monkeypatch.setenv("POSTGRES_DB", "finalayze_db")
        monkeypatch.setenv("REDIS_PASSWORD", "redis_secret")
        # Ensure we are in DEBUG mode (no credential validation)
        monkeypatch.setenv("FINALAYZE_MODE", "debug")

        from config.settings import Settings

        # Should not raise pydantic ValidationError
        settings = Settings()
        assert settings.mode.value == "debug"

    def test_prefixed_vars_still_loaded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """FINALAYZE_-prefixed vars are still picked up correctly."""
        monkeypatch.setenv("FINALAYZE_MODE", "test")
        monkeypatch.setenv("FINALAYZE_REDIS_URL", "redis://custom:6379/1")

        from config.settings import Settings

        settings = Settings()
        assert settings.mode.value == "test"
        assert settings.redis_url == "redis://custom:6379/1"

    def test_unknown_prefixed_vars_ignored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Unknown FINALAYZE_-prefixed vars should also be silently ignored."""
        monkeypatch.setenv("FINALAYZE_MODE", "debug")
        monkeypatch.setenv("FINALAYZE_NONEXISTENT_FIELD", "some_value")
        monkeypatch.setenv("FINALAYZE_ANOTHER_UNKNOWN", "42")

        from config.settings import Settings

        # Should not raise even with unknown FINALAYZE_ vars
        settings = Settings()
        assert settings.mode.value == "debug"
