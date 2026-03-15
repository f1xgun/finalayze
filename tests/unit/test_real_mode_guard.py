"""Tests for real_confirmed preflight guard in Settings.

Validates that the system refuses REAL mode without real_confirmed=True,
ensuring AUT-05 safety requirements are met.
"""

from __future__ import annotations

import pytest


class TestRealModeGuard:
    """Validate real_confirmed guard in Settings model_validator."""

    def test_real_mode_without_confirmed_raises(self) -> None:
        """Settings(mode='real', real_confirmed=False) raises ValueError."""
        from config.settings import Settings

        with pytest.raises((ValueError,), match="real_confirmed"):
            Settings(
                mode="real",
                real_confirmed=False,
                alpaca_api_key="test-key",
                alpaca_secret_key="test-secret",
                database_url="sqlite:///test.db",
            )

    def test_real_mode_with_confirmed_succeeds(self) -> None:
        """Settings(mode='real', real_confirmed=True) succeeds with required fields."""
        from config.settings import Settings

        settings = Settings(
            mode="real",
            real_confirmed=True,
            alpaca_api_key="test-key",
            alpaca_secret_key="test-secret",
            database_url="sqlite:///test.db",
            llm_api_key="test-llm-key",
        )
        assert settings.mode.value == "real"
        assert settings.real_confirmed is True

    def test_sandbox_mode_without_confirmed_succeeds(self) -> None:
        """Settings(mode='sandbox', real_confirmed=False) succeeds (no guard needed)."""
        from config.settings import Settings

        settings = Settings(
            mode="sandbox",
            real_confirmed=False,
        )
        assert settings.mode.value == "sandbox"
