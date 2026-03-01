"""Tests for 6D.6: CORS wildcard lockdown."""

from __future__ import annotations

from unittest.mock import patch

from config.settings import Settings


def test_cors_default_is_empty_list() -> None:
    """Default cors_origins should be an empty list, not ['*']."""
    s = Settings(mode="debug")
    assert s.cors_origins == []


def test_cors_configured_from_settings() -> None:
    """cors_origins should accept a list of origins."""
    s = Settings(
        mode="debug",
        cors_origins=["https://app.example.com", "https://admin.example.com"],
    )
    assert s.cors_origins == ["https://app.example.com", "https://admin.example.com"]


def test_create_app_no_wildcard_cors() -> None:
    """create_app should not set allow_origins=['*'] by default."""
    with patch("config.settings.get_settings") as mock_settings:
        mock_settings.return_value = Settings(mode="debug", cors_origins=[])
        # Re-import to trigger create_app with patched settings
        from finalayze.main import create_app

        app = create_app()
        # Check that CORSMiddleware was added with empty origins
        cors_mw = None
        for mw in app.user_middleware:
            if mw.cls.__name__ == "CORSMiddleware":
                cors_mw = mw
                break
        assert cors_mw is not None
        assert cors_mw.kwargs.get("allow_origins") == []
