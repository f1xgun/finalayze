"""Tests for 6D.12: no default DB password in non-DEBUG/TEST modes."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest
from config.settings import Settings
from pydantic import ValidationError

from finalayze.core.modes import WorkMode


def test_sandbox_mode_fallback_database_url() -> None:
    """SANDBOX mode without database_url should get a fallback (like DEBUG/TEST)."""
    s = Settings(mode=WorkMode.SANDBOX, database_url="")
    assert s.database_url != ""
    assert "finalayze" in s.database_url


def test_debug_mode_fallback_database_url() -> None:
    """DEBUG mode should get a fallback database_url when none is provided."""
    s = Settings(mode=WorkMode.DEBUG, database_url="")
    assert s.database_url != ""
    assert "finalayze" in s.database_url


def test_test_mode_fallback_database_url() -> None:
    """TEST mode should get a fallback database_url when none is provided."""
    s = Settings(mode=WorkMode.TEST, database_url="")
    assert s.database_url != ""


def test_default_database_url_is_empty() -> None:
    """The default database_url field should be empty (no hardcoded password)."""
    # Instantiate in DEBUG mode (which applies fallback) but check the field definition
    assert Settings.model_fields["database_url"].default == ""
