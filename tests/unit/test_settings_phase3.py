"""Unit tests for Phase 3 settings additions."""

from __future__ import annotations

from importlib import reload
from typing import TYPE_CHECKING

import config.settings as settings_module
from config.settings import Settings

if TYPE_CHECKING:
    import pytest

# ── Constants (ruff PLR2004: no magic numbers) ──────────────────────────────
DEFAULT_NEWS_CYCLE_MINUTES = 30
DEFAULT_STRATEGY_CYCLE_MINUTES = 60
DEFAULT_DAILY_RESET_HOUR_UTC = 0
DEFAULT_TELEGRAM_BOT_TOKEN = ""
DEFAULT_TELEGRAM_CHAT_ID = ""

CUSTOM_NEWS_CYCLE_MINUTES = 15
CUSTOM_STRATEGY_CYCLE_MINUTES = 120
CUSTOM_DAILY_RESET_HOUR = 1
TELEGRAM_TEST_TOKEN = "1234567890:AABBccDDeEFfGgHhIiJj"  # noqa: S105
TELEGRAM_TEST_CHAT_ID = "-1001234567890"


class TestPhase3Settings:
    def test_news_cycle_minutes_default(self) -> None:
        s = Settings()
        assert s.news_cycle_minutes == DEFAULT_NEWS_CYCLE_MINUTES

    def test_strategy_cycle_minutes_default(self) -> None:
        s = Settings()
        assert s.strategy_cycle_minutes == DEFAULT_STRATEGY_CYCLE_MINUTES

    def test_daily_reset_hour_utc_default(self) -> None:
        s = Settings()
        assert s.daily_reset_hour_utc == DEFAULT_DAILY_RESET_HOUR_UTC

    def test_telegram_bot_token_default_empty(self) -> None:
        s = Settings()
        assert s.telegram_bot_token == DEFAULT_TELEGRAM_BOT_TOKEN

    def test_telegram_chat_id_default_empty(self) -> None:
        s = Settings()
        assert s.telegram_chat_id == DEFAULT_TELEGRAM_CHAT_ID

    def test_news_cycle_minutes_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FINALAYZE_NEWS_CYCLE_MINUTES", str(CUSTOM_NEWS_CYCLE_MINUTES))
        reload(settings_module)
        s = settings_module.Settings()
        assert s.news_cycle_minutes == CUSTOM_NEWS_CYCLE_MINUTES

    def test_strategy_cycle_minutes_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FINALAYZE_STRATEGY_CYCLE_MINUTES", str(CUSTOM_STRATEGY_CYCLE_MINUTES))
        reload(settings_module)
        s = settings_module.Settings()
        assert s.strategy_cycle_minutes == CUSTOM_STRATEGY_CYCLE_MINUTES

    def test_daily_reset_hour_utc_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FINALAYZE_DAILY_RESET_HOUR_UTC", str(CUSTOM_DAILY_RESET_HOUR))
        reload(settings_module)
        s = settings_module.Settings()
        assert s.daily_reset_hour_utc == CUSTOM_DAILY_RESET_HOUR

    def test_telegram_bot_token_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FINALAYZE_TELEGRAM_BOT_TOKEN", TELEGRAM_TEST_TOKEN)
        reload(settings_module)
        s = settings_module.Settings()
        assert s.telegram_bot_token == TELEGRAM_TEST_TOKEN

    def test_telegram_chat_id_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FINALAYZE_TELEGRAM_CHAT_ID", TELEGRAM_TEST_CHAT_ID)
        reload(settings_module)
        s = settings_module.Settings()
        assert s.telegram_chat_id == TELEGRAM_TEST_CHAT_ID
