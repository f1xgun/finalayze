"""Unit tests for MacroCacheService."""

from __future__ import annotations

import pytest

from finalayze.data.fetchers.cbr import MacroContextProvider, MacroSnapshot
from finalayze.data.macro_cache import MacroCacheService

HISTORY_SIZE = 5


@pytest.fixture
def provider() -> MacroContextProvider:
    return MacroContextProvider()


@pytest.fixture
def cache(provider: MacroContextProvider) -> MacroCacheService:
    return MacroCacheService(provider, history_size=HISTORY_SIZE)


def test_get_returns_none_before_refresh(cache: MacroCacheService) -> None:
    assert cache.get() is None


def test_refresh_returns_snapshot(cache: MacroCacheService) -> None:
    snapshot = cache.refresh()
    assert isinstance(snapshot, MacroSnapshot)
    assert snapshot.key_rate is not None


def test_get_returns_cached_after_refresh(cache: MacroCacheService) -> None:
    cache.refresh()
    result = cache.get()
    assert result is not None
    assert isinstance(result, MacroSnapshot)


def test_history_accumulates(cache: MacroCacheService) -> None:
    cache.refresh()
    cache.refresh()
    history = cache.get_history()
    assert len(history) == 2


def test_history_respects_max_size() -> None:
    provider = MacroContextProvider()
    small_cache = MacroCacheService(provider, history_size=2)
    small_cache.refresh()
    small_cache.refresh()
    small_cache.refresh()
    assert len(small_cache.get_history()) == 2


def test_is_cbr_meeting_day_returns_bool(cache: MacroCacheService) -> None:
    result = cache.is_cbr_meeting_day()
    assert isinstance(result, bool)
