"""Tests for GenericFileCache."""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from finalayze.core.schemas import FXRate
from finalayze.data.fetchers._cache_utils import GenericFileCache


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    return tmp_path / "test_cache"


@pytest.fixture
def cache(cache_dir: Path) -> GenericFileCache:
    return GenericFileCache(cache_dir=cache_dir)


@pytest.fixture
def sample_fx_rates() -> list[FXRate]:
    return [
        FXRate(
            timestamp=datetime(2024, 1, 15, 0, 0, tzinfo=UTC),
            pair="USDRUB",
            rate=Decimal("89.50"),
        ),
        FXRate(
            timestamp=datetime(2024, 1, 16, 0, 0, tzinfo=UTC),
            pair="USDRUB",
            rate=Decimal("89.75"),
        ),
    ]


class TestGenericFileCache:
    def test_auto_creates_cache_dir(self, cache_dir: Path) -> None:
        assert not cache_dir.exists()
        GenericFileCache(cache_dir=cache_dir)
        assert cache_dir.exists()

    def test_cache_miss_returns_none(self, cache: GenericFileCache) -> None:
        result = cache.get("nonexistent_key", FXRate)
        assert result is None

    def test_set_and_get(self, cache: GenericFileCache, sample_fx_rates: list[FXRate]) -> None:
        key = GenericFileCache.make_key("cbr", "USDRUB", date(2024, 1, 1), date(2024, 2, 1))
        cache.set(key, sample_fx_rates)
        result = cache.get(key, FXRate)
        assert result is not None
        assert len(result) == 2
        assert result[0].pair == "USDRUB"
        assert result[0].rate == Decimal("89.50")

    def test_ttl_expired_returns_none(
        self, cache: GenericFileCache, sample_fx_rates: list[FXRate]
    ) -> None:
        key = "expired_key"
        cache.set(key, sample_fx_rates)
        result = cache.get(key, FXRate, ttl_seconds=-1)  # negative TTL = always expired
        assert result is None

    def test_ttl_not_expired(self, cache: GenericFileCache, sample_fx_rates: list[FXRate]) -> None:
        key = "fresh_key"
        cache.set(key, sample_fx_rates)
        result = cache.get(key, FXRate, ttl_seconds=3600)
        assert result is not None
        assert len(result) == 2

    def test_make_key_format(self) -> None:
        key = GenericFileCache.make_key("cbr", "USDRUB", date(2024, 1, 1), date(2024, 12, 31))
        assert key == "cbr__USDRUB__20240101__20241231"

    def test_empty_list_not_cached(self, cache: GenericFileCache) -> None:
        key = "empty_key"
        cache.set(key, [])
        result = cache.get(key, FXRate)
        assert result is None

    def test_corrupt_json_returns_none(self, cache: GenericFileCache) -> None:
        """Corrupt cache file treated as cache miss (not crash)."""
        key = "corrupt_key"
        path = cache._cache_dir / f"{key}.json"
        path.write_text("{corrupt json!!")
        result = cache.get(key, FXRate)
        assert result is None
