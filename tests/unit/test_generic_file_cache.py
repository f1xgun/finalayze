"""Unit tests for GenericFileCache (Layer 2)."""

from __future__ import annotations

import json
import os
import time
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from finalayze.core.schemas import FXRate
from finalayze.data.fetchers._cache_utils import GenericFileCache

# ── Constants (ruff PLR2004: no magic numbers) ──────────────────────────

TTL_SHORT = 1  # seconds
TTL_LONG = 3600  # seconds
RATE_VALUE = Decimal("85.50")
RATE_PAIR = "USDRUB"


@pytest.fixture
def cache(tmp_path: Path) -> GenericFileCache:
    """GenericFileCache backed by a temp directory."""
    return GenericFileCache(cache_dir=tmp_path / "cache")


@pytest.fixture
def sample_fx_rate() -> FXRate:
    return FXRate(
        timestamp=datetime(2024, 1, 15, tzinfo=UTC),
        pair=RATE_PAIR,
        rate=RATE_VALUE,
    )


class TestGenericFileCacheGet:
    def test_miss_on_nonexistent_key(self, cache: GenericFileCache) -> None:
        """Cache miss returns None for unknown key."""
        assert cache.get("nonexistent", FXRate) is None

    def test_hit_returns_models(self, cache: GenericFileCache, sample_fx_rate: FXRate) -> None:
        """Round-trip set/get returns equivalent models."""
        cache.set("fx_key", [sample_fx_rate])
        result = cache.get("fx_key", FXRate)
        assert result is not None
        assert len(result) == 1
        assert result[0].pair == RATE_PAIR
        assert result[0].rate == RATE_VALUE

    def test_corrupt_json_returns_none(self, cache: GenericFileCache) -> None:
        """Corrupt JSON treated as cache miss."""
        key = "corrupt_key"
        path = cache._cache_dir / f"{key}.json"
        path.write_text("NOT_VALID_JSON{{{")
        assert cache.get(key, FXRate) is None

    def test_schema_mismatch_returns_none(self, cache: GenericFileCache) -> None:
        """Cached data with wrong schema treated as cache miss."""
        key = "bad_schema_key"
        path = cache._cache_dir / f"{key}.json"
        # Write data that doesn't match FXRate schema (missing required fields)
        path.write_text('[{"wrong_field": "value"}]')
        result = cache.get(key, FXRate)
        assert result is None

    def test_ttl_miss_when_expired(self, cache: GenericFileCache, sample_fx_rate: FXRate) -> None:
        """Expired entry (TTL elapsed) returns None."""
        cache.set("ttl_key", [sample_fx_rate])
        path = cache._cache_dir / "ttl_key.json"
        old_time = time.time() - TTL_LONG
        os.utime(path, (old_time, old_time))
        assert cache.get("ttl_key", FXRate, ttl_seconds=TTL_SHORT) is None

    def test_ttl_hit_when_fresh(self, cache: GenericFileCache, sample_fx_rate: FXRate) -> None:
        """Fresh entry within TTL returns models."""
        cache.set("fresh_key", [sample_fx_rate])
        result = cache.get("fresh_key", FXRate, ttl_seconds=TTL_LONG)
        assert result is not None
        assert len(result) == 1

    def test_empty_list_returns_none(self, cache: GenericFileCache) -> None:
        """Empty JSON array treated as cache miss."""
        key = "empty_key"
        path = cache._cache_dir / f"{key}.json"
        path.write_text("[]")
        assert cache.get(key, FXRate) is None


class TestGenericFileCacheSet:
    def test_empty_data_writes_nothing(self, cache: GenericFileCache) -> None:
        """set() with empty list does not create a file."""
        cache.set("no_write_key", [])
        assert not (cache._cache_dir / "no_write_key.json").exists()

    def test_set_writes_json(self, cache: GenericFileCache, sample_fx_rate: FXRate) -> None:
        """set() writes valid JSON that can be parsed back."""
        cache.set("write_key", [sample_fx_rate])
        path = cache._cache_dir / "write_key.json"
        assert path.exists()
        data = json.loads(path.read_text())
        assert isinstance(data, list)
        assert len(data) == 1


class TestGenericFileCacheMakeKey:
    def test_make_key_format(self) -> None:
        """make_key produces expected format."""
        from datetime import date

        key = GenericFileCache.make_key("cbr", "USDRUB", date(2024, 1, 1), date(2024, 3, 31))
        assert key == "cbr__USDRUB__20240101__20240331"
