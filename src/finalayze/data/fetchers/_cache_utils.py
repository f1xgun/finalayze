"""Generic file-based cache for Pydantic models (Layer 2).

Companion to CachingFetcher — handles non-Candle types like FXRate,
KeyRateRecord, TurnoverRecord with optional TTL expiry.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from datetime import date

import structlog
from pydantic import BaseModel, ValidationError

_T = TypeVar("_T", bound=BaseModel)
_log = structlog.get_logger()


class GenericFileCache:
    """File-based cache for Pydantic models with optional TTL.

    Not thread-safe — designed for single-threaded backtest use.
    Writes are non-atomic (acceptable for single-process usage).
    """

    def __init__(self, cache_dir: Path = Path(".cache/market_data")) -> None:
        self._cache_dir = cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def get(
        self,
        key: str,
        model_class: type[_T],
        ttl_seconds: int | None = None,
    ) -> list[_T] | None:
        """Read from cache. Returns None on miss, TTL expiry, or corrupt data."""
        path = self._cache_dir / f"{key}.json"
        if not path.exists():
            return None

        if ttl_seconds is not None:
            age = time.time() - path.stat().st_mtime
            if age > ttl_seconds:
                return None

        try:
            raw = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            _log.warning("cache_corrupt", key=key)
            return None

        if not raw:
            return None
        try:
            return [model_class.model_validate(item) for item in raw]
        except ValidationError:
            _log.warning("cache_schema_mismatch", key=key)
            return None

    def set(self, key: str, data: list[BaseModel]) -> None:
        """Write to cache as JSON via model_dump(). Skips empty data."""
        if not data:
            return
        path = self._cache_dir / f"{key}.json"
        path.write_text(json.dumps([item.model_dump(mode="json") for item in data], default=str))

    @staticmethod
    def make_key(source: str, item_id: str, start: date, end: date) -> str:
        """Build cache key: 'source__item_id__YYYYMMDD__YYYYMMDD'."""
        return f"{source}__{item_id}__{start:%Y%m%d}__{end:%Y%m%d}"
