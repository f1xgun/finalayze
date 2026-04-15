"""Redis cache for candles and sentiment data (Layer 2).

Provides async caching with configurable TTL for:
- Candle data: keyed by market:symbol:timeframe, 5 min TTL
- Sentiment scores: keyed by segment, 30 min TTL
- Event type codes: keyed by segment, same TTL as sentiment

Uses redis.asyncio (same pattern as EventBus in core/events.py).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import redis.asyncio

from finalayze.core.schemas import Candle
from finalayze.markets.schedule import MOEX_MARKET_SCHEDULE

if TYPE_CHECKING:
    from datetime import datetime

_CANDLE_TTL_SECONDS = 300  # 5 minutes
_SENTIMENT_TTL_SECONDS = 1800  # 30 minutes
_SENTIMENT_TTL_BUFFER_SECONDS = 1800  # 30 min buffer beyond next open


def _compute_sentiment_ttl(now: datetime) -> int:
    """Compute dynamic TTL for sentiment cache entries.

    During MOEX trading hours, returns the default 30-minute TTL.
    When the market is closed (evenings, weekends), extends TTL to survive
    until the next market open plus a 30-minute buffer, ensuring cached
    sentiment scores are available when trading resumes.

    Args:
        now: Current UTC-aware datetime.

    Returns:
        TTL in seconds, guaranteed >= ``_SENTIMENT_TTL_SECONDS`` (1800).
    """
    if MOEX_MARKET_SCHEDULE.is_market_open(now):
        return _SENTIMENT_TTL_SECONDS
    next_open = MOEX_MARKET_SCHEDULE.next_open(now)
    seconds_to_open = int((next_open - now).total_seconds())
    return max(seconds_to_open + _SENTIMENT_TTL_BUFFER_SECONDS, _SENTIMENT_TTL_SECONDS)


class RedisCache:
    """Async Redis cache for candle and sentiment data."""

    def __init__(self, redis_url: str = "redis://localhost:6379/0") -> None:
        self._redis: redis.asyncio.Redis[str] = redis.asyncio.from_url(
            redis_url, decode_responses=True
        )

    async def get_candles(self, market: str, symbol: str, timeframe: str) -> list[Candle] | None:
        """Retrieve cached candles or None on cache miss."""
        key = f"candles:{market}:{symbol}:{timeframe}"
        raw = await self._redis.get(key)
        if raw is None:
            return None
        items = json.loads(raw)
        return [Candle.model_validate_json(item) for item in items]

    async def set_candles(
        self,
        market: str,
        symbol: str,
        timeframe: str,
        candles: list[Candle],
        ttl: int = _CANDLE_TTL_SECONDS,
    ) -> None:
        """Cache candles with TTL."""
        key = f"candles:{market}:{symbol}:{timeframe}"
        items = [c.model_dump_json() for c in candles]
        await self._redis.set(key, json.dumps(items), ex=ttl)

    async def get_sentiment(self, segment: str) -> float | None:
        """Retrieve cached sentiment score or None on cache miss."""
        key = f"sentiment:{segment}"
        raw = await self._redis.get(key)
        if raw is None:
            return None
        return float(raw)

    async def set_sentiment(
        self,
        segment: str,
        score: float,
        ttl: int = _SENTIMENT_TTL_SECONDS,
    ) -> None:
        """Cache sentiment score with TTL."""
        key = f"sentiment:{segment}"
        await self._redis.set(key, str(score), ex=ttl)

    async def set_event_type(
        self,
        segment: str,
        code: float,
        ttl: int = _SENTIMENT_TTL_SECONDS,
    ) -> None:
        """Cache an event type code for downstream combiner dedup.

        Args:
            segment: Cache key suffix (e.g. ``"ru_blue_chips:SBER"``).
            code: Numeric event type code (e.g. 1.0 for CBR_RATE, 2.0 for EARNINGS).
            ttl: Time-to-live in seconds.
        """
        key = f"event_type:{segment}"
        await self._redis.set(key, str(code), ex=ttl)

    async def get_event_type(self, segment: str) -> float | None:
        """Retrieve cached event type code or ``None`` on cache miss.

        Args:
            segment: Cache key suffix used in :meth:`set_event_type`.

        Returns:
            The cached float code, or ``None`` if the key has expired or was
            never written.
        """
        key = f"event_type:{segment}"
        raw = await self._redis.get(key)
        if raw is None:
            return None
        return float(raw)

    async def close(self) -> None:
        """Close the underlying Redis connection."""
        await self._redis.aclose()  # type: ignore[attr-defined]
