"""Redis cache for candles and sentiment data (Layer 2).

Provides async caching with configurable TTL for:
- Candle data: keyed by market:symbol:timeframe, 5 min TTL
- Sentiment scores: keyed by segment, 30 min TTL

Uses redis.asyncio (same pattern as EventBus in core/events.py).
"""

from __future__ import annotations

import json

import redis.asyncio

from finalayze.core.schemas import Candle

_CANDLE_TTL_SECONDS = 300  # 5 minutes
_SENTIMENT_TTL_SECONDS = 1800  # 30 minutes


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

    async def close(self) -> None:
        """Close the underlying Redis connection."""
        await self._redis.aclose()  # type: ignore[attr-defined]
