"""Redis Streams event bus (Layer 0).

See docs/architecture/DATA_FLOW.md for event flow details.
"""

from __future__ import annotations

import contextlib

import redis.asyncio
from pydantic import BaseModel


class MarketDataEvent(BaseModel):
    """Event carrying a serialized OHLCV candle for a symbol."""

    symbol: str
    market_id: str
    candle: dict[str, object]  # serialized Candle fields


class SignalEvent(BaseModel):
    """Event carrying a trading signal produced by a strategy."""

    strategy_name: str
    symbol: str
    market_id: str
    segment_id: str
    direction: str
    confidence: float


class EventBus:
    """Async event bus backed by Redis Streams (XADD / XREAD)."""

    STREAM_MARKET_DATA = "market_data"
    STREAM_SIGNALS = "signals"
    STREAM_EXECUTION = "execution"

    def __init__(self, redis_url: str = "redis://localhost:6379/0") -> None:
        self._redis: redis.asyncio.Redis[str] = redis.asyncio.from_url(
            redis_url, decode_responses=True
        )

    async def publish(self, stream: str, event: BaseModel) -> str:
        """Publish an event to a Redis Stream.

        Args:
            stream: Name of the stream (use class-level constants).
            event: A Pydantic model instance to serialise as the payload.

        Returns:
            The Redis message ID assigned by the server (e.g. "1234567890-0").
        """
        data: dict[str, str] = {
            "type": type(event).__name__,
            "payload": event.model_dump_json(),
        }
        msg_id: str = await self._redis.xadd(stream, data)
        return msg_id

    async def read(
        self,
        stream: str,
        count: int = 10,
        last_id: str = "0",
    ) -> list[tuple[str, dict[str, str]]]:
        """Read messages from a Redis Stream since *last_id*.

        Args:
            stream: Name of the stream to read from.
            count: Maximum number of messages to return.
            last_id: Only return messages with IDs greater than this value.
                     Use ``"0"`` to read from the beginning, or a previously
                     returned message ID to read only newer messages.

        Returns:
            A flat list of ``(message_id, fields)`` tuples.
            Returns an empty list when there are no new messages.
        """
        raw = await self._redis.xread({stream: last_id}, count=count)
        if not raw:
            return []

        messages: list[tuple[str, dict[str, str]]] = []
        for _stream_name, entries in raw:
            for msg_id, fields in entries:
                messages.append((msg_id, fields))
        return messages

    async def close(self) -> None:
        """Close the underlying Redis connection."""
        await self._redis.aclose()  # type: ignore[attr-defined]

    # ── Consumer group API (at-least-once delivery) ───────────────────────────

    async def create_group(self, stream: str, group: str, start_id: str = "0") -> None:
        """Create a Redis consumer group on *stream*, starting from *start_id*.

        If the stream does not yet exist it is created automatically (mkstream).
        If the group already exists the call is silently ignored.

        Args:
            stream: Stream name (use class-level constants).
            group: Consumer group name (unique per logical consumer process).
            start_id: Message ID at which the group starts consuming.
                      Use ``"0"`` to replay all existing messages, or ``"$"``
                      to consume only new messages written after group creation.
        """
        # ResponseError is raised when the group already exists; suppress it.
        with contextlib.suppress(Exception):
            await self._redis.xgroup_create(stream, group, id=start_id, mkstream=True)

    async def read_group(
        self,
        stream: str,
        group: str,
        consumer: str,
        count: int = 10,
    ) -> list[tuple[str, dict[str, str]]]:
        """Read *count* unacknowledged messages for *consumer* in *group*.

        Uses ``XREADGROUP`` with ``>`` so each message is delivered to at most
        one consumer in the group.  The caller must call ``ack()`` after
        successful processing to remove the message from the PEL.

        Args:
            stream: Stream name.
            group: Consumer group name.
            consumer: Unique name for this consumer instance.
            count: Maximum number of messages to return.

        Returns:
            A flat list of ``(message_id, fields)`` tuples.
        """
        raw = await self._redis.xreadgroup(group, consumer, {stream: ">"}, count=count)
        if not raw:
            return []

        messages: list[tuple[str, dict[str, str]]] = []
        for _stream_name, entries in raw:
            for msg_id, fields in entries:
                messages.append((msg_id, fields))
        return messages

    async def ack(self, stream: str, group: str, *message_ids: str) -> int:
        """Acknowledge one or more messages, removing them from the PEL.

        Args:
            stream: Stream name.
            group: Consumer group name.
            *message_ids: One or more message IDs returned by ``read_group``.

        Returns:
            Number of messages successfully acknowledged.
        """
        result: int = await self._redis.xack(stream, group, *message_ids)  # type: ignore[no-untyped-call]
        return result
