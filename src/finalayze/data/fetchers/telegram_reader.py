"""Telegram channel reader for Russian financial news (Layer 2).

Fetches recent messages from configured Telegram channels and converts
them to :class:`~finalayze.core.schemas.NewsArticle` objects.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING
from uuid import uuid4

import structlog

from finalayze.core.schemas import NewsArticle

if TYPE_CHECKING:
    from telethon import TelegramClient

logger = structlog.get_logger(__name__)

_MAX_MESSAGES_PER_CHANNEL = 50  # Cap per poll cycle
_MIN_TEXT_LENGTH = 10  # Filter noise / media-only messages
_TITLE_MAX_LENGTH = 100


class TelegramChannelReader:
    """Reads financial news messages from Telegram channels.

    Uses Telethon to connect to Telegram and iterate over recent messages
    in configured channels. Returns a list of NewsArticle objects.

    When credentials are not configured (api_id=0 or api_hash=""),
    all fetch operations return an empty list without error.
    """

    def __init__(
        self,
        api_id: int,
        api_hash: str,
        session_name: str = "finalayze_reader",
    ) -> None:
        self._api_id = api_id
        self._api_hash = api_hash
        self._session_name = session_name
        self._configured = api_id != 0 and api_hash != ""

    async def fetch_recent_messages(
        self,
        channels: list[str],
        since_minutes: int = 5,
    ) -> list[NewsArticle]:
        """Fetch recent messages from Telegram channels.

        Args:
            channels: List of Telegram channel usernames (e.g. ``["@fin_news"]``).
            since_minutes: Only return messages from the last N minutes.

        Returns:
            List of NewsArticle objects, may be empty.
        """
        if not self._configured:
            logger.debug("telegram_reader_not_configured")
            return []

        from telethon import TelegramClient  # noqa: PLC0415

        cutoff = datetime.now(UTC) - timedelta(minutes=since_minutes)
        articles: list[NewsArticle] = []

        client: TelegramClient = TelegramClient(
            self._session_name,
            self._api_id,
            self._api_hash,
        )

        async with client:
            for channel in channels:
                try:
                    async for msg in client.iter_messages(
                        channel,
                        offset_date=cutoff,
                        reverse=True,
                        limit=_MAX_MESSAGES_PER_CHANNEL,
                    ):
                        if not msg.text or len(msg.text.strip()) < _MIN_TEXT_LENGTH:
                            continue

                        channel_name = channel.lstrip("@")
                        article = NewsArticle(
                            id=uuid4(),
                            source=f"telegram:{channel}",
                            title=msg.text[:_TITLE_MAX_LENGTH],
                            content=msg.text,
                            url=f"https://t.me/{channel_name}/{msg.id}",
                            language="ru",
                            published_at=msg.date,
                            scope="russia",
                        )
                        articles.append(article)
                except Exception:
                    logger.warning(
                        "telegram_channel_fetch_failed",
                        channel=channel,
                        exc_info=True,
                    )
                    continue

        return articles
