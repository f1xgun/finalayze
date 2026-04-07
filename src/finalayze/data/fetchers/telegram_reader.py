"""Telegram channel reader for Russian financial news (Layer 2).

Fetches recent messages from public Telegram channels via t.me/s/ web preview.
No authentication required — uses plain HTTP GET + HTML parsing.
"""

from __future__ import annotations

from collections import OrderedDict
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import httpx
import structlog
from bs4 import BeautifulSoup, Tag

from finalayze.core.schemas import NewsArticle

logger = structlog.get_logger(__name__)

_MAX_MESSAGES_PER_CHANNEL = 50
_MAX_SEEN_SIZE = 5000
_MIN_TEXT_LENGTH = 10
_TITLE_MAX_LENGTH = 100
_USER_AGENT = "Mozilla/5.0 (compatible; Finalayze/1.0)"
_REQUEST_TIMEOUT = 15


class TelegramChannelReader:
    """Reads financial news from public Telegram channels via web preview.

    Parses ``https://t.me/s/<channel>`` HTML pages — no Telegram API
    credentials needed.  When ``channels`` list is empty, all fetch
    operations return an empty list without error.
    """

    def __init__(self, *, channels: list[str] | None = None) -> None:
        self._channels = channels or []
        self._seen_urls: OrderedDict[str, None] = OrderedDict()

    @property
    def configured(self) -> bool:
        """Whether any channels are set."""
        return len(self._channels) > 0

    async def fetch_recent_messages(
        self,
        channels: list[str] | None = None,
        since_minutes: int = 30,
    ) -> list[NewsArticle]:
        """Fetch recent messages from public Telegram channels.

        Args:
            channels: Override channel list (uses constructor list if *None*).
            since_minutes: Only return messages from the last N minutes.

        Returns:
            List of :class:`NewsArticle`, may be empty.
        """
        target_channels = channels if channels is not None else self._channels
        if not target_channels:
            logger.debug("telegram_reader_no_channels")
            return []

        cutoff = datetime.now(UTC) - timedelta(minutes=since_minutes)
        articles: list[NewsArticle] = []

        async with httpx.AsyncClient(
            headers={"User-Agent": _USER_AGENT},
            timeout=_REQUEST_TIMEOUT,
            follow_redirects=True,
        ) as client:
            for channel in target_channels:
                try:
                    channel_articles = await self._fetch_channel(client, channel, cutoff)
                    articles.extend(channel_articles)
                except Exception:
                    logger.warning(
                        "telegram_channel_fetch_failed",
                        channel=channel,
                        exc_info=True,
                    )
                    continue

        return articles

    async def _fetch_channel(
        self,
        client: httpx.AsyncClient,
        channel: str,
        cutoff: datetime,
    ) -> list[NewsArticle]:
        """Parse a single channel's web preview page."""
        channel_name = channel.lstrip("@")
        url = f"https://t.me/s/{channel_name}"
        resp = await client.get(url)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")
        widgets = soup.select(".tgme_widget_message_wrap")

        articles: list[NewsArticle] = []
        for widget in widgets[-_MAX_MESSAGES_PER_CHANNEL:]:
            article = self._parse_message(widget, channel, cutoff)
            if article is not None:
                articles.append(article)

        logger.info(
            "telegram_channel_fetched",
            channel=channel,
            count=len(articles),
        )
        return articles

    def _parse_message(  # noqa: PLR0911, PLR0912
        self,
        widget: Tag,
        channel: str,
        cutoff: datetime,
    ) -> NewsArticle | None:
        """Extract a NewsArticle from a single message widget, or *None*."""
        # Extract timestamp
        time_tag = widget.select_one("time[datetime]")
        if time_tag is None:
            return None

        dt_str = time_tag.get("datetime", "")
        if isinstance(dt_str, list):
            dt_str = dt_str[0] if dt_str else ""
        if dt_str is None:
            return None
        try:
            published = datetime.fromisoformat(dt_str.replace("+00:00", "+00:00"))
            if published.tzinfo is None:
                published = published.replace(tzinfo=UTC)
        except (ValueError, AttributeError):
            return None

        if published < cutoff:
            return None

        # Extract text
        text_el = widget.select_one(".tgme_widget_message_text")
        if text_el is None:
            return None

        text = text_el.get_text(strip=True)
        if len(text) < _MIN_TEXT_LENGTH:
            return None

        # Extract message link
        channel_name = channel.lstrip("@")
        link_el = widget.select_one(".tgme_widget_message[data-post]")
        if link_el is not None:
            data_post = link_el.get("data-post", "")
            if isinstance(data_post, list):
                data_post = data_post[0] if data_post else ""
            msg_url = f"https://t.me/{data_post}" if data_post else f"https://t.me/{channel_name}"
        else:
            msg_url = f"https://t.me/{channel_name}"

        # URL-based deduplication -- skip messages already seen
        if msg_url in self._seen_urls:
            return None
        self._seen_urls[msg_url] = None
        if len(self._seen_urls) > _MAX_SEEN_SIZE:
            self._seen_urls.popitem(last=False)

        return NewsArticle(
            id=uuid4(),
            source=f"telegram:{channel}",
            title=text[:_TITLE_MAX_LENGTH],
            content=text,
            url=msg_url,
            language="ru",
            published_at=published,
            scope="russia",
        )
