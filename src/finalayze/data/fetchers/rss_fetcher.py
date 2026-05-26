"""RSS news fetcher for Russian financial news feeds (Layer 2).

Parses RSS feeds from RBC, Interfax, TASS, etc. into NewsArticle objects
with URL-based deduplication across polling cycles.
"""

from __future__ import annotations

from collections import OrderedDict
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import feedparser  # type: ignore[import-untyped]
import structlog
from dateutil.parser import parse as dateutil_parse

from finalayze.core.schemas import NewsArticle

if TYPE_CHECKING:
    from finalayze.data.rate_limiter import RateLimiter

log = structlog.get_logger(__name__)
# Module-private alias used by S5.1 logging hooks (mirrors the project-wide
# ``_log`` convention while keeping the legacy ``log`` name in callers).
_log = log

_MAX_SEEN_SIZE = 5000


class RssNewsFetcher:
    """Fetches and deduplicates Russian news articles from RSS feeds.

    Uses ``feedparser`` to parse RSS/Atom feeds and produces
    :class:`~finalayze.core.schemas.NewsArticle` objects with
    ``language="ru"`` and ``scope="russia"``.

    Deduplication is URL-based with an LRU-bounded seen set.
    """

    _MAX_SEEN_SIZE: int = _MAX_SEEN_SIZE

    def __init__(
        self,
        feed_urls: list[str],
        rate_limiter: RateLimiter | None = None,
    ) -> None:
        self._feed_urls = feed_urls
        self._rate_limiter = rate_limiter
        self._seen_urls: OrderedDict[str, None] = OrderedDict()

    def fetch_news(self) -> list[NewsArticle]:
        """Fetch articles from all configured RSS feeds.

        Returns:
            Deduplicated list of NewsArticle objects from all feeds.
        """
        articles: list[NewsArticle] = []

        for url in self._feed_urls:
            if self._rate_limiter is not None:
                self._rate_limiter.acquire()

            try:
                feed = feedparser.parse(url)
            except Exception:
                log.warning("rss_feed_fetch_failed", url=url)
                continue

            for entry in feed.entries:
                try:
                    article = self._parse_entry(entry)
                    if article is not None:
                        articles.append(article)
                except Exception:
                    log.warning("rss_entry_parse_failed", url=url)
                    continue

        return articles

    def _parse_entry(self, entry: Any) -> NewsArticle | None:
        """Parse a single feedparser entry into a NewsArticle.

        Returns None if the entry should be skipped (empty URL or already seen).
        """
        link = entry.get("link", "")
        if not link:
            return None

        # Dedup check
        if link in self._seen_urls:
            return None

        # Add to seen set with LRU eviction
        self._seen_urls[link] = None
        if len(self._seen_urls) > self._MAX_SEEN_SIZE:
            self._seen_urls.popitem(last=False)

        title = entry.get("title", "")
        content = entry.get("summary", "") or entry.get("description", "")
        published_at = self._parse_published(entry)
        # S5.1: skip entries whose timestamp can't be parsed instead of
        # tagging them with "now". Otherwise stale or undated feed items
        # silently contaminate sentiment / signal generation.
        if published_at is None:
            _log.warning("news_invalid_published_at", url=link)
            return None

        return NewsArticle(
            id=uuid4(),
            source="rss",
            title=title,
            content=content,
            url=link,
            language="ru",
            published_at=published_at,
            scope="russia",
        )

    def _parse_published(self, entry: Any) -> datetime | None:
        """Extract publication datetime from a feedparser entry.

        Returns ``None`` when neither ``published_parsed`` nor ``published``
        yields a valid timestamp — the caller drops the entry rather than
        substituting ``datetime.now(UTC)`` (S5.1).
        """
        tp = entry.get("published_parsed")
        if tp is not None:
            try:
                return datetime(
                    tp[0],
                    tp[1],
                    tp[2],
                    tp[3],
                    tp[4],
                    tp[5],
                    tzinfo=UTC,
                )
            except (TypeError, ValueError, IndexError):
                pass

        # Try dateutil as fallback
        published_str = entry.get("published", "")
        if published_str:
            try:
                dt = dateutil_parse(published_str)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=UTC)
                return dt
            except (ValueError, TypeError):
                pass

        return None
