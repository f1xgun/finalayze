"""NewsAPI REST fetcher for news articles (Layer 2)."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import httpx

from finalayze.core.exceptions import DataFetchError, RateLimitError
from finalayze.core.schemas import NewsArticle

_BASE_URL = "https://newsapi.org/v2/everything"
_HTTP_OK = 200
_HTTP_RATE_LIMIT = 429
_STATUS_OK = "ok"


class NewsApiFetcher:
    """Fetches news articles from the NewsAPI v2 REST API.

    Uses the ``/v2/everything`` endpoint for keyword-based article search.
    Returns a list of :class:`~finalayze.core.schemas.NewsArticle` objects.
    """

    def __init__(self, api_key: str, language: str = "en") -> None:
        self._api_key = api_key
        self._language = language

    def fetch_news(
        self,
        query: str,
        from_date: datetime,
        to_date: datetime,
        page_size: int = 20,
    ) -> list[NewsArticle]:
        """Fetch news articles matching the query in the given date range.

        Args:
            query: Search keywords (e.g. ticker symbol or topic).
            from_date: Start of the date range (inclusive, UTC-aware).
            to_date: End of the date range (exclusive, UTC-aware).
            page_size: Max results per request (1-100, NewsAPI limit).

        Returns:
            List of NewsArticle objects, may be empty if no results.

        Raises:
            RateLimitError: When the API returns HTTP 429.
            DataFetchError: On HTTP errors or API-level error responses.
        """
        params: dict[str, str | int] = {
            "q": query,
            "from": from_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "to": to_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "language": self._language,
            "sortBy": "publishedAt",
            "pageSize": page_size,
            "apiKey": self._api_key,
        }

        with httpx.Client() as client:
            try:
                response = client.get(_BASE_URL, params=params)
            except httpx.HTTPError as exc:
                msg = f"NewsAPI HTTP request failed: {exc}"
                raise DataFetchError(msg) from exc

        if response.status_code == _HTTP_RATE_LIMIT:
            msg = "NewsAPI rate limit exceeded"
            raise RateLimitError(msg)

        if response.status_code != _HTTP_OK:
            msg = f"NewsAPI HTTP error: {response.status_code}"
            raise DataFetchError(msg)

        data = response.json()
        if data.get("status") != _STATUS_OK:
            msg = f"NewsAPI error: {data.get('message', 'unknown')}"
            raise DataFetchError(msg)

        return [self._parse_article(raw) for raw in data.get("articles", [])]

    def _parse_article(self, raw: dict[str, object]) -> NewsArticle:
        """Convert a raw NewsAPI article dict to a NewsArticle schema."""
        source_obj = raw.get("source") or {}
        source_name = (
            source_obj.get("name", "unknown") if isinstance(source_obj, dict) else "unknown"
        )
        published_raw = raw.get("publishedAt", "")
        try:
            # NewsAPI returns RFC 3339 timestamps like "2024-01-03T10:00:00Z"
            published_at = datetime.fromisoformat(str(published_raw))
        except (ValueError, TypeError):
            published_at = datetime.now(UTC)

        return NewsArticle(
            id=uuid4(),
            source=str(source_name),
            title=str(raw.get("title") or ""),
            content=str(raw.get("content") or raw.get("description") or ""),
            url=str(raw.get("url") or ""),
            language=self._language,
            published_at=published_at,
        )
