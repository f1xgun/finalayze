"""S5.1 — News fetchers must drop articles with un-parseable timestamps.

The audit flagged ``NewsApiFetcher`` and ``RssNewsFetcher`` for silently
substituting ``datetime.now(UTC)`` when ``published_at`` could not be
parsed. A stale or corrupt news item tagged with "now" can:
  * Leak into the same-day signal pipeline as if it were fresh.
  * Poison training datasets (look-ahead bias).
  * Skew sentiment averages with old / undated articles.

Contract:
  NEWSTS-01: NewsApiFetcher drops articles with missing / unparseable
             ``publishedAt`` and logs ``news_invalid_published_at``.
  NEWSTS-02: NewsApiFetcher keeps valid timestamps untouched (TZ-aware).
  NEWSTS-03: RssNewsFetcher drops entries when neither
             ``published_parsed`` nor ``published`` can be parsed.
  NEWSTS-04: RssNewsFetcher still publishes entries with a valid date.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch


def _fake_newsapi_response(articles: list[dict]) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"status": "ok", "articles": articles}
    return resp


# ─── NEWSTS-01 ───────────────────────────────────────────────────────────────
def test_newsapi_drops_article_with_unparseable_published_at() -> None:
    from finalayze.data.fetchers.newsapi import NewsApiFetcher

    fetcher = NewsApiFetcher(api_key="k")
    raw = [
        {
            "source": {"name": "Reuters"},
            "title": "Bad date item",
            "content": "...",
            "url": "https://example.com/bad",
            "publishedAt": "not-a-date",
        },
        {
            "source": {"name": "Reuters"},
            "title": "Good date item",
            "content": "...",
            "url": "https://example.com/good",
            "publishedAt": "2024-01-03T10:00:00Z",
        },
    ]
    with (
        patch("httpx.Client") as mock_client_cls,
        patch("finalayze.data.fetchers.newsapi._log") as mock_log,
    ):
        client = mock_client_cls.return_value.__enter__.return_value
        client.get.return_value = _fake_newsapi_response(raw)
        articles = fetcher.fetch_news(
            "test",
            from_date=datetime(2024, 1, 1, tzinfo=UTC),
            to_date=datetime(2024, 1, 4, tzinfo=UTC),
        )

    assert len(articles) == 1
    assert articles[0].title == "Good date item"
    # The dropped article must surface in logs so it isn't a silent failure.
    warned = [
        c
        for c in mock_log.warning.call_args_list
        if c.args and c.args[0] == "news_invalid_published_at"
    ]
    assert warned, f"expected warning, got: {mock_log.warning.call_args_list}"


def test_newsapi_drops_article_with_empty_published_at() -> None:
    from finalayze.data.fetchers.newsapi import NewsApiFetcher

    fetcher = NewsApiFetcher(api_key="k")
    raw = [
        {
            "source": {"name": "Reuters"},
            "title": "No date",
            "content": "...",
            "url": "https://example.com/none",
            # publishedAt missing entirely
        },
    ]
    with patch("httpx.Client") as mock_client_cls:
        client = mock_client_cls.return_value.__enter__.return_value
        client.get.return_value = _fake_newsapi_response(raw)
        articles = fetcher.fetch_news(
            "test",
            from_date=datetime(2024, 1, 1, tzinfo=UTC),
            to_date=datetime(2024, 1, 4, tzinfo=UTC),
        )

    assert articles == []


# ─── NEWSTS-02 ───────────────────────────────────────────────────────────────
def test_newsapi_keeps_valid_timestamp() -> None:
    """Valid published_at must round-trip unchanged (TZ-aware)."""
    from finalayze.data.fetchers.newsapi import NewsApiFetcher

    fetcher = NewsApiFetcher(api_key="k")
    raw = [
        {
            "source": {"name": "Reuters"},
            "title": "Good",
            "content": "...",
            "url": "https://example.com/g",
            "publishedAt": "2024-01-03T10:00:00Z",
        },
    ]
    with patch("httpx.Client") as mock_client_cls:
        client = mock_client_cls.return_value.__enter__.return_value
        client.get.return_value = _fake_newsapi_response(raw)
        articles = fetcher.fetch_news(
            "test",
            from_date=datetime(2024, 1, 1, tzinfo=UTC),
            to_date=datetime(2024, 1, 4, tzinfo=UTC),
        )

    assert len(articles) == 1
    assert articles[0].published_at == datetime(2024, 1, 3, 10, 0, 0, tzinfo=UTC)
    assert articles[0].published_at.tzinfo is not None


def _make_rss_entry(payload: dict[str, object]) -> MagicMock:
    """Build a feedparser-style entry with ``.get(key, default)``."""
    entry = MagicMock()
    entry.get = lambda key, default=None: payload.get(key, default)
    return entry


# ─── NEWSTS-03 ───────────────────────────────────────────────────────────────
def test_rss_drops_entry_with_unparseable_dates() -> None:
    from finalayze.data.fetchers.rss_fetcher import RssNewsFetcher

    fetcher = RssNewsFetcher(feed_urls=["https://example.com/feed"])
    bad_entry = _make_rss_entry(
        {
            "title": "Bad date",
            "summary": "...",
            "link": "https://example.com/bad",
            "published_parsed": None,
            "published": "not-a-date",
        }
    )
    good_entry = _make_rss_entry(
        {
            "title": "Good",
            "summary": "...",
            "link": "https://example.com/good",
            "published": "Mon, 02 Jan 2024 10:00:00 GMT",
        }
    )

    feed = MagicMock()
    feed.entries = [bad_entry, good_entry]
    with (
        patch("feedparser.parse", return_value=feed),
        patch("finalayze.data.fetchers.rss_fetcher._log") as mock_log,
    ):
        articles = fetcher.fetch_news()

    # Only the good one survives
    titles = [a.title for a in articles]
    assert titles == ["Good"]
    warned = [
        c
        for c in mock_log.warning.call_args_list
        if c.args and c.args[0] == "news_invalid_published_at"
    ]
    assert warned


# ─── NEWSTS-04 ───────────────────────────────────────────────────────────────
def test_rss_keeps_valid_entries() -> None:
    from finalayze.data.fetchers.rss_fetcher import RssNewsFetcher

    fetcher = RssNewsFetcher(feed_urls=["https://example.com/feed"])
    good = _make_rss_entry(
        {
            "title": "Good",
            "summary": "...",
            "link": "https://example.com/g",
            "published": "Mon, 02 Jan 2024 10:00:00 GMT",
        }
    )

    feed = MagicMock()
    feed.entries = [good]
    with patch("feedparser.parse", return_value=feed):
        articles = fetcher.fetch_news()

    assert len(articles) == 1
    assert articles[0].published_at.tzinfo is not None
