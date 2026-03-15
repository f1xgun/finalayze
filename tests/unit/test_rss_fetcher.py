"""Unit tests for RssNewsFetcher (Layer 2)."""

from __future__ import annotations

import time
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest

from finalayze.data.fetchers.rss_fetcher import RssNewsFetcher


def _make_entry(
    title: str = "Test headline",
    summary: str = "Some content",
    link: str = "https://example.com/article1",
    published_parsed: tuple[int, ...] | None = (2026, 3, 15, 10, 0, 0, 5, 74, 0),
) -> dict[str, object]:
    """Build a fake feedparser entry dict."""
    entry: dict[str, object] = {
        "title": title,
        "summary": summary,
        "link": link,
    }
    if published_parsed is not None:
        entry["published_parsed"] = time.struct_time(published_parsed)
    return entry


def _make_feed(entries: list[dict[str, object]], bozo: bool = False) -> MagicMock:
    feed = MagicMock()
    feed.entries = entries
    feed.bozo = bozo
    return feed


@pytest.fixture
def fetcher() -> RssNewsFetcher:
    return RssNewsFetcher(feed_urls=["https://rbc.ru/rss", "https://tass.com/rss"])


class TestFetchNewsParsing:
    """RssNewsFetcher.fetch_news parses entries into NewsArticle list."""

    @patch("finalayze.data.fetchers.rss_fetcher.feedparser")
    def test_parses_entries_into_news_articles(
        self, mock_fp: MagicMock, fetcher: RssNewsFetcher
    ) -> None:
        entry = _make_entry(title="RBC headline", summary="Content here", link="https://rbc.ru/1")
        mock_fp.parse.return_value = _make_feed([entry])

        articles = fetcher.fetch_news()

        # 2 feeds return same entry with same link -> deduplicated to 1
        assert len(articles) == 1
        art = articles[0]
        assert art.title == "RBC headline"
        assert art.content == "Content here"
        assert art.url == "https://rbc.ru/1"
        assert art.language == "ru"
        assert art.scope == "russia"
        assert art.source == "rss"
        assert art.published_at.year == 2026


class TestDeduplication:
    """Articles are deduplicated by URL."""

    @patch("finalayze.data.fetchers.rss_fetcher.feedparser")
    def test_dedup_within_single_fetch(self, mock_fp: MagicMock, fetcher: RssNewsFetcher) -> None:
        """Same URL in two feeds -> 1 article."""
        entry = _make_entry(link="https://rbc.ru/same")
        mock_fp.parse.return_value = _make_feed([entry])

        articles = fetcher.fetch_news()
        assert len(articles) == 1

    @patch("finalayze.data.fetchers.rss_fetcher.feedparser")
    def test_dedup_across_fetch_cycles(self, mock_fp: MagicMock, fetcher: RssNewsFetcher) -> None:
        """Second call skips previously seen URLs."""
        entry = _make_entry(link="https://rbc.ru/same")
        mock_fp.parse.return_value = _make_feed([entry])

        first = fetcher.fetch_news()
        second = fetcher.fetch_news()

        assert len(first) == 1
        assert len(second) == 0

    @patch("finalayze.data.fetchers.rss_fetcher.feedparser")
    def test_dedup_bounded_eviction(
        self,
        mock_fp: MagicMock,
    ) -> None:
        """After MAX_SEEN_SIZE entries, old URLs are evicted."""
        small_fetcher = RssNewsFetcher(feed_urls=["https://rbc.ru/rss"])
        # Override max size for test
        small_fetcher._MAX_SEEN_SIZE = 3  # type: ignore[attr-defined]

        # Feed 4 unique URLs (exceeds max of 3)
        for i in range(4):
            entry = _make_entry(link=f"https://rbc.ru/{i}")
            mock_fp.parse.return_value = _make_feed([entry])
            small_fetcher.fetch_news()

        # Oldest URL (0) should have been evicted, so re-fetching it should work
        entry = _make_entry(link="https://rbc.ru/0")
        mock_fp.parse.return_value = _make_feed([entry])
        result = small_fetcher.fetch_news()
        assert len(result) == 1


class TestMalformedEntries:
    """Malformed feed entries are skipped gracefully."""

    @patch("finalayze.data.fetchers.rss_fetcher.feedparser")
    def test_missing_link_skipped(self, mock_fp: MagicMock, fetcher: RssNewsFetcher) -> None:
        entry_no_link = _make_entry(link="")
        entry_ok = _make_entry(link="https://rbc.ru/ok")
        mock_fp.parse.return_value = _make_feed([entry_no_link, entry_ok])

        articles = fetcher.fetch_news()
        assert len(articles) == 1
        assert articles[0].url == "https://rbc.ru/ok"

    @patch("finalayze.data.fetchers.rss_fetcher.feedparser")
    def test_bozo_feed_still_returns_articles(
        self, mock_fp: MagicMock, fetcher: RssNewsFetcher
    ) -> None:
        """feedparser bozo=True with entries still returns articles."""
        entry = _make_entry(link="https://rbc.ru/bozo")
        mock_fp.parse.return_value = _make_feed([entry], bozo=True)

        articles = fetcher.fetch_news()
        assert len(articles) == 1

    @patch("finalayze.data.fetchers.rss_fetcher.feedparser")
    def test_empty_feed_returns_empty_list(
        self, mock_fp: MagicMock, fetcher: RssNewsFetcher
    ) -> None:
        mock_fp.parse.return_value = _make_feed([])

        articles = fetcher.fetch_news()
        assert articles == []

    @patch("finalayze.data.fetchers.rss_fetcher.feedparser")
    def test_missing_published_parsed_uses_fallback(
        self, mock_fp: MagicMock, fetcher: RssNewsFetcher
    ) -> None:
        entry = _make_entry(link="https://rbc.ru/nopub", published_parsed=None)
        mock_fp.parse.return_value = _make_feed([entry])

        articles = fetcher.fetch_news()
        assert len(articles) == 1
        # Should have a valid datetime (fallback to now)
        assert articles[0].published_at.tzinfo is not None
