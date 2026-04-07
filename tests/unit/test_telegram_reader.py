"""Unit tests for TelegramChannelReader (Layer 2).

Tests the HTTP-based reader that parses t.me/s/ web preview pages.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from finalayze.data.fetchers.telegram_reader import (
    _MAX_SEEN_SIZE,
    TelegramChannelReader,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NOW = datetime.now(UTC)


def _make_html(messages: list[dict]) -> str:
    """Build a minimal t.me/s/ HTML page with message widgets."""
    widgets = []
    for msg in messages:
        text = msg.get("text", "")
        msg_id = msg.get("id", 1)
        channel = msg.get("channel", "test_channel")
        dt = msg.get("dt", _NOW - timedelta(minutes=2))
        dt_str = dt.isoformat()

        text_div = f'<div class="tgme_widget_message_text">{text}</div>' if text else ""
        widgets.append(
            f'<div class="tgme_widget_message_wrap">'
            f'  <div class="tgme_widget_message" data-post="{channel}/{msg_id}">'
            f"    {text_div}"
            f'    <time datetime="{dt_str}"></time>'
            f"  </div>"
            f"</div>"
        )
    body = "\n".join(widgets)
    return f"<html><body>{body}</body></html>"


def _mock_response(html: str, status: int = 200) -> httpx.Response:
    return httpx.Response(
        status_code=status, text=html, request=httpx.Request("GET", "https://t.me/s/test")
    )


# ---------------------------------------------------------------------------
# Tests — unconfigured
# ---------------------------------------------------------------------------


class TestTelegramChannelReaderUnconfigured:
    @pytest.mark.asyncio
    async def test_no_channels_returns_empty(self) -> None:
        reader = TelegramChannelReader(channels=[])
        result = await reader.fetch_recent_messages()
        assert result == []

    @pytest.mark.asyncio
    async def test_none_channels_returns_empty(self) -> None:
        reader = TelegramChannelReader()
        result = await reader.fetch_recent_messages()
        assert result == []

    def test_configured_property_false(self) -> None:
        reader = TelegramChannelReader()
        assert reader.configured is False

    def test_configured_property_true(self) -> None:
        reader = TelegramChannelReader(channels=["@rbc_news"])
        assert reader.configured is True


_ASYNC_CLIENT_PATH = "finalayze.data.fetchers.telegram_reader.httpx.AsyncClient"

# ---------------------------------------------------------------------------
# Tests — configured (mocked HTTP)
# ---------------------------------------------------------------------------


class TestTelegramChannelReaderConfigured:
    @pytest.mark.asyncio
    async def test_fetch_returns_news_articles(self) -> None:
        html = _make_html(
            [
                {
                    "text": "Breaking: Sberbank reports record profits in Q4",
                    "id": 100,
                    "channel": "fin_news",
                }
            ]
        )
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(return_value=_mock_response(html))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch(_ASYNC_CLIENT_PATH, return_value=mock_client):
            reader = TelegramChannelReader(channels=["@fin_news"])
            articles = await reader.fetch_recent_messages()

        assert len(articles) == 1
        assert articles[0].content == "Breaking: Sberbank reports record profits in Q4"
        assert articles[0].language == "ru"
        assert articles[0].scope == "russia"

    @pytest.mark.asyncio
    async def test_source_format(self) -> None:
        html = _make_html(
            [
                {
                    "text": "Test message with enough text",
                    "id": 42,
                    "channel": "fin_news",
                }
            ]
        )
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(return_value=_mock_response(html))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch(_ASYNC_CLIENT_PATH, return_value=mock_client):
            reader = TelegramChannelReader(channels=["@fin_news"])
            articles = await reader.fetch_recent_messages()

        assert articles[0].source == "telegram:@fin_news"

    @pytest.mark.asyncio
    async def test_url_contains_data_post(self) -> None:
        html = _make_html(
            [
                {
                    "text": "Another test message content",
                    "id": 42,
                    "channel": "fin_news",
                }
            ]
        )
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(return_value=_mock_response(html))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch(_ASYNC_CLIENT_PATH, return_value=mock_client):
            reader = TelegramChannelReader(channels=["@fin_news"])
            articles = await reader.fetch_recent_messages()

        assert articles[0].url == "https://t.me/fin_news/42"

    @pytest.mark.asyncio
    async def test_skips_messages_without_text(self) -> None:
        html = _make_html(
            [
                {"text": "", "id": 1},
                {"text": "Valid message with content", "id": 2},
            ]
        )
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(return_value=_mock_response(html))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch(_ASYNC_CLIENT_PATH, return_value=mock_client):
            reader = TelegramChannelReader(channels=["@test"])
            articles = await reader.fetch_recent_messages()

        assert len(articles) == 1
        assert articles[0].content == "Valid message with content"

    @pytest.mark.asyncio
    async def test_skips_short_messages(self) -> None:
        html = _make_html(
            [
                {"text": "Short", "id": 1},
                {"text": "This message is long enough to be accepted", "id": 2},
            ]
        )
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(return_value=_mock_response(html))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch(_ASYNC_CLIENT_PATH, return_value=mock_client):
            reader = TelegramChannelReader(channels=["@test"])
            articles = await reader.fetch_recent_messages()

        assert len(articles) == 1

    @pytest.mark.asyncio
    async def test_multiple_channels_combined(self) -> None:
        html_ch1 = _make_html(
            [
                {
                    "text": "Message from channel one here",
                    "id": 1,
                    "channel": "ch1",
                }
            ]
        )
        html_ch2 = _make_html(
            [
                {
                    "text": "Message from channel two here",
                    "id": 2,
                    "channel": "ch2",
                }
            ]
        )

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(
            side_effect=[
                _mock_response(html_ch1),
                _mock_response(html_ch2),
            ]
        )
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch(_ASYNC_CLIENT_PATH, return_value=mock_client):
            reader = TelegramChannelReader(channels=["@ch1", "@ch2"])
            articles = await reader.fetch_recent_messages()

        assert len(articles) == 2
        sources = {a.source for a in articles}
        assert sources == {"telegram:@ch1", "telegram:@ch2"}

    @pytest.mark.asyncio
    async def test_title_truncated_to_100_chars(self) -> None:
        long_text = "A" * 200
        html = _make_html([{"text": long_text, "id": 1}])
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(return_value=_mock_response(html))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch(_ASYNC_CLIENT_PATH, return_value=mock_client):
            reader = TelegramChannelReader(channels=["@test"])
            articles = await reader.fetch_recent_messages()

        assert len(articles[0].title) == 100
        assert articles[0].content == long_text

    @pytest.mark.asyncio
    async def test_channel_error_continues_others(self) -> None:
        html_good = _make_html(
            [
                {
                    "text": "Good message from second channel",
                    "id": 1,
                    "channel": "good",
                }
            ]
        )

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(
            side_effect=[
                httpx.HTTPStatusError(
                    "Not Found",
                    request=httpx.Request("GET", "https://t.me/s/bad"),
                    response=_mock_response("", 404),
                ),
                _mock_response(html_good),
            ]
        )
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch(_ASYNC_CLIENT_PATH, return_value=mock_client):
            reader = TelegramChannelReader(channels=["@bad_channel", "@good_channel"])
            articles = await reader.fetch_recent_messages()

        assert len(articles) == 1
        assert articles[0].source == "telegram:@good_channel"

    @pytest.mark.asyncio
    async def test_filters_old_messages(self) -> None:
        old_dt = _NOW - timedelta(hours=2)
        recent_dt = _NOW - timedelta(minutes=5)
        html = _make_html(
            [
                {"text": "Old message should be filtered out", "id": 1, "dt": old_dt},
                {"text": "Recent message should be included", "id": 2, "dt": recent_dt},
            ]
        )
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(return_value=_mock_response(html))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch(_ASYNC_CLIENT_PATH, return_value=mock_client):
            reader = TelegramChannelReader(channels=["@test"])
            articles = await reader.fetch_recent_messages(since_minutes=30)

        assert len(articles) == 1
        assert "Recent" in articles[0].content

    @pytest.mark.asyncio
    async def test_channels_override(self) -> None:
        """Passing channels to fetch_recent_messages overrides constructor list."""
        html = _make_html(
            [
                {
                    "text": "Override channel message here",
                    "id": 1,
                    "channel": "override",
                }
            ]
        )
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(return_value=_mock_response(html))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch(_ASYNC_CLIENT_PATH, return_value=mock_client):
            reader = TelegramChannelReader(channels=["@original"])
            articles = await reader.fetch_recent_messages(channels=["@override"])

        assert len(articles) == 1
        assert articles[0].source == "telegram:@override"


# ---------------------------------------------------------------------------
# Tests -- message deduplication (NEWS-04)
# ---------------------------------------------------------------------------


class TestTelegramChannelReaderDedup:
    """URL-based message deduplication prevents duplicate LLM processing."""

    @pytest.mark.asyncio
    async def test_dedup_skips_same_url_on_second_fetch(self) -> None:
        """Same message URL returned on first fetch, skipped on second."""
        html = _make_html(
            [
                {"text": "Breaking news about Sberbank today", "id": 100, "channel": "fin_news"},
            ]
        )
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(return_value=_mock_response(html))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "finalayze.data.fetchers.telegram_reader.httpx.AsyncClient",
            return_value=mock_client,
        ):
            reader = TelegramChannelReader(channels=["@fin_news"])

            first = await reader.fetch_recent_messages()
            assert len(first) == 1

            second = await reader.fetch_recent_messages()
            assert len(second) == 0

    @pytest.mark.asyncio
    async def test_dedup_different_urls_returned_normally(self) -> None:
        """Messages with different URLs are both returned."""
        html = _make_html(
            [
                {"text": "First message about markets today", "id": 100, "channel": "fin_news"},
                {"text": "Second different message about bonds", "id": 101, "channel": "fin_news"},
            ]
        )
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(return_value=_mock_response(html))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "finalayze.data.fetchers.telegram_reader.httpx.AsyncClient",
            return_value=mock_client,
        ):
            reader = TelegramChannelReader(channels=["@fin_news"])
            articles = await reader.fetch_recent_messages()

        assert len(articles) == 2

    @pytest.mark.asyncio
    async def test_dedup_evicts_oldest_beyond_max_size(self) -> None:
        """Seen URL set does not grow beyond _MAX_SEEN_SIZE; oldest evicted."""
        reader = TelegramChannelReader(channels=["@test"])

        # Manually populate _seen_urls to capacity
        for i in range(_MAX_SEEN_SIZE):
            reader._seen_urls[f"https://t.me/test/{i}"] = None

        assert len(reader._seen_urls) == _MAX_SEEN_SIZE

        # The very first URL should still be in the set
        assert "https://t.me/test/0" in reader._seen_urls

        # Add one more -- should evict the oldest (index 0)
        reader._seen_urls[f"https://t.me/test/{_MAX_SEEN_SIZE}"] = None
        if len(reader._seen_urls) > _MAX_SEEN_SIZE:
            reader._seen_urls.popitem(last=False)

        assert len(reader._seen_urls) == _MAX_SEEN_SIZE
        assert "https://t.me/test/0" not in reader._seen_urls
        assert f"https://t.me/test/{_MAX_SEEN_SIZE}" in reader._seen_urls

    def test_max_seen_size_constant(self) -> None:
        """_MAX_SEEN_SIZE is 5000."""
        assert _MAX_SEEN_SIZE == 5000
