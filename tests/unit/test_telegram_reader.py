"""Unit tests for TelegramChannelReader (Layer 2)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from finalayze.data.fetchers.telegram_reader import TelegramChannelReader

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_msg(*, text: str | None, msg_id: int = 1, minutes_ago: int = 2) -> SimpleNamespace:
    """Create a mock Telegram message."""
    return SimpleNamespace(
        text=text,
        id=msg_id,
        date=datetime.now(UTC) - timedelta(minutes=minutes_ago),
    )


async def _async_iter(items: list[SimpleNamespace]):  # noqa: RUF029
    """Helper to create an async iterator from a list."""
    for item in items:
        yield item


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTelegramChannelReaderUnconfigured:
    """Tests for unconfigured reader (no credentials)."""

    @pytest.mark.asyncio
    async def test_unconfigured_api_id_zero_returns_empty(self) -> None:
        reader = TelegramChannelReader(api_id=0, api_hash="somehash")
        result = await reader.fetch_recent_messages(channels=["@test_channel"])
        assert result == []

    @pytest.mark.asyncio
    async def test_unconfigured_api_hash_empty_returns_empty(self) -> None:
        reader = TelegramChannelReader(api_id=12345, api_hash="")
        result = await reader.fetch_recent_messages(channels=["@test_channel"])
        assert result == []


class TestTelegramChannelReaderConfigured:
    """Tests for configured reader with mocked Telethon client."""

    @pytest.mark.asyncio
    async def test_fetch_returns_news_articles(self) -> None:
        msgs = [
            _make_msg(text="Breaking: Sberbank reports record profits in Q4", msg_id=100),
        ]
        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.iter_messages = MagicMock(return_value=_async_iter(msgs))

        with patch("telethon.TelegramClient", return_value=mock_client):
            reader = TelegramChannelReader(api_id=12345, api_hash="abc123")
            articles = await reader.fetch_recent_messages(channels=["@fin_news"])

        assert len(articles) == 1
        article = articles[0]
        assert article.content == "Breaking: Sberbank reports record profits in Q4"
        assert article.language == "ru"
        assert article.scope == "russia"

    @pytest.mark.asyncio
    async def test_source_format(self) -> None:
        msgs = [_make_msg(text="Test message with enough text", msg_id=42)]
        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.iter_messages = MagicMock(return_value=_async_iter(msgs))

        with patch("telethon.TelegramClient", return_value=mock_client):
            reader = TelegramChannelReader(api_id=12345, api_hash="abc123")
            articles = await reader.fetch_recent_messages(channels=["@fin_news"])

        assert articles[0].source == "telegram:@fin_news"

    @pytest.mark.asyncio
    async def test_url_format(self) -> None:
        msgs = [_make_msg(text="Another test message content", msg_id=42)]
        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.iter_messages = MagicMock(return_value=_async_iter(msgs))

        with patch("telethon.TelegramClient", return_value=mock_client):
            reader = TelegramChannelReader(api_id=12345, api_hash="abc123")
            articles = await reader.fetch_recent_messages(channels=["@fin_news"])

        assert articles[0].url == "https://t.me/fin_news/42"

    @pytest.mark.asyncio
    async def test_skips_messages_without_text(self) -> None:
        msgs = [
            _make_msg(text=None, msg_id=1),
            _make_msg(text="Valid message with content", msg_id=2),
        ]
        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.iter_messages = MagicMock(return_value=_async_iter(msgs))

        with patch("telethon.TelegramClient", return_value=mock_client):
            reader = TelegramChannelReader(api_id=12345, api_hash="abc123")
            articles = await reader.fetch_recent_messages(channels=["@test"])

        assert len(articles) == 1
        assert articles[0].content == "Valid message with content"

    @pytest.mark.asyncio
    async def test_skips_short_messages(self) -> None:
        msgs = [
            _make_msg(text="Short", msg_id=1),
            _make_msg(text="This message is long enough to be accepted", msg_id=2),
        ]
        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.iter_messages = MagicMock(return_value=_async_iter(msgs))

        with patch("telethon.TelegramClient", return_value=mock_client):
            reader = TelegramChannelReader(api_id=12345, api_hash="abc123")
            articles = await reader.fetch_recent_messages(channels=["@test"])

        assert len(articles) == 1

    @pytest.mark.asyncio
    async def test_multiple_channels_combined(self) -> None:
        msgs_ch1 = [_make_msg(text="Message from channel one here", msg_id=1)]
        msgs_ch2 = [_make_msg(text="Message from channel two here", msg_id=2)]

        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.iter_messages = MagicMock(
            side_effect=[_async_iter(msgs_ch1), _async_iter(msgs_ch2)]
        )

        with patch("telethon.TelegramClient", return_value=mock_client):
            reader = TelegramChannelReader(api_id=12345, api_hash="abc123")
            articles = await reader.fetch_recent_messages(channels=["@ch1", "@ch2"])

        assert len(articles) == 2
        sources = {a.source for a in articles}
        assert sources == {"telegram:@ch1", "telegram:@ch2"}

    @pytest.mark.asyncio
    async def test_title_truncated_to_100_chars(self) -> None:
        long_text = "A" * 200
        msgs = [_make_msg(text=long_text, msg_id=1)]
        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.iter_messages = MagicMock(return_value=_async_iter(msgs))

        with patch("telethon.TelegramClient", return_value=mock_client):
            reader = TelegramChannelReader(api_id=12345, api_hash="abc123")
            articles = await reader.fetch_recent_messages(channels=["@test"])

        title_len = 100
        assert len(articles[0].title) == title_len
        assert articles[0].content == long_text

    @pytest.mark.asyncio
    async def test_channel_error_continues_others(self) -> None:
        """If one channel raises, others still get processed."""
        msgs_ch2 = [_make_msg(text="Good message from second channel", msg_id=1)]

        def _side_effect(channel, **_kwargs):
            if channel == "@bad_channel":
                raise Exception("Connection failed")  # noqa: TRY002
            return _async_iter(msgs_ch2)

        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.iter_messages = MagicMock(side_effect=_side_effect)

        with patch("telethon.TelegramClient", return_value=mock_client):
            reader = TelegramChannelReader(api_id=12345, api_hash="abc123")
            channels = ["@bad_channel", "@good_channel"]
            articles = await reader.fetch_recent_messages(channels=channels)

        assert len(articles) == 1
        assert articles[0].source == "telegram:@good_channel"
