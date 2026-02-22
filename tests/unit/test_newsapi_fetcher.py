"""Unit tests for NewsApiFetcher."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest

from finalayze.core.exceptions import DataFetchError, RateLimitError
from finalayze.core.schemas import NewsArticle
from finalayze.data.fetchers.newsapi import NewsApiFetcher

# Constants
_API_KEY = "test-key"
_FROM_DATE = datetime(2024, 1, 1, tzinfo=UTC)
_TO_DATE = datetime(2024, 1, 7, tzinfo=UTC)

_SAMPLE_RESPONSE = {
    "status": "ok",
    "totalResults": 1,
    "articles": [
        {
            "source": {"name": "Reuters"},
            "title": "Fed raises rates",
            "description": "Brief description",
            "content": "Full content here",
            "url": "https://reuters.com/article/1",
            "publishedAt": "2024-01-03T10:00:00Z",
        }
    ],
}


class TestNewsApiFetcherInit:
    def test_init_stores_api_key(self) -> None:
        fetcher = NewsApiFetcher(api_key=_API_KEY)
        assert fetcher._api_key == _API_KEY


class TestNewsApiFetcherSuccess:
    def test_returns_news_articles(self) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = _SAMPLE_RESPONSE

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.get.return_value = mock_response
            mock_client_cls.return_value = mock_client

            fetcher = NewsApiFetcher(api_key=_API_KEY)
            articles = fetcher.fetch_news("Fed", _FROM_DATE, _TO_DATE)

        assert len(articles) == 1
        assert isinstance(articles[0], NewsArticle)
        assert articles[0].title == "Fed raises rates"
        assert articles[0].source == "Reuters"
        assert articles[0].language == "en"

    def test_empty_results_returns_empty_list(self) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "ok", "totalResults": 0, "articles": []}

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.get.return_value = mock_response
            mock_client_cls.return_value = mock_client

            fetcher = NewsApiFetcher(api_key=_API_KEY)
            articles = fetcher.fetch_news("xyz", _FROM_DATE, _TO_DATE)

        assert articles == []


class TestNewsApiFetcherErrors:
    def test_rate_limit_raises_rate_limit_error(self) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 429

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.get.return_value = mock_response
            mock_client_cls.return_value = mock_client

            fetcher = NewsApiFetcher(api_key=_API_KEY)
            with pytest.raises(RateLimitError):
                fetcher.fetch_news("test", _FROM_DATE, _TO_DATE)

    def test_api_error_status_raises_data_fetch_error(self) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "error", "message": "apiKeyInvalid"}

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.get.return_value = mock_response
            mock_client_cls.return_value = mock_client

            fetcher = NewsApiFetcher(api_key=_API_KEY)
            with pytest.raises(DataFetchError):
                fetcher.fetch_news("test", _FROM_DATE, _TO_DATE)
