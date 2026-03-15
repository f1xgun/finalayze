"""Unit tests for EntityExtractor (Layer 3)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from finalayze.analysis.entity_extractor import EntityExtractor
from finalayze.core.schemas import NewsArticle


def _make_article(
    title: str = "Test headline",
    content: str = "Test content",
) -> NewsArticle:
    return NewsArticle(
        id=uuid4(),
        source="rss",
        title=title,
        content=content,
        url="https://rbc.ru/test",
        language="ru",
        published_at=datetime(2026, 3, 15, tzinfo=UTC),
    )


@pytest.fixture
def mock_llm() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def extractor(mock_llm: AsyncMock) -> EntityExtractor:
    return EntityExtractor(llm_client=mock_llm)


class TestEntityExtraction:
    """EntityExtractor.extract returns MOEX tickers from Russian text."""

    @pytest.mark.asyncio
    async def test_extracts_single_ticker(
        self, mock_llm: AsyncMock, extractor: EntityExtractor
    ) -> None:
        mock_llm.complete.return_value = json.dumps({"tickers": ["SBER"], "scope": "company"})
        article = _make_article(title="Сбербанк повысил дивиденды")

        result = await extractor.extract(article)

        assert result == ["SBER"]

    @pytest.mark.asyncio
    async def test_extracts_multiple_tickers(
        self, mock_llm: AsyncMock, extractor: EntityExtractor
    ) -> None:
        mock_llm.complete.return_value = json.dumps(
            {"tickers": ["SBER", "GAZP", "LKOH"], "scope": "company"}
        )
        article = _make_article(title="Сбербанк, Газпром и Лукойл")

        result = await extractor.extract(article)

        assert result == ["SBER", "GAZP", "LKOH"]

    @pytest.mark.asyncio
    async def test_no_companies_returns_empty(
        self, mock_llm: AsyncMock, extractor: EntityExtractor
    ) -> None:
        mock_llm.complete.return_value = json.dumps({"tickers": [], "scope": "market"})
        article = _make_article(title="Индекс Мосбиржи вырос")

        result = await extractor.extract(article)

        assert result == []

    @pytest.mark.asyncio
    async def test_malformed_json_returns_empty(
        self, mock_llm: AsyncMock, extractor: EntityExtractor
    ) -> None:
        mock_llm.complete.return_value = "not valid json at all"
        article = _make_article()

        result = await extractor.extract(article)

        assert result == []

    @pytest.mark.asyncio
    async def test_filters_invalid_tickers(
        self, mock_llm: AsyncMock, extractor: EntityExtractor
    ) -> None:
        """Tickers not in MOEX universe are filtered out."""
        mock_llm.complete.return_value = json.dumps(
            {"tickers": ["SBER", "FAKE", "AAPL"], "scope": "company"}
        )
        article = _make_article()

        result = await extractor.extract(article)

        assert result == ["SBER"]

    @pytest.mark.asyncio
    async def test_markdown_wrapped_json(
        self, mock_llm: AsyncMock, extractor: EntityExtractor
    ) -> None:
        """LLM wrapping response in markdown code fences."""
        mock_llm.complete.return_value = '```json\n{"tickers": ["GAZP"], "scope": "company"}\n```'
        article = _make_article()

        result = await extractor.extract(article)

        assert result == ["GAZP"]
