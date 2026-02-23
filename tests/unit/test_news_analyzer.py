"""Unit tests for NewsAnalyzer."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from finalayze.analysis.news_analyzer import NewsAnalyzer
from finalayze.core.schemas import NewsArticle, SentimentResult

_ARTICLE_EN = NewsArticle(
    id=uuid4(),
    source="reuters",
    title="Fed raises rates",
    content="The Federal Reserve raised rates by 25bps.",
    url="https://reuters.com/1",
    language="en",
    published_at=datetime(2024, 1, 3, tzinfo=UTC),
)

_ARTICLE_RU = NewsArticle(
    id=uuid4(),
    source="interfax",
    title="ЦБ повысил ставку",
    content="Центральный банк повысил ключевую ставку до 16%.",
    url="https://interfax.ru/1",
    language="ru",
    published_at=datetime(2024, 1, 3, tzinfo=UTC),
)

_SENTIMENT_VALUE = 0.6
_CONFIDENCE_VALUE = 0.85
_SENTIMENT_RU_VALUE = -0.7
_CONFIDENCE_RU_VALUE = 0.9


class TestNewsAnalyzer:
    @pytest.mark.asyncio
    async def test_analyze_en_returns_sentiment_result(self) -> None:
        mock_llm = AsyncMock()
        mock_llm.complete.return_value = json.dumps(
            {
                "sentiment": _SENTIMENT_VALUE,
                "confidence": _CONFIDENCE_VALUE,
                "reasoning": "Rate hike positive for USD",
            }
        )
        analyzer = NewsAnalyzer(llm_client=mock_llm)
        result = await analyzer.analyze(_ARTICLE_EN)
        assert isinstance(result, SentimentResult)
        assert result.sentiment == pytest.approx(_SENTIMENT_VALUE)
        assert result.confidence == pytest.approx(_CONFIDENCE_VALUE)

    @pytest.mark.asyncio
    async def test_analyze_ru_uses_russian_prompt(self) -> None:
        mock_llm = AsyncMock()
        mock_llm.complete.return_value = json.dumps(
            {
                "sentiment": _SENTIMENT_RU_VALUE,
                "confidence": _CONFIDENCE_RU_VALUE,
                "reasoning": "Ставка повышена — негатив",
            }
        )
        analyzer = NewsAnalyzer(llm_client=mock_llm)
        result = await analyzer.analyze(_ARTICLE_RU)
        # verify the system prompt passed was the RU prompt
        call_kwargs = mock_llm.complete.call_args
        system_arg = call_kwargs[0][1] if call_kwargs[0] else call_kwargs[1]["system"]
        assert "ЦБ" in system_arg or "финансовых" in system_arg
        assert result.sentiment == pytest.approx(_SENTIMENT_RU_VALUE)

    @pytest.mark.asyncio
    async def test_parse_error_returns_zero_sentiment(self) -> None:
        mock_llm = AsyncMock()
        mock_llm.complete.return_value = "not valid json at all"
        analyzer = NewsAnalyzer(llm_client=mock_llm)
        result = await analyzer.analyze(_ARTICLE_EN)
        assert result.sentiment == 0.0
        assert result.confidence == 0.0
        assert "parse_error" in result.reasoning
