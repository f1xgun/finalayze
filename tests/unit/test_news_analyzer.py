"""Unit tests for NewsAnalyzer."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from finalayze.analysis.news_analyzer import NewsAnalyzer
from finalayze.core.schemas import SentimentResult

_ARTICLE_EN_KWARGS = {
    "id": uuid4(),
    "source": "reuters",
    "title": "Fed raises rates",
    "content": "The Federal Reserve raised rates by 25bps.",
    "url": "https://reuters.com/1",
    "language": "en",
    "published_at": datetime(2024, 1, 3, tzinfo=UTC),
}

_ARTICLE_RU_KWARGS = {
    "id": uuid4(),
    "source": "interfax",
    "title": "\u0426\u0411 \u043f\u043e\u0432\u044b\u0441\u0438\u043b \u0441\u0442\u0430\u0432\u043a\u0443",
    "content": "\u0426\u0435\u043d\u0442\u0440\u0430\u043b\u044c\u043d\u044b\u0439 \u0431\u0430\u043d\u043a \u043f\u043e\u0432\u044b\u0441\u0438\u043b \u043a\u043b\u044e\u0447\u0435\u0432\u0443\u044e \u0441\u0442\u0430\u0432\u043a\u0443 \u0434\u043e 16%.",
    "url": "https://interfax.ru/1",
    "language": "ru",
    "published_at": datetime(2024, 1, 3, tzinfo=UTC),
}

_SENTIMENT_VALUE = 0.6
_CONFIDENCE_VALUE = 0.85
_SENTIMENT_RU_VALUE = -0.7
_CONFIDENCE_RU_VALUE = 0.9


def _make_article(**overrides):  # type: ignore[no-untyped-def]
    from finalayze.core.schemas import NewsArticle

    return NewsArticle(**{**_ARTICLE_EN_KWARGS, **overrides})


class TestNewsAnalyzer:
    @pytest.mark.asyncio
    async def test_analyze_calls_parse_structured(self) -> None:
        """parse_structured must be called with SentimentResult response model."""
        mock_llm = AsyncMock()
        mock_llm.parse_structured.return_value = SentimentResult(
            sentiment=_SENTIMENT_VALUE,
            confidence=_CONFIDENCE_VALUE,
            reasoning="Rate hike positive for USD",
        )
        analyzer = NewsAnalyzer(llm_client=mock_llm)
        article = _make_article()
        result = await analyzer.analyze(article)

        mock_llm.parse_structured.assert_awaited_once()
        call_args = mock_llm.parse_structured.call_args
        # Third positional arg or response_model kwarg should be SentimentResult
        assert SentimentResult in call_args[0] or call_args[1].get("response_model") is SentimentResult
        assert isinstance(result, SentimentResult)
        assert result.sentiment == pytest.approx(_SENTIMENT_VALUE)
        assert result.confidence == pytest.approx(_CONFIDENCE_VALUE)

    @pytest.mark.asyncio
    async def test_analyze_ru_uses_russian_prompt(self) -> None:
        mock_llm = AsyncMock()
        mock_llm.parse_structured.return_value = SentimentResult(
            sentiment=_SENTIMENT_RU_VALUE,
            confidence=_CONFIDENCE_RU_VALUE,
            reasoning="\u0421\u0442\u0430\u0432\u043a\u0430 \u043f\u043e\u0432\u044b\u0448\u0435\u043d\u0430 \u2014 \u043d\u0435\u0433\u0430\u0442\u0438\u0432",
        )
        analyzer = NewsAnalyzer(llm_client=mock_llm)
        article = _make_article(**_ARTICLE_RU_KWARGS)
        result = await analyzer.analyze(article)

        call_args = mock_llm.parse_structured.call_args
        system_arg = call_args[0][1]  # second positional arg is system prompt
        assert "\u0426\u0411" in system_arg or "\u0444\u0438\u043d\u0430\u043d\u0441\u043e\u0432\u044b\u0445" in system_arg
        assert result.sentiment == pytest.approx(_SENTIMENT_RU_VALUE)

    @pytest.mark.asyncio
    async def test_parse_error_returns_fallback(self) -> None:
        """When parse_structured raises, return neutral fallback."""
        mock_llm = AsyncMock()
        mock_llm.parse_structured.side_effect = Exception("LLM parse failed")
        analyzer = NewsAnalyzer(llm_client=mock_llm)
        result = await analyzer.analyze(_make_article())
        assert result.sentiment == 0.0
        assert result.confidence == 0.0

    @pytest.mark.asyncio
    async def test_timeout_returns_fallback(self) -> None:
        """When LLM call times out (5s), return fallback instead of crashing."""
        mock_llm = AsyncMock()
        mock_llm.parse_structured.side_effect = TimeoutError()
        analyzer = NewsAnalyzer(llm_client=mock_llm)
        result = await analyzer.analyze(_make_article())
        assert result.sentiment == 0.0
        assert result.confidence == 0.0

    @pytest.mark.asyncio
    async def test_analyze_wraps_with_wait_for_timeout(self) -> None:
        """Verify that analyze uses asyncio.wait_for with 5s timeout."""
        mock_llm = AsyncMock()

        async def slow_parse(*args, **kwargs):  # type: ignore[no-untyped-def]
            await asyncio.sleep(10)  # Longer than the 5s timeout
            return SentimentResult(sentiment=0.5, confidence=0.5, reasoning="late")

        mock_llm.parse_structured = slow_parse
        analyzer = NewsAnalyzer(llm_client=mock_llm)
        result = await analyzer.analyze(_make_article())
        # Should have timed out and returned fallback
        assert result.sentiment == 0.0
        assert result.confidence == 0.0

    @pytest.mark.asyncio
    async def test_no_json_loads_used(self) -> None:
        """Ensure json.loads is not used (no import json needed)."""
        import inspect

        from finalayze.analysis import news_analyzer as mod

        source = inspect.getsource(mod)
        assert "json.loads" not in source


class TestSentimentResultFields:
    def test_fallback_has_is_fallback_true(self) -> None:
        """_FALLBACK in news_analyzer.py must have is_fallback=True."""
        from finalayze.analysis.news_analyzer import _FALLBACK

        assert _FALLBACK.is_fallback is True

    def test_normal_result_has_is_fallback_false(self) -> None:
        """A normal SentimentResult defaults to is_fallback=False."""
        result = SentimentResult(sentiment=0.5, confidence=0.8, reasoning="test")
        assert result.is_fallback is False

    def test_sentiment_result_tickers_default_empty(self) -> None:
        """SentimentResult.tickers defaults to empty list."""
        result = SentimentResult(sentiment=0.5, confidence=0.8, reasoning="test")
        assert result.tickers == []

    def test_sentiment_result_accepts_tickers(self) -> None:
        """SentimentResult accepts a tickers list."""
        result = SentimentResult(
            sentiment=0.5, confidence=0.8, reasoning="test", tickers=["SBER", "GAZP"]
        )
        assert result.tickers == ["SBER", "GAZP"]
