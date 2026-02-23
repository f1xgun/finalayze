"""Unit tests for EventClassifier."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from finalayze.analysis.event_classifier import EventClassifier, EventType
from finalayze.core.schemas import NewsArticle

_ARTICLE = NewsArticle(
    id=uuid4(),
    source="reuters",
    title="Fed raises rates",
    content="Federal Reserve raised rates.",
    url="https://reuters.com/1",
    language="en",
    published_at=datetime(2024, 1, 1, tzinfo=UTC),
)


class TestEventType:
    def test_all_expected_values_exist(self) -> None:
        assert EventType.EARNINGS == "earnings"
        assert EventType.FDA == "fda"
        assert EventType.MACRO == "macro"
        assert EventType.GEOPOLITICAL == "geopolitical"
        assert EventType.CBR_RATE == "cbr_rate"
        assert EventType.OIL_PRICE == "oil_price"
        assert EventType.SANCTIONS == "sanctions"
        assert EventType.OTHER == "other"


class TestEventClassifier:
    @pytest.mark.asyncio
    async def test_classify_known_event_type(self) -> None:
        mock_llm = AsyncMock()
        mock_llm.complete.return_value = "macro"
        classifier = EventClassifier(llm_client=mock_llm)
        result = await classifier.classify(_ARTICLE)
        assert result == EventType.MACRO

    @pytest.mark.asyncio
    async def test_unknown_response_returns_other(self) -> None:
        mock_llm = AsyncMock()
        mock_llm.complete.return_value = "random_unknown_value"
        classifier = EventClassifier(llm_client=mock_llm)
        result = await classifier.classify(_ARTICLE)
        assert result == EventType.OTHER

    @pytest.mark.asyncio
    async def test_whitespace_stripped_from_response(self) -> None:
        mock_llm = AsyncMock()
        mock_llm.complete.return_value = "  earnings  \n"
        classifier = EventClassifier(llm_client=mock_llm)
        result = await classifier.classify(_ARTICLE)
        assert result == EventType.EARNINGS
