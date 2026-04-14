"""Unit tests for EventClassifier."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from finalayze.analysis.event_classifier import (
    EventClassifier,
    EventClassifierResult,
    EventType,
)
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
    async def test_classify_calls_parse_structured(self) -> None:
        """classify must call parse_structured with EventClassifierResult."""
        mock_llm = AsyncMock()
        mock_llm.parse_structured.return_value = EventClassifierResult(
            event_types=["macro"],
        )
        classifier = EventClassifier(llm_client=mock_llm)
        result = await classifier.classify(_ARTICLE)
        mock_llm.parse_structured.assert_awaited_once()
        assert result == EventType.MACRO

    @pytest.mark.asyncio
    async def test_classify_empty_event_types_returns_other(self) -> None:
        mock_llm = AsyncMock()
        mock_llm.parse_structured.return_value = EventClassifierResult(
            event_types=[],
        )
        classifier = EventClassifier(llm_client=mock_llm)
        result = await classifier.classify(_ARTICLE)
        assert result == EventType.OTHER

    @pytest.mark.asyncio
    async def test_classify_unknown_event_type_returns_other(self) -> None:
        mock_llm = AsyncMock()
        mock_llm.parse_structured.return_value = EventClassifierResult(
            event_types=["definitely_not_a_real_event"],
        )
        classifier = EventClassifier(llm_client=mock_llm)
        result = await classifier.classify(_ARTICLE)
        assert result == EventType.OTHER

    @pytest.mark.asyncio
    async def test_classify_fda_via_clinical_trial(self) -> None:
        """'clinical_trial' in prompt vocabulary should map to EventType.FDA."""
        mock_llm = AsyncMock()
        mock_llm.parse_structured.return_value = EventClassifierResult(
            event_types=["clinical_trial"],
        )
        classifier = EventClassifier(llm_client=mock_llm)
        result = await classifier.classify(_ARTICLE)
        assert result == EventType.FDA

    @pytest.mark.asyncio
    async def test_classify_earnings_type(self) -> None:
        mock_llm = AsyncMock()
        mock_llm.parse_structured.return_value = EventClassifierResult(
            event_types=["earnings"],
        )
        classifier = EventClassifier(llm_client=mock_llm)
        result = await classifier.classify(_ARTICLE)
        assert result == EventType.EARNINGS

    @pytest.mark.asyncio
    async def test_parse_error_returns_other(self) -> None:
        """When parse_structured raises, return EventType.OTHER."""
        mock_llm = AsyncMock()
        mock_llm.parse_structured.side_effect = Exception("parse failed")
        classifier = EventClassifier(llm_client=mock_llm)
        result = await classifier.classify(_ARTICLE)
        assert result == EventType.OTHER

    @pytest.mark.asyncio
    async def test_timeout_returns_other(self) -> None:
        """When LLM call times out, return EventType.OTHER fallback."""
        mock_llm = AsyncMock()
        mock_llm.parse_structured.side_effect = TimeoutError()
        classifier = EventClassifier(llm_client=mock_llm)
        result = await classifier.classify(_ARTICLE)
        assert result == EventType.OTHER

    @pytest.mark.asyncio
    async def test_classify_wraps_with_wait_for_timeout(self) -> None:
        """Verify that classify uses asyncio.wait_for with 5s timeout."""
        mock_llm = AsyncMock()

        async def slow_parse(*args, **kwargs):  # type: ignore[no-untyped-def]
            await asyncio.sleep(10)
            return EventClassifierResult(event_types=["macro"])

        mock_llm.parse_structured = slow_parse
        classifier = EventClassifier(llm_client=mock_llm)
        result = await classifier.classify(_ARTICLE)
        assert result == EventType.OTHER

    @pytest.mark.asyncio
    async def test_no_json_loads_used(self) -> None:
        """Ensure json.loads is not used in event_classifier module."""
        import inspect

        from finalayze.analysis import event_classifier as mod

        source = inspect.getsource(mod)
        assert "json.loads" not in source
