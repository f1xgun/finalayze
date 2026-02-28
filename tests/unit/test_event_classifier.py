"""Unit tests for EventClassifier."""

from __future__ import annotations

import json
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

    # ── #143: JSON response parsing ─────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_classify_parses_json_event_types_list(self) -> None:
        """Classifier must parse the rich JSON returned by classify_event.txt (#143)."""
        mock_llm = AsyncMock()
        response = json.dumps(
            {
                "event_types": ["macro"],
                "scope": "us",
                "affected_sectors": ["financials"],
                "affected_tickers": [],
                "impact_magnitude": 0.8,
                "reasoning": "Fed raised rates.",
            }
        )
        mock_llm.complete.return_value = response
        classifier = EventClassifier(llm_client=mock_llm)
        result = await classifier.classify(_ARTICLE)
        assert result == EventType.MACRO

    @pytest.mark.asyncio
    async def test_classify_parses_json_earnings_type(self) -> None:
        mock_llm = AsyncMock()
        response = json.dumps(
            {
                "event_types": ["earnings"],
                "scope": "us",
                "affected_sectors": ["tech"],
                "affected_tickers": ["AAPL"],
                "impact_magnitude": 0.9,
                "reasoning": "Q1 earnings beat.",
            }
        )
        mock_llm.complete.return_value = response
        classifier = EventClassifier(llm_client=mock_llm)
        result = await classifier.classify(_ARTICLE)
        assert result == EventType.EARNINGS

    @pytest.mark.asyncio
    async def test_classify_json_unknown_event_type_returns_other(self) -> None:
        mock_llm = AsyncMock()
        response = json.dumps(
            {
                "event_types": ["definitely_not_a_real_event"],
                "scope": "global",
                "affected_sectors": [],
                "affected_tickers": [],
                "impact_magnitude": 0.1,
                "reasoning": "Unknown event.",
            }
        )
        mock_llm.complete.return_value = response
        classifier = EventClassifier(llm_client=mock_llm)
        result = await classifier.classify(_ARTICLE)
        assert result == EventType.OTHER

    @pytest.mark.asyncio
    async def test_classify_json_empty_event_types_returns_other(self) -> None:
        mock_llm = AsyncMock()
        response = json.dumps(
            {
                "event_types": [],
                "scope": "global",
                "affected_sectors": [],
                "affected_tickers": [],
                "impact_magnitude": 0.0,
                "reasoning": "Nothing.",
            }
        )
        mock_llm.complete.return_value = response
        classifier = EventClassifier(llm_client=mock_llm)
        result = await classifier.classify(_ARTICLE)
        assert result == EventType.OTHER

    @pytest.mark.asyncio
    async def test_classify_json_fda_via_clinical_trial_maps_correctly(self) -> None:
        """'clinical_trial' in prompt vocabulary should map to EventType.FDA."""
        mock_llm = AsyncMock()
        response = json.dumps(
            {
                "event_types": ["clinical_trial"],
                "scope": "us",
                "affected_sectors": ["healthcare"],
                "affected_tickers": [],
                "impact_magnitude": 0.7,
                "reasoning": "Phase 3 trial results.",
            }
        )
        mock_llm.complete.return_value = response
        classifier = EventClassifier(llm_client=mock_llm)
        result = await classifier.classify(_ARTICLE)
        assert result == EventType.FDA
