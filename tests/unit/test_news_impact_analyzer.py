"""Tests for NewsImpactAnalyzer (Layer 3)."""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from finalayze.analysis.event_classifier import EventType
from finalayze.analysis.news_impact_analyzer import (
    _FALLBACK_RESULT,
    NewsImpactAnalyzer,
    NewsImpactResult,
    SectorImpactDetail,
)
from finalayze.core.schemas import NewsArticle


def _make_article(language: str = "ru", title: str = "Test", content: str = "Body") -> NewsArticle:
    return NewsArticle(
        id=uuid4(),
        source="test",
        title=title,
        content=content,
        url="https://example.com",
        language=language,
        published_at=datetime.now(tz=UTC),
    )


def _valid_json_response() -> str:
    return json.dumps(
        {
            "event_type": "cbr_rate",
            "sentiment": -0.7,
            "confidence": 0.85,
            "reasoning": "CBR raised rate to 21%",
            "affected_sectors": [
                {
                    "sector": "banking",
                    "direction": -1,
                    "magnitude": 0.8,
                    "reasoning": "Higher rates hurt bank margins",
                },
            ],
            "direct_tickers": ["SBER", "VTBR"],
        }
    )


class TestNewsImpactAnalyzer:
    """Tests for NewsImpactAnalyzer.analyze()."""

    def setup_method(self) -> None:
        self.mock_llm = AsyncMock()
        self.analyzer = NewsImpactAnalyzer(self.mock_llm)

    @pytest.mark.asyncio
    async def test_analyze_calls_llm_once(self) -> None:
        """analyze() must call LLM exactly once per article (NEWS-09)."""
        self.mock_llm.complete.return_value = _valid_json_response()
        article = _make_article()
        await self.analyzer.analyze(article)
        self.mock_llm.complete.assert_called_once()

    @pytest.mark.asyncio
    async def test_analyze_valid_response(self) -> None:
        """Valid JSON response parsed into NewsImpactResult."""
        self.mock_llm.complete.return_value = _valid_json_response()
        result = await self.analyzer.analyze(_make_article())

        assert isinstance(result, NewsImpactResult)
        assert result.event_type == EventType.CBR_RATE
        assert result.sentiment == pytest.approx(-0.7)
        assert result.confidence == pytest.approx(0.85)
        assert result.reasoning == "CBR raised rate to 21%"
        assert len(result.affected_sectors) == 1
        assert result.affected_sectors[0].sector == "banking"
        assert result.affected_sectors[0].direction == -1
        assert result.affected_sectors[0].magnitude == pytest.approx(0.8)
        assert result.direct_tickers == ["SBER", "VTBR"]

    @pytest.mark.asyncio
    async def test_analyze_malformed_json_returns_fallback(self) -> None:
        """Malformed JSON returns safe fallback."""
        self.mock_llm.complete.return_value = "not json at all"
        result = await self.analyzer.analyze(_make_article())

        assert result.sentiment == 0.0
        assert result.confidence == 0.0
        assert result.event_type == EventType.OTHER
        assert result.affected_sectors == []
        assert result.direct_tickers == []

    @pytest.mark.asyncio
    async def test_analyze_code_fence_stripping(self) -> None:
        """Code fences around JSON are stripped."""
        self.mock_llm.complete.return_value = f"```json\n{_valid_json_response()}\n```"
        result = await self.analyzer.analyze(_make_article())
        assert result.event_type == EventType.CBR_RATE

    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_after_threshold(self) -> None:
        """After 5 consecutive failures, returns fallback without LLM call."""
        self.mock_llm.complete.side_effect = Exception("LLM down")

        # Trigger 5 failures
        for _ in range(5):
            await self.analyzer.analyze(_make_article())

        # Reset mock to track next call
        self.mock_llm.complete.reset_mock()

        # 6th call should skip LLM
        result = await self.analyzer.analyze(_make_article())
        self.mock_llm.complete.assert_not_called()
        assert result == _FALLBACK_RESULT

    @pytest.mark.asyncio
    async def test_circuit_breaker_resets_on_success(self) -> None:
        """Circuit breaker resets consecutive_failures on success."""
        # 3 failures
        self.mock_llm.complete.side_effect = Exception("fail")
        for _ in range(3):
            await self.analyzer.analyze(_make_article())

        # Then success
        self.mock_llm.complete.side_effect = None
        self.mock_llm.complete.return_value = _valid_json_response()
        result = await self.analyzer.analyze(_make_article())
        assert result.event_type == EventType.CBR_RATE
        # Internal state should be reset
        assert self.analyzer._consecutive_failures == 0

    @pytest.mark.asyncio
    async def test_language_selection_ru(self) -> None:
        """Russian article uses Russian prompt."""
        self.mock_llm.complete.return_value = _valid_json_response()
        await self.analyzer.analyze(_make_article(language="ru"))

        call_args = self.mock_llm.complete.call_args
        system_prompt = call_args[0][1] if len(call_args[0]) > 1 else call_args[1].get("system", "")
        assert "финансовый аналитик" in system_prompt

    @pytest.mark.asyncio
    async def test_language_selection_en(self) -> None:
        """English article uses English prompt."""
        self.mock_llm.complete.return_value = _valid_json_response()
        await self.analyzer.analyze(_make_article(language="en"))

        call_args = self.mock_llm.complete.call_args
        system_prompt = call_args[0][1] if len(call_args[0]) > 1 else call_args[1].get("system", "")
        assert "financial analyst" in system_prompt

    @pytest.mark.asyncio
    async def test_direction_clamped(self) -> None:
        """Direction values clamped to {-1, +1}."""
        response = json.dumps(
            {
                "event_type": "macro",
                "sentiment": 0.5,
                "confidence": 0.5,
                "reasoning": "test",
                "affected_sectors": [
                    {"sector": "banking", "direction": 0.5, "magnitude": 0.5, "reasoning": "test"},
                    {"sector": "oil_gas", "direction": -3, "magnitude": 0.5, "reasoning": "test"},
                ],
                "direct_tickers": [],
            }
        )
        self.mock_llm.complete.return_value = response
        result = await self.analyzer.analyze(_make_article())

        assert result.affected_sectors[0].direction == 1  # 0.5 -> +1
        assert result.affected_sectors[1].direction == -1  # -3 -> -1

    @pytest.mark.asyncio
    async def test_magnitude_clamped(self) -> None:
        """Magnitude clamped to [0.0, 1.0]."""
        response = json.dumps(
            {
                "event_type": "macro",
                "sentiment": 0.5,
                "confidence": 0.5,
                "reasoning": "test",
                "affected_sectors": [
                    {"sector": "banking", "direction": 1, "magnitude": 1.5, "reasoning": "test"},
                    {"sector": "oil_gas", "direction": -1, "magnitude": -0.3, "reasoning": "test"},
                ],
                "direct_tickers": [],
            }
        )
        self.mock_llm.complete.return_value = response
        result = await self.analyzer.analyze(_make_article())

        assert result.affected_sectors[0].magnitude == pytest.approx(1.0)
        assert result.affected_sectors[1].magnitude == pytest.approx(0.0)

    @pytest.mark.asyncio
    async def test_direct_tickers_filtered(self) -> None:
        """Only valid MOEX tickers pass through."""
        response = json.dumps(
            {
                "event_type": "macro",
                "sentiment": 0.5,
                "confidence": 0.5,
                "reasoning": "test",
                "affected_sectors": [],
                "direct_tickers": ["SBER", "AAPL", "FAKE", "LKOH"],
            }
        )
        self.mock_llm.complete.return_value = response
        result = await self.analyzer.analyze(_make_article())
        assert result.direct_tickers == ["SBER", "LKOH"]

    @pytest.mark.asyncio
    async def test_unknown_event_type_maps_to_other(self) -> None:
        """Unknown event_type maps to EventType.OTHER."""
        response = json.dumps(
            {
                "event_type": "totally_unknown",
                "sentiment": 0.0,
                "confidence": 0.5,
                "reasoning": "test",
                "affected_sectors": [],
                "direct_tickers": [],
            }
        )
        self.mock_llm.complete.return_value = response
        result = await self.analyzer.analyze(_make_article())
        assert result.event_type == EventType.OTHER

    @pytest.mark.asyncio
    async def test_sentiment_clamped(self) -> None:
        """Sentiment clamped to [-1.0, 1.0]."""
        response = json.dumps(
            {
                "event_type": "macro",
                "sentiment": 2.5,
                "confidence": -0.5,
                "reasoning": "test",
                "affected_sectors": [],
                "direct_tickers": [],
            }
        )
        self.mock_llm.complete.return_value = response
        result = await self.analyzer.analyze(_make_article())
        assert result.sentiment == pytest.approx(1.0)
        assert result.confidence == pytest.approx(0.0)

    @pytest.mark.asyncio
    async def test_circuit_breaker_half_open_after_cooldown(self) -> None:
        """After cooldown, circuit breaker allows one retry."""
        self.mock_llm.complete.side_effect = Exception("fail")
        for _ in range(5):
            await self.analyzer.analyze(_make_article())

        # Simulate time passing beyond cooldown
        self.analyzer._circuit_open_until = time.monotonic() - 1

        self.mock_llm.complete.side_effect = None
        self.mock_llm.complete.return_value = _valid_json_response()
        result = await self.analyzer.analyze(_make_article())
        assert result.event_type == EventType.CBR_RATE
        assert self.analyzer._consecutive_failures == 0

    @pytest.mark.asyncio
    async def test_sector_detail_partial_parse_error(self) -> None:
        """If one sector entry is malformed, others still parsed."""
        response = json.dumps(
            {
                "event_type": "macro",
                "sentiment": 0.5,
                "confidence": 0.5,
                "reasoning": "test",
                "affected_sectors": [
                    {"sector": "banking", "direction": -1, "magnitude": 0.8, "reasoning": "ok"},
                    "not_a_dict",
                    {"sector": "oil_gas", "direction": 1, "magnitude": 0.5, "reasoning": "ok"},
                ],
                "direct_tickers": [],
            }
        )
        self.mock_llm.complete.return_value = response
        result = await self.analyzer.analyze(_make_article())
        assert len(result.affected_sectors) == 2
        assert result.affected_sectors[0].sector == "banking"
        assert result.affected_sectors[1].sector == "oil_gas"

    @pytest.mark.asyncio
    async def test_known_event_types_mapped(self) -> None:
        """Known event types from the prompt vocabulary are correctly mapped."""
        for event_str, expected in [
            ("cbr_rate", EventType.CBR_RATE),
            ("oil_price", EventType.OIL_PRICE),
            ("sanctions", EventType.SANCTIONS),
            ("earnings", EventType.EARNINGS),
            ("geopolitical", EventType.GEOPOLITICAL),
            ("macro", EventType.MACRO),
        ]:
            response = json.dumps(
                {
                    "event_type": event_str,
                    "sentiment": 0.0,
                    "confidence": 0.5,
                    "reasoning": "test",
                    "affected_sectors": [],
                    "direct_tickers": [],
                }
            )
            self.mock_llm.complete.return_value = response
            # Reset circuit breaker state
            self.analyzer._consecutive_failures = 0
            result = await self.analyzer.analyze(_make_article())
            assert result.event_type == expected, f"{event_str} -> {result.event_type}"
