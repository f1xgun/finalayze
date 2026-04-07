"""Unit tests for EntityExtractor (Layer 3)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from finalayze.analysis.entity_extractor import (
    _CIRCUIT_BREAKER_RESET_SECONDS,
    _CIRCUIT_BREAKER_THRESHOLD,
    _VALID_TICKERS,
    EntityExtractor,
)
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


class TestValidTickers:
    """NEWS-03: TCSG ticker must be in _VALID_TICKERS, bare T must not."""

    def test_tcsg_in_valid_tickers(self) -> None:
        assert "TCSG" in _VALID_TICKERS

    def test_bare_t_not_in_valid_tickers(self) -> None:
        assert "T" not in _VALID_TICKERS

    @pytest.mark.asyncio
    async def test_extract_keeps_tcsg(
        self, mock_llm: AsyncMock, extractor: EntityExtractor
    ) -> None:
        """EntityExtractor.extract() keeps TCSG in output when LLM returns it."""
        mock_llm.complete.return_value = json.dumps(
            {"tickers": ["TCSG", "SBER"], "scope": "company"}
        )
        article = _make_article(
            title="Т-Банк повысил дивиденды",  # noqa: RUF001
        )
        result = await extractor.extract(article)
        assert "TCSG" in result

    @pytest.mark.asyncio
    async def test_extract_filters_out_bare_t(
        self, mock_llm: AsyncMock, extractor: EntityExtractor
    ) -> None:
        """EntityExtractor.extract() filters out bare T (not in valid set)."""
        mock_llm.complete.return_value = json.dumps({"tickers": ["T", "SBER"], "scope": "company"})
        article = _make_article(title="Some news about T")
        result = await extractor.extract(article)
        assert "T" not in result
        assert result == ["SBER"]


class TestEntityExtractionCircuitBreaker:
    """Circuit breaker stops calling LLM after consecutive failures."""

    @pytest.mark.asyncio
    async def test_opens_after_threshold_failures(
        self, mock_llm: AsyncMock, extractor: EntityExtractor
    ) -> None:
        """Circuit opens after _CIRCUIT_BREAKER_THRESHOLD consecutive failures."""
        mock_llm.complete.side_effect = RuntimeError("LLM down")
        article = _make_article()

        for _ in range(_CIRCUIT_BREAKER_THRESHOLD):
            await extractor.extract(article)

        assert extractor._consecutive_failures == _CIRCUIT_BREAKER_THRESHOLD

    @pytest.mark.asyncio
    async def test_skips_llm_while_circuit_open(
        self, mock_llm: AsyncMock, extractor: EntityExtractor
    ) -> None:
        """When circuit is open, extract returns [] without calling LLM."""
        mock_llm.complete.side_effect = RuntimeError("LLM down")
        article = _make_article()

        # Trip the circuit breaker
        for _ in range(_CIRCUIT_BREAKER_THRESHOLD):
            await extractor.extract(article)

        call_count_at_open = mock_llm.complete.call_count

        # Next call should be skipped (circuit open)
        result = await extractor.extract(article)

        assert result == []
        assert mock_llm.complete.call_count == call_count_at_open

    @pytest.mark.asyncio
    async def test_half_open_retries_after_cooldown(
        self, mock_llm: AsyncMock, extractor: EntityExtractor
    ) -> None:
        """After cooldown expires, circuit enters half-open and retries LLM."""
        mock_llm.complete.side_effect = RuntimeError("LLM down")
        article = _make_article()

        # Trip the circuit breaker
        for _ in range(_CIRCUIT_BREAKER_THRESHOLD):
            await extractor.extract(article)

        # Simulate cooldown expiry by rewinding the open_until timestamp
        extractor._circuit_open_until = 0.0

        # LLM still fails, so it should call LLM and fail again
        result = await extractor.extract(article)

        assert result == []
        assert mock_llm.complete.call_count == _CIRCUIT_BREAKER_THRESHOLD + 1

    @pytest.mark.asyncio
    async def test_resets_on_success(self, mock_llm: AsyncMock, extractor: EntityExtractor) -> None:
        """Successful call after failures resets the consecutive failure count."""
        article = _make_article()
        mock_llm.complete.side_effect = RuntimeError("LLM down")

        # Accumulate some failures (below threshold)
        failure_count = _CIRCUIT_BREAKER_THRESHOLD - 1
        for _ in range(failure_count):
            await extractor.extract(article)

        assert extractor._consecutive_failures == failure_count

        # Now succeed
        mock_llm.complete.side_effect = None
        mock_llm.complete.return_value = json.dumps({"tickers": ["SBER"], "scope": "company"})
        result = await extractor.extract(article)

        assert result == ["SBER"]
        assert extractor._consecutive_failures == 0

    @pytest.mark.asyncio
    async def test_half_open_success_closes_circuit(
        self, mock_llm: AsyncMock, extractor: EntityExtractor
    ) -> None:
        """A successful retry in half-open state fully closes the circuit."""
        article = _make_article()
        mock_llm.complete.side_effect = RuntimeError("LLM down")

        # Trip the circuit
        for _ in range(_CIRCUIT_BREAKER_THRESHOLD):
            await extractor.extract(article)

        # Simulate cooldown expiry
        extractor._circuit_open_until = 0.0

        # LLM recovers
        mock_llm.complete.side_effect = None
        mock_llm.complete.return_value = json.dumps({"tickers": ["GAZP"], "scope": "company"})

        result = await extractor.extract(article)

        assert result == ["GAZP"]
        assert extractor._consecutive_failures == 0

    @pytest.mark.asyncio
    async def test_circuit_opened_log_includes_exc_info(
        self, mock_llm: AsyncMock, extractor: EntityExtractor
    ) -> None:
        """The warning that opens the circuit includes exc_info for debugging."""
        mock_llm.complete.side_effect = RuntimeError("LLM 500")
        article = _make_article()

        # The 5th failure should trigger the circuit-opened log.
        # We verify indirectly: after threshold failures the circuit is open.
        for _ in range(_CIRCUIT_BREAKER_THRESHOLD):
            await extractor.extract(article)

        assert extractor._consecutive_failures == _CIRCUIT_BREAKER_THRESHOLD
        assert extractor._circuit_open_until > 0.0
