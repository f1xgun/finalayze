"""Integration test: mocked LLM -> NewsAnalyzer -> EventClassifier -> ImpactEstimator
-> EventDrivenStrategy produces BUY signal for affected segments.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock

import pytest

from finalayze.analysis.event_classifier import EventClassifier, EventType
from finalayze.analysis.impact_estimator import ImpactEstimator
from finalayze.analysis.llm_client import LLMClient
from finalayze.analysis.news_analyzer import NewsAnalyzer
from finalayze.core.schemas import Candle, NewsArticle, SignalDirection
from finalayze.strategies.event_driven import EventDrivenStrategy

# ── Constants ──────────────────────────────────────────────────────────────
SENTIMENT_BULLISH = 0.85
CONFIDENCE_HIGH = 0.90
SEGMENT_US_TECH = "us_tech"
SYMBOL_AAPL = "AAPL"
MARKET_US = "us"
NUM_CANDLES = 10
CANDLE_CLOSE = Decimal("150.00")
MIN_CONFIDENCE = 0.50
NEGATIVE_SENTIMENT = -0.85
NEUTRAL_SENTIMENT = 0.1  # below min_sentiment threshold

ACTIVE_SEGMENTS = [
    "us_tech",
    "us_finance",
    "us_healthcare",
    "us_energy",
]


def _make_us_article() -> NewsArticle:
    return NewsArticle(
        id=__import__("uuid").uuid4(),
        source="Bloomberg",
        title="Apple beats earnings expectations",
        content="Apple Inc. reported quarterly earnings that exceeded analyst forecasts.",
        url="https://bloomberg.com/1",
        language="en",
        published_at=datetime(2026, 1, 15, 12, 0, tzinfo=UTC),
        scope="us",
    )


def _make_candles(symbol: str = SYMBOL_AAPL, n: int = NUM_CANDLES) -> list[Candle]:
    base = datetime(2025, 1, 1, 14, 30, tzinfo=UTC)
    return [
        Candle(
            symbol=symbol,
            market_id=MARKET_US,
            timeframe="1d",
            timestamp=base + timedelta(days=i),
            open=CANDLE_CLOSE,
            high=CANDLE_CLOSE,
            low=CANDLE_CLOSE,
            close=CANDLE_CLOSE,
            volume=1_000_000,
        )
        for i in range(n)
    ]


@pytest.mark.integration
class TestNewsToSignalPipeline:
    """Tests the full chain: LLM (mocked) -> analysis -> strategy signal."""

    def _make_llm_client(
        self,
        sentiment: float = SENTIMENT_BULLISH,
        event_type: str = "earnings",
    ) -> LLMClient:
        """Create a mock LLM client that returns canned sentiment and event responses."""
        client = AsyncMock(spec=LLMClient)
        sentiment_response = json.dumps(
            {"sentiment": sentiment, "confidence": CONFIDENCE_HIGH, "reasoning": "strong earnings"}
        )
        # First call -> sentiment JSON, second call -> event type string
        client.complete = AsyncMock(side_effect=[sentiment_response, event_type])
        return client

    @pytest.mark.asyncio
    async def test_positive_sentiment_produces_buy_signal(self) -> None:
        llm = self._make_llm_client(sentiment=SENTIMENT_BULLISH, event_type="earnings")
        news_analyzer = NewsAnalyzer(llm_client=llm)
        event_classifier = EventClassifier(llm_client=llm)
        impact_estimator = ImpactEstimator()
        strategy = EventDrivenStrategy()

        article = _make_us_article()
        candles = _make_candles()

        sentiment_result = await news_analyzer.analyze(article)
        event = await event_classifier.classify(article)
        impacts = impact_estimator.estimate(article, event, sentiment_result, ACTIVE_SEGMENTS)

        # At least some US segments should be impacted
        assert len(impacts) > 0

        # For each impacted segment, run strategy
        signals = []
        for impact in impacts:
            signal = strategy.generate_signal(
                symbol=SYMBOL_AAPL,
                candles=candles,
                segment_id=impact.segment_id,
                sentiment_score=impact.sentiment,
            )
            if signal is not None:
                signals.append(signal)

        # Bullish sentiment -> at least one BUY signal
        assert any(s.direction == SignalDirection.BUY for s in signals)

    @pytest.mark.asyncio
    async def test_negative_sentiment_produces_sell_signal(self) -> None:
        llm = self._make_llm_client(sentiment=NEGATIVE_SENTIMENT, event_type="macro")
        news_analyzer = NewsAnalyzer(llm_client=llm)
        event_classifier = EventClassifier(llm_client=llm)
        impact_estimator = ImpactEstimator()
        strategy = EventDrivenStrategy()

        article = _make_us_article()
        candles = _make_candles()

        sentiment_result = await news_analyzer.analyze(article)
        event = await event_classifier.classify(article)
        impacts = impact_estimator.estimate(article, event, sentiment_result, ACTIVE_SEGMENTS)

        signals = []
        for impact in impacts:
            signal = strategy.generate_signal(
                symbol=SYMBOL_AAPL,
                candles=candles,
                segment_id=impact.segment_id,
                sentiment_score=impact.sentiment,
            )
            if signal is not None:
                signals.append(signal)

        assert any(s.direction == SignalDirection.SELL for s in signals)

    @pytest.mark.asyncio
    async def test_neutral_sentiment_produces_no_signal(self) -> None:
        llm = self._make_llm_client(sentiment=NEUTRAL_SENTIMENT, event_type="other")
        news_analyzer = NewsAnalyzer(llm_client=llm)
        event_classifier = EventClassifier(llm_client=llm)
        impact_estimator = ImpactEstimator()
        strategy = EventDrivenStrategy()

        article = _make_us_article()
        candles = _make_candles()

        sentiment_result = await news_analyzer.analyze(article)
        event = await event_classifier.classify(article)
        impacts = impact_estimator.estimate(article, event, sentiment_result, ACTIVE_SEGMENTS)

        signals = []
        for impact in impacts:
            signal = strategy.generate_signal(
                symbol=SYMBOL_AAPL,
                candles=candles,
                segment_id=impact.segment_id,
                sentiment_score=impact.sentiment,
            )
            if signal is not None:
                signals.append(signal)

        assert len(signals) == 0

    @pytest.mark.asyncio
    async def test_llm_parse_error_falls_back_to_neutral(self) -> None:
        """If the LLM returns invalid JSON, NewsAnalyzer returns neutral (0.0) sentiment."""
        llm = AsyncMock(spec=LLMClient)
        llm.complete = AsyncMock(return_value="not valid json {{ }}")
        news_analyzer = NewsAnalyzer(llm_client=llm)

        article = _make_us_article()
        result = await news_analyzer.analyze(article)

        assert result.sentiment == 0.0
        assert result.confidence == 0.0

    @pytest.mark.asyncio
    async def test_unknown_event_type_falls_back_to_other(self) -> None:
        """If the LLM returns an unknown event label, classifier returns EventType.OTHER."""
        llm = AsyncMock(spec=LLMClient)
        llm.complete = AsyncMock(return_value="alien_invasion")
        event_classifier = EventClassifier(llm_client=llm)

        article = _make_us_article()
        event = await event_classifier.classify(article)

        assert event == EventType.OTHER
