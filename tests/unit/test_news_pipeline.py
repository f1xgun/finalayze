"""Unit tests for news pipeline budget cap and threading safety."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from finalayze.core.schemas import NewsArticle, SentimentResult


def _make_article(i: int = 0) -> NewsArticle:
    """Create a test article with unique ID."""
    return NewsArticle(
        id=uuid4(),
        source="test",
        title=f"Article {i}",
        content=f"Content of article {i}",
        url=f"https://test.com/{i}",
        language="en",
        published_at=datetime(2024, 1, 1, tzinfo=UTC),
    )


def _make_trading_loop(**overrides: Any) -> Any:
    """Create a TradingLoop with minimal mocks for news cycle testing."""
    from finalayze.core.trading_loop import TradingLoop

    settings = MagicMock()
    settings.max_position_pct = 0.1
    settings.max_positions_per_market = 5
    settings.mode.can_submit_orders.return_value = False

    loop = object.__new__(TradingLoop)
    loop._settings = settings
    loop._news_fetcher = MagicMock()
    loop._news_analyzer = AsyncMock()
    loop._news_analyzer.analyze = AsyncMock(
        return_value=SentimentResult(sentiment=0.1, confidence=0.5, reasoning="test")
    )
    loop._event_classifier = AsyncMock()

    from finalayze.analysis.event_classifier import EventType

    loop._event_classifier.classify = AsyncMock(return_value=EventType.OTHER)
    loop._impact_estimator = MagicMock()
    loop._impact_estimator.estimate.return_value = []
    loop._alerter = MagicMock()
    loop._registry = MagicMock()
    loop._registry.list_by_market.return_value = []
    loop._fetchers = {}
    loop._cache = None
    loop._sentiment_cache = {}

    import threading

    loop._sentiment_lock = threading.Lock()
    loop._async_loop = None
    loop._async_thread = None

    for key, val in overrides.items():
        setattr(loop, f"_{key}", val)

    return loop


class TestBudgetCap:
    def test_budget_cap_limits_articles(self) -> None:
        """When more than 20 articles fetched, only 20 are processed."""
        articles = [_make_article(i) for i in range(30)]
        loop = _make_trading_loop()
        loop._news_fetcher.fetch_news.return_value = articles

        process_calls: list[NewsArticle] = []

        def track_process(article: NewsArticle) -> None:
            process_calls.append(article)

        loop._process_news_article = track_process
        loop._news_cycle()

        from finalayze.core.trading_loop import _MAX_ARTICLES_PER_CYCLE

        assert len(process_calls) == _MAX_ARTICLES_PER_CYCLE
        assert _MAX_ARTICLES_PER_CYCLE == 20  # noqa: PLR2004

    def test_budget_cap_metric_incremented(self) -> None:
        """When articles > 20, MetricsCollector.inc_news_budget_cap_hit is called."""
        articles = [_make_article(i) for i in range(25)]
        loop = _make_trading_loop()
        loop._news_fetcher.fetch_news.return_value = articles
        loop._process_news_article = MagicMock()

        with patch("finalayze.api.metrics.MetricsCollector") as mock_metrics:
            loop._news_cycle()
            mock_metrics.inc_news_budget_cap_hit.assert_called_once()

    def test_no_budget_cap_under_limit(self) -> None:
        """When articles <= 20, all are processed and no cap metric fired."""
        article_count = 15
        articles = [_make_article(i) for i in range(article_count)]
        loop = _make_trading_loop()
        loop._news_fetcher.fetch_news.return_value = articles

        process_calls: list[NewsArticle] = []

        def track_process(article: NewsArticle) -> None:
            process_calls.append(article)

        loop._process_news_article = track_process

        with patch("finalayze.api.metrics.MetricsCollector") as mock_metrics:
            loop._news_cycle()
            mock_metrics.inc_news_budget_cap_hit.assert_not_called()
            assert len(process_calls) == article_count


class TestSentimentLockSafety:
    def test_sentiment_lock_not_in_async_methods(self) -> None:
        """_sentiment_lock must not be acquired in any async method."""
        import ast
        import inspect

        from finalayze.core import trading_loop as mod

        source = inspect.getsource(mod.TradingLoop)
        tree = ast.parse(source)

        # Find all async methods that reference _sentiment_lock
        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef):
                method_source = ast.get_source_segment(source, node) or ""
                assert "_sentiment_lock" not in method_source, (
                    f"async method {node.name} must not reference _sentiment_lock"
                )
