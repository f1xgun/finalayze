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


class TestCredibilityMap:
    def test_credibility_map_rss(self) -> None:
        """RSS sources (rbc, interfax, tass) return 0.8."""
        from finalayze.core.trading_loop import get_credibility

        assert get_credibility("rbc") == 0.8  # noqa: PLR2004
        assert get_credibility("interfax") == 0.8  # noqa: PLR2004
        assert get_credibility("tass") == 0.8  # noqa: PLR2004

    def test_credibility_map_telegram(self) -> None:
        """Telegram source returns 0.7."""
        from finalayze.core.trading_loop import get_credibility

        assert get_credibility("telegram") == 0.7  # noqa: PLR2004

    def test_credibility_map_unknown(self) -> None:
        """Unknown source returns default 0.5."""
        from finalayze.core.trading_loop import get_credibility

        assert get_credibility("unknown_source") == 0.5  # noqa: PLR2004

    def test_credibility_case_insensitive(self) -> None:
        """Credibility lookup is case insensitive."""
        from finalayze.core.trading_loop import get_credibility

        assert get_credibility("RBC") == 0.8  # noqa: PLR2004
        assert get_credibility("Telegram") == 0.7  # noqa: PLR2004


class TestTickerValidation:
    def test_ticker_validation_filters_unknown(self) -> None:
        """Only tickers present in InstrumentRegistry are returned."""
        from finalayze.core.exceptions import InstrumentNotFoundError
        from finalayze.core.trading_loop import validate_tickers

        registry = MagicMock()

        def fake_get(symbol: str, market_id: str) -> Any:
            if symbol in ("SBER", "GAZP"):
                return MagicMock()
            raise InstrumentNotFoundError(f"{symbol} not found")

        registry.get.side_effect = fake_get

        result = validate_tickers(["SBER", "FAKE123", "GAZP"], registry, "moex")
        assert result == ["SBER", "GAZP"]

    def test_ticker_validation_logs_rejected(self) -> None:
        """Rejected tickers produce a structured log warning."""
        from finalayze.core.exceptions import InstrumentNotFoundError
        from finalayze.core.trading_loop import validate_tickers

        registry = MagicMock()
        registry.get.side_effect = InstrumentNotFoundError("not found")

        with patch("finalayze.core.trading_loop._log") as mock_log:
            validate_tickers(["FAKE"], registry, "moex")
            mock_log.warning.assert_called_once()
            call_args = mock_log.warning.call_args
            assert call_args[0][0] == "entity_not_in_registry"

    def test_ticker_validation_empty_list(self) -> None:
        """Empty ticker list returns empty list."""
        from finalayze.core.trading_loop import validate_tickers

        registry = MagicMock()
        result = validate_tickers([], registry, "moex")
        assert result == []

    def test_validate_tickers_called_in_process_article(self) -> None:
        """_process_news_article calls validate_tickers when sentiment has tickers."""
        loop = _make_trading_loop()
        loop._fetchers = {"moex": MagicMock()}  # provide market_id

        article = _make_article()
        # Mock _run_async to return sentiment with tickers + event
        from finalayze.analysis.event_classifier import EventType

        sentiment_with_tickers = SentimentResult(
            sentiment=0.3, confidence=0.7, reasoning="test", tickers=["SBER", "FAKE"]
        )
        loop._run_async = MagicMock(return_value=(sentiment_with_tickers, EventType.OTHER))

        with patch(
            "finalayze.core.trading_loop.validate_tickers", return_value=["SBER"]
        ) as mock_vt:
            loop._process_news_article(article)
            mock_vt.assert_called_once_with(["SBER", "FAKE"], loop._registry, "moex")

    def test_empty_tickers_skips_validation(self) -> None:
        """_process_news_article does not call validate_tickers when tickers is empty."""
        loop = _make_trading_loop()

        article = _make_article()
        from finalayze.analysis.event_classifier import EventType

        sentiment_no_tickers = SentimentResult(
            sentiment=0.3, confidence=0.7, reasoning="test", tickers=[]
        )
        loop._run_async = MagicMock(return_value=(sentiment_no_tickers, EventType.OTHER))

        with patch("finalayze.core.trading_loop.validate_tickers") as mock_vt:
            loop._process_news_article(article)
            mock_vt.assert_not_called()

    def test_credibility_set_on_articles_in_news_cycle(self) -> None:
        """_news_cycle sets credibility_score on articles before processing."""
        loop = _make_trading_loop()
        article = _make_article()
        loop._news_fetcher.fetch_news.return_value = [article]

        processed_articles: list[NewsArticle] = []

        def track(art: NewsArticle) -> None:
            processed_articles.append(art)

        loop._process_news_article = track
        loop._news_cycle()

        assert len(processed_articles) == 1
        assert processed_articles[0].credibility_score is not None


class TestSentimentScoreModelCredibility:
    def test_sentiment_score_model_has_credibility(self) -> None:
        """SentimentScoreModel has a credibility column."""
        from finalayze.core.models import SentimentScoreModel

        assert hasattr(SentimentScoreModel, "credibility")


class TestLLMLiveness:
    def _run_cycle_with_failure(self, loop: Any, *, fail: bool) -> None:
        """Run a news cycle where all articles either fail or succeed."""
        article = _make_article()
        loop._news_fetcher.fetch_news.return_value = [article]

        if fail:
            loop._process_news_article = MagicMock(side_effect=Exception("LLM fail"))
        else:
            loop._process_news_article = MagicMock()

        loop._news_cycle()

    def test_llm_liveness_no_alert_under_threshold(self) -> None:
        """2 consecutive all-fail cycles do not trigger alert."""
        loop = _make_trading_loop()
        loop._llm_consecutive_failures = 0

        self._run_cycle_with_failure(loop, fail=True)
        self._run_cycle_with_failure(loop, fail=True)

        loop._alerter.on_error.assert_not_called()

    def test_llm_liveness_alert_at_threshold(self) -> None:
        """3 consecutive all-fail cycles trigger TelegramAlerter.on_error."""
        loop = _make_trading_loop()
        loop._llm_consecutive_failures = 0

        self._run_cycle_with_failure(loop, fail=True)
        self._run_cycle_with_failure(loop, fail=True)
        self._run_cycle_with_failure(loop, fail=True)

        loop._alerter.on_error.assert_called()
        call_args = loop._alerter.on_error.call_args
        assert call_args[0][0] == "LLMLiveness"

    def test_llm_liveness_reset_on_success(self) -> None:
        """After 2 fails, 1 success resets counter. 2 more fails do not alert."""
        loop = _make_trading_loop()
        loop._llm_consecutive_failures = 0

        self._run_cycle_with_failure(loop, fail=True)
        self._run_cycle_with_failure(loop, fail=True)
        # Success resets
        self._run_cycle_with_failure(loop, fail=False)
        # 2 more fails
        self._run_cycle_with_failure(loop, fail=True)
        self._run_cycle_with_failure(loop, fail=True)

        loop._alerter.on_error.assert_not_called()

    def test_llm_liveness_prometheus_counter(self) -> None:
        """Prometheus counter increments on each failure cycle."""
        from finalayze.api.metrics import llm_liveness_failures

        loop = _make_trading_loop()
        loop._llm_consecutive_failures = 0

        before = llm_liveness_failures._value.get()
        self._run_cycle_with_failure(loop, fail=True)
        self._run_cycle_with_failure(loop, fail=True)
        self._run_cycle_with_failure(loop, fail=True)
        after = llm_liveness_failures._value.get()

        expected_increment = 3
        assert after - before == expected_increment  # noqa: PLR2004

    def test_llm_liveness_re_alert_on_sustained_failure(self) -> None:
        """After 3 fails + alert, 3 more fails trigger a second alert."""
        loop = _make_trading_loop()
        loop._llm_consecutive_failures = 0

        # First 3 fails -> first alert
        for _ in range(3):
            self._run_cycle_with_failure(loop, fail=True)
        assert loop._alerter.on_error.call_count == 1

        # 3 more fails -> second alert (at failure count 6)
        for _ in range(3):
            self._run_cycle_with_failure(loop, fail=True)
        assert loop._alerter.on_error.call_count >= 2  # noqa: PLR2004


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
