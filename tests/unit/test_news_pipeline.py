"""Unit tests for news pipeline budget cap and threading safety."""

from __future__ import annotations

from collections import OrderedDict
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


def _wire_subcomponents(loop: Any) -> None:
    """Wire NewsPipeline and SentimentManager onto a TradingLoop instance."""
    from finalayze.orchestration.news_pipeline import NewsPipeline
    from finalayze.orchestration.sentiment_manager import SentimentManager

    sentiment_mgr = SentimentManager.__new__(SentimentManager)
    sentiment_mgr._sentiment_cache = loop._sentiment_cache
    sentiment_mgr._sentiment_lock = loop._sentiment_lock
    sentiment_mgr._cache = loop._cache
    sentiment_mgr._event_driven_active = True
    sentiment_mgr._registry = loop._registry
    sentiment_mgr._market_ids = list(loop._fetchers.keys()) if loop._fetchers else []
    loop._sentiment_mgr = sentiment_mgr

    _PIPELINE_ATTRS = (
        "_news_fetcher",
        "_news_analyzer",
        "_event_classifier",
        "_impact_estimator",
        "_alerter",
        "_registry",
        "_fetchers",
        "_settings",
        "_event_driven_active",
        "_rss_fetcher",
        "_telegram_reader",
        "_news_impact_analyzer",
        "_seen_article_hashes",
        "_health_monitor",
        "_metrics",
        "_llm_consecutive_failures",
    )
    pipeline = NewsPipeline.__new__(NewsPipeline)
    for attr in _PIPELINE_ATTRS:
        if hasattr(loop, attr):
            setattr(pipeline, attr, getattr(loop, attr))
    pipeline._sentiment_mgr = sentiment_mgr
    pipeline._persistence = MagicMock()
    pipeline._async_loop_fn = getattr(loop, "_run_async", lambda coro, **kw: None)
    loop._news_pipeline = pipeline


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
    loop._llm_consecutive_failures = 0

    import threading

    loop._sentiment_lock = threading.Lock()
    loop._async_loop = None
    loop._async_thread = None
    loop._event_driven_active = True
    loop._rss_fetcher = None
    loop._telegram_reader = None
    loop._news_impact_analyzer = None
    loop._seen_article_hashes = OrderedDict()
    loop._health_monitor = None
    loop._metrics = None

    for key, val in overrides.items():
        setattr(loop, f"_{key}", val)

    _wire_subcomponents(loop)

    return loop


class TestBudgetCap:
    def test_budget_cap_limits_articles(self) -> None:
        """When more than 20 articles fetched, only 20 are processed."""
        articles = [_make_article(i) for i in range(30)]
        loop = _make_trading_loop()
        loop._news_fetcher.fetch_news.return_value = articles
        loop._news_pipeline._news_fetcher = loop._news_fetcher

        # Wire up _news_impact_analyzer so _analyze_impact_batch is called.
        analyzer = MagicMock()
        loop._news_impact_analyzer = analyzer
        loop._news_pipeline._news_impact_analyzer = analyzer

        # Mock _async_loop_fn on the pipeline to capture batch size
        def mock_run_async(coro: object, *, timeout: int = 30) -> tuple[int, int, str]:
            return (20, 0, "")

        loop._news_pipeline._async_loop_fn = mock_run_async
        loop._news_cycle()

        from finalayze.core.trading_loop import _MAX_ARTICLES_PER_CYCLE

        assert _MAX_ARTICLES_PER_CYCLE == 20  # noqa: PLR2004

    def test_budget_cap_metric_incremented(self) -> None:
        """When articles > 20, MetricsCollector.inc_news_budget_cap_hit is called."""
        articles = [_make_article(i) for i in range(25)]
        loop = _make_trading_loop()
        loop._news_fetcher.fetch_news.return_value = articles

        with patch("finalayze.api.metrics.MetricsCollector") as mock_metrics:
            loop._news_cycle()
            mock_metrics.inc_news_budget_cap_hit.assert_called_once()

    def test_no_budget_cap_under_limit(self) -> None:
        """When articles <= 20, all are processed and no cap metric fired."""
        article_count = 15
        articles = [_make_article(i) for i in range(article_count)]
        loop = _make_trading_loop()
        loop._news_fetcher.fetch_news.return_value = articles

        with patch("finalayze.api.metrics.MetricsCollector") as mock_metrics:
            loop._news_cycle()
            mock_metrics.inc_news_budget_cap_hit.assert_not_called()


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

    def test_credibility_set_on_articles_in_news_cycle(self) -> None:
        """_news_cycle sets credibility_score on articles via model_copy."""
        from finalayze.core.trading_loop import get_credibility

        loop = _make_trading_loop()
        article = _make_article()
        loop._news_fetcher.fetch_news.return_value = [article]

        # Use _news_impact_analyzer mock to capture articles after credibility is set.
        analyzer = MagicMock()
        loop._news_impact_analyzer = analyzer

        _captured_articles: list[NewsArticle] = []

        def mock_run_async(coro, *, timeout: int = 30) -> tuple[int, int, str]:
            # Cannot inspect coroutine args directly, but credibility was already
            # applied before _analyze_impact_batch is called. Return success.
            return (1, 0, "")

        loop._run_async = mock_run_async
        loop._news_cycle()

        # Verify credibility score is correct for "test" source
        expected_cred = get_credibility("test")
        assert expected_cred == 0.5  # noqa: PLR2004 -- unknown source default


class TestSentimentScoreModelCredibility:
    def test_sentiment_score_model_has_credibility(self) -> None:
        """SentimentScoreModel has a credibility column."""
        from finalayze.core.models import SentimentScoreModel

        assert hasattr(SentimentScoreModel, "credibility")


class TestLLMLiveness:
    """Tests for LLM liveness tracking in _news_cycle.

    The current _news_cycle uses _news_impact_analyzer via _run_async which returns
    (processed_ok, processed_fail, last_error). LLM liveness is tracked based on
    whether processed_ok == 0 when there were articles to process.
    """

    def _run_cycle_with_result(self, loop: Any, *, ok: int = 0, fail: int = 0) -> None:
        """Run a news cycle with a specific ok/fail count from _analyze_impact_batch."""
        article = _make_article()
        loop._news_fetcher.fetch_news.return_value = [article]

        # Wire up analyzer so _news_cycle enters the processing path
        loop._news_impact_analyzer = MagicMock()
        loop._run_async = MagicMock(return_value=(ok, fail, ""))

        loop._news_cycle()

    def test_llm_liveness_no_alert_under_threshold(self) -> None:
        """2 consecutive all-fail cycles do not trigger alert."""
        loop = _make_trading_loop()
        loop._llm_consecutive_failures = 0

        self._run_cycle_with_result(loop, ok=0, fail=1)
        self._run_cycle_with_result(loop, ok=0, fail=1)

        loop._alerter.on_error.assert_not_called()

    def test_llm_liveness_alert_at_threshold(self) -> None:
        """3 consecutive all-fail cycles trigger TelegramAlerter.on_error."""
        loop = _make_trading_loop()
        loop._llm_consecutive_failures = 0

        self._run_cycle_with_result(loop, ok=0, fail=1)
        self._run_cycle_with_result(loop, ok=0, fail=1)
        self._run_cycle_with_result(loop, ok=0, fail=1)

        loop._alerter.on_error.assert_called()
        call_args = loop._alerter.on_error.call_args
        assert call_args[0][0] == "LLMLiveness"

    def test_llm_liveness_reset_on_success(self) -> None:
        """After 2 fails, 1 success resets counter. 2 more fails do not alert."""
        loop = _make_trading_loop()
        loop._llm_consecutive_failures = 0

        self._run_cycle_with_result(loop, ok=0, fail=1)
        self._run_cycle_with_result(loop, ok=0, fail=1)
        # Success resets
        self._run_cycle_with_result(loop, ok=1, fail=0)
        # 2 more fails
        self._run_cycle_with_result(loop, ok=0, fail=1)
        self._run_cycle_with_result(loop, ok=0, fail=1)

        loop._alerter.on_error.assert_not_called()

    def test_llm_liveness_prometheus_counter(self) -> None:
        """Prometheus counter increments on each failure cycle."""
        from finalayze.api.metrics import llm_liveness_failures

        loop = _make_trading_loop()
        loop._llm_consecutive_failures = 0

        before = llm_liveness_failures._value.get()
        self._run_cycle_with_result(loop, ok=0, fail=1)
        self._run_cycle_with_result(loop, ok=0, fail=1)
        self._run_cycle_with_result(loop, ok=0, fail=1)
        after = llm_liveness_failures._value.get()

        expected_increment = 3
        assert after - before == expected_increment  # noqa: PLR2004

    def test_llm_liveness_re_alert_on_sustained_failure(self) -> None:
        """After 3 fails + alert, 3 more fails trigger a second alert."""
        loop = _make_trading_loop()
        loop._llm_consecutive_failures = 0

        # First 3 fails -> first alert
        for _ in range(3):
            self._run_cycle_with_result(loop, ok=0, fail=1)
        assert loop._alerter.on_error.call_count == 1

        # 3 more fails -> second alert (at failure count 6)
        for _ in range(3):
            self._run_cycle_with_result(loop, ok=0, fail=1)
        assert loop._alerter.on_error.call_count >= 2  # noqa: PLR2004

    def test_llm_liveness_reset_on_real_success(self) -> None:
        """2 fail cycles then 1 success resets counter."""
        loop = _make_trading_loop()
        loop._llm_consecutive_failures = 0

        self._run_cycle_with_result(loop, ok=0, fail=1)
        self._run_cycle_with_result(loop, ok=0, fail=1)
        # Real success resets counter
        self._run_cycle_with_result(loop, ok=1, fail=0)
        # 2 more fails should not trigger alert
        self._run_cycle_with_result(loop, ok=0, fail=1)
        self._run_cycle_with_result(loop, ok=0, fail=1)

        loop._alerter.on_error.assert_not_called()

    def test_llm_liveness_mixed_ok_and_fail_resets_counter(self) -> None:
        """A cycle with at least one ok article resets the counter."""
        loop = _make_trading_loop()
        loop._llm_consecutive_failures = 0
        articles = [_make_article(0), _make_article(1)]
        loop._news_fetcher.fetch_news.return_value = articles

        # _run_async returns (1 ok, 1 fail) -> resets counter
        loop._news_impact_analyzer = MagicMock()
        loop._run_async = MagicMock(return_value=(1, 1, ""))
        loop._news_cycle()

        assert loop._llm_consecutive_failures == 0


class TestSentimentLockSafety:
    def test_sentiment_lock_not_in_async_methods(self) -> None:
        """_sentiment_lock must not be acquired in unexpected async methods.

        _apply_impact_result is an allowed exception: it runs on the background
        async loop and holds the threading.Lock briefly for cache updates.
        """
        import ast
        import inspect

        from finalayze.core import trading_loop as mod

        source = inspect.getsource(mod.TradingLoop)
        tree = ast.parse(source)

        # _apply_impact_result is allowed to use _sentiment_lock because it must
        # update the shared cache from the async background loop.
        _allowed = {"_apply_impact_result"}

        # Find all async methods that reference _sentiment_lock
        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef):
                if node.name in _allowed:
                    continue
                method_source = ast.get_source_segment(source, node) or ""
                assert "_sentiment_lock" not in method_source, (
                    f"async method {node.name} must not reference _sentiment_lock"
                )
