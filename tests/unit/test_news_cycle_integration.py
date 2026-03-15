"""Integration tests for TradingLoop._news_cycle with RSS + Telegram + EntityExtractor.

Validates that the news cycle fetches from multiple sources with independent
error handling and runs entity extraction before sentiment processing.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from finalayze.core.schemas import NewsArticle


def _make_article(title: str = "Test", source: str = "rss") -> NewsArticle:
    return NewsArticle(
        id=uuid4(),
        source=source,
        title=title,
        content=f"Content of {title}",
        url=f"https://example.com/{title.replace(' ', '-')}",
        language="ru",
        published_at=datetime.now(UTC),
        scope="russia",
    )


def _make_loop(
    *,
    rss_fetcher: object | None = None,
    telegram_reader: object | None = None,
    entity_extractor: object | None = None,
    news_fetcher: object | None = None,
    settings: object | None = None,
) -> object:
    """Build a TradingLoop with mocked dependencies."""
    from finalayze.core.trading_loop import TradingLoop

    mock_settings = settings or MagicMock()
    if settings is None:
        mock_settings.news_cycle_minutes = 15
        mock_settings.news_poll_interval_minutes = 5
        mock_settings.strategy_cycle_minutes = 30
        mock_settings.daily_reset_hour_utc = 0
        mock_settings.max_position_pct = 0.1
        mock_settings.max_positions_per_market = 10
        mock_settings.daily_loss_limit_pct = 0.05
        mock_settings.kelly_fraction = 0.5
        mock_settings.ml_enabled = False
        mock_settings.telegram_channels = ["@test_channel"]

    mock_news_fetcher = news_fetcher or MagicMock()

    loop = TradingLoop(
        settings=mock_settings,
        fetchers={},
        news_fetcher=mock_news_fetcher,
        news_analyzer=MagicMock(),
        event_classifier=MagicMock(),
        impact_estimator=MagicMock(),
        strategy=MagicMock(),
        broker_router=MagicMock(),
        circuit_breakers={},
        cross_market_breaker=MagicMock(),
        alerter=MagicMock(),
        instrument_registry=MagicMock(),
        rss_fetcher=rss_fetcher,
        telegram_reader=telegram_reader,
        entity_extractor=entity_extractor,
    )
    return loop


class TestNewsCycleRss:
    """Test RSS fetcher integration in _news_cycle."""

    def test_rss_articles_fetched_and_processed(self) -> None:
        """_news_cycle fetches from RSS fetcher and processes articles."""
        rss = MagicMock()
        articles = [_make_article("RSS Article 1"), _make_article("RSS Article 2")]
        rss.fetch_news.return_value = articles

        loop = _make_loop(rss_fetcher=rss)
        # Mock _process_news_article to track calls
        loop._process_news_article = MagicMock()  # type: ignore[attr-defined]

        loop._news_cycle()  # type: ignore[attr-defined]

        rss.fetch_news.assert_called_once()
        assert loop._process_news_article.call_count == 2  # type: ignore[attr-defined]

    def test_rss_failure_does_not_block_telegram(self) -> None:
        """RSS failure does not prevent Telegram fetch."""
        rss = MagicMock()
        rss.fetch_news.side_effect = RuntimeError("RSS down")

        tg = MagicMock()
        tg_articles = [_make_article("TG Article", source="telegram:@chan")]
        tg.fetch_recent_messages = AsyncMock(return_value=tg_articles)

        loop = _make_loop(rss_fetcher=rss, telegram_reader=tg)
        loop._process_news_article = MagicMock()  # type: ignore[attr-defined]

        loop._news_cycle()  # type: ignore[attr-defined]

        # Telegram articles should still be processed
        assert loop._process_news_article.call_count == 1  # type: ignore[attr-defined]


class TestNewsCycleTelegram:
    """Test Telegram reader integration in _news_cycle."""

    def test_telegram_articles_fetched_and_processed(self) -> None:
        """_news_cycle fetches from Telegram reader and processes articles."""
        tg = MagicMock()
        tg_articles = [_make_article("TG News", source="telegram:@rbc")]
        tg.fetch_recent_messages = AsyncMock(return_value=tg_articles)

        loop = _make_loop(telegram_reader=tg)
        loop._process_news_article = MagicMock()  # type: ignore[attr-defined]

        loop._news_cycle()  # type: ignore[attr-defined]

        assert loop._process_news_article.call_count == 1  # type: ignore[attr-defined]

    def test_telegram_failure_does_not_block_rss(self) -> None:
        """Telegram failure does not prevent RSS fetch."""
        rss = MagicMock()
        rss_articles = [_make_article("RSS News")]
        rss.fetch_news.return_value = rss_articles

        tg = MagicMock()
        tg.fetch_recent_messages = AsyncMock(side_effect=RuntimeError("TG down"))

        loop = _make_loop(rss_fetcher=rss, telegram_reader=tg)
        loop._process_news_article = MagicMock()  # type: ignore[attr-defined]

        loop._news_cycle()  # type: ignore[attr-defined]

        # RSS articles should still be processed
        assert loop._process_news_article.call_count == 1  # type: ignore[attr-defined]


class TestEntityExtraction:
    """Test entity extraction integration in _news_cycle."""

    def test_entity_extraction_populates_symbols(self) -> None:
        """Entity extraction runs on each article and populates symbols."""
        rss = MagicMock()
        article = _make_article("Sberbank grows")
        rss.fetch_news.return_value = [article]

        extractor = MagicMock()
        extractor.extract = AsyncMock(return_value=["SBER"])

        loop = _make_loop(rss_fetcher=rss, entity_extractor=extractor)
        loop._process_news_article = MagicMock()  # type: ignore[attr-defined]

        loop._news_cycle()  # type: ignore[attr-defined]

        extractor.extract.assert_called_once()
        # The processed article should have symbols populated
        processed_article = loop._process_news_article.call_args[0][0]  # type: ignore[attr-defined]
        assert processed_article.symbols == ["SBER"]


class TestNewsCycleInterval:
    """Test scheduler interval configuration."""

    def test_news_interval_uses_poll_minutes_when_rss_configured(self) -> None:
        """News cycle uses settings.news_poll_interval_minutes when RSS is configured."""
        rss = MagicMock()
        rss.fetch_news.return_value = []

        mock_settings = MagicMock()
        mock_settings.news_cycle_minutes = 15
        mock_settings.news_poll_interval_minutes = 5
        mock_settings.strategy_cycle_minutes = 30
        mock_settings.daily_reset_hour_utc = 0
        mock_settings.weekly_digest_hour_utc = 10
        mock_settings.max_position_pct = 0.1
        mock_settings.max_positions_per_market = 10
        mock_settings.daily_loss_limit_pct = 0.05
        mock_settings.kelly_fraction = 0.5
        mock_settings.ml_enabled = False
        mock_settings.telegram_channels = ["@test_channel"]
        mock_settings.mode = MagicMock()
        mock_settings.mode.value = "sandbox"

        loop = _make_loop(rss_fetcher=rss, settings=mock_settings)

        # Start the scheduler to check the interval
        with patch.object(loop, "_stop_event", MagicMock()) as stop_event:  # type: ignore[attr-defined]
            stop_event.wait.return_value = True  # Don't block

            with patch("finalayze.core.trading_loop.BackgroundScheduler") as mock_sched_cls:
                mock_sched = MagicMock()
                mock_sched_cls.return_value = mock_sched

                loop.start()  # type: ignore[attr-defined]

                # Find the news_cycle add_job call
                add_job_calls = mock_sched.add_job.call_args_list
                news_call = [c for c in add_job_calls if c[1].get("id") == "news_cycle"]
                assert len(news_call) == 1
                assert news_call[0][1]["minutes"] == 5  # news_poll_interval_minutes


class TestLegacyFallback:
    """Test that legacy NewsAPI is used when no RSS/Telegram configured."""

    def test_legacy_fallback_when_no_new_fetchers(self) -> None:
        """Without RSS/Telegram, legacy NewsAPI fetcher is used."""
        legacy = MagicMock()
        legacy_articles = [_make_article("Legacy news")]
        legacy.fetch_news.return_value = legacy_articles

        loop = _make_loop(news_fetcher=legacy)
        loop._process_news_article = MagicMock()  # type: ignore[attr-defined]

        loop._news_cycle()  # type: ignore[attr-defined]

        legacy.fetch_news.assert_called_once()
        assert loop._process_news_article.call_count == 1  # type: ignore[attr-defined]
