"""Integration tests for TradingLoop._news_cycle with NewsImpactAnalyzer pipeline.

Validates that the news cycle fetches from multiple sources with independent
error handling and runs NewsImpactAnalyzer for per-ticker sentiment.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from finalayze.analysis.event_classifier import EventType
from finalayze.analysis.news_impact_analyzer import NewsImpactResult, SectorImpactDetail
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


def _make_impact_result(
    *,
    sentiment: float = 0.6,
    confidence: float = 0.8,
    sectors: list[SectorImpactDetail] | None = None,
    direct_tickers: list[str] | None = None,
) -> NewsImpactResult:
    """Build a NewsImpactResult for testing."""
    return NewsImpactResult(
        event_type=EventType.CBR_RATE,
        sentiment=sentiment,
        confidence=confidence,
        reasoning="test reasoning",
        affected_sectors=sectors or [],
        direct_tickers=direct_tickers or [],
    )


def _make_loop(
    *,
    rss_fetcher: object | None = None,
    telegram_reader: object | None = None,
    news_impact_analyzer: object | None = None,
    sector_ticker_mapper: object | None = None,
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
        news_impact_analyzer=news_impact_analyzer,
        sector_ticker_mapper=sector_ticker_mapper,
    )
    # Pre-set event_driven guard so news cycle tests proceed without reading YAMLs
    loop._event_driven_active = True  # type: ignore[attr-defined]
    return loop


class TestNewsCycleRss:
    """Test RSS fetcher integration in _news_cycle."""

    def test_rss_articles_fetched_and_processed(self) -> None:
        """_news_cycle fetches from RSS fetcher and processes articles."""
        rss = MagicMock()
        articles = [_make_article("RSS Article 1"), _make_article("RSS Article 2")]
        rss.fetch_news.return_value = articles

        analyzer = MagicMock()
        analyzer.analyze = AsyncMock(return_value=_make_impact_result())

        loop = _make_loop(rss_fetcher=rss, news_impact_analyzer=analyzer)
        loop._analyze_impact_batch = AsyncMock(return_value=(2, 0, ""))  # type: ignore[attr-defined]

        loop._news_cycle()  # type: ignore[attr-defined]

        rss.fetch_news.assert_called_once()
        loop._analyze_impact_batch.assert_called_once()  # type: ignore[attr-defined]
        processed = loop._analyze_impact_batch.call_args[0][0]  # type: ignore[attr-defined]
        assert len(processed) == 2

    def test_rss_failure_does_not_block_telegram(self) -> None:
        """RSS failure does not prevent Telegram fetch."""
        rss = MagicMock()
        rss.fetch_news.side_effect = RuntimeError("RSS down")

        tg = MagicMock()
        tg_articles = [_make_article("TG Article", source="telegram:@chan")]
        tg.fetch_recent_messages = AsyncMock(return_value=tg_articles)

        analyzer = MagicMock()
        analyzer.analyze = AsyncMock(return_value=_make_impact_result())

        loop = _make_loop(
            rss_fetcher=rss,
            telegram_reader=tg,
            news_impact_analyzer=analyzer,
        )
        loop._analyze_impact_batch = AsyncMock(return_value=(1, 0, ""))  # type: ignore[attr-defined]

        loop._news_cycle()  # type: ignore[attr-defined]

        # Telegram articles should still be processed
        loop._analyze_impact_batch.assert_called_once()  # type: ignore[attr-defined]
        processed = loop._analyze_impact_batch.call_args[0][0]  # type: ignore[attr-defined]
        assert len(processed) == 1


class TestNewsCycleTelegram:
    """Test Telegram reader integration in _news_cycle."""

    def test_telegram_articles_fetched_and_processed(self) -> None:
        """_news_cycle fetches from Telegram reader and processes articles."""
        tg = MagicMock()
        tg_articles = [_make_article("TG News", source="telegram:@rbc")]
        tg.fetch_recent_messages = AsyncMock(return_value=tg_articles)

        analyzer = MagicMock()
        analyzer.analyze = AsyncMock(return_value=_make_impact_result())

        loop = _make_loop(telegram_reader=tg, news_impact_analyzer=analyzer)
        loop._analyze_impact_batch = AsyncMock(return_value=(1, 0, ""))  # type: ignore[attr-defined]

        loop._news_cycle()  # type: ignore[attr-defined]

        loop._analyze_impact_batch.assert_called_once()  # type: ignore[attr-defined]
        processed = loop._analyze_impact_batch.call_args[0][0]  # type: ignore[attr-defined]
        assert len(processed) == 1

    def test_telegram_failure_does_not_block_rss(self) -> None:
        """Telegram failure does not prevent RSS fetch."""
        rss = MagicMock()
        rss_articles = [_make_article("RSS News")]
        rss.fetch_news.return_value = rss_articles

        tg = MagicMock()
        tg.fetch_recent_messages = AsyncMock(side_effect=RuntimeError("TG down"))

        analyzer = MagicMock()
        analyzer.analyze = AsyncMock(return_value=_make_impact_result())

        loop = _make_loop(
            rss_fetcher=rss,
            telegram_reader=tg,
            news_impact_analyzer=analyzer,
        )
        loop._analyze_impact_batch = AsyncMock(return_value=(1, 0, ""))  # type: ignore[attr-defined]

        loop._news_cycle()  # type: ignore[attr-defined]

        # RSS articles should still be processed
        loop._analyze_impact_batch.assert_called_once()  # type: ignore[attr-defined]
        processed = loop._analyze_impact_batch.call_args[0][0]  # type: ignore[attr-defined]
        assert len(processed) == 1


class TestNewsImpactPipeline:
    """Test NewsImpactAnalyzer integration replacing EntityExtractor + CombinedAnalyzer."""

    def test_news_cycle_calls_impact_analyzer_not_entity_extractor(self) -> None:
        """_news_cycle calls NewsImpactAnalyzer.analyze(), not EntityExtractor."""
        rss = MagicMock()
        rss.fetch_news.return_value = [_make_article("CBR raised rates")]

        analyzer = MagicMock()
        analyzer.analyze = AsyncMock(return_value=_make_impact_result())

        loop = _make_loop(rss_fetcher=rss, news_impact_analyzer=analyzer)
        loop._analyze_impact_batch = AsyncMock(return_value=(1, 0, ""))  # type: ignore[attr-defined]

        loop._news_cycle()  # type: ignore[attr-defined]

        # Should call _analyze_impact_batch, not _process_articles_batch
        loop._analyze_impact_batch.assert_called_once()  # type: ignore[attr-defined]
        assert not hasattr(loop, "_entity_extractor")
        assert not hasattr(loop, "_combined_analyzer")

    def test_per_ticker_sentiment_from_sector_impact(self) -> None:
        """Article with affected_sectors=['banking'] populates per-ticker cache
        for SBER, VTBR, TCSG (NEWS-08)."""
        from finalayze.analysis.sector_ticker_mapper import SectorTickerMapper

        mapper = SectorTickerMapper()
        result = _make_impact_result(
            sentiment=0.6,
            confidence=0.8,
            sectors=[
                SectorImpactDetail(
                    sector="banking",
                    direction=1,
                    magnitude=0.7,
                    reasoning="CBR rate cut positive for banks",
                ),
            ],
        )

        loop = _make_loop(
            news_impact_analyzer=MagicMock(),
            sector_ticker_mapper=mapper,
        )

        # Set up fetchers and registry so _collect_active_segments / _get_segment_tickers work
        mock_instruments = []
        for ticker in ["SBER", "VTBR", "TCSG", "LKOH"]:
            instr = MagicMock()
            instr.symbol = ticker
            instr.segment_id = "ru_blue_chips"
            mock_instruments.append(instr)
        loop._fetchers = {"moex": MagicMock()}  # type: ignore[attr-defined]
        loop._registry.list_by_market.return_value = mock_instruments  # type: ignore[attr-defined]

        import asyncio as _aio

        _aio.run(loop._apply_impact_result(result))  # type: ignore[attr-defined]

        # Check per-ticker cache entries
        cache = loop._sentiment_cache  # type: ignore[attr-defined]
        assert ("ru_blue_chips", "SBER") in cache
        assert ("ru_blue_chips", "VTBR") in cache
        assert ("ru_blue_chips", "TCSG") in cache
        # LKOH is not in banking sector, should NOT have entry
        assert ("ru_blue_chips", "LKOH") not in cache

    def test_direct_ticker_sentiment(self) -> None:
        """Article with direct_tickers=['SBER'] gets per-ticker entry."""
        from finalayze.analysis.sector_ticker_mapper import SectorTickerMapper

        mapper = SectorTickerMapper()
        result = _make_impact_result(
            sentiment=0.5,
            confidence=0.9,
            direct_tickers=["SBER"],
        )

        loop = _make_loop(
            news_impact_analyzer=MagicMock(),
            sector_ticker_mapper=mapper,
        )

        mock_instruments = []
        for ticker in ["SBER", "VTBR"]:
            instr = MagicMock()
            instr.symbol = ticker
            instr.segment_id = "ru_blue_chips"
            mock_instruments.append(instr)
        loop._fetchers = {"moex": MagicMock()}  # type: ignore[attr-defined]
        loop._registry.list_by_market.return_value = mock_instruments  # type: ignore[attr-defined]

        import asyncio as _aio

        _aio.run(loop._apply_impact_result(result))  # type: ignore[attr-defined]

        cache = loop._sentiment_cache  # type: ignore[attr-defined]
        assert ("ru_blue_chips", "SBER") in cache
        # VTBR was not mentioned, should NOT have entry
        assert ("ru_blue_chips", "VTBR") not in cache

    def test_sentiment_formula_sector(self) -> None:
        """Sentiment score = sector.magnitude * sector.direction * article.sentiment."""
        from finalayze.analysis.sector_ticker_mapper import SectorTickerMapper

        mapper = SectorTickerMapper()
        result = _make_impact_result(
            sentiment=-0.5,
            confidence=0.8,
            sectors=[
                SectorImpactDetail(
                    sector="banking",
                    direction=-1,
                    magnitude=0.6,
                    reasoning="negative for banks",
                ),
            ],
        )

        loop = _make_loop(
            news_impact_analyzer=MagicMock(),
            sector_ticker_mapper=mapper,
        )

        instr = MagicMock()
        instr.symbol = "SBER"
        instr.segment_id = "ru_blue_chips"
        loop._fetchers = {"moex": MagicMock()}  # type: ignore[attr-defined]
        loop._registry.list_by_market.return_value = [instr]  # type: ignore[attr-defined]

        import asyncio as _aio

        _aio.run(loop._apply_impact_result(result))  # type: ignore[attr-defined]

        cache = loop._sentiment_cache  # type: ignore[attr-defined]
        score, _ts = cache[("ru_blue_chips", "SBER")]
        # formula: magnitude * direction * sentiment = 0.6 * -1 * -0.5 = 0.3
        # EMA: 0.0 * 0.7 + 0.3 * 0.3 = 0.09
        expected = 0.0 * 0.7 + (0.6 * (-1) * (-0.5)) * 0.3
        assert abs(score - expected) < 1e-9


class TestPerTickerSentimentRead:
    """Test _read_decayed_sentiment and _get_sentiment with per-ticker keying."""

    def test_read_decayed_sentiment_per_ticker(self) -> None:
        """_read_decayed_sentiment(seg_id, ticker) returns per-ticker score."""
        loop = _make_loop()
        now = time.monotonic()

        with loop._sentiment_lock:  # type: ignore[attr-defined]
            loop._sentiment_cache[("ru_blue_chips", "SBER")] = (0.8, now)  # type: ignore[attr-defined]

        result = loop._read_decayed_sentiment("ru_blue_chips", "SBER")  # type: ignore[attr-defined]
        assert abs(result - 0.8) < 0.01

    def test_read_decayed_sentiment_fallback_to_segment_average(self) -> None:
        """_read_decayed_sentiment(seg_id, 'UNKNOWN') falls back to segment average."""
        loop = _make_loop()
        now = time.monotonic()

        with loop._sentiment_lock:  # type: ignore[attr-defined]
            loop._sentiment_cache[("ru_blue_chips", "SBER")] = (0.6, now)  # type: ignore[attr-defined]
            loop._sentiment_cache[("ru_blue_chips", "VTBR")] = (0.4, now)  # type: ignore[attr-defined]

        result = loop._read_decayed_sentiment(  # type: ignore[attr-defined]
            "ru_blue_chips", "UNKNOWN_TICKER"
        )
        # Should be average of 0.6 and 0.4 = 0.5
        assert abs(result - 0.5) < 0.01

    def test_read_decayed_sentiment_default_when_no_entries(self) -> None:
        """_read_decayed_sentiment returns default when no entries for segment."""
        loop = _make_loop()
        result = loop._read_decayed_sentiment("nonexistent", "SBER")  # type: ignore[attr-defined]
        assert result == 0.0

    def test_get_sentiment_passes_ticker(self) -> None:
        """_get_sentiment(seg_id, ticker) passes ticker to _read_decayed_sentiment."""
        loop = _make_loop()
        loop._cache = None  # type: ignore[attr-defined]
        now = time.monotonic()

        with loop._sentiment_lock:  # type: ignore[attr-defined]
            loop._sentiment_cache[("ru_blue_chips", "SBER")] = (0.8, now)  # type: ignore[attr-defined]

        result = loop._get_sentiment("ru_blue_chips", "SBER")  # type: ignore[attr-defined]
        assert abs(result - 0.8) < 0.01

    def test_sector_only_article_produces_nonzero_sentiment(self) -> None:
        """Sector-only articles (no direct_tickers) produce non-zero sentiment
        for mapped tickers (NEWS-08)."""
        from finalayze.analysis.sector_ticker_mapper import SectorTickerMapper

        mapper = SectorTickerMapper()
        result = _make_impact_result(
            sentiment=0.7,
            confidence=0.9,
            sectors=[
                SectorImpactDetail(
                    sector="banking",
                    direction=1,
                    magnitude=0.8,
                    reasoning="positive for banks",
                ),
            ],
            direct_tickers=[],  # No direct tickers!
        )

        loop = _make_loop(
            news_impact_analyzer=MagicMock(),
            sector_ticker_mapper=mapper,
        )

        instr = MagicMock()
        instr.symbol = "SBER"
        instr.segment_id = "ru_blue_chips"
        loop._fetchers = {"moex": MagicMock()}  # type: ignore[attr-defined]
        loop._registry.list_by_market.return_value = [instr]  # type: ignore[attr-defined]

        import asyncio as _aio

        _aio.run(loop._apply_impact_result(result))  # type: ignore[attr-defined]

        cache = loop._sentiment_cache  # type: ignore[attr-defined]
        score, _ts = cache[("ru_blue_chips", "SBER")]
        assert score != 0.0, "Sector-only article must produce non-zero sentiment"


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

        analyzer = MagicMock()
        analyzer.analyze = AsyncMock(return_value=_make_impact_result())

        loop = _make_loop(news_fetcher=legacy, news_impact_analyzer=analyzer)
        loop._analyze_impact_batch = AsyncMock(return_value=(1, 0, ""))  # type: ignore[attr-defined]

        loop._news_cycle()  # type: ignore[attr-defined]

        legacy.fetch_news.assert_called_once()
        loop._analyze_impact_batch.assert_called_once()  # type: ignore[attr-defined]
        processed = loop._analyze_impact_batch.call_args[0][0]  # type: ignore[attr-defined]
        assert len(processed) == 1
