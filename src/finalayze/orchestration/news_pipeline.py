"""News pipeline orchestration (Phase 1.5 -- orchestrator).

Extracted from trading_loop.py: manages news fetching, deduplication, impact
analysis, and sentiment updates.

Scheduled by TradingLoop via APScheduler as _news_cycle job.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections import OrderedDict
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from config.settings import Settings

    from finalayze.analysis.news_impact_analyzer import NewsImpactAnalyzer, NewsImpactResult
    from finalayze.analysis.sector_ticker_mapper import SectorTickerMapper
    from finalayze.api.alerts import TelegramAlerter
    from finalayze.core.schemas import NewsArticle
    from finalayze.data.cache import RedisCache
    from finalayze.data.fetchers.newsapi import NewsApiFetcher
    from finalayze.data.fetchers.rss_fetcher import RssNewsFetcher
    from finalayze.data.fetchers.telegram_reader import TelegramChannelReader
    from finalayze.markets.instruments import InstrumentRegistry
    from finalayze.orchestration.db_persistence import TradingPersistence
    from finalayze.orchestration.sentiment_manager import SentimentManager

# ── Constants ──────────────────────────────────────────────────────────────
_NEWS_QUERY = "stock market finance"
_NEWS_LOOKBACK_HOURS = 2
_ARTICLE_DEDUP_MAX_SIZE = 5000  # max hashes to track
_ARTICLE_DEDUP_TTL_HOURS = 24  # skip articles seen within this window

_log = structlog.get_logger()


class NewsPipeline:
    """Orchestrates news fetching, impact analysis, and sentiment updates.

    Pulled from TradingLoop for Phase 1.5 refactor. Manages:
    - RSS feed fetching (sync)
    - Telegram channel reading (async)
    - Legacy NewsAPI fallback (sync)
    - Impact analysis with circuit breaker (async batch with semaphore)
    - Sentiment cache + Redis + DB updates
    - Deduplication with TTL window
    """

    def __init__(
        self,
        rss_fetcher: RssNewsFetcher | None,
        telegram_reader: TelegramChannelReader | None,
        news_fetcher: NewsApiFetcher | None,
        news_impact_analyzer: NewsImpactAnalyzer | None,
        sector_ticker_mapper: SectorTickerMapper | None,
        sentiment_mgr: SentimentManager,
        persistence: TradingPersistence,
        registry: InstrumentRegistry,
        cache: RedisCache | None,
        settings: Settings,
        alerter: TelegramAlerter | None = None,
        async_loop_fn: Any | None = None,
    ) -> None:
        """Initialize NewsPipeline with all dependencies.

        Args:
            rss_fetcher: Optional RSS fetcher for synchronous news
            telegram_reader: Optional Telegram channel reader (async)
            news_fetcher: Optional legacy NewsAPI fetcher (fallback)
            news_impact_analyzer: LLM-based impact analyzer
            sector_ticker_mapper: Maps sectors to tickers
            sentiment_mgr: Sentiment cache manager
            persistence: DB persistence layer
            registry: Instrument registry
            cache: Optional Redis cache for sentiment
            settings: Application settings
            alerter: Optional Telegram alerter
            async_loop_fn: Callable that runs async code (trading_loop._run_async)
        """
        self._rss_fetcher = rss_fetcher
        self._telegram_reader = telegram_reader
        self._news_fetcher = news_fetcher
        self._news_impact_analyzer = news_impact_analyzer
        self._sector_ticker_mapper = sector_ticker_mapper
        self._sentiment_mgr = sentiment_mgr
        self._persistence = persistence
        self._registry = registry
        self._cache = cache
        self._settings = settings
        self._alerter = alerter
        self._async_loop_fn = async_loop_fn

        # Deduplication window: SHA-256 hash of (url + title) with TTL
        self._seen_article_hashes: OrderedDict[str, float] = OrderedDict()

    def run_news_cycle(self) -> None:
        """Fetch news from RSS, Telegram, and legacy NewsAPI; analyze and update sentiment.

        Called by APScheduler every news_cycle_minutes.
        """
        if not self._sentiment_mgr.is_event_driven_active():
            _log.debug("news_cycle_skipped_no_event_driven")
            return

        articles: list[NewsArticle] = []

        # RSS feeds (sync -- runs in APScheduler thread)
        if self._rss_fetcher is not None:
            try:
                rss_articles = self._rss_fetcher.fetch_news()
                articles.extend(rss_articles)
                _log.info("news_rss_fetched", count=len(rss_articles))
            except Exception:
                _log.warning("news_rss_fetch_failed", exc_info=True)

        # Telegram channels (async -- bridge via _async_loop_fn)
        if self._telegram_reader is not None:
            try:
                tg_channels = self._settings.telegram_channels
                if tg_channels and self._async_loop_fn is not None:
                    tg_articles = self._async_loop_fn(
                        self._telegram_reader.fetch_recent_messages(
                            channels=tg_channels,
                            since_minutes=self._settings.news_poll_interval_minutes,
                        )
                    )
                    articles.extend(tg_articles)
                    _log.info("news_telegram_fetched", count=len(tg_articles))
            except Exception:
                _log.warning("news_telegram_fetch_failed", exc_info=True)

        # Legacy NewsAPI fallback (unchanged behavior)
        if not articles and self._news_fetcher is not None:
            now = datetime.now(UTC)
            from_date = now - timedelta(hours=_NEWS_LOOKBACK_HOURS)
            try:
                articles = self._news_fetcher.fetch_news(
                    query=_NEWS_QUERY,
                    from_date=from_date,
                    to_date=now,
                )
                _log.info("news_legacy_fetched", count=len(articles))
            except Exception:
                _log.warning("news_legacy_fetch_failed", exc_info=True)
                return

        # Analyze articles via NewsImpactAnalyzer (single LLM call per article)
        # Large timeout: with rate-limited LLM, batches may take minutes.
        _batch_timeout = 1800
        processed_ok = 0
        processed_fail = 0
        if self._news_impact_analyzer is not None and articles and self._async_loop_fn is not None:
            processed_ok, processed_fail, _ = self._async_loop_fn(
                self._analyze_impact_batch(articles), timeout=_batch_timeout
            )

        # Single summary line for the entire news cycle
        log_fn = _log.info if processed_fail == 0 else _log.warning
        log_fn(
            "news_cycle_complete",
            articles=len(articles),
            processed_ok=processed_ok,
            processed_fail=processed_fail,
        )

    def _is_article_duplicate(self, article: NewsArticle) -> bool:
        """Check if article was already processed within the TTL window.

        Uses SHA-256 of (url + title) as the dedup key. Evicts entries
        older than _ARTICLE_DEDUP_TTL_HOURS and caps at _ARTICLE_DEDUP_MAX_SIZE.
        """
        key = hashlib.sha256(f"{article.url}|{article.title}".encode()).hexdigest()
        now = time.monotonic()

        # Evict expired entries (oldest first, since OrderedDict preserves insertion order)
        cutoff = now - _ARTICLE_DEDUP_TTL_HOURS * 3600
        while self._seen_article_hashes:
            oldest_key, oldest_ts = next(iter(self._seen_article_hashes.items()))
            if oldest_ts < cutoff:
                del self._seen_article_hashes[oldest_key]
            else:
                break

        if key in self._seen_article_hashes:
            return True

        self._seen_article_hashes[key] = now
        # Cap size
        while len(self._seen_article_hashes) > _ARTICLE_DEDUP_MAX_SIZE:
            self._seen_article_hashes.popitem(last=False)

        return False

    async def _analyze_impact_batch(self, articles: list[NewsArticle]) -> tuple[int, int, str]:
        """Analyze all articles via NewsImpactAnalyzer with bounded concurrency.

        Uses an inline circuit breaker: after 5 consecutive LLM failures,
        remaining articles are skipped to avoid wasting minutes on retries.

        Returns:
            (ok_count, fail_count, last_error_type) for summary logging.
        """
        # Deduplicate articles already seen within TTL window (OPS-03)
        unique_articles = [a for a in articles if not self._is_article_duplicate(a)]
        skipped_count = len(articles) - len(unique_articles)
        if skipped_count > 0:
            _log.info(
                "news_articles_deduplicated",
                skipped=skipped_count,
                remaining=len(unique_articles),
            )
        articles = unique_articles
        if not articles:
            return 0, 0, ""

        sem = asyncio.Semaphore(5)
        ok_count = 0
        fail_count = 0
        last_error = ""
        consecutive_failures = 0
        _fail_threshold = 5
        analyzer = self._news_impact_analyzer
        assert analyzer is not None

        async def _process_one(article: NewsArticle) -> bool:
            nonlocal consecutive_failures, last_error
            if consecutive_failures >= _fail_threshold:
                return False
            async with sem:
                try:
                    result = await analyzer.analyze(article)
                    _log.info(
                        "news_article_analyzed",
                        article_title=article.title[:80],
                        article_url=article.url,
                        event_type=result.event_type,
                        sentiment=round(result.sentiment, 3),
                        confidence=round(result.confidence, 3),
                        sectors=[s.sector for s in result.affected_sectors],
                        direct_tickers=result.direct_tickers,
                    )
                    # Fire-and-forget news article persistence (PERSIST-03)
                    await self._persistence.persist_news_article_async(article, result)
                    await self._apply_impact_result(result)
                    consecutive_failures = 0
                    return True
                except Exception as exc:
                    consecutive_failures += 1
                    last_error = type(exc).__name__
                    _log.debug(
                        "news_article_analysis_failed",
                        article_title=article.title[:80],
                        error_type=last_error,
                        error=str(exc)[:200],
                    )
                    if consecutive_failures == _fail_threshold:
                        _log.warning(
                            "news_processing_circuit_opened",
                            error=last_error,
                            consecutive_failures=consecutive_failures,
                        )
                    return False

        results = await asyncio.gather(*[_process_one(a) for a in articles])
        for success in results:
            if success:
                ok_count += 1
            else:
                fail_count += 1
        return ok_count, fail_count, last_error

    async def _apply_impact_result(self, result: NewsImpactResult) -> None:
        """Apply NewsImpactResult to per-ticker sentiment cache.

        Must be async because it is called from _process_one which runs on
        _async_loop. Using sync _run_async / _persist_to_db from within
        _async_loop would deadlock (submit + block on the same loop).
        """
        active_segments = self._sentiment_mgr.collect_active_segments()
        mapper = self._sector_ticker_mapper
        if mapper is None:
            return

        # Build ticker -> score mapping from sectors
        ticker_scores: dict[str, float] = {}
        for sector_impact in result.affected_sectors:
            tickers = mapper.map_sectors([sector_impact.sector])
            score = sector_impact.magnitude * sector_impact.direction * result.sentiment
            for ticker in tickers:
                # Take the strongest impact if ticker appears in multiple sectors
                if ticker not in ticker_scores or abs(score) > abs(ticker_scores[ticker]):
                    ticker_scores[ticker] = score

        # Direct tickers get the raw sentiment * confidence
        for ticker in result.direct_tickers:
            direct_score = result.sentiment * result.confidence
            if ticker not in ticker_scores or abs(direct_score) > abs(ticker_scores[ticker]):
                ticker_scores[ticker] = direct_score

        # Update cache for all active segments containing these tickers
        redis_updates: list[tuple[str, str, float]] = []
        for seg_id in active_segments:
            seg_tickers = self._sentiment_mgr.get_segment_tickers(seg_id)
            for ticker in seg_tickers:
                if ticker in ticker_scores:
                    # Read existing decayed sentiment (with lock held internally)
                    existing = self._sentiment_mgr.read_decayed_sentiment(seg_id, ticker)
                    new_score = existing * 0.7 + ticker_scores[ticker] * 0.3
                    # Update cache (with lock held internally)
                    self._sentiment_mgr.update_sentiment(seg_id, ticker, new_score)
                    redis_updates.append((seg_id, ticker, new_score))

        # Fire-and-forget sentiment persistence (PERSIST-04) — async path
        await self._persist_sentiment_scores_async(
            ticker_scores, active_segments, result.confidence
        )

        _log.info(
            "news_impact_applied",
            tickers_updated=len(ticker_scores),
            segments_affected=len(active_segments),
            cache_entries_written=len(redis_updates),
        )

        # Redis write — already on _async_loop, just await directly
        if self._cache is not None:
            for seg_id, ticker, score in redis_updates:
                try:
                    await self._cache.set_sentiment(f"{seg_id}:{ticker}", score)
                except Exception:
                    _log.debug("Failed to write sentiment to Redis cache")

    async def _persist_sentiment_scores_async(
        self,
        ticker_scores: dict[str, float],
        active_segments: list[str],
        confidence: float,
    ) -> None:
        """Async variant of _persist_sentiment_scores for use on _async_loop."""
        if not ticker_scores:
            return
        has_non_ru = any(not s.startswith("ru_") for s in active_segments)
        market_id = "us" if has_non_ru else "moex"
        await self._persistence.persist_sentiment_batch_async(ticker_scores, market_id, confidence)
