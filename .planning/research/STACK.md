# Technology Stack — v10.0 Runtime LLM Trading Agents

**Project:** Finalayze v10.0
**Researched:** 2026-04-14
**Confidence:** HIGH (codebase direct reads + verified web research)

---

## Premise: What Already Exists (Do Not Re-Research)

These capabilities are shipped and verified. v10.0 extends them without rewrites.

| Already Exists | Location |
|----------------|----------|
| `LLMClient` ABC with 5 providers + `FallbackLLMClient` | `src/finalayze/analysis/llm_client.py` |
| `_CachingLLMClient` — SHA-256 LRU cache (1000 entries), rate limiter, retry with backoff | `src/finalayze/analysis/llm_client.py` |
| `parse_structured(prompt, system, response_model)` — typed Pydantic output on all clients | `src/finalayze/analysis/llm_client.py` |
| `NewsAnalyzer`, `EntityExtractor`, `EventClassifier`, `ImpactEstimator`, `NewsImpactAnalyzer` | `src/finalayze/analysis/` |
| `RssNewsFetcher` — feedparser-based, URL dedup, LRU seen-set | `src/finalayze/data/fetchers/rss_fetcher.py` |
| `TelegramChannelReader` — httpx + BeautifulSoup, t.me/s/ scraping, no auth required | `src/finalayze/data/fetchers/telegram_reader.py` |
| `TradingLoop._news_cycle()` — APScheduler interval job, RSS + Telegram + NewsAPI pipeline | `src/finalayze/orchestration/trading_loop.py` |
| `EventDrivenStrategy` — credibility-gated signal from `sentiment_score` | `src/finalayze/strategies/event_driven.py` |
| `SentimentScoreModel` — TimescaleDB hypertable (symbol, market_id, timestamp, news_sentiment, composite_sentiment) | `src/finalayze/core/models.py` |
| `NewsArticleModel` — DB persistence with LLM analysis JSONB | `src/finalayze/core/models.py` |
| `AnomalyDetector` — z-score + threshold anomaly detection with per-metric cooldown | `src/finalayze/monitoring/anomaly_detector.py` |
| `EventBus` — Redis Streams with consumer groups | `src/finalayze/core/events.py` |
| `APScheduler` `BackgroundScheduler` with `ThreadPoolExecutor` | `src/finalayze/orchestration/trading_loop.py` |
| `structlog`, `Prometheus`, Streamlit dashboard | Throughout |
| `feedparser>=6.0.12`, `beautifulsoup4>=4.14.3`, `httpx>=0.28.0` | `pyproject.toml` |
| RSS URLs: RBC, Interfax, TASS, banki.ru, Vedomosti, Kommersant | `config/settings.py:news_rss_urls` |
| Telegram channels: @markettwits, @AK47pfl, @cbrstocks, @investorbiz, @raborynok | `config/settings.py:telegram_channels` |

---

## v10.0 Gap Analysis

| Requirement | Gap | Solution |
|-------------|-----|---------|
| MOEX news sources — real-time RSS | Existing RSS list has 8 feeds but `MOEX ISS /sitenews` feed not included; no dedicated MOEX official exchange news | Add `https://www.moex.com/export/news.aspx?cat=200` (all MOEX news) and `https://www.moex.com/export/news.aspx?cat=202` (listing news) to `news_rss_urls` in config — zero new libraries, pure config change |
| T-Pulse (Tinkoff social feed) | T-Pulse API (`https://www.tinkoff.ru/api/invest-gw/social/v1/post/instrument/{ticker}`) is a public REST endpoint (no auth required, cursor pagination). `tpulse-py` wrapper exists but last committed December 2021 and is unmaintained. | Use `httpx` (already installed) directly against the T-Pulse REST endpoint. No new library needed. Implement thin `TPulseFetcher` class in `data/fetchers/tpulse.py` following the `TelegramChannelReader` pattern (httpx + Pydantic parse + dedup). HIGH RISK: T-Pulse API is unofficial, undocumented, authentication changed in June 2024, and may be blocked or require session cookies. Treat as BEST-EFFORT source with broad exception handling. |
| Portfolio Review Agent — daily LLM analysis | No agent exists. `parse_structured()` is available on all LLM clients. `PortfolioSnapshotModel` exists but is sparsely populated. | New `PortfolioReviewAgent` class in `analysis/portfolio_review.py` (Layer 3). Uses existing `parse_structured()` with a new `PortfolioReviewOutput` Pydantic model. Scheduled via APScheduler daily job in `TradingLoop`. No new libraries. |
| Anomaly Interpreter Agent — LLM explanation | `AnomalyDetector.check()` returns `list[str]` of triggered metric names and fires Telegram alerts. No LLM explanation step. | New `AnomalyInterpreterAgent` in `analysis/anomaly_interpreter.py` (Layer 3). Called from `SandboxMonitorService` as fire-and-forget via `asyncio.create_task()` (stored in a set to prevent GC). Uses `parse_structured()` for structured explanation output. No new libraries. |
| Sentiment persistence for ML features | `SentimentScoreModel` hypertable exists with correct schema (symbol, timestamp, news_sentiment, composite_sentiment). `_persist_sentiment_scores_async()` already writes to it in `TradingLoop`. | Gap is rolling aggregation views for ML feature extraction (1d, 7d, 30d windows). Implement via TimescaleDB `CREATE MATERIALIZED VIEW` with `time_bucket()` — pure SQL migration, no new Python library. |
| Rolling sentiment aggregation for ML | No materialized view or query helper for rolling sentiment features. ML feature pipeline (`technical.py`) has no sentiment features yet. | New Alembic migration (004) adds `sentiment_rolling` continuous aggregate view. New helper `fetch_rolling_sentiment()` in `data/fetchers/` queries the view and returns a DataFrame. This feeds into `technical.py` as new features in a future phase. |
| EventDrivenStrategy on live feed | `EventDrivenStrategy` is already enabled on ru_* segments (15% weight per CLAUDE.md). `_news_cycle()` skips if `not self._any_event_driven_enabled()`. | Gap is latency SLA verification and credibility cap. `_any_event_driven_enabled()` check exists. The 0.7 credibility cap must be added to `EventDrivenStrategy.generate_signal()` (currently uses `confidence = min(1.0, abs(sentiment) * credibility)` — cap the credibility input). Config change + small code edit, no new library. |

---

## Recommended Stack

### Core Technologies — No New Packages Required

| Technology | Version (installed) | Purpose | Why |
|------------|--------------------|---------|----|
| `anthropic` / `openai` / OpenRouter clients | `>=0.42.0` / `>=1.50.0` | LLM backbone for Portfolio Review Agent and Anomaly Interpreter Agent | Already abstracted behind `LLMClient` ABC with `parse_structured()`. New agents call `parse_structured()` — no provider change needed. |
| `pydantic` v2 | `>=2.10.0` | `PortfolioReviewOutput`, `AnomalyExplanationOutput` Pydantic models for structured agent outputs | Already the standard for all typed outputs in the codebase. `parse_structured()` enforces type safety at LLM response parse time. |
| `feedparser` | `>=6.0.12` | RSS ingestion for MOEX ISS news feeds and existing financial news sources | Already installed. Adding MOEX ISS feeds is a config-only change (`news_rss_urls`). |
| `httpx` | `>=0.28.0` | T-Pulse REST API calls (thin `TPulseFetcher`), existing Telegram web scraping | Already installed. `TelegramChannelReader` is the reference pattern for httpx-based news fetching. |
| `beautifulsoup4` | `>=4.14.3` | HTML parsing for t.me/s/ Telegram channel previews | Already installed. No change. |
| `apscheduler` | `>=3.10.4` | Scheduling `PortfolioReviewAgent` daily job, existing `news_cycle` | Already installed. New agents register additional jobs in `TradingLoop.start()`. |
| `SQLAlchemy` 2.0 async | `>=2.0.36` | Writing/reading `sentiment_scores` hypertable, rolling aggregation queries | Already installed. Rolling sentiment queries use `session.execute(text(...))` pattern already used for raw SQL in the codebase. |
| `asyncpg` | `>=0.30.0` | Async PostgreSQL driver for TimescaleDB | Already installed. No change. |
| `redis` | `>=5.2.0` | EventBus for news article events | Already installed. No change. |
| `structlog` | `>=24.4.0` | Structured logging for agent decisions and news pipeline events | Already installed. Bind `agent_name`, `anomaly_type`, `portfolio_value` to log events. |

### New Modules to Create (Zero New Package Dependencies)

| Module | Location | Layer | Responsibility |
|--------|----------|-------|---------------|
| `PortfolioReviewAgent` | `src/finalayze/analysis/portfolio_review.py` | Layer 3 | Accepts `PortfolioState`, calls `LLMClient.parse_structured()` with `PortfolioReviewOutput` model. Advisory only — no trades. Logs to structlog, sends Telegram summary. |
| `AnomalyInterpreterAgent` | `src/finalayze/analysis/anomaly_interpreter.py` | Layer 3 | Accepts `list[str]` of triggered anomaly names + `CycleMetrics`. Calls `parse_structured()` with `AnomalyExplanationOutput`. Fire-and-forget — uses `asyncio.create_task()` with strong reference set. |
| `TPulseFetcher` | `src/finalayze/data/fetchers/tpulse.py` | Layer 2 | Thin httpx wrapper for `https://www.tinkoff.ru/api/invest-gw/social/v1/post/instrument/{ticker}`. Returns `list[NewsArticle]`. Broad exception handling — BEST-EFFORT only. Dedup via LRU set (same pattern as `TelegramChannelReader`). |
| `fetch_rolling_sentiment()` | `src/finalayze/data/fetchers/sentiment_queries.py` | Layer 2 | Queries `sentiment_rolling` continuous aggregate view. Returns `pd.DataFrame` with columns `(symbol, date, sentiment_1d, sentiment_7d, sentiment_30d)`. Used by ML feature pipeline. |
| Alembic migration 004 | `alembic/versions/004_sentiment_rolling.py` | DB | `CREATE MATERIALIZED VIEW sentiment_rolling` with TimescaleDB `time_bucket()` for 1d, 7d, 30d rolling averages. Auto-refresh policy via `add_continuous_aggregate_policy`. |
| `PortfolioReviewOutput` Pydantic model | `src/finalayze/core/schemas.py` | Layer 0 | Typed output for LLM portfolio review: `overall_assessment`, `risk_flags: list[str]`, `recommended_actions: list[str]`, `confidence: float`. |
| `AnomalyExplanationOutput` Pydantic model | `src/finalayze/core/schemas.py` | Layer 0 | Typed output for LLM anomaly explanation: `explanation: str`, `likely_cause: str`, `severity: Literal["low", "medium", "high"]`, `suggested_response: str`. |

---

## T-Pulse Integration: Risk Assessment

T-Pulse is Tinkoff's investor social network. Research findings:

- **API endpoint:** `https://www.tinkoff.ru/api/invest-gw/social/v1/post/instrument/{ticker}?limit=50&appName=invest&platform=web&cursor={cursor}` (confirmed from community sources, 2021-era docs)
- **Authentication:** Historically public (no auth required), cursor-based pagination returning JSON
- **Risk:** The endpoint is undocumented and unofficial. Authentication changed in June 2024 (SMS auth for full access broke). The endpoint may return empty results, require session cookies, or be geo-blocked. The `tpulse-py` library (last commit: Dec 2021) is unmaintained and should NOT be used as a dependency.
- **Confidence:** LOW — verified from artydev.ru blog posts and community GitHub repos, not official T-Bank docs
- **Implementation strategy:** Implement `TPulseFetcher` with broad exception handling, a `is_available()` health check, and disable flag in settings (`tpulse_enabled: bool = False` default). Enable only after manual verification in sandbox environment. Never block the news cycle on T-Pulse failure.

**Decision:** Implement but default-disabled. Cost = ~100 lines of httpx code. Risk of wasting time is low.

---

## TimescaleDB Rolling Sentiment: Implementation Pattern

TimescaleDB continuous aggregates support rolling 1d/7d/30d windows natively via `time_bucket()` + `avg()`. The `sentiment_scores` table is already a hypertable (TimescaleDB extension applied at migration 002).

Rolling aggregation approach — SQL-only, no new Python library:

```sql
-- Migration 004: sentiment rolling continuous aggregate
CREATE MATERIALIZED VIEW sentiment_rolling
WITH (timescaledb.continuous) AS
SELECT
    symbol,
    market_id,
    time_bucket('1 day', timestamp) AS bucket,
    avg(news_sentiment) FILTER (WHERE timestamp >= time_bucket('1 day', timestamp) - INTERVAL '1 day')  AS sentiment_1d,
    avg(news_sentiment) FILTER (WHERE timestamp >= time_bucket('1 day', timestamp) - INTERVAL '7 days') AS sentiment_7d,
    avg(news_sentiment) FILTER (WHERE timestamp >= time_bucket('1 day', timestamp) - INTERVAL '30 days') AS sentiment_30d,
    count(*) AS article_count_7d
FROM sentiment_scores
GROUP BY symbol, market_id, time_bucket('1 day', timestamp)
WITH NO DATA;

SELECT add_continuous_aggregate_policy('sentiment_rolling',
    start_offset => INTERVAL '35 days',
    end_offset   => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour');
```

The Python query helper reads from this view using the existing `session.execute(text(...))` pattern. Window functions in continuous aggregates require `timescaledb.enable_cagg_window_functions = 'on'` (set per-session or in postgresql.conf).

**Alternative considered:** Python-side rolling with `pandas.rolling()` on raw data pulled from DB. Rejected because: pulls all raw rows from DB every ML training run (expensive for 730 days × 20 symbols), re-computes on every call, loses TimescaleDB's incremental refresh optimization.

---

## Fire-and-Forget Async Pattern: Implementation Note

`AnomalyInterpreterAgent` runs as fire-and-forget — anomaly detection happens in APScheduler thread, LLM call is async. The correct pattern for Python 3.12+ (prevents silent task GC):

```python
# In SandboxMonitorService or AnomalyDetector callback:
_background_tasks: set[asyncio.Task] = set()

def _fire_and_forget(coro: Coroutine) -> None:
    """Schedule async coroutine without blocking; keep strong reference."""
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
```

This is the documented fix for Python 3.12+ task GC silently killing fire-and-forget tasks. The pattern is used in `TradingLoop._persist_sentiment_scores()` (fire-and-forget DB writes) — extend the same set to cover agent invocations.

---

## MOEX News Sources: Verified RSS Feed Catalog

Based on research of currently active feeds (confidence: MEDIUM — verified URLs via web research, availability subject to change):

| Source | URL | Category | Why |
|--------|-----|----------|-----|
| MOEX All News | `https://www.moex.com/export/news.aspx?cat=200` | Exchange official | Listing decisions, trading halts, circuit breaker activations — high-impact for MOEX trading |
| MOEX Listing News | `https://www.moex.com/export/news.aspx?cat=202` | Exchange official | Symbol additions/removals, delistings — critical for instrument registry |
| RBC Finance | `https://rssexport.rbc.ru/rbcnews/news/30/full.rss` | Business media | Already in config |
| Interfax | `https://www.interfax.ru/rss.asp` | Business media | Already in config |
| TASS | `https://tass.com/rss/v2.xml` | State news agency | Already in config |
| Vedomosti | `https://www.vedomosti.ru/rss/news` | Business media | Already in config |
| Kommersant | `https://www.kommersant.ru/RSS/news.xml` | Business media | Already in config |

**New additions for v10.0:** MOEX cat=200 and cat=202 feeds. These are official exchange feeds and much more reliable than social sources for trading-relevant events.

---

## Portfolio Review Agent: Design Constraints

Expert validation (v10.0 planning) rejected Pre-Trade Reasoning Agent — LLM modifiers in the sizing pipeline break determinism, calibration, and backtestability. The Portfolio Review Agent is **advisory only**:

- Input: `PortfolioState` snapshot (positions, equity, drawdown, recent signals)
- Output: `PortfolioReviewOutput` Pydantic model — no code paths that modify strategy or risk parameters
- Delivery: Telegram daily summary message only (human reads it, no automated action)
- Scheduling: APScheduler cron job at `daily_reset_hour_utc` (same slot as existing daily reset)
- LLM call: `parse_structured()` with 2000-token max, Russian-language prompt, OpenRouter free model

The advisory constraint means the agent is safe to add incrementally without breaking backtestability guarantees.

---

## Alternatives Considered

| Recommended | Alternative | Why Not |
|-------------|-------------|---------|
| `httpx` for T-Pulse (existing) | `tpulse-py` library (pip) | Last commit Dec 2021, unmaintained, adds a fragile dependency for an unofficial API. Direct httpx is 50 lines and fully under our control. |
| TimescaleDB continuous aggregate (SQL migration) | Python `pandas.rolling()` on raw DB data | Pulls all raw rows per ML training run — expensive at 730 days × 20 symbols. TimescaleDB incremental refresh is O(new data only). |
| `asyncio.create_task()` with strong reference set | `asyncio.ensure_future()` or `threading.Thread` | `ensure_future` deprecated in 3.10+. Threading loses the async context of the LLM client. Strong-reference set pattern is documented Python 3.12+ fix for fire-and-forget task GC. |
| New `PortfolioReviewAgent` in `analysis/` (Layer 3) | Pydantic AI framework | Pydantic AI is a full agent framework (16K GitHub stars, stable 1.x since late 2025). Overkill for two advisory agents — adds external dependency and new abstraction layer over existing `parse_structured()` which already handles structured output idiomatically. |
| MOEX ISS `/sitenews` via `feedparser` (RSS) | `aiomoex` library for `/sitenews` endpoint | `aiomoex` (v2.2.0, May 2025) focuses only on price/OHLCV data — no news functions. The MOEX RSS feed at `export/news.aspx` is the correct, supported news channel. `feedparser` already handles it. |
| Telethon (MTProto) for Telegram | Current httpx + BeautifulSoup scraping of t.me/s/ | Telethon 1.42.0 (late 2025) is the better long-term choice for authenticated reading of private channels. However, current `TelegramChannelReader` works for all configured public channels without credentials. Telethon requires phone number registration (credentials management overhead). Defer to a future phase if private channels are needed. |
| Default-disabled T-Pulse (`tpulse_enabled: bool = False`) | Skip T-Pulse entirely | Costs ~100 lines. If the API still works, T-Pulse posts from retail investors provide social sentiment signal unavailable in mainstream RSS. Low implementation cost justifies opportunistic inclusion. |

---

## What NOT to Add

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| `tpulse-py` (PyPI) | Unmaintained since Dec 2021; unofficial API. Adds fragile third-party dependency for undocumented endpoint | Direct `httpx` implementation in `TPulseFetcher` |
| `pydantic-ai` framework | Overkill for 2 advisory agents; adds abstraction over existing `parse_structured()` | Existing `LLMClient.parse_structured()` |
| `aiomoex` | Only OHLCV data, no news endpoint | MOEX ISS RSS feed via `feedparser` |
| `Telethon` (MTProto) | Requires phone number registration, session file management; current httpx scraping is sufficient for public channels | Keep existing `TelegramChannelReader`; defer Telethon if private channels needed |
| Separate news microservice / Celery workers | Architecture overkill; APScheduler `news_cycle` already handles polling reliably | Extend `TradingLoop._news_cycle()` with T-Pulse source |
| PostgreSQL window functions on raw `sentiment_scores` table | Expensive full-table scan per ML training run | TimescaleDB continuous aggregate materialized view |
| OpenAI Structured Outputs beta (`anthropic-beta: structured-outputs-2025-11-13`) | Existing `parse_structured()` uses JSON mode + Pydantic validation which already works reliably across all 5 providers | Existing `_CachingLLMClient.parse_structured()` implementation |
| LLM modifiers in sizing pipeline | Unanimously rejected by expert panel — breaks backtestability, determinism, and calibration | Portfolio Review Agent is advisory-only (Telegram output, no strategy modification) |

---

## Installation

**No new packages.** Zero additions to `pyproject.toml`.

Verify environment:

```bash
# All imports must succeed:
uv run python -c "from finalayze.analysis.llm_client import LLMClient; print('LLMClient OK')"
uv run python -c "from finalayze.data.fetchers.rss_fetcher import RssNewsFetcher; print('RSS OK')"
uv run python -c "from finalayze.core.models import SentimentScoreModel; print('SentimentScore schema OK')"
uv run python -c "import httpx; print('httpx OK')"
uv run python -c "import feedparser; print('feedparser OK')"

# Verify MOEX news feeds are reachable:
curl -s "https://www.moex.com/export/news.aspx?cat=200" | head -5
```

---

## Version Compatibility

| Package | Installed | Feature Used | v10.0 Constraint |
|---------|-----------|-------------|-----------------|
| `feedparser` | `>=6.0.12` | MOEX ISS RSS feeds (new cat=200, cat=202) | No change; existing `RssNewsFetcher` handles all RSS feeds identically |
| `httpx` | `>=0.28.0` | `TPulseFetcher` REST calls | Existing sync client pattern from `TelegramChannelReader`; no version bump needed |
| `beautifulsoup4` | `>=4.14.3` | Existing Telegram scraping | No change |
| `anthropic` | `>=0.42.0` | `PortfolioReviewAgent`, `AnomalyInterpreterAgent` via `parse_structured()` | No change; `parse_structured()` already works with Anthropic provider |
| `openai` | `>=1.50.0` | Same agents via OpenRouter / OpenAI provider | No change |
| `SQLAlchemy` | `>=2.0.36` | `fetch_rolling_sentiment()` raw SQL query against new continuous aggregate | Use `session.execute(text(...))` — existing pattern already in codebase |
| `asyncpg` | `>=0.30.0` | DB driver for TimescaleDB continuous aggregate | No change |
| `apscheduler` | `>=3.10.4` | Portfolio review daily job | Extend `TradingLoop.start()` with new `add_job()` call — no scheduler version change |
| `pydantic` v2 | `>=2.10.0` | `PortfolioReviewOutput`, `AnomalyExplanationOutput` | Standard `BaseModel` with `model_config = ConfigDict(frozen=True)` — no version constraint |

---

## Sources

- `src/finalayze/analysis/llm_client.py` — `parse_structured()` signature and `_CachingLLMClient` confirmed (HIGH confidence, direct read)
- `src/finalayze/data/fetchers/rss_fetcher.py` — `RssNewsFetcher` pattern confirmed (HIGH confidence, direct read)
- `src/finalayze/data/fetchers/telegram_reader.py` — httpx + BeautifulSoup pattern confirmed (HIGH confidence, direct read)
- `src/finalayze/orchestration/trading_loop.py` — `_news_cycle()` structure, APScheduler integration, fire-and-forget sentiment persistence confirmed (HIGH confidence, direct read)
- `src/finalayze/core/models.py` — `SentimentScoreModel` hypertable schema confirmed (HIGH confidence, direct read)
- `src/finalayze/monitoring/anomaly_detector.py` — `check()` returns `list[str]` of triggered metrics confirmed (HIGH confidence, direct read)
- `config/settings.py` — existing RSS URLs and telegram channels confirmed (HIGH confidence, direct read)
- `pyproject.toml` — all installed library versions confirmed (HIGH confidence, direct read)
- `https://www.moex.com/s355` — MOEX official RSS feed catalog, cat=200 and cat=202 URLs verified (MEDIUM confidence, WebFetch)
- artydev.ru/posts/pulse-parser/ — T-Pulse REST API endpoint structure `https://www.tinkoff.ru/api/invest-gw/social/v1/post/instrument/{ticker}` (LOW confidence — community blog, unofficial, authentication changes noted in June 2024)
- github.com/meanother/tpulse-py — last commit Dec 2021, confirms unmaintained status (MEDIUM confidence, direct GitHub read)
- Python asyncio docs + mkennedy.codes fire-and-forget article — `asyncio.create_task()` + strong reference set pattern for Python 3.12+ (HIGH confidence, official docs + verified community source)
- TimescaleDB continuous aggregates docs — `time_bucket()` + `add_continuous_aggregate_policy()` pattern (MEDIUM confidence, WebSearch-verified official docs)
- WebSearch (Anthropic structured outputs, Nov 2025) — confirmed `parse_structured()` approach is equivalent to new beta API; existing implementation adequate (MEDIUM confidence)

---

*Stack research for: Finalayze v10.0 Runtime LLM Trading Agents*
*Researched: 2026-04-14*
