# Phase 7: News Pipeline and Go-Live - Research

**Researched:** 2026-03-15
**Domain:** Russian news ingestion (RSS + Telegram), LLM sentiment analysis, event_driven strategy enablement, real MOEX trading
**Confidence:** HIGH

## Summary

Phase 7 builds the Russian news ingestion pipeline (RSS feeds from RBC, Interfax, TASS + Telegram channel reader via Telethon), wires LLM-based sentiment analysis into the existing `event_driven` strategy, enables it on all `ru_*` segments, and deploys the first real MOEX trades on a 500K RUB account.

The codebase already has substantial infrastructure for this phase: `NewsAnalyzer` with Russian prompts, `EventClassifier`, `ImpactEstimator`, `EventDrivenStrategy` with sanctions proximity scoring, and the `TradingLoop._news_cycle()` method that fetches articles, analyzes sentiment, and updates the per-segment sentiment cache consumed by `StrategyCombiner`. The current news source is `NewsApiFetcher` (English-only, US-focused). This phase replaces/supplements it with Russian-language RSS and Telegram sources.

**Primary recommendation:** Build an `RssNewsFetcher` and `TelegramChannelReader` that produce `NewsArticle` objects, integrate them into the existing `_news_cycle()` flow in `TradingLoop`, enable `event_driven` in all `ru_*` YAML presets with weight 0.15, then deploy to real trading after sandbox validation passes.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Use feedparser library for RSS fetching (mature, handles RSS/Atom)
- Target 3 sources: RBC, Interfax, TASS (major Russian business wires)
- Use Telethon (async) for Telegram channel reading (NWS-03)
- Poll news sources every 5 minutes (balance freshness vs rate limits)
- LLM entity extraction for mapping news articles to MOEX tickers (more accurate than keyword dict)
- Individual article analysis (reuse existing NewsAnalyzer.analyze() pattern)
- Weighted average in StrategyCombiner for combining news sentiment with technical signals
- Use OpenRouter client with free model for cost efficiency (user preference)
- Starting capital: 500K RUB (matches PROJECT.md constraint)
- Require passing sandbox validation report before go-live (re-confirm AUT-04)
- Same risk limits as sandbox (proven, no reason to change)
- Kill switch: existing circuit breaker + add Telegram /stop write command
- Enable event_driven on ALL ru_* segments (ru_blue_chips, ru_energy, ru_finance, ru_tech)
- Initial weight: 0.15 (15%) -- meaningful but technical signals still dominate
- Event types: geopolitical, sanctions, cbr_rate, commodity_price, earnings (already in presets)
- Backtest validation required before enabling (mandatory per WORKFLOW.md)

### Claude's Discretion
- RSS URL selection within the 3 confirmed sources (validate at implementation)
- Telegram channel selection for financial sentiment
- LLM prompt design for entity extraction (extend existing sentiment_ru.txt or separate prompt)
- Deduplication strategy for news articles across sources

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| NWS-01 | Russian news RSS feed reader (RBC, Interfax, TASS) | feedparser 6.0.11, verified RSS URLs, RssNewsFetcher pattern from existing NewsApiFetcher |
| NWS-02 | LLM analysis of Russian news via existing NewsAnalyzer + Russian prompts | Existing NewsAnalyzer.analyze() + sentiment_ru.txt prompt; add entity extraction prompt for ticker mapping |
| NWS-03 | Telegram channel reading for financial sentiment (Telethon) | Telethon 1.42.0, async iter_messages(), requires api_id/api_hash credentials |
| NWS-04 | News-driven signal generation (event impact -> trading decision) | Existing EventClassifier + ImpactEstimator + EventDrivenStrategy pipeline already wired |
| NWS-05 | event_driven strategy enabled on MOEX segments | Update 4 ru_* YAML presets: enabled=true, weight=0.15; backtest validation required |
| AUT-05 | Real money deployment on small account (first real MOEX trades) | Switch mode from SANDBOX to REAL, set real_confirmed=true, 500K RUB capital |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| feedparser | 6.0.11 | RSS/Atom feed parsing | De facto standard for Python RSS parsing; handles encoding, dates, bozo detection |
| Telethon | 1.42.0 | Telegram MTProto client | Async-native, reads channel messages without bot API limitations, Python 3.9+ |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| openai (via OpenRouter) | already installed | LLM API calls | Entity extraction + sentiment analysis via OpenRouterClient |
| httpx | already installed | HTTP client for RSS fetch | feedparser.parse() accepts URL directly, but httpx gives timeout/retry control |
| APScheduler | already installed | News polling scheduler | Already wired in TradingLoop for news_cycle job |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| feedparser | aiohttp + xml.etree | feedparser handles edge cases (encoding, malformed feeds, date normalization) |
| Telethon | pyrogram | Telethon is more mature, better async support, user decision locked |
| Telethon | python-telegram-bot | Bot API cannot read arbitrary public channels; MTProto needed |

**Installation:**
```bash
uv add feedparser telethon
```

## Architecture Patterns

### Recommended Project Structure
```
src/finalayze/
  data/fetchers/
    rss_fetcher.py          # RssNewsFetcher (NWS-01)
    telegram_reader.py      # TelegramChannelReader (NWS-03)
  analysis/
    entity_extractor.py     # LLM-based ticker extraction (NWS-02)
    prompts/
      entity_extraction.txt # Prompt for MOEX ticker extraction from Russian text
  strategies/presets/
    ru_blue_chips.yaml      # event_driven: enabled: true, weight: 0.15 (NWS-05)
    ru_energy.yaml          # same
    ru_finance.yaml         # same
    ru_tech.yaml            # same
  core/
    trading_loop.py         # Wire new fetchers into _news_cycle() (NWS-04)
    telegram_bot.py         # Add /stop command (AUT-05)
config/
  settings.py              # Add news_rss_urls, telegram_api_id, telegram_api_hash, telegram_channels
```

### Pattern 1: News Fetcher Producing NewsArticle
**What:** RSS and Telegram readers produce `NewsArticle` objects that feed into the existing `_process_news_article()` pipeline.
**When to use:** All news ingestion paths.
**Example:**
```python
# Source: existing NewsApiFetcher pattern in data/fetchers/newsapi.py
class RssNewsFetcher:
    """Fetches Russian news from RSS feeds, returns NewsArticle objects."""

    def __init__(self, feed_urls: list[str], rate_limiter: RateLimiter | None = None) -> None:
        self._feed_urls = feed_urls
        self._rate_limiter = rate_limiter
        self._seen_ids: set[str] = set()  # deduplication by URL hash

    def fetch_news(self) -> list[NewsArticle]:
        """Fetch new articles from all configured RSS feeds."""
        articles: list[NewsArticle] = []
        for url in self._feed_urls:
            feed = feedparser.parse(url)
            for entry in feed.entries:
                article = self._parse_entry(entry, source=url)
                if article and article.url not in self._seen_ids:
                    self._seen_ids.add(article.url)
                    articles.append(article)
        return articles
```

### Pattern 2: Telegram Channel Reader (Async)
**What:** Telethon client reads messages from configured financial Telegram channels.
**When to use:** NWS-03 implementation.
**Example:**
```python
# Source: Telethon 1.42.0 docs
from telethon import TelegramClient

class TelegramChannelReader:
    def __init__(self, api_id: int, api_hash: str, session_name: str = "finalayze") -> None:
        self._client = TelegramClient(session_name, api_id, api_hash)
        self._channels: list[str] = []

    async def fetch_recent_messages(
        self, channels: list[str], since_minutes: int = 5
    ) -> list[NewsArticle]:
        cutoff = datetime.now(UTC) - timedelta(minutes=since_minutes)
        articles: list[NewsArticle] = []
        async with self._client:
            for channel in channels:
                async for msg in self._client.iter_messages(channel, offset_date=cutoff, reverse=True):
                    if msg.text:
                        articles.append(self._to_article(msg, channel))
        return articles
```

### Pattern 3: Entity Extraction via LLM
**What:** Separate LLM prompt extracts MOEX tickers from Russian news text.
**When to use:** After sentiment analysis, before ImpactEstimator routing.
**Example:**
```python
# New prompt: entity_extraction.txt
# Instructs LLM to return JSON with {"tickers": ["SBER", "GAZP"], "scope": "russia"}
class EntityExtractor:
    def __init__(self, llm_client: LLMClient) -> None:
        self._llm = llm_client

    async def extract(self, article: NewsArticle) -> list[str]:
        """Extract MOEX ticker symbols from article text."""
        system = self._load_prompt()
        raw = await self._llm.complete(f"Title: {article.title}\n\nContent: {article.content}", system)
        data = json.loads(raw)
        return data.get("tickers", [])
```

### Pattern 4: Modified News Cycle
**What:** `TradingLoop._news_cycle()` calls both RSS and Telegram fetchers instead of just NewsApiFetcher.
**When to use:** Integration point in trading_loop.py.
**Example:**
```python
def _news_cycle(self) -> None:
    articles: list[NewsArticle] = []
    # RSS feeds
    try:
        articles.extend(self._rss_fetcher.fetch_news())
    except Exception:
        _log.warning("RSS fetch failed", exc_info=True)
    # Telegram channels
    try:
        tg_articles = self._run_async(self._telegram_reader.fetch_recent_messages(...))
        articles.extend(tg_articles)
    except Exception:
        _log.warning("Telegram fetch failed", exc_info=True)
    # Existing processing pipeline
    for article in articles:
        try:
            self._process_news_article(article)
        except Exception:
            _log.exception("Error processing article %s", article.id)
```

### Anti-Patterns to Avoid
- **Blocking RSS fetch in async context:** feedparser.parse() is synchronous; run it in the APScheduler thread (already sync), do NOT wrap in asyncio.to_thread unless needed.
- **Unbounded seen_ids set:** Cap deduplication set size (e.g., LRU or time-based eviction) to prevent memory leak in long-running process.
- **Hardcoded RSS URLs:** Put them in Settings so they can be changed via env vars without code changes.
- **Telegram session file conflicts:** Use a unique session name and store session file in a stable path (not /tmp).

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| RSS parsing | Custom XML parser | feedparser | Handles encoding detection, malformed feeds, date formats, bozo flag |
| Telegram channel reading | Bot API polling | Telethon MTProto | Bot API cannot read arbitrary public channels; needs user/bot membership |
| News deduplication | Complex content hashing | URL-based dedup with bounded set | RSS entries have unique URLs; content hashing is brittle with minor edits |
| Ticker extraction from Russian text | Regex/keyword dictionary | LLM entity extraction | Russian company names have many forms (Sberbank, Сбербанк, СБЕР, SBER) |
| Sentiment analysis | Custom NLP model | Existing NewsAnalyzer + OpenRouter | Already built and tested; free model via OpenRouter is cost-effective |

**Key insight:** The existing analysis pipeline (NewsAnalyzer -> EventClassifier -> ImpactEstimator -> EventDrivenStrategy) is already complete and tested. This phase only needs to wire new data sources (RSS + Telegram) into the existing pipeline and enable the strategy.

## Common Pitfalls

### Pitfall 1: RSS Feed Encoding Issues
**What goes wrong:** Russian RSS feeds use various encodings (UTF-8, Windows-1251, KOI8-R); naive parsing produces garbled text.
**Why it happens:** Feed headers may declare wrong encoding, or content may mix encodings.
**How to avoid:** feedparser handles encoding detection automatically via its sanitization layer. Verify output contains valid Cyrillic after parsing.
**Warning signs:** `feed.bozo` flag is True, garbled characters in article titles.

### Pitfall 2: Telethon Session Authentication
**What goes wrong:** Telethon requires interactive phone number verification on first run; cannot be done headlessly.
**Why it happens:** Telegram MTProto requires phone number + verification code for user accounts.
**How to avoid:** Run initial authentication manually once, then persist the `.session` file. For deployment, use a bot account if it has been added to the channels, or use a dedicated Telegram user account.
**Warning signs:** `SessionPasswordNeededError` or interactive prompts in production.

### Pitfall 3: RSS Feed URL Stability
**What goes wrong:** Russian news sites may change RSS feed URLs, break feeds, or rate-limit.
**Why it happens:** RSS is not a primary delivery channel for most Russian media.
**How to avoid:** Validate URLs at implementation time. Add health monitoring (log when feeds return 0 articles or bozo=True). Graceful degradation -- missing one feed should not block the cycle.
**Warning signs:** Persistent empty feed responses, HTTP 403/429 errors.

### Pitfall 4: LLM Cost Runaway
**What goes wrong:** Polling every 5 minutes across 3 RSS feeds + Telegram channels generates many LLM calls.
**Why it happens:** Each article requires sentiment analysis + entity extraction = 2+ LLM calls.
**How to avoid:** OpenRouter free model (already decided). Deduplication prevents re-analyzing seen articles. Rate limit LLM calls per cycle (e.g., max 10 articles per 5-min cycle). LRU cache in `_CachingLLMClient` already prevents duplicate prompts.
**Warning signs:** OpenRouter rate limit errors, increasing latency in news cycle.

### Pitfall 5: Sentiment Cache Stale Data
**What goes wrong:** Old sentiment scores influence signals long after news is stale.
**Why it happens:** Exponential moving average (`existing * 0.7 + new * 0.3`) decays slowly. If no new articles arrive, old sentiment persists indefinitely.
**How to avoid:** Add sentiment decay -- reduce cached scores toward 0.0 over time (e.g., halve every 24 hours without new articles). Or clear cache at daily reset.
**Warning signs:** Strong sentiment scores persisting for days without new confirming news.

### Pitfall 6: Real Trading Safety
**What goes wrong:** Bugs in news pipeline cause erroneous signals that execute real trades.
**Why it happens:** Moving from sandbox to real mode with a new data source is inherently risky.
**How to avoid:** Require sandbox validation report (AUT-04 already complete). Add /stop command to kill switch. Start with circuit breaker limits proven in sandbox. Monitor first 24h manually.
**Warning signs:** Unusual trade frequency, conflicting signals, large position sizes.

## Code Examples

### Verified: Existing News Processing Pipeline
```python
# Source: src/finalayze/core/trading_loop.py lines 671-724
def _news_cycle(self) -> None:
    """Fetch latest news, analyze sentiment, update _sentiment_cache."""
    # Current: uses NewsApiFetcher
    # Phase 7: replace/augment with RssNewsFetcher + TelegramChannelReader
    articles = self._news_fetcher.fetch_news(...)
    for article in articles:
        self._process_news_article(article)

def _process_news_article(self, article: NewsArticle) -> None:
    """Analyze article -> classify event -> estimate impact -> update cache."""
    sentiment, event = self._run_async(self._analyze_article(article))
    impacts = self._impact_estimator.estimate(article, event, sentiment, active_segments)
    with self._sentiment_lock:
        for impact in impacts:
            existing = self._sentiment_cache.get(impact.segment_id, 0.0)
            new_score = existing * 0.7 + impact.sentiment * 0.3
            self._sentiment_cache[impact.segment_id] = new_score
```

### Verified: EventDrivenStrategy Signal Generation
```python
# Source: src/finalayze/strategies/event_driven.py lines 103-175
def generate_signal(self, symbol, candles, segment_id,
                    sentiment_score=0.0, ...) -> Signal | None:
    # Uses sanctions proximity scoring for Russian equities
    # Price-move guard prevents trading on already-priced-in news
    # Returns Signal with direction, confidence, features dict
```

### Verified: YAML Preset Structure for event_driven
```yaml
# Source: src/finalayze/strategies/presets/ru_blue_chips.yaml lines 38-43
event_driven:
  enabled: false   # -> change to true
  weight: 0.00     # -> change to 0.15
  params:
    min_sentiment: 0.6
    event_types: [geopolitical, sanctions, cbr_rate, commodity_price, earnings]
```

### Verified: RSS Feed URLs (Validated)
```python
# RBC: CONFIRMED working RSS 2.0 feed with full article content
RSS_RBC = "https://rssexport.rbc.ru/rbcnews/news/30/full.rss"

# Interfax: RSS feed endpoint (confirmed via web check)
RSS_INTERFAX = "https://www.interfax.ru/rss.asp"

# TASS: RSS v2 XML feed (confirmed via multiple sources)
RSS_TASS = "https://tass.com/rss/v2.xml"
```

### Verified: Settings Fields to Add
```python
# Source: config/settings.py pattern
# New fields for Phase 7:
news_rss_urls: list[str] = [
    "https://rssexport.rbc.ru/rbcnews/news/30/full.rss",
    "https://www.interfax.ru/rss.asp",
    "https://tass.com/rss/v2.xml",
]
news_poll_interval_minutes: int = 5
telegram_api_id: int = 0       # FINALAYZE_TELEGRAM_API_ID
telegram_api_hash: str = ""    # FINALAYZE_TELEGRAM_API_HASH
telegram_channels: list[str] = []  # FINALAYZE_TELEGRAM_CHANNELS
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| NewsApiFetcher (EN-only) | RSS + Telegram (RU-native) | Phase 7 | Russian news directly ingested, no translation needed |
| event_driven disabled | event_driven enabled at 0.15 weight | Phase 7 | News sentiment influences MOEX trading signals |
| Sandbox-only MOEX | Real MOEX trading | Phase 7 | System executes live trades on 500K RUB |

**Deprecated/outdated:**
- NewsApiFetcher remains available for US news but is not used for MOEX/Russian news
- feedparser 5.x is outdated; use 6.0.11+

## Open Questions

1. **Telegram Channel Selection**
   - What we know: Need financial/market channels in Russian
   - What's unclear: Which specific channels provide best signal-to-noise ratio
   - Recommendation: Start with well-known channels (e.g., @raborynok, @markettwits, @cbloginvest); make configurable via Settings

2. **Interfax RSS Feed Reliability**
   - What we know: `https://www.interfax.ru/rss.asp` exists and returns content
   - What's unclear: Whether this feed includes economics-only content or all news
   - Recommendation: Validate at implementation; may need `/rss.asp?sec=economics` or similar filter. Fallback to main feed if section-specific unavailable. LOW confidence on exact URL path.

3. **TASS RSS Language**
   - What we know: `https://tass.com/rss/v2.xml` is the English-language feed
   - What's unclear: Whether Russian-language feed is at `tass.ru/rss/v2.xml` or different path
   - Recommendation: Validate both `tass.com` and `tass.ru` at implementation time. Russian-language feed preferred for MOEX context.

4. **Telethon Session Persistence in Docker**
   - What we know: Telethon stores `.session` SQLite file for authentication
   - What's unclear: How to persist session across Docker restarts
   - Recommendation: Mount session file via Docker volume; document initial auth process

5. **Entity Extraction Accuracy**
   - What we know: LLM-based extraction is more flexible than keyword matching
   - What's unclear: Free model accuracy for Russian ticker extraction
   - Recommendation: Build entity extraction prompt, test with sample articles, measure accuracy before relying on it for live trading

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (already configured) |
| Config file | pyproject.toml [tool.pytest.ini_options] |
| Quick run command | `uv run pytest tests/unit/ -x --timeout=30` |
| Full suite command | `uv run pytest tests/ --timeout=120` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| NWS-01 | RSS fetcher parses RBC/Interfax/TASS feeds into NewsArticle | unit | `uv run pytest tests/unit/test_rss_fetcher.py -x` | No -- Wave 0 |
| NWS-01 | RSS fetcher deduplicates articles by URL | unit | `uv run pytest tests/unit/test_rss_fetcher.py::test_deduplication -x` | No -- Wave 0 |
| NWS-02 | LLM entity extraction returns MOEX tickers from Russian text | unit | `uv run pytest tests/unit/test_entity_extractor.py -x` | No -- Wave 0 |
| NWS-02 | NewsAnalyzer.analyze() returns valid SentimentResult for Russian articles | unit | `uv run pytest tests/unit/test_news_analyzer.py -x` | Yes |
| NWS-03 | Telegram reader fetches messages from channels as NewsArticle | unit | `uv run pytest tests/unit/test_telegram_reader.py -x` | No -- Wave 0 |
| NWS-04 | News processing pipeline routes impact to correct segments | unit | `uv run pytest tests/unit/test_impact_estimator.py -x` | Yes |
| NWS-05 | event_driven strategy generates signals from sentiment scores | unit | `uv run pytest tests/unit/test_event_driven_strategy.py -x` | Yes |
| NWS-05 | YAML presets have event_driven enabled with weight 0.15 | unit | `uv run pytest tests/unit/test_event_driven_presets.py -x` | No -- Wave 0 |
| AUT-05 | Settings accept real_confirmed=True with required credentials | unit | `uv run pytest tests/unit/test_settings.py -x` | Partial (exists but needs go-live test) |
| AUT-05 | /stop Telegram command triggers trading halt | unit | `uv run pytest tests/unit/test_telegram_webhook.py -x` | Partial (exists but no /stop test) |

### Sampling Rate
- **Per task commit:** `uv run pytest tests/unit/ -x --timeout=30`
- **Per wave merge:** `uv run pytest tests/ --timeout=120`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/unit/test_rss_fetcher.py` -- covers NWS-01 (RSS parsing, dedup, error handling)
- [ ] `tests/unit/test_entity_extractor.py` -- covers NWS-02 (ticker extraction from Russian text)
- [ ] `tests/unit/test_telegram_reader.py` -- covers NWS-03 (Telethon channel reading)
- [ ] `tests/unit/test_event_driven_presets.py` -- covers NWS-05 (YAML preset validation)
- [ ] Framework install: `uv add feedparser telethon` -- new dependencies

## Sources

### Primary (HIGH confidence)
- Existing codebase: `src/finalayze/analysis/news_analyzer.py`, `event_classifier.py`, `impact_estimator.py`, `strategies/event_driven.py`, `core/trading_loop.py`, `data/fetchers/newsapi.py`
- feedparser 6.0.11 docs: https://feedparser.readthedocs.io/
- Telethon 1.42.0 docs: https://docs.telethon.dev/
- RBC RSS feed verified working: `https://rssexport.rbc.ru/rbcnews/news/30/full.rss`

### Secondary (MEDIUM confidence)
- Interfax RSS: `https://www.interfax.ru/rss.asp` (confirmed endpoint exists, content not fully verified)
- TASS RSS: `https://tass.com/rss/v2.xml` (from multiple aggregator sources; may be English-only)
- Telegram financial channels: community-known channels, need validation

### Tertiary (LOW confidence)
- TASS Russian-language RSS URL (may differ from English `tass.com`)
- Interfax section-specific RSS feeds (economics vs all news)
- Free model accuracy for Russian entity extraction

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - feedparser and Telethon are well-established, versions verified
- Architecture: HIGH - existing pipeline is complete; only new data sources + enablement needed
- Pitfalls: HIGH - common issues well-documented across community; encoding, auth, feed stability
- RSS URLs: MEDIUM - RBC confirmed working; Interfax and TASS need validation at implementation

**Research date:** 2026-03-15
**Valid until:** 2026-04-15 (stable domain; RSS URLs may need re-validation)
