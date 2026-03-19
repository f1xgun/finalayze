---
phase: 07-news-pipeline-and-go-live
plan: 01
subsystem: data, analysis
tags: [rss, feedparser, llm, entity-extraction, moex, news-pipeline]

requires:
  - phase: 06-sandbox-validation
    provides: "Validated sandbox stack and trading loop"
provides:
  - "RssNewsFetcher class for Russian RSS feed ingestion"
  - "EntityExtractor class for LLM-based MOEX ticker extraction"
  - "Settings fields for news RSS and Telegram channel config"
  - "Entity extraction LLM prompt with Russian company mappings"
affects: [07-02, 07-03, trading-loop, news-pipeline]

tech-stack:
  added: [feedparser, telethon, dateutil]
  patterns: [OrderedDict LRU dedup, code-fence stripping, frozenset ticker validation]

key-files:
  created:
    - src/finalayze/data/fetchers/rss_fetcher.py
    - src/finalayze/analysis/entity_extractor.py
    - src/finalayze/analysis/prompts/entity_extraction.txt
    - tests/unit/test_rss_fetcher.py
    - tests/unit/test_entity_extractor.py
  modified:
    - config/settings.py
    - pyproject.toml
    - uv.lock

key-decisions:
  - "feedparser for RSS parsing (well-maintained, handles malformed feeds gracefully)"
  - "OrderedDict-based LRU dedup with configurable MAX_SEEN_SIZE=5000"
  - "29 known MOEX tickers as frozenset for fast validation of LLM output"
  - "Regex-based markdown code fence stripping for robust LLM response parsing"

patterns-established:
  - "RSS fetcher pattern: sync fetch with feedparser, URL-based dedup, NewsArticle output"
  - "Entity extraction pattern: LLM prompt + JSON parse + frozenset validation"

requirements-completed: [NWS-01, NWS-02]

duration: 5min
completed: 2026-03-15
---

# Phase 7 Plan 01: RSS Fetcher & Entity Extractor Summary

**RSS news fetcher with URL dedup and LLM-based MOEX ticker extraction from Russian financial news**

## Performance

- **Duration:** 5 min
- **Started:** 2026-03-15T19:11:31Z
- **Completed:** 2026-03-15T19:16:09Z
- **Tasks:** 2
- **Files modified:** 8

## Accomplishments
- RssNewsFetcher parses RSS feeds from RBC, Interfax, TASS into NewsArticle objects with bounded URL dedup
- EntityExtractor extracts MOEX tickers from Russian news text via LLM with frozenset validation
- Settings extended with news_rss_urls, poll interval, Telegram API credentials
- 14 unit tests covering all behaviors (8 RSS + 6 entity extraction)

## Task Commits

Each task was committed atomically:

1. **Task 1: RSS fetcher + Settings fields + tests** - `67b5064` (feat)
2. **Task 2: EntityExtractor + prompt + tests** - `f243e08` (feat)

_Note: TDD tasks - tests written first (RED), then implementation (GREEN)_

## Files Created/Modified
- `src/finalayze/data/fetchers/rss_fetcher.py` - RSS feed fetcher with dedup (Layer 2)
- `src/finalayze/analysis/entity_extractor.py` - LLM MOEX ticker extractor (Layer 3)
- `src/finalayze/analysis/prompts/entity_extraction.txt` - Extraction prompt with company mappings
- `config/settings.py` - Added news_rss_urls, news_poll_interval_minutes, telegram_api_id/hash/channels
- `tests/unit/test_rss_fetcher.py` - 8 tests for RSS fetcher
- `tests/unit/test_entity_extractor.py` - 6 tests for entity extractor
- `pyproject.toml` - Added feedparser and telethon dependencies
- `uv.lock` - Updated lockfile

## Decisions Made
- feedparser for RSS parsing (handles malformed feeds with bozo flag gracefully)
- OrderedDict-based LRU dedup bounded at 5000 entries (FIFO eviction)
- 29 known MOEX tickers as frozenset for O(1) validation of LLM output
- Regex-based markdown code fence stripping for robust LLM response parsing
- dateutil.parser as fallback for non-standard published timestamps

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- RSS fetcher and entity extractor ready for TradingLoop integration (Plan 02)
- Settings has all news pipeline config fields for downstream use
- Entity extraction prompt covers 29 major MOEX tickers

---
*Phase: 07-news-pipeline-and-go-live*
*Completed: 2026-03-15*
