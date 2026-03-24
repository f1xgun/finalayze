---
phase: 27-intelligent-news-impact-analysis
plan: 01
subsystem: analysis
tags: [llm, news, sentiment, moex, sector-mapping, pydantic]

# Dependency graph
requires:
  - phase: existing
    provides: LLMClient ABC, EventType enum, _VALID_TICKERS, NewsArticle schema
provides:
  - NewsImpactAnalyzer -- single LLM call per article returning event_type + sentiment + sectors + tickers
  - SectorTickerMapper -- static 14-sector MOEX taxonomy to ticker mapping
  - Russian and English prompt templates for sector-aware impact analysis
affects: [27-02, event_driven strategy, news pipeline]

# Tech tracking
tech-stack:
  added: []
  patterns: [single-LLM-call analysis, sector-aware impact, static sector registry]

key-files:
  created:
    - src/finalayze/analysis/news_impact_analyzer.py
    - src/finalayze/analysis/sector_ticker_mapper.py
    - src/finalayze/analysis/prompts/analyze_impact_ru.txt
    - src/finalayze/analysis/prompts/analyze_impact_en.txt
    - tests/unit/test_news_impact_analyzer.py
    - tests/unit/test_sector_ticker_mapper.py
  modified: []

key-decisions:
  - "Reuse _VALID_TICKERS from entity_extractor and _PROMPT_TO_EVENT_TYPE from event_classifier rather than duplicating"
  - "Copy _CODE_FENCE_RE pattern locally instead of importing private symbol"
  - "Direction clamping: >= 0 maps to +1, < 0 maps to -1 (simple sign-based)"

patterns-established:
  - "Single LLM call pattern: one analyze() call returns all structured fields"
  - "Sector taxonomy: 14 MOEX sectors as canonical vocabulary for LLM prompts and static mapping"

requirements-completed: [NEWS-05, NEWS-06, NEWS-09]

# Metrics
duration: 3min
completed: 2026-03-24
---

# Phase 27 Plan 01: News Impact Analyzer Summary

**Single-LLM-call NewsImpactAnalyzer with 14-sector MOEX taxonomy and SectorTickerMapper for static sector-to-ticker resolution**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-24T08:55:49Z
- **Completed:** 2026-03-24T08:59:06Z
- **Tasks:** 1
- **Files created:** 6

## Accomplishments
- NewsImpactAnalyzer replaces 2-call EntityExtractor+CombinedAnalyzer with single LLM invocation (NEWS-05, NEWS-09)
- SectorTickerMapper provides static 14-sector MOEX taxonomy mapping to concrete tickers (NEWS-06)
- Circuit breaker with 5-failure threshold and 5-minute cooldown for LLM resilience
- 26 unit tests covering parsing, clamping, fallback, circuit breaker, language selection, ticker filtering

## Task Commits

Each task was committed atomically:

1. **Task 1: Create SectorTickerMapper and NewsImpactAnalyzer with tests** - `bdc06e8` (feat)

## Files Created/Modified
- `src/finalayze/analysis/news_impact_analyzer.py` - Unified LLM analyzer with NewsImpactResult, SectorImpactDetail models
- `src/finalayze/analysis/sector_ticker_mapper.py` - Static SECTOR_TICKERS registry and SectorTickerMapper class
- `src/finalayze/analysis/prompts/analyze_impact_ru.txt` - Russian prompt for sector-aware impact analysis
- `src/finalayze/analysis/prompts/analyze_impact_en.txt` - English prompt for sector-aware impact analysis
- `tests/unit/test_news_impact_analyzer.py` - 16 tests for NewsImpactAnalyzer
- `tests/unit/test_sector_ticker_mapper.py` - 10 tests for SectorTickerMapper

## Decisions Made
- Reused `_VALID_TICKERS` from `entity_extractor` and `_PROMPT_TO_EVENT_TYPE` from `event_classifier` to avoid duplication
- Copied `_CODE_FENCE_RE` regex pattern locally rather than importing private symbol cross-module
- Direction clamping uses simple sign-based logic: >= 0 maps to +1, < 0 maps to -1
- Prompt includes `sector_specific` as event_type option (maps to OTHER via existing vocabulary)

## Deviations from Plan

None - plan executed exactly as written.

## Known Stubs

None - all data paths are wired through LLMClient and static registry.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- NewsImpactAnalyzer and SectorTickerMapper are standalone, tested modules ready for Plan 02 integration
- Plan 02 will wire these into the news pipeline replacing EntityExtractor + CombinedAnalyzer

---
*Phase: 27-intelligent-news-impact-analysis*
*Completed: 2026-03-24*
