---
phase: 27-intelligent-news-impact-analysis
plan: 02
subsystem: orchestration
tags: [news-pipeline, sentiment, per-ticker, trading-loop, sector-mapping]

requires:
  - phase: 27-01
    provides: NewsImpactAnalyzer and SectorTickerMapper classes

provides:
  - Per-ticker sentiment cache in TradingLoop keyed by (segment_id, ticker)
  - Single-call news pipeline using NewsImpactAnalyzer instead of EntityExtractor + CombinedAnalyzer
  - Sector-only articles produce non-zero sentiment for mapped tickers

affects: [event_driven, strategies, backtest]

tech-stack:
  added: []
  patterns:
    - "Per-ticker sentiment cache with tuple keys (segment_id, ticker)"
    - "Segment average fallback when ticker has no direct sentiment entry"
    - "_apply_impact_result bridges NewsImpactResult to per-ticker cache"

key-files:
  created: []
  modified:
    - src/finalayze/orchestration/trading_loop.py
    - src/finalayze/main.py
    - tests/unit/test_news_cycle_integration.py
    - tests/unit/core/test_trading_loop.py
    - tests/unit/test_trading_loop_thread_safety.py

key-decisions:
  - "Sector sentiment formula: magnitude * direction * article.sentiment (not confidence-weighted)"
  - "Direct ticker score: sentiment * confidence (stronger than sector-derived score)"
  - "Fallback to segment average when ticker has no per-ticker entry"
  - "Redis cache key format: seg_id:ticker for per-ticker entries"

patterns-established:
  - "Per-ticker sentiment keying: (segment_id, ticker) tuple as cache key"
  - "_apply_impact_result as bridge between NewsImpactResult and sentiment cache"

requirements-completed: [NEWS-07, NEWS-08]

duration: 13min
completed: 2026-03-24
---

# Phase 27 Plan 02: TradingLoop Integration Summary

**Per-ticker sentiment via NewsImpactAnalyzer+SectorTickerMapper replacing 2-call EntityExtractor+CombinedAnalyzer pipeline**

## Performance

- **Duration:** 13 min
- **Started:** 2026-03-24T09:02:08Z
- **Completed:** 2026-03-24T09:15:30Z
- **Tasks:** 1
- **Files modified:** 5

## Accomplishments
- Replaced EntityExtractor + CombinedAnalyzer (2-call pipeline) with NewsImpactAnalyzer (single LLM call) in TradingLoop
- Changed _sentiment_cache from flat segment_id keys to (segment_id, ticker) tuple keys for per-ticker sentiment
- Sector-only articles (e.g., "CBR raised rates") now produce non-zero sentiment for banking tickers (SBER, VTBR, TCSG) via SectorTickerMapper
- Updated _get_sentiment and _process_instrument to pass ticker for per-ticker lookups

## Task Commits

Each task was committed atomically:

1. **Task 1 (RED): Failing tests for per-ticker sentiment** - `3917552` (test)
2. **Task 1 (GREEN): Wire NewsImpactAnalyzer with per-ticker cache** - `3c92072` (feat)

## Files Created/Modified
- `src/finalayze/orchestration/trading_loop.py` - Replaced news pipeline, per-ticker cache, new methods
- `src/finalayze/main.py` - Updated TradingLoop construction with NewsImpactAnalyzer + SectorTickerMapper
- `tests/unit/test_news_cycle_integration.py` - Rewritten for new pipeline (15 tests)
- `tests/unit/core/test_trading_loop.py` - Updated sentiment cache tests for tuple keys (10 tests)
- `tests/unit/test_trading_loop_thread_safety.py` - Updated Redis lock scope test for new API (6 tests)

## Decisions Made
- Sector sentiment formula: `magnitude * direction * sentiment` (not confidence-weighted) -- matches CONTEXT.md specification
- Direct ticker score uses `sentiment * confidence` (stronger than sector-derived) -- prioritizes explicit mentions
- Segment average fallback when no per-ticker entry exists -- ensures backward compatibility for unknown tickers
- Redis cache key uses `seg_id:ticker` format -- simple string key for Redis compatibility

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Updated test_trading_loop_thread_safety.py**
- **Found during:** Task 1 (GREEN phase)
- **Issue:** `test_redis_write_outside_lock` called removed method `_process_news_article`
- **Fix:** Rewrote test to use `_apply_impact_result` with NewsImpactResult
- **Files modified:** tests/unit/test_trading_loop_thread_safety.py
- **Committed in:** 3c92072

**2. [Rule 1 - Bug] Removed unused TYPE_CHECKING imports**
- **Found during:** Task 1 (GREEN phase)
- **Issue:** `EventType` and `SentimentResult` imports became unused after removing old methods
- **Fix:** Removed unused imports to pass ruff F401 check
- **Files modified:** src/finalayze/orchestration/trading_loop.py
- **Committed in:** 3c92072

---

**Total deviations:** 2 auto-fixed (2 bugs)
**Impact on plan:** Both fixes necessary for correctness. No scope creep.

## Issues Encountered
None - plan executed as written.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Per-ticker sentiment pipeline is fully operational
- All NEWS-07 and NEWS-08 requirements implemented
- Phase 27 (intelligent-news-impact-analysis) is complete
- Event_driven strategy now receives per-ticker sentiment scores instead of flat segment scores

## Known Stubs
None - all data paths are wired to live sources.

---
*Phase: 27-intelligent-news-impact-analysis*
*Completed: 2026-03-24*
