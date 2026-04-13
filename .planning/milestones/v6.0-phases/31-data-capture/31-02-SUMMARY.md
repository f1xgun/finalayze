---
phase: 31-data-capture
plan: 02
subsystem: database
tags: [sqlalchemy, prometheus, fire-and-forget, persistence, orm, news, sentiment]

requires:
  - phase: 31-data-capture
    provides: "_persist_to_db fire-and-forget helper, db_write_failures counter"
provides:
  - "News article persistence after LLM analysis via _persist_news_article_async"
  - "Sentiment score batch persistence via _persist_sentiment_batch_async"
  - "Content hash (SHA-256) for news article deduplication"
affects: [31-data-capture, monitoring, observability]

tech-stack:
  added: []
  patterns: ["await persist in async context (no _persist_to_db deadlock)", "extracted _persist_sentiment_scores helper for branch limit"]

key-files:
  created: []
  modified:
    - src/finalayze/orchestration/trading_loop.py
    - tests/unit/core/test_db_persistence.py

key-decisions:
  - "Used direct await in _process_one (async) instead of _persist_to_db to avoid deadlock on shared _async_loop"
  - "Extracted _persist_sentiment_scores helper method to stay under ruff PLR0912 branch limit (12)"
  - "Market ID derived from segment prefix (ru_ -> moex, else us) using any() for compact branch count"
  - "Content stored as SHA-256 hash (32 chars) in content column; summary holds first 500 chars of article text"

patterns-established:
  - "await persist in async context + try/except for fire-and-forget (vs _persist_to_db in sync context)"

requirements-completed: [PERSIST-03, PERSIST-04]

duration: 3min
completed: 2026-03-30
---

# Phase 31 Plan 02: News Article and Sentiment Score Persistence Summary

**Fire-and-forget DB persistence for news articles (with SHA-256 content hash) and batch sentiment scores after LLM analysis**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-30T21:26:30Z
- **Completed:** 2026-03-30T21:29:28Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- Added `_persist_news_article_async` that maps NewsArticle + NewsImpactResult to NewsArticleModel with SHA-256 content hash
- Added `_persist_sentiment_batch_async` for batch insert of per-ticker sentiment scores in single DB session
- Wired news persistence in `_analyze_impact_batch._process_one` using direct `await` (avoids event loop deadlock)
- Wired sentiment persistence in `_apply_impact_result` via extracted `_persist_sentiment_scores` helper using `_persist_to_db`
- 9 new tests covering method existence, coroutine return, content hash, parameter signatures, and failure isolation

## Task Commits

Each task was committed atomically:

1. **Task 1+2: Wire news article and sentiment persistence** - `2d3107e` (feat)

_Note: Both tasks combined in single commit as they share test file and are tightly coupled_

## Files Created/Modified
- `src/finalayze/orchestration/trading_loop.py` - Added _persist_news_article_async, _persist_sentiment_batch_async, _persist_sentiment_scores methods; wired in _analyze_impact_batch and _apply_impact_result
- `tests/unit/core/test_db_persistence.py` - 9 new tests in TestNewsArticlePersistence and TestSentimentPersistence classes

## Decisions Made
- Used direct `await` for news persistence inside `_process_one` (async context) instead of `_persist_to_db` which would deadlock by calling `_run_async` from the same event loop
- Extracted `_persist_sentiment_scores` helper to keep `_apply_impact_result` under ruff PLR0912 branch limit (was 13, limit 12)
- Derived market_id from segment prefixes using `any(not s.startswith("ru_") for s in active_segments)` for compact branch count
- Stored content hash in `content` column and article summary (first 500 chars) in `summary` column

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Extracted _persist_sentiment_scores to fix PLR0912**
- **Found during:** Task 2
- **Issue:** Adding sentiment persistence block to `_apply_impact_result` pushed branch count to 13 (limit 12)
- **Fix:** Extracted `_persist_sentiment_scores` helper method, used `any()` instead of for-loop for market_id derivation
- **Files modified:** src/finalayze/orchestration/trading_loop.py
- **Verification:** `ruff check` passes clean
- **Committed in:** 2d3107e

---

**Total deviations:** 1 auto-fixed (1 lint fix)
**Impact on plan:** Minor structural refactor for lint compliance. No scope creep.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- All 4 persistence types wired: orders, signals, news articles, sentiment scores
- Phase 31 (data-capture) complete -- all PERSIST requirements fulfilled
- DB tables populated during live trading via fire-and-forget semantics

---
*Phase: 31-data-capture*
*Completed: 2026-03-30*
