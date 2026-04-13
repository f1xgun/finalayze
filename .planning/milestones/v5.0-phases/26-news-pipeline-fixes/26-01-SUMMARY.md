---
phase: 26-news-pipeline-fixes
plan: 01
subsystem: orchestration
tags: [news-pipeline, sentiment-decay, event-driven, trading-loop]

# Dependency graph
requires:
  - phase: 22-orchestration-extraction
    provides: "TradingLoop in orchestration/ module"
provides:
  - "News cycle skip guard when event_driven disabled across all segments"
  - "_read_decayed_sentiment with 4-hour half-life exponential decay"
  - "_any_event_driven_enabled with preset YAML scanning and caching"
affects: [news-pipeline, sentiment, trading-loop]

# Tech tracking
tech-stack:
  added: []
  patterns: ["Exponential time-decay for cached sentiment scores", "Preset YAML scanning with result caching for feature flags"]

key-files:
  created: []
  modified:
    - src/finalayze/orchestration/trading_loop.py
    - tests/unit/core/test_trading_loop.py
    - tests/unit/test_news_cycle_integration.py

key-decisions:
  - "Reused _get_segment_min_confidence YAML scanning pattern for _any_event_driven_enabled"
  - "Cache event_driven check result once per process lifetime (no invalidation needed)"

patterns-established:
  - "Sentiment decay: score * exp(-lambda * hours_elapsed) with lambda = ln(2)/4h"
  - "Feature flag scanning: iterate presets/*.yaml, cache result in instance variable"

requirements-completed: [NEWS-01, NEWS-02]

# Metrics
duration: 5min
completed: 2026-03-24
---

# Phase 26 Plan 01: News Pipeline Fixes Summary

**News cycle skip guard when event_driven is disabled plus 4-hour half-life exponential sentiment decay**

## Performance

- **Duration:** 5 min
- **Started:** 2026-03-24T10:25:00Z
- **Completed:** 2026-03-24T10:30:00Z
- **Tasks:** 1
- **Files modified:** 3

## Accomplishments
- _news_cycle() returns immediately when no segment preset has event_driven enabled, saving LLM tokens
- Sentiment scores now decay exponentially with a 4-hour half-life (50% at 4h, 25% at 8h)
- All 17 tests pass (10 new + 7 pre-existing integration tests)

## Task Commits

Each task was committed atomically:

1. **Task 1: Add news cycle skip guard and sentiment time-decay** - `e280363` (feat)
   - TDD RED commit (pre-existing): `f67f6e6` (test)

**Plan metadata:** TBD (docs: complete plan)

## Files Created/Modified
- `src/finalayze/orchestration/trading_loop.py` - Added _any_event_driven_enabled(), _read_decayed_sentiment(), updated _get_sentiment() fallback
- `tests/unit/core/test_trading_loop.py` - 10 tests for skip guard and sentiment decay (from RED commit)
- `tests/unit/test_news_cycle_integration.py` - Fixed pre-existing tests to set _event_driven_active=True

## Decisions Made
- Reused the _get_segment_min_confidence YAML scanning pattern for _any_event_driven_enabled
- Cache result once per process lifetime -- no invalidation needed since preset files don't change at runtime
- Decay uses time.monotonic() for clock-skew safety

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed pre-existing integration tests broken by skip guard**
- **Found during:** Task 1 (GREEN implementation)
- **Issue:** 7 pre-existing tests in test_news_cycle_integration.py started failing because _any_event_driven_enabled() reads real preset YAMLs (all have event_driven disabled)
- **Fix:** Set _event_driven_active=True in _make_loop() helper so tests bypass the guard
- **Files modified:** tests/unit/test_news_cycle_integration.py
- **Verification:** All 17 tests pass
- **Committed in:** e280363 (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Auto-fix necessary to keep pre-existing tests passing. No scope creep.

## Issues Encountered
None

## Known Stubs
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- News pipeline skip guard active -- zero LLM waste when event_driven is disabled
- Sentiment decay operational -- stale scores age out naturally
- Ready for 26-02 (ticker mismatch and Telegram dedup)

---
## Self-Check: PASSED

All files exist. All commits verified.

*Phase: 26-news-pipeline-fixes*
*Completed: 2026-03-24*
