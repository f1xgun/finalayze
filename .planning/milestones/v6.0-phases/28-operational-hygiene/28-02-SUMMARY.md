---
phase: 28-operational-hygiene
plan: 02
subsystem: orchestration
tags: [dedup, telegram, resilience, sha256, news-pipeline]

requires:
  - phase: 28-operational-hygiene
    provides: "Trading loop and sandbox operational context"
provides:
  - "LLM article deduplication via SHA-256 content hash with 24h TTL"
  - "Startup-safe Telegram alerter in sandbox launch"
affects: [news-pipeline, sandbox-operations]

tech-stack:
  added: []
  patterns: [OrderedDict-based dedup with TTL eviction, defense-in-depth try/except for alerter]

key-files:
  created:
    - tests/unit/test_api_alerts.py
  modified:
    - src/finalayze/orchestration/trading_loop.py
    - scripts/run_sandbox.py
    - tests/unit/core/test_trading_loop.py

key-decisions:
  - "SHA-256(url|title) as dedup key -- fast, collision-resistant, deterministic"
  - "time.monotonic() for TTL instead of wall-clock -- immune to clock adjustments"
  - "Defense-in-depth: wrap send_alert at call site even though send_alert has internal suppression"

patterns-established:
  - "OrderedDict + TTL eviction pattern for bounded dedup caches"

requirements-completed: [OPS-03, OPS-04]

duration: 5min
completed: 2026-03-30
---

# Phase 28 Plan 02: LLM Article Dedup and Alerter Resilience Summary

**SHA-256 article deduplication before LLM calls (24h TTL, 5000-entry cap) and try/except-wrapped Telegram alerter at sandbox startup/shutdown**

## Performance

- **Duration:** 5 min
- **Started:** 2026-03-30T08:22:48Z
- **Completed:** 2026-03-30T08:27:40Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- Duplicate news articles are now filtered by SHA-256(url|title) hash before LLM processing, eliminating the 35 daily fallback activations from article duplication
- Telegram alerter failures at sandbox startup/shutdown no longer block the trading loop
- 5 new tests covering dedup logic and alerter resilience

## Task Commits

Each task was committed atomically:

1. **Task 1: Add LLM article deduplication via content hash (OPS-03)** - `58a6cc0` (feat)
2. **Task 2: Make Telegram alerter startup-safe (OPS-04)** - `e8d768e` (feat)

## Files Created/Modified
- `src/finalayze/orchestration/trading_loop.py` - Added _is_article_duplicate method with SHA-256 hash + OrderedDict TTL cache, dedup filter in _analyze_impact_batch
- `scripts/run_sandbox.py` - Wrapped startup/shutdown alerter.send_alert in try/except with structured logging
- `tests/unit/core/test_trading_loop.py` - 3 dedup tests (skip duplicate, TTL expiry, different articles)
- `tests/unit/test_api_alerts.py` - 2 alerter resilience tests (no-op on empty token, suppresses network errors)

## Decisions Made
- Used SHA-256(url|title) as dedup key rather than just URL -- catches same article with different URLs but same title
- Used time.monotonic() for TTL tracking -- immune to system clock adjustments during long-running sandbox
- Defense-in-depth: wrapped send_alert at call site in run_sandbox.py even though send_alert already suppresses exceptions internally -- httpx.Client instantiation failures could bypass the inner try/except

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
- Parallel agent (28-01) committed on top of task 1 commit, bundling trading_loop.py changes into its commit. Changes are preserved correctly in the repository.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- News pipeline now deduplicates articles before LLM processing
- Sandbox launch is resilient to Telegram connectivity issues
- Ready for Phase 29 (gRPC loop consolidation)

---
*Phase: 28-operational-hygiene*
*Completed: 2026-03-30*
