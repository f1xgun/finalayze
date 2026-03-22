---
phase: 20-async-correctness-and-resource-management
plan: 03
subsystem: core
tags: [httpx, telegram, lifecycle, resource-management, async]

requires:
  - phase: none
    provides: standalone fix
provides:
  - "Idempotent TelegramAlerter.close() with _closed guard"
  - "Lifespan shutdown wiring for both TelegramAlerter instances"
affects: [core-alerts, main-lifespan, resource-management]

tech-stack:
  added: []
  patterns: ["Idempotent async close with _closed flag guard"]

key-files:
  created:
    - tests/unit/core/test_alerts_lifecycle.py
  modified:
    - src/finalayze/core/alerts.py
    - src/finalayze/main.py

key-decisions:
  - "Used _closed boolean flag for idempotent close (simple, no lock needed since close is always called from same async context)"
  - "Close alerters after trading loop thread join to ensure no in-flight messages"

patterns-established:
  - "Idempotent close pattern: _closed flag checked at top of close(), set before any resource release"

requirements-completed: [RES-03]

duration: 5min
completed: 2026-03-22
---

# Phase 20 Plan 03: TelegramAlerter Shutdown Wiring Summary

**Idempotent TelegramAlerter.close() wired into FastAPI lifespan shutdown for both alerter instances (trading loop + bot handler), preventing httpx.AsyncClient resource leaks**

## Performance

- **Duration:** 5 min
- **Started:** 2026-03-22T20:41:37Z
- **Completed:** 2026-03-22T20:46:00Z
- **Tasks:** 1
- **Files modified:** 3

## Accomplishments
- Added `_closed` flag to TelegramAlerter for idempotent close()
- Wired trading loop alerter close into lifespan shutdown
- Wired bot handler alerter close into lifespan shutdown
- 5 lifecycle tests covering close, queue stop, idempotency, and flag behavior

## Task Commits

Each task was committed atomically (TDD: RED + GREEN):

1. **Task 1 (RED): Lifecycle tests** - `6485560` (test)
2. **Task 1 (GREEN): Implementation** - `ea36cc2` (feat)

## Files Created/Modified
- `tests/unit/core/test_alerts_lifecycle.py` - 5 lifecycle tests for TelegramAlerter.close()
- `src/finalayze/core/alerts.py` - Added _closed flag and idempotent close() guard
- `src/finalayze/main.py` - Shutdown wiring for both alerter instances in lifespan

## Decisions Made
- Used `_closed` boolean flag for idempotency (no lock needed -- close is always called from same async context in lifespan)
- Close alerters after trading loop thread join to ensure no in-flight alert messages

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- httpx resource leaks on shutdown are now fixed
- Both TelegramAlerter instances properly cleaned up during application exit

---
*Phase: 20-async-correctness-and-resource-management*
*Completed: 2026-03-22*
