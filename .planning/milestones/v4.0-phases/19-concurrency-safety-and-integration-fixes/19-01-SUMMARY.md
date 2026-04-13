---
phase: 19-concurrency-safety-and-integration-fixes
plan: 01
subsystem: execution, data
tags: [asyncio, threading, concurrency, session-management, grpc]

# Dependency graph
requires: []
provides:
  - "Thread-safe async broker with correct lock types (asyncio.Lock for async, threading.Lock for sync)"
  - "Leak-free macro snapshot persistence with async-with session scoping"
  - "Double-check locking for event loop creation in TinkoffBroker"
affects: [execution, data, trading-loop]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "asyncio.Lock for async double-check pattern (not threading.Lock)"
    - "threading.Lock only for sync code paths and thread-safe init guards"
    - "async-with context manager for DB session lifecycle"

key-files:
  created:
    - tests/unit/test_tinkoff_broker_concurrency.py
    - tests/unit/test_macro_cache_session.py
  modified:
    - src/finalayze/execution/tinkoff_broker.py
    - src/finalayze/data/macro_cache.py

key-decisions:
  - "Keep _client_lock as threading.Lock for sync _get_client (APScheduler compatibility)"
  - "Separate _loop_init_lock from _client_lock for independent concerns"
  - "Fire-and-forget pattern for macro persistence failure (log warning, don't re-raise)"

patterns-established:
  - "asyncio.Lock for all async double-check patterns in broker code"
  - "async-with for all DB session usage (never bare session assignment)"

requirements-completed: [CONC-02, CONC-03, CONC-04]

# Metrics
duration: 4min
completed: 2026-03-22
---

# Phase 19 Plan 01: Concurrency Safety Summary

**asyncio.Lock for async broker paths, threading.Lock double-check for loop init, async-with session scoping in macro_cache**

## Performance

- **Duration:** 4 min
- **Started:** 2026-03-22T20:18:46Z
- **Completed:** 2026-03-22T20:23:14Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- Replaced threading.Lock with asyncio.Lock in TinkoffBroker async code paths (CONC-02), eliminating deadlock risk
- Added double-check locking with threading.Lock for event loop creation in _run_async (CONC-03), eliminating TOCTOU race
- Fixed macro_cache session leak by switching to async-with context manager with automatic rollback (CONC-04)
- Added 12 tests covering lock types, source code patterns, session lifecycle, and error handling

## Task Commits

Each task was committed atomically:

1. **Task 1: Fix TinkoffBroker lock types and event loop TOCTOU** - `7bddbe8` (fix)
2. **Task 2: Fix macro_cache session leak** - `5dac006` (fix)

## Files Created/Modified
- `src/finalayze/execution/tinkoff_broker.py` - Added _async_lock (asyncio.Lock) and _loop_init_lock (threading.Lock), replaced threading.Lock usage in async methods
- `src/finalayze/data/macro_cache.py` - Replaced bare session with async-with context manager, added error handling
- `tests/unit/test_tinkoff_broker_concurrency.py` - 7 tests for lock types and usage patterns
- `tests/unit/test_macro_cache_session.py` - 5 tests for session scoping and error handling

## Decisions Made
- Kept _client_lock as threading.Lock for sync _get_client method (called from APScheduler sync threads)
- Created separate _loop_init_lock rather than reusing _client_lock for clarity and independent concerns
- macro_cache persistence uses fire-and-forget with warning log on failure (existing pattern, not worth changing)

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
- threading.Lock() returns _thread.lock object, not threading.Lock type -- adjusted test assertions to use attribute checks instead of isinstance()
- MacroSnapshot constructor in test had wrong kwargs (days_to_next_cbr, next_cbr_date don't exist) -- fixed test fixture

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Concurrency defects in TinkoffBroker and macro_cache are resolved
- Ready for remaining plans in phase 19

---
*Phase: 19-concurrency-safety-and-integration-fixes*
*Completed: 2026-03-22*
