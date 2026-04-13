---
phase: 20-async-correctness-and-resource-management
plan: 02
subsystem: api, execution, data
tags: [asyncio, run_in_executor, grpc, timeout, structured-logging, fastapi]

# Dependency graph
requires:
  - phase: 19-concurrency-safety
    provides: "Thread-safe TinkoffBroker with lock fixes"
provides:
  - "Non-blocking portfolio API endpoint via run_in_executor"
  - "Structured logging in TinkoffBroker.close() instead of exception suppression"
  - "Configurable gRPC timeout (default 60s) for TinkoffFetcher"
affects: [api, execution, data-fetchers]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "run_in_executor for sync broker calls in async FastAPI endpoints"
    - "Structured warning logging for resource cleanup failures"
    - "asyncio.wait_for wrapping for gRPC calls with configurable timeout"

key-files:
  created:
    - "tests/unit/test_tinkoff_broker_close.py"
  modified:
    - "src/finalayze/api/v1/portfolio.py"
    - "src/finalayze/execution/tinkoff_broker.py"
    - "src/finalayze/data/fetchers/tinkoff_data.py"
    - "tests/unit/test_api_portfolio.py"
    - "tests/unit/test_tinkoff_data.py"

key-decisions:
  - "Used default ThreadPoolExecutor (None) for run_in_executor -- appropriate for I/O-bound broker calls"
  - "Split __aexit__ and loop.stop into separate try/except blocks for independent failure handling"
  - "Default gRPC timeout 60s -- long enough for slow MOEX responses, short enough to prevent indefinite hangs"

patterns-established:
  - "run_in_executor pattern: wrap sync broker calls in async endpoints to prevent event loop starvation"
  - "Structured cleanup logging: log resource name + error_type + error message on cleanup failure"

requirements-completed: [ASYNC-03, RES-01, RES-02]

# Metrics
duration: 4min
completed: 2026-03-22
---

# Phase 20 Plan 02: Blocking API, Suppressed Exceptions, and Missing gRPC Timeout Summary

**Non-blocking portfolio endpoint via run_in_executor, structured close() failure logging, and configurable 60s gRPC timeout for TinkoffFetcher**

## Performance

- **Duration:** 4 min
- **Started:** 2026-03-22T20:41:35Z
- **Completed:** 2026-03-22T20:45:39Z
- **Tasks:** 3
- **Files modified:** 6

## Accomplishments
- Portfolio API endpoint no longer blocks FastAPI event loop during broker I/O (uses run_in_executor)
- TinkoffBroker.close() logs structured warnings on cleanup failure instead of silently suppressing all exceptions
- TinkoffFetcher gRPC calls have configurable timeout (default 60s) preventing indefinite hangs

## Task Commits

Each task was committed atomically:

1. **Task 1: Wrap portfolio broker calls with run_in_executor** - `2dacff3` (feat)
2. **Task 2: Log TinkoffBroker.close() cleanup failures** - `1f760f5` (fix)
3. **Task 3: Add configurable timeout to TinkoffFetcher gRPC calls** - `2bbb677` (feat)

## Files Created/Modified
- `src/finalayze/api/v1/portfolio.py` - Added asyncio.get_running_loop().run_in_executor() around sync broker.get_portfolio()
- `src/finalayze/execution/tinkoff_broker.py` - Replaced contextlib.suppress(Exception) with explicit try/except + structured logging
- `src/finalayze/data/fetchers/tinkoff_data.py` - Added grpc_timeout parameter and asyncio.wait_for() wrapping
- `tests/unit/test_api_portfolio.py` - Added 3 tests for run_in_executor usage and error handling
- `tests/unit/test_tinkoff_broker_close.py` - Created 5 tests for close() logging behavior
- `tests/unit/test_tinkoff_data.py` - Added 4 tests for gRPC timeout configuration

## Decisions Made
- Used default ThreadPoolExecutor (None) for run_in_executor -- appropriate for I/O-bound broker calls, avoids custom pool management
- Split __aexit__ and loop.stop into separate try/except blocks so one failure doesn't prevent the other from being attempted
- Default gRPC timeout of 60 seconds balances MOEX API latency with hang prevention

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Known Stubs
None - all functionality is fully wired.

## Next Phase Readiness
- All three async/resource fixes verified with 29 passing tests
- Ready for remaining phase 20 plans

---
*Phase: 20-async-correctness-and-resource-management*
*Completed: 2026-03-22*
