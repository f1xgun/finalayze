---
phase: 31-data-capture
plan: 01
subsystem: database
tags: [sqlalchemy, prometheus, fire-and-forget, persistence, orm]

requires:
  - phase: 22-orchestration-extraction
    provides: orchestration/ module with TradingLoop
provides:
  - "_persist_to_db fire-and-forget helper in TradingLoop"
  - "Order persistence after fill via _persist_order_async"
  - "Signal persistence after generation via _persist_signal_async"
  - "db_write_failures Prometheus counter with table label"
affects: [31-data-capture, monitoring, observability]

tech-stack:
  added: []
  patterns: ["fire-and-forget DB persistence with Prometheus failure counter"]

key-files:
  created:
    - tests/unit/core/test_db_persistence.py
  modified:
    - src/finalayze/orchestration/trading_loop.py
    - src/finalayze/api/metrics.py

key-decisions:
  - "Removed strategy_name and market_id params from async persist methods to avoid ARG002 lint -- OrderModel links to signal via FK, signal already carries market_id"
  - "Deferred import of db_write_failures in _persist_to_db to avoid circular dependency"

patterns-established:
  - "_persist_to_db(coro, table=...) pattern for any future fire-and-forget DB writes"

requirements-completed: [PERSIST-01, PERSIST-02, PERSIST-05]

duration: 6min
completed: 2026-03-30
---

# Phase 31 Plan 01: DB Persistence for Orders and Signals Summary

**Fire-and-forget DB persistence for orders and signals via _persist_to_db helper with Prometheus failure counter**

## Performance

- **Duration:** 6 min
- **Started:** 2026-03-30T21:17:57Z
- **Completed:** 2026-03-30T21:24:18Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- Added `_persist_to_db` fire-and-forget helper that swallows exceptions, logs at WARNING, and increments `db_write_failures` Prometheus counter
- Wired order persistence after fill in `_submit_order` via `_persist_order_async`
- Wired signal persistence after generation in strategy cycle via `_persist_signal_async`
- 10 tests covering exception isolation, logging, counter behavior, and persistence wiring

## Task Commits

Each task was committed atomically:

1. **Task 1: Add db_write_failures counter and _persist_to_db helper** - `da9ab9d` (test)
2. **Task 2: Wire order and signal persistence in strategy cycle** - `94c237f` (feat)

_Note: TDD tasks combined RED+GREEN in single commits for simplicity_

## Files Created/Modified
- `src/finalayze/api/metrics.py` - Added db_write_failures Prometheus Counter with table label
- `src/finalayze/orchestration/trading_loop.py` - Added _persist_to_db, _persist_order_async, _persist_signal_async methods; wired in strategy cycle
- `tests/unit/core/test_db_persistence.py` - 10 tests for fire-and-forget persistence behavior

## Decisions Made
- Removed `strategy_name` param from `_persist_order_async` -- OrderModel has FK to signals table, no strategy_name column
- Removed `market_id` param from `_persist_signal_async` -- signal already carries `signal.market_id`
- Used deferred import for `db_write_failures` in `_persist_to_db` to avoid circular dependency (orchestration L5 -> api L6)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Removed unused parameters from async persist methods**
- **Found during:** Task 2
- **Issue:** Plan specified `strategy_name` on `_persist_order_async` and `market_id` on `_persist_signal_async` but neither was used, causing ruff ARG002 lint errors
- **Fix:** Removed unused params, simplified call sites
- **Files modified:** src/finalayze/orchestration/trading_loop.py
- **Verification:** ruff check passes clean
- **Committed in:** 94c237f

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Minor API simplification. No scope creep.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Order and signal persistence wired and tested
- Plan 31-02 can proceed with news article and sentiment score persistence
- `_persist_to_db` pattern ready for reuse in plan 02

---
*Phase: 31-data-capture*
*Completed: 2026-03-30*
