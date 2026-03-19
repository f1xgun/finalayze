---
phase: 03-bond-data-pipeline
plan: 02
subsystem: data
tags: [cbr, yield-curve, ofz-in, indexation, macro, timescaledb, orm]

requires:
  - phase: 01-position-sizing
    provides: "CBR meeting calendar, MacroSnapshot, MacroContextProvider"
provides:
  - "Extended MacroSnapshot with yield_curve, breakeven_inflation, usdrub, ofzin_indexation_coefficient"
  - "CBR yield curve fetcher (12 maturity points from HTML)"
  - "OFZ-IN indexation coefficient fetcher"
  - "MacroSnapshotModel ORM for TimescaleDB persistence"
  - "MacroCacheService DB persistence on refresh"
affects: [03-bond-data-pipeline, 04-bond-execution]

tech-stack:
  added: [lxml.html]
  patterns: [sync-async-boundary, fire-and-forget-db-write, graceful-degradation]

key-files:
  created:
    - tests/unit/test_cbr_yield_curve.py
    - tests/unit/test_macro_persistence.py
    - tests/unit/test_ofzin_indexation.py
  modified:
    - src/finalayze/data/fetchers/cbr.py
    - src/finalayze/core/models.py
    - src/finalayze/data/macro_cache.py

key-decisions:
  - "Yield curve parsed from CBR HTML table using lxml.html (not XML API)"
  - "Async DB persistence via asyncio.run/create_task boundary in sync refresh()"
  - "DB failures never crash refresh cycle (logged warning, graceful degradation)"
  - "Backtest mode returns None for yield_curve and indexation (no HTTP calls)"

patterns-established:
  - "Sync-async boundary: sync APScheduler callback fires async DB write via asyncio.run or loop.create_task"
  - "Graceful degradation: CBR fetchers return None on HTTP error, never raise to caller"
  - "Optional DB persistence: db_session_factory=None preserves backward compatibility"

requirements-completed: [BDP-03]

duration: 5min
completed: 2026-03-14
---

# Phase 3 Plan 2: Macro Data & Yield Curve Summary

**Extended MacroSnapshot with CBR yield curve (12 maturities), OFZ-IN indexation coefficient, and TimescaleDB persistence via async boundary in MacroCacheService**

## Performance

- **Duration:** 5 min
- **Started:** 2026-03-14T17:38:46Z
- **Completed:** 2026-03-14T17:44:00Z
- **Tasks:** 3
- **Files modified:** 6

## Accomplishments
- Extended MacroSnapshot with yield_curve, breakeven_inflation, usdrub, ofzin_indexation_coefficient fields
- Added CBR yield curve fetcher parsing HTML into 12-point maturity dict via lxml.html
- Added OFZ-IN indexation coefficient fetcher for inflation-linked bond nominal adjustment
- Created MacroSnapshotModel ORM with JSONB yield_curve and all macro fields
- Added DB persistence to MacroCacheService with sync/async boundary (backward compatible)
- 31 total tests across 3 test files

## Task Commits

Each task was committed atomically:

1. **Task 1: Extend MacroSnapshot, add yield curve fetching, create ORM model** - `8339b10` (feat)
2. **Task 2: Add DB persistence to MacroCacheService with async boundary** - `391dc30` (feat)
3. **Task 3: Add OFZ-IN indexation coefficient fetching to CBRFetcher** - `ef73c43` (feat)

_Note: TDD tasks -- each had RED (failing test) then GREEN (implementation) phases_

## Files Created/Modified
- `src/finalayze/data/fetchers/cbr.py` - Extended MacroSnapshot, added fetch_yield_curve, fetch_ofzin_indexation_coefficient
- `src/finalayze/core/models.py` - Added MacroSnapshotModel ORM with JSONB yield_curve
- `src/finalayze/data/macro_cache.py` - Added DB persistence with async boundary
- `tests/unit/test_cbr_yield_curve.py` - 11 tests for yield curve parsing and MacroSnapshot fields
- `tests/unit/test_macro_persistence.py` - 5 tests for DB persistence and backward compatibility
- `tests/unit/test_ofzin_indexation.py` - 9 tests for indexation coefficient fetching

## Decisions Made
- Used lxml.html for yield curve HTML parsing (CBR returns HTML table, not XML)
- Async DB writes via asyncio.run (no running loop) or loop.create_task (running loop)
- DB failures logged but never crash refresh cycle (fire-and-forget pattern)
- Backtest mode (MacroContextProvider) returns None for all new fields (no HTTP calls)
- Yield curve stored as JSONB in PostgreSQL (flexible schema for maturity points)

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- MacroSnapshot now has all fields needed by bond pricing (yield curves for QuantLib)
- OFZ-IN indexation coefficient ready for inflation-linked bond nominal adjustment
- MacroSnapshotModel ORM ready for TimescaleDB hypertable creation
- Plan 03 (T-Bank bond data) can proceed with this macro data foundation

---
*Phase: 03-bond-data-pipeline*
*Completed: 2026-03-14*
