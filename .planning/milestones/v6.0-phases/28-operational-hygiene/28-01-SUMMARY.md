---
phase: 28-operational-hygiene
plan: 01
subsystem: orchestration, config
tags: [market-hours, trading-loop, moex, tickers, schedule]

requires:
  - phase: none
    provides: n/a
provides:
  - Market-hours gate in TradingLoop._strategy_cycle
  - Corrected MOEX ticker lists (HEAD replaces HHRU)
affects: [29-grpc-loop-consolidation, sandbox-stability]

tech-stack:
  added: []
  patterns:
    - "Market-hours guard pattern: check SCHEDULES before cycle execution"

key-files:
  created:
    - tests/unit/config/__init__.py
    - tests/unit/config/test_segments.py
  modified:
    - src/finalayze/orchestration/trading_loop.py
    - config/segments.py
    - tests/unit/core/test_trading_loop.py

key-decisions:
  - "Unknown markets (no SCHEDULES entry) assumed open -- safe default to avoid blocking new markets"
  - "PLR0915 suppressed on _strategy_cycle -- method inherently long, guard adds 10 lines"

patterns-established:
  - "Market-hours guard: iterate registered_markets, check SCHEDULES, skip if all closed"

requirements-completed: [OPS-01, OPS-02]

duration: 3min
completed: 2026-03-30
---

# Phase 28 Plan 01: Operational Hygiene Summary

**Market-hours gate in strategy cycle to skip off-hours computation, and HHRU->HEAD ticker fix in ru_tech segment**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-30T08:22:40Z
- **Completed:** 2026-03-30T08:25:40Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments
- Strategy cycle now checks all registered markets via SCHEDULES before executing; skips with structured log when all are closed
- Replaced stale HHRU ticker with HEAD in ru_tech segment (HeadHunter MOEX rebrand)
- 5 new tests covering market-hours gate and stale ticker validation

## Task Commits

Each task was committed atomically:

1. **Task 1: Add market-hours guard to strategy cycle** - `a363908` (feat)
2. **Task 2: Fix stale MOEX tickers in segment config** - `07a429b` (fix)

## Files Created/Modified
- `src/finalayze/orchestration/trading_loop.py` - Added SCHEDULES import and market-hours gate in _strategy_cycle
- `config/segments.py` - Replaced HHRU with HEAD in ru_tech symbols
- `tests/unit/core/test_trading_loop.py` - Added TestMarketHoursGate class with 2 tests
- `tests/unit/config/test_segments.py` - Created with 3 tests for stale ticker validation
- `tests/unit/config/__init__.py` - Created package init

## Decisions Made
- Unknown markets (no SCHEDULES entry) assumed open -- safe default to avoid accidentally blocking new market integrations
- Added noqa PLR0915 to _strategy_cycle since the method is inherently long and the guard is a small addition

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed test assertion on non-mock method**
- **Found during:** Task 1 (market-hours gate tests)
- **Issue:** Test tried assert_not_called() on real _strategy_cycle_impl method (not a mock)
- **Fix:** Mock _strategy_cycle_impl before calling _strategy_cycle in both tests
- **Files modified:** tests/unit/core/test_trading_loop.py
- **Verification:** Both tests pass
- **Committed in:** a363908 (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Minor test fix, no scope change.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Market-hours guard ready; will reduce wasted cycles in sandbox
- Stale tickers fixed; ru_tech segment ready for data fetching
- Phase 29 (gRPC loop consolidation) can proceed independently

---
*Phase: 28-operational-hygiene*
*Completed: 2026-03-30*
