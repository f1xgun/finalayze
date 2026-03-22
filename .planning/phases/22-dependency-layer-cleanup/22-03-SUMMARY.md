---
phase: 22-dependency-layer-cleanup
plan: 03
subsystem: core, api
tags: [event-bus, redis-streams, dead-code, api-stubs, http-501]

requires:
  - phase: 22-dependency-layer-cleanup
    provides: "orchestration module extraction (plan 01), layer assignment (plan 02)"
provides:
  - "Clean EventBus with only STREAM_COUPONS (dead streams removed)"
  - "Explicit 501 Not Implemented for 7 stub API endpoints"
  - "Removed unused MarketDataEvent and SignalEvent model classes"
affects: [api, core]

tech-stack:
  added: []
  patterns:
    - "Stub endpoints return 501 with detail message instead of empty 200"
    - "TYPE_CHECKING guard for pydantic BaseModel in events.py"

key-files:
  created: []
  modified:
    - src/finalayze/core/events.py
    - src/finalayze/api/v1/signals.py
    - src/finalayze/api/v1/trades.py
    - src/finalayze/api/v1/news.py
    - src/finalayze/api/v1/ml.py
    - tests/unit/test_events.py
    - tests/unit/test_api_signals_risk.py
    - tests/unit/test_api_trades.py
    - tests/unit/conftest.py

key-decisions:
  - "Removed MarketDataEvent and SignalEvent classes -- unused in src/, tests updated to use local _TestEvent"
  - "Kept Pydantic response models on 501 endpoints for OpenAPI documentation"

patterns-established:
  - "501 Not Implemented pattern: raise HTTPException(status_code=501, detail='Not yet implemented')"

requirements-completed: [DEAD-01, DEAD-02]

duration: 5min
completed: 2026-03-23
---

# Phase 22 Plan 03: Dead Event Bus Streams and Stub API Cleanup Summary

**Removed 3 dead EventBus stream constants and 2 unused event models; converted 7 stub API endpoints from empty-200 to explicit 501 Not Implemented**

## Performance

- **Duration:** 5 min
- **Started:** 2026-03-22T21:14:42Z
- **Completed:** 2026-03-22T21:19:39Z
- **Tasks:** 2
- **Files modified:** 9

## Accomplishments
- EventBus now has only STREAM_COUPONS (used by bond_discovery.py); STREAM_MARKET_DATA, STREAM_SIGNALS, STREAM_EXECUTION removed
- MarketDataEvent and SignalEvent classes removed (no consumers in src/)
- 7 stub endpoints (signals, strategies/performance, trades, trades/analytics, trade detail, news, ml/status) now return HTTP 501 instead of silently returning empty data
- All tests updated and passing (28 tests across affected files)

## Task Commits

Each task was committed atomically:

1. **Task 1: Remove dead event bus streams** - `3eac311` (feat)
2. **Task 2: Convert stub API endpoints to 501** - `3c65ae6` (feat)
3. **Ruff fix: BaseModel TYPE_CHECKING** - `62f704f` (fix)

## Files Created/Modified
- `src/finalayze/core/events.py` - Removed dead streams and event classes, BaseModel to TYPE_CHECKING
- `src/finalayze/api/v1/signals.py` - 501 for list_signals and strategies_performance
- `src/finalayze/api/v1/trades.py` - 501 for list_trades, trade_analytics, get_trade
- `src/finalayze/api/v1/news.py` - 501 for list_news
- `src/finalayze/api/v1/ml.py` - 501 for ml_status
- `tests/unit/test_events.py` - Updated to use local _TestEvent, removed dead constant assertions
- `tests/unit/test_api_signals_risk.py` - Updated to expect 501
- `tests/unit/test_api_trades.py` - Updated to expect 501
- `tests/unit/conftest.py` - Fixed import path for moved TradingLoop

## Decisions Made
- Removed MarketDataEvent and SignalEvent from events.py since they had no consumers in src/; tests updated to use a local _TestEvent model
- Kept Pydantic response models (SignalsResponse, TradesResponse, etc.) on 501 endpoints to preserve OpenAPI documentation
- Moved BaseModel import to TYPE_CHECKING block to satisfy ruff TC002

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Fixed conftest.py import for moved TradingLoop**
- **Found during:** Task 1 (running test verification)
- **Issue:** tests/unit/conftest.py imported from finalayze.core.trading_loop which was moved to orchestration/ by plan 22-01
- **Fix:** Updated import to finalayze.orchestration.trading_loop
- **Files modified:** tests/unit/conftest.py
- **Verification:** All tests pass
- **Committed in:** 3eac311 (Task 1 commit)

**2. [Rule 1 - Bug] Fixed ruff TC002 for BaseModel import**
- **Found during:** Post-Task 2 verification
- **Issue:** After removing MarketDataEvent/SignalEvent, BaseModel was only used as type hint -- ruff TC002 requires moving to TYPE_CHECKING
- **Fix:** Moved import to TYPE_CHECKING block
- **Files modified:** src/finalayze/core/events.py
- **Verification:** ruff check passes
- **Committed in:** 62f704f

---

**Total deviations:** 2 auto-fixed (1 blocking, 1 bug)
**Impact on plan:** Both fixes necessary for correctness. No scope creep.

## Issues Encountered
None beyond the auto-fixed deviations.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Phase 22 (dependency-layer-cleanup) is now complete (all 3 plans done)
- EventBus is clean with only active infrastructure
- API endpoints clearly communicate unimplemented status
- Ready for next v4.0 phase

---
*Phase: 22-dependency-layer-cleanup*
*Completed: 2026-03-23*
