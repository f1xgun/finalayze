---
phase: 30-broker-resilience
plan: 01
subsystem: execution
tags: [grpc, tinkoff, broker, resilience, portfolio-cache, reconnect]

requires:
  - phase: 29-core-stability
    provides: gRPC loop injection and TinkoffBroker refactoring
provides:
  - Portfolio cache fallback on T-Bank error 70001
  - Auto-reconnect gRPC channel after 5 consecutive 70001 errors
  - Staleness-aware cached portfolio with age logging
affects: [orchestration, trading-loop, broker-resilience]

tech-stack:
  added: []
  patterns: [cache-on-success-fallback-on-failure, consecutive-error-threshold-reconnect]

key-files:
  created: []
  modified:
    - src/finalayze/execution/tinkoff_broker.py
    - tests/unit/test_broker.py

key-decisions:
  - "Cache portfolio in-memory on success, return stale cache on 70001 -- no threading locks needed (single APScheduler thread)"
  - "Auto-reconnect after 5 consecutive 70001 errors, reset counter after reconnect or success"
  - "Non-70001 errors still raise BrokerError unchanged -- no fallback for unknown failures"

patterns-established:
  - "70001 fallback: cache-on-success, return-stale-on-70001, reconnect-after-threshold"

requirements-completed: [GRPC-02, GRPC-03]

duration: 3min
completed: 2026-03-30
---

# Phase 30 Plan 01: Broker Resilience Summary

**Portfolio cache fallback on T-Bank 70001 errors with auto-reconnect after 5 consecutive failures**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-30T21:02:39Z
- **Completed:** 2026-03-30T21:06:06Z
- **Tasks:** 1 (TDD: RED + GREEN)
- **Files modified:** 2

## Accomplishments
- TinkoffBroker.get_portfolio() caches successful results with timestamp for fallback
- On 70001 error with cache: returns stale PortfolioState with "portfolio_using_cached" log including cache_age_seconds
- Auto-reconnects gRPC channel after 5 consecutive 70001 errors, logs "grpc_channel_reconnected_70001"
- 8 new unit tests covering all cache/fallback/reconnect paths, all passing

## Task Commits

Each task was committed atomically:

1. **Task 1 (RED): Failing tests for portfolio fallback** - `84bda65` (test)
2. **Task 1 (GREEN): Portfolio cache + 70001 reconnect implementation** - `515f890` (feat)

## Files Created/Modified
- `src/finalayze/execution/tinkoff_broker.py` - Added _last_known_portfolio cache, _handle_70001_fallback method, consecutive error tracking with auto-reconnect
- `tests/unit/test_broker.py` - 8 new tests: TestPortfolioFallbackCacheOnSuccess (2), TestPortfolioFallback70001WithCache (2), TestPortfolioFallback70001NoCache (1), TestPortfolioFallbackNon70001 (1), TestPortfolioFallbackAutoReconnect (2)

## Decisions Made
- Cache portfolio in-memory on success, return stale on 70001 -- no threading locks needed since get_portfolio is always called from same APScheduler thread
- Auto-reconnect threshold set to 5 (matching plan spec) -- balances recovery speed vs reconnect overhead
- Non-70001 errors raise BrokerError unchanged -- fallback is specific to known intermittent T-Bank issue

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Known Stubs
None - all functionality fully wired.

## Next Phase Readiness
- Broker resilience for 70001 errors is complete
- Trading loop will now survive intermittent T-Bank portfolio fetch failures
- Bond broker client sync after reconnect is out of scope (noted in plan as future concern)

## Self-Check: PASSED

---
*Phase: 30-broker-resilience*
*Completed: 2026-03-30*
