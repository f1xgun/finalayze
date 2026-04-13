---
phase: 25-data-validation-and-infrastructure
plan: 02
subsystem: data
tags: [grpc, tinkoff, caching, persistent-channel, asyncio]

requires:
  - phase: 25-data-validation-and-infrastructure
    provides: data validation wiring (plan 01)
provides:
  - Persistent gRPC channel for all TinkoffFetcher methods (not just candles)
  - Brent crude caching via GenericFileCache in MarketDataLoader
affects: [backtest, execution, data]

tech-stack:
  added: []
  patterns: [persistent background event loop for gRPC, _cached_fetch for commodity data]

key-files:
  created: []
  modified:
    - src/finalayze/data/fetchers/tinkoff_data.py
    - src/finalayze/data/loader.py
    - tests/unit/test_tinkoff_persistent_client.py
    - tests/unit/test_tinkoff_data.py
    - tests/unit/test_market_data_loader.py

key-decisions:
  - "Bond async methods refactored to use _get_services_async instead of creating per-call clients"
  - "Brent cache uses Candle model class and cache_id BZ_F (underscore replaces equals sign)"

patterns-established:
  - "All TinkoffFetcher methods use _run_async + _get_services_async for persistent gRPC channel"
  - "Commodity data (Brent) cached via GenericFileCache like FX and turnover data"

requirements-completed: [INFRA-01, INFRA-02]

duration: 8min
completed: 2026-03-24
---

# Phase 25 Plan 02: Persistent gRPC Channel and Brent Caching Summary

**Persistent gRPC channel for all TinkoffFetcher bond methods and Brent crude caching via GenericFileCache**

## Performance

- **Duration:** 8 min
- **Started:** 2026-03-24T07:29:35Z
- **Completed:** 2026-03-24T07:37:22Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments
- All TinkoffFetcher methods (bond candles, bond info, coupons, accrued interest, amortization) now use persistent gRPC channel via _run_async and _get_services_async
- Brent crude candles cached via _cached_fetch in MarketDataLoader with optional brent_cache parameter
- Updated 45 tests to work with persistent channel pattern (replaced asyncio.run mocks with _run_async mocks)

## Task Commits

Each task was committed atomically:

1. **Task 1: Add persistent gRPC channel to TinkoffFetcher**
   - `292f811` (test: failing tests for bond methods persistent channel)
   - `78e6158` (feat: persistent gRPC channel for all TinkoffFetcher methods)

2. **Task 2: Cache Brent crude candles in MarketDataLoader**
   - `8e780f7` (test: failing tests for Brent caching)
   - `668e9e8` (feat: cache Brent crude candles via _cached_fetch)

## Files Created/Modified
- `src/finalayze/data/fetchers/tinkoff_data.py` - Refactored bond methods to use persistent channel pattern
- `src/finalayze/data/loader.py` - Added brent_cache parameter and _cached_fetch for Brent
- `tests/unit/test_tinkoff_persistent_client.py` - Added tests for bond methods persistent channel
- `tests/unit/test_tinkoff_data.py` - Updated mocks from asyncio.run to _run_async pattern
- `tests/unit/test_market_data_loader.py` - Added tests for Brent caching behavior

## Decisions Made
- Bond async methods refactored to use _get_services_async (shared persistent services) instead of creating per-call clients via _make_client
- Brent cache uses Candle as model_class and "BZ_F" as cache_id (underscore replaces equals sign from ticker "BZ=F")
- Test mocks updated from patching asyncio.run to patching instance _run_async method

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Updated test mocks from asyncio.run to _run_async**
- **Found during:** Task 1 (persistent gRPC channel)
- **Issue:** Existing tests in test_tinkoff_data.py mocked asyncio.run, but fetch_candles already uses _run_async, causing tests to hit real gRPC endpoint
- **Fix:** Changed all asyncio.run mocks to patch.object(fetcher, "_run_async", ...) pattern
- **Files modified:** tests/unit/test_tinkoff_data.py
- **Verification:** All 45 tests pass
- **Committed in:** 78e6158 (part of Task 1 commit)

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Fix necessary to make tests work with persistent channel pattern. No scope creep.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- TinkoffFetcher persistent channel complete -- all methods reuse single gRPC connection
- Brent caching ready -- callers can pass GenericFileCache for repeated backtest runs
- brent_cache parameter is optional (default None) so no breaking changes to existing callers

---
*Phase: 25-data-validation-and-infrastructure*
*Completed: 2026-03-24*
