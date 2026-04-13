---
phase: 25-data-validation-and-infrastructure
plan: 01
subsystem: data
tags: [moex-iss, data-normalizer, staleness-detection, trading-loop]

requires:
  - phase: 22-dependency-layer-cleanup
    provides: orchestration module extraction (trading_loop.py in orchestration/)
provides:
  - DataNormalizer wired into live trading loop before signal generation
  - Candle staleness detection with 48h threshold in _process_instrument
  - IMOEX volume uses correct column (share volume row[5], not turnover row[4])
affects: [live-trading, backtest-parity, moex-data]

tech-stack:
  added: []
  patterns:
    - "DataNormalizer validation gate before generate_signal in _process_instrument"
    - "Staleness threshold as module-level constant (_STALENESS_THRESHOLD_HOURS)"

key-files:
  created: [tests/unit/test_data_validation.py]
  modified: [src/finalayze/data/fetchers/moex_iss.py, src/finalayze/orchestration/trading_loop.py]

key-decisions:
  - "48h staleness threshold (2x daily timeframe) as module-level constant"
  - "DataNormalizer import at module level (Layer 2 into Layer 5 is valid)"

patterns-established:
  - "Data validation gate pattern: normalize -> staleness check -> signal generation"

requirements-completed: [DATA-01, DATA-02, DATA-03]

duration: 5min
completed: 2026-03-24
---

# Phase 25 Plan 01: Data Validation Wiring Summary

**DataNormalizer candle validation, 48h staleness detection, and IMOEX volume column fix wired into live trading loop**

## Performance

- **Duration:** 5 min
- **Started:** 2026-03-24
- **Completed:** 2026-03-24
- **Tasks:** 1
- **Files modified:** 3

## Accomplishments
- IMOEX volume field uses share volume (row[5]) instead of turnover (row[4]) -- DATA-03
- DataNormalizer.normalize_batch() validates all fetched candles before generate_signal() -- DATA-01
- Stale candle detection (>48h) skips instrument with warning log -- DATA-02
- 7 regression tests covering all three data validation requirements

## Task Commits

Each task was committed atomically:

1. **Task 1 (RED): Add failing tests for data validation wiring** - `5371122` (test)
2. **Task 1 (GREEN): Fix test setup for data validation wiring** - `e280363` (feat)

_Note: Implementation code (moex_iss.py, trading_loop.py) was already committed in the RED step since the fixes were straightforward. The GREEN step fixed test mock attribute mismatches._

## Files Created/Modified
- `tests/unit/test_data_validation.py` - 7 tests covering DATA-01/02/03 requirements
- `src/finalayze/data/fetchers/moex_iss.py` - IMOEX volume uses row[5] (share_volume)
- `src/finalayze/orchestration/trading_loop.py` - DataNormalizer wiring + staleness check in _process_instrument

## Decisions Made
- 48h staleness threshold chosen as 2x daily timeframe -- configurable via _STALENESS_THRESHOLD_HOURS constant
- DataNormalizer imported at module level (not TYPE_CHECKING) since it is Layer 2 and trading_loop is Layer 5

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed test mock attribute name mismatch**
- **Found during:** Task 1 GREEN step
- **Issue:** Tests used `_stop_loss_states` but actual TradingLoop uses `_stop_states`
- **Fix:** Changed attribute name in test mocks to `_stop_states`
- **Files modified:** tests/unit/test_data_validation.py
- **Verification:** All 7 tests pass
- **Committed in:** e280363

**2. [Rule 1 - Bug] Added missing mock attributes to staleness test setups**
- **Found during:** Task 1 GREEN step
- **Issue:** Staleness tests missing _stop_loss_lock, _stop_states, _sentiment_cache, _sentiment_lock, _cache attributes needed by _process_instrument code path
- **Fix:** Added all required mock attributes to both staleness test methods
- **Files modified:** tests/unit/test_data_validation.py
- **Verification:** All 7 tests pass
- **Committed in:** e280363

---

**Total deviations:** 2 auto-fixed (2 bugs in test setup)
**Impact on plan:** Both fixes necessary for tests to exercise the correct code path. No scope creep.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Data validation pipeline is complete for live trading loop
- All existing moex_iss and trading_loop tests continue to pass (19 total verified)

---
*Phase: 25-data-validation-and-infrastructure*
*Completed: 2026-03-24*

## Self-Check: PASSED
