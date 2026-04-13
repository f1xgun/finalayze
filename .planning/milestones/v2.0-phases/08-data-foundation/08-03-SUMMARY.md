---
phase: 08-data-foundation
plan: 03
subsystem: data
tags: [dividends, moex, tinkoff-api, yaml, dividend-gap, look-ahead-bias]

requires:
  - phase: 08-data-foundation
    provides: "MOEX config recalibration (vol_target, toxic symbols)"
provides:
  - "Expanded MOEX dividend calendar (262 events, 38 symbols) with status field"
  - "DividendEntry status field (paid/cancelled/reduced) with backward-compatible default"
  - "DividendGapStrategy skips cancelled/reduced dividends in signal generation"
  - "Batch fetch script for T-Invest API dividend data"
  - "Calendar validation tests (8 tests)"
affects: [dividend_gap, backtest, strategies]

tech-stack:
  added: []
  patterns: ["status field on DividendEntry for look-ahead bias elimination"]

key-files:
  created:
    - scripts/fetch_moex_dividends.py
    - tests/unit/test_dividend_calendar.py
  modified:
    - src/finalayze/strategies/dividend_gap.py
    - src/finalayze/strategies/presets/moex_dividends.yaml

key-decisions:
  - "T-Invest API does not distinguish cancelled dividends -- manual overrides required for known events"
  - "DividendEntry status defaults to 'paid' for backward compatibility"
  - "Only 'paid' dividends trigger BUY signals -- cancelled and reduced are skipped"

patterns-established:
  - "Manual override pattern for API-unavailable data corrections in fetch scripts"
  - "Status field filtering in signal generation to prevent look-ahead bias"

requirements-completed: [DATA-03]

duration: 3min
completed: 2026-03-20
---

# Phase 8 Plan 3: MOEX Dividend Calendar Expansion Summary

**262 dividend events across 38 MOEX symbols with status field eliminating look-ahead bias from cancelled dividends (GAZP 2022)**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-20T07:54:02Z
- **Completed:** 2026-03-20T07:57:00Z
- **Tasks:** 3
- **Files modified:** 4

## Accomplishments

- Expanded MOEX dividend calendar from 43 events / 6 symbols to 262 events / 38 symbols
- Added `status` field to DividendEntry (paid/cancelled/reduced) with backward-compatible default
- DividendGapStrategy now skips cancelled/reduced dividends, eliminating look-ahead bias
- GAZP 2022 cancelled dividend (52.53 RUB) correctly marked with status: cancelled
- Reusable batch fetch script using T-Invest API with manual override mechanism

## Task Commits

Each task was committed atomically:

1. **Task 1: Create dividend batch fetch script** - `89d4d02` (feat)
2. **Task 2: Run dividend fetch script** - checkpoint (user ran script: 262 events, 38 symbols)
3. **Task 3: Add DividendEntry status field and calendar validation tests** (TDD)
   - RED: `22f29a9` (test: add failing tests)
   - GREEN: `7cd017d` (feat: add status field, skip cancelled dividends, fix fetch script)

## Files Created/Modified

- `scripts/fetch_moex_dividends.py` - Batch fetch MOEX dividends from T-Invest API with manual overrides
- `src/finalayze/strategies/dividend_gap.py` - DividendEntry with status field, generate_signal skips non-paid
- `src/finalayze/strategies/presets/moex_dividends.yaml` - 262 events across 38 symbols with status field
- `tests/unit/test_dividend_calendar.py` - 8 tests: status field, signal skip, calendar validation

## Decisions Made

- T-Invest API does not provide a dividend_type field distinguishing cancelled dividends -- all API-fetched entries marked as "paid", known cancelled events applied via manual overrides
- DividendEntry.status defaults to "paid" for backward compatibility with existing code
- Only "paid" dividends trigger BUY signals -- cancelled and reduced dividends are skipped in generate_signal

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed fetch script id_type parameter**
- **Found during:** Task 2 (checkpoint - user reported fix needed)
- **Issue:** `id_type` was integer `1` instead of `InstrumentIdType.INSTRUMENT_ID_TYPE_TICKER` enum value
- **Fix:** Changed to use proper SDK enum
- **Files modified:** scripts/fetch_moex_dividends.py
- **Verification:** Script ran successfully, fetched 262 events
- **Committed in:** 7cd017d (Task 3 commit)

**2. [Rule 3 - Blocking] Added timeframe field to test Candle construction**
- **Found during:** Task 3 (TDD GREEN)
- **Issue:** Candle schema requires `timeframe` field, test helper was missing it
- **Fix:** Added `timeframe="1d"` to test helper
- **Files modified:** tests/unit/test_dividend_calendar.py
- **Committed in:** 7cd017d (Task 3 commit)

---

**Total deviations:** 2 auto-fixed (1 bug, 1 blocking)
**Impact on plan:** Both fixes necessary for correctness. No scope creep.

## Issues Encountered

- TCSG ticker not found via T-Invest API (rebranded to T-Technology) -- not critical, excluded from calendar

## User Setup Required

None - dividend calendar is pre-populated via fetch script. Re-running requires FINALAYZE_TINKOFF_TOKEN.

## Next Phase Readiness

- Dividend calendar ready for backtesting with look-ahead bias eliminated
- DividendGapStrategy correctly handles cancelled dividends
- Calendar can be refreshed by re-running `scripts/fetch_moex_dividends.py` with valid token

---
*Phase: 08-data-foundation*
*Completed: 2026-03-20*
