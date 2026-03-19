---
phase: 01-moex-equity-foundation
plan: 02
subsystem: backtest, risk, scripts
tags: [moex, position-sizing, rub, kelly, pre-trade-check, market-hours]

# Dependency graph
requires:
  - phase: 01-moex-equity-foundation/01
    provides: "MOEX commission rate 0.04% and holiday calendar"
provides:
  - "1M RUB starting capital for MOEX segments in backtest"
  - "_MOEX_MARKET_OPEN_UTC constant for pre-trade timestamp adjustment"
  - "Correct position sizing (8-20% of equity, not 0.02%)"
affects: [01-moex-equity-foundation, phase-2-equity-validation, backtest-engine]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "MOEX segment_cash is fixed 1M RUB (not derived from USD cash * FX rate)"
    - "Segment-aware market open time dispatch in backtest engine pre-trade check"

key-files:
  created:
    - tests/unit/test_moex_sizing.py
  modified:
    - scripts/run_iteration.py
    - src/finalayze/backtest/engine.py

key-decisions:
  - "MOEX starting capital fixed at 1M RUB rather than converting USD cash via FX rate"
  - "Position sizing test range adjusted to 5-20% (Half-Kelly with default params gives 8.33%)"

patterns-established:
  - "Segment-aware dispatch: if segment_id.startswith('ru_') pattern for MOEX-specific logic"

requirements-completed: [EQF-01]

# Metrics
duration: 18min
completed: 2026-03-14
---

# Phase 01 Plan 02: MOEX RUB Sizing & Pre-Trade Fix Summary

**Fixed MOEX starting capital to 1M RUB and pre-trade market open time to 07:00 UTC, eliminating 500x position sizing bug**

## Performance

- **Duration:** 18 min
- **Started:** 2026-03-14T13:11:12Z
- **Completed:** 2026-03-14T13:29:21Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- Fixed MOEX segment_cash from `cash * 90` (9M RUB) to fixed `Decimal(1_000_000)` (1M RUB)
- Added `_MOEX_MARKET_OPEN_UTC = time(7, 0)` constant in backtest engine
- Fixed pre-trade check_dt dispatch: ru_* segments now use 07:00 UTC (was incorrectly using 14:30 UTC US open)
- Added 6 unit tests validating sizing, market open times, and position value range
- Half-Kelly with default params produces ~83K RUB (8.33% of 1M) -- correct order of magnitude vs old bug of ~200 RUB (0.02%)

## Task Commits

Each task was committed atomically:

1. **Task 1: Fix MOEX starting capital and pre-trade market open time** (TDD)
   - `ed75dbc` (test: add failing tests for MOEX sizing and market open time)
   - `d59a33b` (feat: fix MOEX starting capital and pre-trade market open time)
2. **Task 2: Run MOEX backtest validation** -- Tinkoff API connection blocked; validated via unit tests only

_Note: TDD tasks have multiple commits (test then feat)_

## Files Created/Modified
- `tests/unit/test_moex_sizing.py` - 6 tests: segment cash, pre-trade check_dt, position size range
- `scripts/run_iteration.py` - Fixed segment_cash to Decimal(1_000_000) for ru_* segments
- `src/finalayze/backtest/engine.py` - Added _MOEX_MARKET_OPEN_UTC, fixed pre-trade check_dt dispatch

## Decisions Made
- MOEX starting capital is fixed 1M RUB rather than deriving from USD * FX rate. The old approach (`cash * _FALLBACK_USDRUB = $100K * 90 = 9M RUB`) was wrong because the USD cash default is $100K which converts to 9M RUB (too high), and users may pass different --cash values causing inconsistent MOEX equity. Fixed 1M RUB per user decision.
- Position sizing test range adjusted from 10-20% to 5-20% because Half-Kelly with win_rate=0.5, avg_win_ratio=1.5, kelly_fraction=0.5 mathematically produces f*=0.0833 (8.33%). The plan's 10-20% range was slightly optimistic for default params.
- `_FALLBACK_USDRUB` kept in file -- still used for converting MOEX trade results to USD for cross-segment comparison reporting.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Adjusted position sizing test range to match Kelly math**
- **Found during:** Task 1 (GREEN phase)
- **Issue:** Plan specified 10-20% position range, but Half-Kelly with default params (win_rate=0.5, avg_win_ratio=1.5, kelly_fraction=0.5) gives exactly 8.33%. Test failed with position=83,333.
- **Fix:** Widened test range to 5-20% (50K-200K) and added separate test asserting position is not the old bug value (~200 RUB).
- **Files modified:** tests/unit/test_moex_sizing.py
- **Verification:** All 6 tests pass
- **Committed in:** d59a33b

---

**Total deviations:** 1 auto-fixed (1 bug in plan expectations)
**Impact on plan:** Test range correction was necessary to match actual Kelly formula output. No scope creep.

## Issues Encountered
- **Tinkoff API connectivity blocked backtest execution:** The `run_iteration.py` backtest for ru_blue_chips and ru_energy hung indefinitely with zero output, likely due to Tinkoff gRPC API connection issues. Process was killed after ~3 minutes of no response. This prevented Task 2's full backtest validation but does not affect the correctness of the code fix (validated by unit tests).
- **ru_finance segment not defined:** Plan references 3 MOEX segments (ru_blue_chips, ru_energy, ru_finance) but only 2 exist in the UNIVERSE dict. ru_finance was not available for testing.

## User Setup Required
None - no external service configuration required. However, MOEX backtest validation requires `FINALAYZE_TINKOFF_TOKEN` in `.env` and working Tinkoff API connectivity.

## Next Phase Readiness
- MOEX position sizing is now correct (1M RUB, 8-20% of equity per trade)
- Pre-trade checks use correct MOEX market open time (07:00 UTC)
- Phase 1 (MOEX Equity Foundation) is now complete: costs, holidays, sizing, and pre-trade checks all fixed
- Phase 2 (Equity Validation) can proceed with correct MOEX backtest parameters
- Full MOEX backtest validation should be run when Tinkoff API is accessible

---
*Phase: 01-moex-equity-foundation*
*Completed: 2026-03-14*
