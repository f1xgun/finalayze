---
phase: 08-data-foundation
plan: 02
subsystem: backtest
tags: [atr, volatility, moex, structural-break, stop-loss, backtest-config]

# Dependency graph
requires:
  - phase: 08-data-foundation/01
    provides: "MOEX segment recalibration and toxic symbol removal"
provides:
  - "BacktestConfig.exclude_periods field for date range exclusion"
  - "MOEX_2022_BREAK constant for Feb 21 - Apr 1 2022 structural break"
  - "filter_candles_by_exclusion() reusable helper in stop_loss.py"
  - "ATR and chandelier stop computations filter excluded periods"
  - "run_iteration.py and run_strategy_isolation.py pass exclusion for ru_* segments"
affects: [backtest, risk, strategies, moex-profitability]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Date range exclusion via tuple[tuple[str, str], ...] for structural break handling"
    - "filter_candles_by_exclusion as reusable candle filter before vol/ATR computation"

key-files:
  created: []
  modified:
    - src/finalayze/backtest/config.py
    - src/finalayze/risk/stop_loss.py
    - src/finalayze/backtest/engine.py
    - scripts/run_iteration.py
    - scripts/run_strategy_isolation.py
    - tests/unit/test_backtest_engine.py

key-decisions:
  - "exclude_periods stored as tuple of string pairs for JSON serializability and frozen dataclass compatibility"
  - "filter_candles_by_exclusion placed in stop_loss.py (Layer 4) for reuse by both ATR and chandelier computations"
  - "Exclusion applied to ATR computation and chandelier stop updates but OHLCV data preserved for position tracking"

patterns-established:
  - "Structural break exclusion: pass exclude_periods to BacktestConfig, filtered in ATR/vol computations"
  - "MOEX-specific constants (MOEX_2022_BREAK) in backtest/config.py alongside other MOEX-aware logic"

requirements-completed: [DATA-04]

# Metrics
duration: 6min
completed: 2026-03-20
---

# Phase 08 Plan 02: MOEX 2022 Structural Break Exclusion Summary

**Exclude Feb 21 - Apr 1 2022 MOEX closure from vol/ATR calculations via BacktestConfig.exclude_periods, preventing 3-5x vol distortion**

## Performance

- **Duration:** 6 min
- **Started:** 2026-03-19T22:05:31Z
- **Completed:** 2026-03-19T22:11:39Z
- **Tasks:** 2
- **Files modified:** 6

## Accomplishments
- BacktestConfig now has exclude_periods field with MOEX_2022_BREAK constant for the Feb-Apr 2022 structural break
- ATR and chandelier stop computations filter out candles in excluded date ranges, preventing inflated volatility estimates
- Both run scripts (run_iteration.py, run_strategy_isolation.py) automatically apply exclusion for ru_* segments
- 5 new tests validate exclusion behavior; all 13 backtest engine tests pass (backward compatible)

## Task Commits

Each task was committed atomically:

1. **Task 1 (RED): Add failing tests** - `5fa198f` (test)
2. **Task 1 (GREEN): Implement exclude_periods and filtering** - `bbb3090` (feat)
3. **Task 2: Wire MOEX_2022_BREAK into run scripts** - `d3d7986` (feat)

_TDD approach: failing tests committed first, then implementation._

## Files Created/Modified
- `src/finalayze/backtest/config.py` - Added exclude_periods field and MOEX_2022_BREAK constant
- `src/finalayze/risk/stop_loss.py` - Added filter_candles_by_exclusion() and exclude_periods param to compute_atr_stop_loss()
- `src/finalayze/backtest/engine.py` - Passes exclude_periods to ATR/chandelier stop computations
- `scripts/run_iteration.py` - Imports MOEX_2022_BREAK, passes to BacktestConfig for ru_* segments
- `scripts/run_strategy_isolation.py` - Imports MOEX_2022_BREAK, passes to BacktestConfig for ru_* segments
- `tests/unit/test_backtest_engine.py` - 5 new tests for exclusion behavior

## Decisions Made
- Used tuple of string pairs for exclude_periods (JSON serializable, frozen dataclass compatible)
- Placed filter_candles_by_exclusion in stop_loss.py for reuse by engine's chandelier logic
- Candles during excluded period remain in OHLCV data for position tracking -- only vol/ATR windows skip them

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
- Test data initially had extreme candles outside the ATR window (last 15 candles were all normal), causing the ATR-with-exclusion test to fail with equal values. Fixed by restructuring test data so extreme candles fall within the ATR computation window.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Structural break exclusion ready for MOEX backtests
- Vol/ATR estimates will be accurate for ru_* segments going forward
- No blockers for subsequent phases

---
*Phase: 08-data-foundation*
*Completed: 2026-03-20*
