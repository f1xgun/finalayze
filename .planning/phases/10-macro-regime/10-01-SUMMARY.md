---
phase: 10-macro-regime
plan: 01
subsystem: risk
tags: [cbr, yield-curve, position-sizing, moex, macro-regime]

requires:
  - phase: 09-strategy-wiring
    provides: BrentGateStep and RubOilRegimeStep in sizing pipeline
provides:
  - CBRRegimeStep scaling ru_* positions by yield curve slope
  - SectorAllocationStep scaling ru_energy by Brent and ru_finance by CBR direction
  - Static yield curve slope data (2022-2025) and CBR helper functions
affects: [10-macro-regime, backtest-iteration]

tech-stack:
  added: []
  patterns: [macro-regime sizing steps in pipeline, static yield curve data for backtesting]

key-files:
  created: []
  modified:
    - src/finalayze/data/fetchers/cbr.py
    - src/finalayze/risk/position_sizing_pipeline.py
    - src/finalayze/backtest/config.py
    - src/finalayze/backtest/engine.py
    - scripts/run_iteration.py
    - tests/unit/test_cbr_meeting_calendar.py
    - tests/unit/test_moex_sizing.py

key-decisions:
  - "Yield curve slope data is static dict keyed by YYYY-MM for backtest reproducibility"
  - "CBRRegimeStep inserted after BrentGate, before Copula/EVT/MetaLabel/HardCaps"
  - "SectorAllocationStep handles only ru_energy and ru_finance; other segments pass through"

patterns-established:
  - "Macro regime sizing steps follow same PositionSizingStep Protocol as existing steps"
  - "get_yield_slope_bps uses latest-key-lte-target lookup pattern"

requirements-completed: [MACRO-01, MACRO-03]

duration: 5min
completed: 2026-03-20
---

# Phase 10 Plan 01: CBR Regime and Sector Allocation Sizing Steps Summary

**CBRRegimeStep scales MOEX equity positions by yield curve slope tier (1.2x/1.0x/0.6x), SectorAllocationStep scales ru_energy by Brent-in-RUB and ru_finance by CBR direction**

## Performance

- **Duration:** 5 min
- **Started:** 2026-03-20T09:11:48Z
- **Completed:** 2026-03-20T09:17:00Z
- **Tasks:** 2
- **Files modified:** 7

## Accomplishments
- Added static yield curve slope data (17 monthly data points, 2022-2025) and 3 CBR helper functions
- CBRRegimeStep: steepening (>100bps) -> 1.2x, flat (0-100bps) -> 1.0x, inverted (<0bps) -> 0.6x for ru_* segments
- SectorAllocationStep: ru_energy Brent thresholds (>6000->1.3x, <4000->0.7x), ru_finance CBR direction (cut->1.2x, hike->0.8x)
- Both steps wired into engine pipeline after BrentGate, before Copula/EVT/MetaLabel/HardCaps
- 62 tests pass including 22 new tests covering all scaling and passthrough cases

## Task Commits

Each task was committed atomically:

1. **Task 1: CBR helpers + CBRRegimeStep + SectorAllocationStep (TDD)** - `54a21de` (test: RED), `1425745` (feat: GREEN)
2. **Task 2: Wire into backtest engine and run_iteration.py** - `eb36da9` (feat)

_Note: Task 1 used TDD with separate RED and GREEN commits._

## Files Created/Modified
- `src/finalayze/data/fetchers/cbr.py` - Added _YIELD_CURVE_SLOPE_BPS, get_recent_cbr_decisions, is_cutting_cycle, get_yield_slope_bps
- `src/finalayze/risk/position_sizing_pipeline.py` - Added CBRRegimeStep and SectorAllocationStep classes
- `src/finalayze/backtest/config.py` - Added yield_slope_bps and cbr_direction fields to BacktestConfig
- `src/finalayze/backtest/engine.py` - Wired CBRRegimeStep and SectorAllocationStep into _build_sizing_pipeline
- `scripts/run_iteration.py` - Compute yield slope and CBR direction, pass to BacktestConfig
- `tests/unit/test_cbr_meeting_calendar.py` - 12 new tests for CBR helpers and yield slope
- `tests/unit/test_moex_sizing.py` - 13 new tests for CBRRegimeStep and SectorAllocationStep

## Decisions Made
- Yield curve slope data stored as static dict for backtest reproducibility (no live API calls during backtest)
- CBRRegimeStep activates for all ru_* segments OR when yield_slope_bps is non-zero
- SectorAllocationStep only activates when cbr_direction is non-empty
- run_iteration.py computes yield slope and CBR direction using today's date (live mode)

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- CBRRegimeStep and SectorAllocationStep are ready for backtest iteration validation
- Plan 10-02 can proceed with additional macro regime features
- Blocker resolved: OFZ yield curve slope data source now available as static data

---
*Phase: 10-macro-regime*
*Completed: 2026-03-20*
