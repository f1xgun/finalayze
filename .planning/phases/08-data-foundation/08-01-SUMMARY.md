---
phase: 08-data-foundation
plan: 01
subsystem: config
tags: [moex, volatility, segments, presets, yaml]

# Dependency graph
requires: []
provides:
  - MOEX-calibrated vol_target (0.40) in all ru_* YAML presets
  - Cleaned MOEX universe without toxic symbols (GAZP, VTBR, SNGS, SNGSP, IRAO, ALRS)
  - Consistent min_combined_confidence (0.38) across ru_* presets
  - event_driven disabled in all ru_* presets
affects: [08-data-foundation, 09-moex-strategies, backtest-iteration]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "MOEX vol_target at 0.40 to match 35-45% annualized volatility"
    - "Toxic symbol exclusion based on negative PnL contribution analysis"

key-files:
  created: []
  modified:
    - src/finalayze/strategies/presets/ru_blue_chips.yaml
    - src/finalayze/strategies/presets/ru_energy.yaml
    - src/finalayze/strategies/presets/ru_finance.yaml
    - src/finalayze/strategies/presets/ru_tech.yaml
    - config/segments.py
    - tests/unit/test_config.py

key-decisions:
  - "vol_target 0.40 chosen to match MOEX blue chip annualized vol (35-45%)"
  - "Toxic symbols removed based on ~60% negative PnL contribution analysis"

patterns-established:
  - "MOEX presets use vol_target: 0.40 (not US-calibrated 0.19-0.22)"
  - "min_combined_confidence: 0.38 standardized across all ru_* segments"

requirements-completed: [DATA-01, DATA-02]

# Metrics
duration: 2min
completed: 2026-03-20
---

# Phase 8 Plan 01: MOEX Config Recalibration Summary

**MOEX vol_target raised to 0.40 and 5 toxic symbols removed from ru_* segments to fix position sizing and PnL drag**

## Performance

- **Duration:** 2 min
- **Started:** 2026-03-19T22:05:04Z
- **Completed:** 2026-03-19T22:07:30Z
- **Tasks:** 2
- **Files modified:** 6

## Accomplishments
- Recalibrated vol_target from 0.19-0.22 to 0.40 across all ru_* YAML presets to match MOEX 35-45% annualized volatility
- Removed 5 toxic symbols (GAZP, VTBR, SNGS/SNGSP, IRAO, ALRS) responsible for ~60% of negative MOEX PnL
- Standardized min_combined_confidence to 0.38 and disabled event_driven strategy (no real-time news feed)
- Added test_toxic_symbols_excluded_from_moex_segments validation test

## Task Commits

Each task was committed atomically:

1. **Task 1: Update ru_* YAML presets** - `893ecea` (feat)
2. **Task 2: Remove toxic symbols from MOEX segments** - `c77ee6e` (feat)

## Files Created/Modified
- `src/finalayze/strategies/presets/ru_blue_chips.yaml` - vol_target 0.19->0.40, confidence 0.15->0.38, event_driven disabled
- `src/finalayze/strategies/presets/ru_energy.yaml` - vol_target 0.22->0.40, confidence 0.15->0.38, event_driven disabled
- `src/finalayze/strategies/presets/ru_finance.yaml` - vol_target 0.21->0.40, confidence 0.15->0.38, event_driven disabled
- `src/finalayze/strategies/presets/ru_tech.yaml` - vol_target 0.20->0.40, confidence 0.30->0.38, event_driven disabled
- `config/segments.py` - Removed GAZP, VTBR, SNGS, SNGSP, IRAO, ALRS from respective segments
- `tests/unit/test_config.py` - Added toxic symbol exclusion test

## Decisions Made
- Used vol_target 0.40 to match MOEX blue chip annualized volatility (35-45%), preventing position sizing from crushing allocations to 25-42% of intended size
- Removed toxic symbols based on PnL contribution analysis rather than fundamental analysis

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- MOEX config foundation is ready for valid backtests
- Next plans in phase 08 can build on cleaned universe and calibrated parameters

## Self-Check: PASSED

All 6 files verified present. Both commit hashes (893ecea, c77ee6e) confirmed in git log.

---
*Phase: 08-data-foundation*
*Completed: 2026-03-20*
