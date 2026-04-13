---
phase: 11-advanced-strategies-and-ml
plan: 01
subsystem: strategies
tags: [pairs-trading, cointegration, kalman, moex, preferred-shares]

requires:
  - phase: 08-data-pipeline-moex
    provides: MOEX instrument registry with FIGIs
provides:
  - PairsStrategy allow_short parameter for long-only market gating
  - cointegration_start date filtering for post-structural-break validation
  - ru_blue_chips pairs config with SBER/SBERP, TATN/TATNP
  - TATNP FIGI for T-Invest data fetching
affects: [backtest-iteration, strategy-combiner, moex-live-trading]

tech-stack:
  added: []
  patterns:
    - "allow_short param pattern for long-only market constraints"
    - "cointegration_start date filtering to exclude structural breaks"

key-files:
  created: []
  modified:
    - src/finalayze/strategies/pairs.py
    - src/finalayze/strategies/presets/ru_blue_chips.yaml
    - src/finalayze/markets/instruments.py
    - tests/unit/test_pairs_strategy.py

key-decisions:
  - "allow_short=False suppresses SELL signals at _compute_signal level, not generate_signal"
  - "cointegration_start filters candles before cointegration test and spread computation"
  - "Weights rebalanced by reducing 4 strategies 0.03 each to free 0.12 for pairs"

patterns-established:
  - "Long-only market constraint via allow_short parameter in strategy params"
  - "Date-based data filtering via cointegration_start for structural break avoidance"

requirements-completed: [ADV-01]

duration: 4min
completed: 2026-03-20
---

# Phase 11 Plan 01: MOEX Preferred Share Pairs Trading Summary

**PairsStrategy wired for MOEX pref/ord arbitrage with allow_short gating and post-2022 cointegration filtering**

## Performance

- **Duration:** 4 min
- **Started:** 2026-03-20T20:26:02Z
- **Completed:** 2026-03-20T20:30:02Z
- **Tasks:** 1 (TDD: RED + GREEN)
- **Files modified:** 4

## Accomplishments
- PairsStrategy supports allow_short=False to suppress SELL signals on long-only MOEX market
- cointegration_start parameter restricts cointegration validation to post-2022 data only
- ru_blue_chips preset enabled with SBER/SBERP and TATN/TATNP pairs, z_entry=2.0, Kalman filter
- TATNP instrument now has FIGI (BBG004S68CP5) for T-Invest data fetching
- 5 new tests covering allow_short gating, config validation, and date filtering

## Task Commits

Each task was committed atomically:

1. **Task 1 (RED): Failing tests for allow_short and pairs config** - `132dce1` (test)
2. **Task 1 (GREEN): Implement allow_short, cointegration_start, TATNP FIGI, preset update** - `3c6df3c` (feat)

## Files Created/Modified
- `src/finalayze/strategies/pairs.py` - Added allow_short and cointegration_start params to _compute_signal
- `src/finalayze/strategies/presets/ru_blue_chips.yaml` - Enabled pairs with SBER/SBERP, TATN/TATNP, rebalanced weights
- `src/finalayze/markets/instruments.py` - Added FIGI BBG004S68CP5 to TATNP instrument
- `tests/unit/test_pairs_strategy.py` - 5 new tests for allow_short, config, and cointegration_start

## Decisions Made
- allow_short=False suppresses SELL at _compute_signal level (not generate_signal) for cleaner separation
- cointegration_start filters candles before both cointegration test and spread computation
- Weights rebalanced: momentum 0.12->0.09, mean_reversion 0.12->0.09, rsi2_connors 0.17->0.14, dual_momentum 0.17->0.14 to free 0.12 for pairs

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed test approach for allow_short tests**
- **Found during:** Task 1 (TDD GREEN)
- **Issue:** Tests used generate_signal with ru_blue_chips segment but AAPL/MSFT symbols not in SBER/SBERP pairs config
- **Fix:** Changed tests to call _compute_signal directly with allow_short parameter, decoupling from preset config
- **Files modified:** tests/unit/test_pairs_strategy.py
- **Verification:** All 24 pairs tests pass
- **Committed in:** 3c6df3c (Task 1 GREEN commit)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Test approach fix was necessary for correctness. No scope creep.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Pairs strategy ready for backtest iteration on ru_blue_chips segment
- Cointegration validation will use only post-2023 data as intended
- TATNP FIGI available for live T-Invest data fetching

## Self-Check: PASSED

- All 4 modified files exist on disk
- Commit 132dce1 (test RED) verified in git log
- Commit 3c6df3c (feat GREEN) verified in git log
- All 7 acceptance criteria grep checks pass
- All 24 pairs tests pass

---
*Phase: 11-advanced-strategies-and-ml*
*Completed: 2026-03-20*
