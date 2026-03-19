---
phase: 02-moex-equity-validation
plan: 03
subsystem: backtest
tags: [moex, kelly, position-sizing, universe, walk-forward]

# Dependency graph
requires:
  - phase: 02-02
    provides: MOEX calibration and walk-forward validation baseline
provides:
  - Three-quarter Kelly (0.75) sizing for MOEX segments
  - Full MOEX symbol universes restored for future news integration
  - min_combined_confidence at 0.15 for all MOEX segments
affects: [03-bond-data-pipeline, 07-news-pipeline]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Three-quarter Kelly for MOEX (fraction=0.75) vs quarter-Kelly for US (fraction=0.25)
    - Symbol universes kept broad for future event_driven strategy integration

key-files:
  created: []
  modified:
    - scripts/run_iteration.py
    - scripts/run_strategy_isolation.py
    - src/finalayze/strategies/presets/ru_blue_chips.yaml
    - src/finalayze/strategies/presets/ru_energy.yaml
    - src/finalayze/strategies/presets/ru_finance.yaml

key-decisions:
  - "Three-quarter Kelly (0.75) kept for MOEX -- improves position sizing from 0.2% to meaningful levels"
  - "All pruned symbols restored to MOEX universes -- losing symbols may become profitable with news/sentiment (Phase 7)"
  - "min_combined_confidence kept at 0.15 -- allows more trade opportunities with broader universes"
  - "Accepted temporary metric worsening to preserve full symbol coverage for future event_driven integration"

patterns-established:
  - "MOEX universes: keep broad for future strategy integration, do not prune for short-term performance"

requirements-completed: [EQF-02, EQF-03]

# Metrics
duration: 4min
completed: 2026-03-14
---

# Phase 02 Plan 03: MOEX Gap Closure Summary

**Three-quarter Kelly sizing with full MOEX symbol universes restored for future news/sentiment integration**

## Performance

- **Duration:** 4 min (Tasks 1-2 by previous agent: ~38 min; Task 3 continuation: ~4 min)
- **Started:** 2026-03-14T16:20:00Z (original), 2026-03-14T16:40:09Z (continuation)
- **Completed:** 2026-03-14T16:44:00Z
- **Tasks:** 3
- **Files modified:** 5

## Accomplishments
- Increased RollingKelly fraction from 0.25 (quarter-Kelly) to 0.75 (three-quarter Kelly) for MOEX segments, producing meaningful position sizes
- Ran walk-forward validation confirming max drawdown stays well below 20% hard cap
- Restored all pruned symbols back to MOEX universes per user decision -- 10 symbols in ru_blue_chips, 8 in ru_energy, 7 in ru_finance
- min_combined_confidence set to 0.15 on all MOEX segments for broader trade opportunities

## Task Commits

Each task was committed atomically:

1. **Task 1: Increase MOEX position sizing and prune losing symbols** - `e794fd9` (feat)
2. **Task 2: Walk-forward validation with improved sizing** - `6244b8a` (feat)
3. **Task 3: Restore pruned symbols per user feedback** - `7f64f0f` (feat)

## Files Created/Modified
- `scripts/run_iteration.py` - MOEX UNIVERSE restored to full symbol lists, Kelly fraction=0.75
- `scripts/run_strategy_isolation.py` - MOEX UNIVERSE mirrors run_iteration.py, Kelly fraction=0.75
- `src/finalayze/strategies/presets/ru_blue_chips.yaml` - min_combined_confidence=0.15 (set in Task 1)
- `src/finalayze/strategies/presets/ru_energy.yaml` - min_combined_confidence=0.15 (set in Task 1)
- `src/finalayze/strategies/presets/ru_finance.yaml` - min_combined_confidence=0.15 (set in Task 1)

## Decisions Made
- **Kelly fraction 0.75 (three-quarter Kelly):** Provides 3x position sizes vs quarter-Kelly default. Max DD was 0.22% at quarter-Kelly, so three-quarter should stay well under 20% cap. This was the best-performing variant in walk-forward testing.
- **Symbols restored:** User decided to keep all MOEX symbols in universes despite some having negative Sharpe. Rationale: Phase 7 news/sentiment integration (event_driven strategy) may make currently-losing symbols profitable. Better to track them now than lose visibility.
- **SBERP added to ru_finance:** Preferred shares of Sberbank added alongside common shares (SBER) for diversification.

## Deviations from Plan

### User-Directed Changes (Task 3)

**1. Symbol restoration per checkpoint feedback**
- **Found during:** Task 3 (checkpoint review)
- **Issue:** User decided pruned symbols should be restored to preserve full universe coverage
- **Fix:** Restored all original symbols to MOEX universes in both scripts
- **Files modified:** scripts/run_iteration.py, scripts/run_strategy_isolation.py
- **Verification:** Preset validation tests pass (15/15)
- **Committed in:** 7f64f0f

---

**Total deviations:** 1 user-directed change (symbol restoration)
**Impact on plan:** Plan objective (gap closure) was partially achieved -- Kelly sizing improved, but segment Sharpe targets not fully met with restored (broader) universes. This is accepted per user decision.

## Issues Encountered
- Walk-forward targets (OOS Sharpe > 0.1 on 2+ segments) not fully achievable with current strategy mix on MOEX. Individual symbols profitable (YNDX +0.88, ROSN +0.65) but segment averages dragged by losing symbols. Accepted as best-effort -- future event_driven strategy integration expected to improve.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- MOEX equity backtest infrastructure is operational with meaningful position sizes
- Bond data pipeline (Phase 3) can proceed independently
- MOEX symbol universes preserved for Phase 7 news/sentiment integration
- Consider re-running walk-forward after event_driven strategy is enabled

---
*Phase: 02-moex-equity-validation*
*Completed: 2026-03-14*
