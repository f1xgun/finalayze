---
phase: 11-advanced-strategies-and-ml
plan: 02
subsystem: ml
tags: [moex, features, cbr, fx, brent, macro, ml-pipeline]

# Dependency graph
requires:
  - phase: 08-moex-data-sources
    provides: MoexMarketData schema, existing 4 MOEX features
provides:
  - 7 new MOEX ML feature columns (CBR rate/delta/direction, USDRUB return/vol, Brent return)
  - FEATURE_SCHEMA_VERSION 3
affects: [ml-training, backtest, strategy-tuning]

# Tech tracking
tech-stack:
  added: []
  patterns: [cbr-direction-one-hot, log-return-clipping, forward-fill-key-rates]

key-files:
  created: []
  modified:
    - src/finalayze/ml/features/technical.py
    - src/finalayze/ml/loader.py
    - tests/unit/test_features_moex.py

key-decisions:
  - "KeyRateRecord.rate is already decimal fraction (0.16=16%) -- no /100 normalization needed"
  - "CBR direction one-hot: cut=(1,0), hike=(0,1), hold=(0,0)"
  - "IMOEX relative strength already covered by cross-asset relative_strength_21d -- no duplicate"

patterns-established:
  - "CBR rate epsilon constant (_CBR_RATE_EPSILON) for float comparison"
  - "Log return clipping to [-0.15, 0.15] for FX/commodity returns"

requirements-completed: [ADV-02]

# Metrics
duration: 4min
completed: 2026-03-20
---

# Phase 11 Plan 02: MOEX ML Features Summary

**7 new MOEX ML feature columns (CBR rate level/delta/direction one-hot, USDRUB return/vol, Brent return) with 2-bar lag and schema version bump to 3**

## Performance

- **Duration:** 4 min
- **Started:** 2026-03-20T20:26:04Z
- **Completed:** 2026-03-20T20:30:20Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- Added 7 new MOEX-specific ML feature columns to compute_features() pipeline
- CBR monetary policy features: rate level, rate delta, direction one-hot encoding
- FX/commodity return features: USDRUB log return, USDRUB 20d rolling vol, Brent log return
- All features use 2-bar lag to avoid look-ahead bias with appropriate clipping
- Bumped FEATURE_SCHEMA_VERSION to 3 (old v2 models rejected at load time)

## Task Commits

Each task was committed atomically:

1. **Task 1: Add 7 CBR/FX/Brent feature columns** - `fc31a29` (test: RED), `58c6431` (feat: GREEN)
2. **Task 2: Bump FEATURE_SCHEMA_VERSION to 3** - `67efb8d` (chore)

_Note: Task 1 used TDD (test -> implementation commits)_

## Files Created/Modified
- `src/finalayze/ml/features/technical.py` - 3 new feature functions + wiring into compute_features()
- `src/finalayze/ml/loader.py` - FEATURE_SCHEMA_VERSION 2 -> 3
- `tests/unit/test_features_moex.py` - 24 new tests for CBR/FX/Brent features

## Decisions Made
- KeyRateRecord.rate is already a decimal fraction (0.16 = 16%), so no /100 normalization is needed (plan suggested dividing by 100, but that would double-normalize)
- IMOEX relative strength is already covered by cross-asset relative_strength_21d -- no duplicate feature needed
- CBR direction one-hot: cut=(1,0), hike=(0,1), hold=(0,0) via epsilon comparison

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Prevented double-normalization of CBR key rates**
- **Found during:** Task 1 (CBR feature implementation)
- **Issue:** Plan specified dividing rates by 100, but KeyRateRecord.rate is already decimal fraction (0.16 = 16% per schema docstring)
- **Fix:** Used rates as-is without /100 normalization
- **Files modified:** src/finalayze/ml/features/technical.py
- **Verification:** Tests pass with correct rate values
- **Committed in:** 58c6431

---

**Total deviations:** 1 auto-fixed (1 bug prevention)
**Impact on plan:** Essential correctness fix. No scope creep.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- 11 total MOEX-specific feature columns now available (4 existing + 7 new)
- Models must be retrained with schema v3 to use new features
- Ready for ML training and strategy tuning phases

---
*Phase: 11-advanced-strategies-and-ml*
*Completed: 2026-03-20*
