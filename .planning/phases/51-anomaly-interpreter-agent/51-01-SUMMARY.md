---
phase: 51-anomaly-interpreter-agent
plan: 01
subsystem: analysis
tags: [pydantic, statistics, anomaly-detection, z-score, tdd]

# Dependency graph
requires: []
provides:
  - AnomalyDetector class with rolling z-score price anomaly detection
  - AnomalyResult frozen Pydantic schema for anomaly data
  - Volume spike detection via rolling mean ratio
affects: [51-02, anomaly-interpreter-agent]

# Tech tracking
tech-stack:
  added: []
  patterns: [rolling-z-score, volume-ratio-threshold, pure-computation-layer3]

key-files:
  created:
    - src/finalayze/analysis/anomaly_detector.py
    - tests/unit/test_anomaly_detector.py
  modified: []

key-decisions:
  - "Pure computation module with no IO, no LLM, no alerting -- clean Layer 3"
  - "Configurable thresholds via constructor (sigma=3.0, window=20, vol_ratio=2.0)"
  - "Zero std deviation returns None instead of raising ZeroDivisionError"

patterns-established:
  - "AnomalyDetector pattern: stateless detector with check() returning Optional[Result]"
  - "TYPE_CHECKING import for Candle to avoid circular Layer 0 dependency"

requirements-completed: [ANMI-01, ANMI-03]

# Metrics
duration: 2min
completed: 2026-04-15
---

# Phase 51 Plan 01: Anomaly Detector Summary

**Statistical anomaly detector with rolling 20-bar z-score for >3-sigma price moves and >2x volume spikes, TDD-driven with 8 unit tests**

## Performance

- **Duration:** 2 min 30s
- **Started:** 2026-04-15T08:20:21Z
- **Completed:** 2026-04-15T08:22:51Z
- **Tasks:** 1 (TDD: RED + GREEN)
- **Files modified:** 2

## Accomplishments
- AnomalyDetector class with check() method detecting price and volume anomalies
- AnomalyResult frozen Pydantic v2 schema with all required fields
- 8 unit tests covering all detection paths including edge cases
- Full verification: ruff check clean, ruff format clean, mypy strict clean

## Task Commits

Each task was committed atomically:

1. **Task 1 RED: Failing tests for AnomalyDetector** - `66f33de` (test)
2. **Task 1 GREEN: Implement AnomalyDetector** - `65ab315` (feat)

## Files Created/Modified
- `src/finalayze/analysis/anomaly_detector.py` - AnomalyDetector class + AnomalyResult schema, pure computation Layer 3
- `tests/unit/test_anomaly_detector.py` - 8 unit tests: insufficient data, price anomaly, volume anomaly, both, normal range, zero std, frozen model

## Decisions Made
- Used statistics.stdev for z-score calculation (stdlib, no external dependency needed)
- TYPE_CHECKING guard for Candle import to keep Layer 3 clean
- Configurable thresholds via constructor parameters for flexibility in Plan 02 wiring

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed ruff B017/PT011 lint error in test**
- **Found during:** Task 1 GREEN phase verification
- **Issue:** `pytest.raises(Exception)` too broad per ruff rules B017 and PT011
- **Fix:** Changed to `pytest.raises(ValidationError)` with explicit pydantic import
- **Files modified:** tests/unit/test_anomaly_detector.py
- **Verification:** ruff check passes clean
- **Committed in:** 65ab315 (part of GREEN commit)

---

**Total deviations:** 1 auto-fixed (1 bug fix)
**Impact on plan:** Minor lint fix, no scope creep.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- AnomalyDetector ready for Plan 02 wiring into TradingLoop
- AnomalyResult schema provides the structured data needed by the interpreter agent
- No blockers or concerns

---
*Phase: 51-anomaly-interpreter-agent*
*Completed: 2026-04-15*
