---
phase: 32-critical-sandbox-fixes
plan: 04
subsystem: ml
tags: [calibration, brier-score, walk-forward, ensemble, isotonic-regression]

requires:
  - phase: 32-03
    provides: "profit_factor wiring in _evaluate_fold_metrics"
provides:
  - "EnsembleCalibrator.predict_proba batch method for array calibration"
  - "Per-fold calibrator fitting in walk-forward loop using cal_idx data"
  - "Calibrated Brier score in walk-forward fold evaluation"
affects: [ml-training, sandbox-validation]

tech-stack:
  added: []
  patterns: ["Per-fold calibrator fitting on calibration split before evaluation"]

key-files:
  created: []
  modified:
    - src/finalayze/ml/calibration.py
    - scripts/train_models.py
    - tests/unit/test_ml_quality_gates.py

key-decisions:
  - "predict_proba returns raw copy when unfitted (safe fallback, no silent errors)"
  - "Per-fold calibrator uses same predict_proba pattern as _fit_and_save_calibrator"

patterns-established:
  - "Batch calibration via predict_proba for array inputs; single-value via calibrate()"

requirements-completed: [SANDBOX-FIX-10]

duration: 2min
completed: 2026-04-07
---

# Phase 32 Plan 04: Calibrated Brier Walk-Forward Wiring Summary

**Per-fold EnsembleCalibrator fitted on cal_idx and wired to _evaluate_fold_metrics so walk-forward Brier score uses calibrated probabilities**

## Performance

- **Duration:** 2 min
- **Started:** 2026-04-07T19:12:13Z
- **Completed:** 2026-04-07T19:14:14Z
- **Tasks:** 1
- **Files modified:** 3

## Accomplishments
- Added `predict_proba` batch method to `EnsembleCalibrator` (vectorized calibration for arrays)
- Wired per-fold calibrator fitting on `cal_f`/`cal_l` data inside walk-forward loop
- `_evaluate_fold_metrics` now receives `fold_calibrator` and uses calibrated probabilities for Brier score
- Added `cal_l` extraction (calibration labels) that was missing from walk-forward data preparation
- 4 new tests covering predict_proba (fitted, unfitted, isotonic) and wiring integration

## Task Commits

Each task was committed atomically:

1. **Task 1 (RED): Add failing tests for predict_proba and wiring** - `57de63c` (test)
2. **Task 1 (GREEN): Wire per-fold calibrator into walk-forward Brier evaluation** - `263c2e3` (feat)

## Files Created/Modified
- `src/finalayze/ml/calibration.py` - Added `predict_proba(np.ndarray) -> np.ndarray` batch method to EnsembleCalibrator
- `scripts/train_models.py` - Added `cal_l` extraction, per-fold calibrator fitting, and `calibrator=fold_calibrator` argument to `_evaluate_fold_metrics` call
- `tests/unit/test_ml_quality_gates.py` - 4 new tests for predict_proba and walk-forward calibrator wiring

## Decisions Made
- `predict_proba` returns a copy of input when not fitted (safe fallback, avoids mutating caller's array)
- Per-fold calibrator fitting mirrors the existing `_fit_and_save_calibrator` pattern for consistency
- Import of `EnsembleCalibrator` placed inside loop with `noqa: PLC0415` to match existing script conventions

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- SANDBOX-FIX-10 gap closed: walk-forward Brier evaluation now uses calibrated probabilities
- All 10 SANDBOX-FIX gaps are now addressed across plans 01-04
- Ready for phase verification

---
*Phase: 32-critical-sandbox-fixes*
*Completed: 2026-04-07*

## Self-Check: PASSED
