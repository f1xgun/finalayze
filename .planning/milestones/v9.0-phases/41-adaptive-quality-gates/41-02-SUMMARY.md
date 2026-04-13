---
phase: 41-adaptive-quality-gates
plan: "02"
subsystem: ml-training
tags: [moex, walk-forward, fold-generation, quality-gates, min-signals]
dependency_graph:
  requires: [41-01]
  provides: [MOEX-fold-constants, parametric-generate-folds, min-signals-routing]
  affects: [scripts/auto_ml_research.py]
tech_stack:
  added: []
  patterns: [parametric-fold-generation, segment-aware-branching]
key_files:
  created:
    - tests/unit/test_auto_ml_research_folds.py
  modified:
    - scripts/auto_ml_research.py
decisions:
  - "Use _US_MIN_SIGNALS=50 local constant instead of importing private _MIN_SIGNALS from quality_gates — avoids cross-module private imports"
  - "Use calendar-day timestamps in tests (not weekday-filtered) so n_days directly maps to the timedelta math in generate_folds()"
  - "730 calendar days with US constants (window=740 days) yields 0 folds — confirms MOEX constants are mandatory for MOEX segments"
metrics:
  duration: "4m"
  completed_date: "2026-04-13"
  tasks_completed: 2
  files_changed: 2
---

# Phase 41 Plan 02: MOEX Fold Constants and min_signals Routing Summary

MOEX-specific walk-forward fold constants (8mo/1mo/3mo/21-day purge/2mo step) added to auto_ml_research.py with parametric generate_folds() and segment-aware min_signals routing through _run_fold to evaluate_fold.

## What Was Built

### Task 1: MOEX fold constants + parametrize generate_folds + wire min_signals

Added six new constants to `scripts/auto_ml_research.py`:

```python
_MOEX_WF_TRAIN_MONTHS = 8
_MOEX_WF_CAL_MONTHS = 1
_MOEX_WF_TEST_MONTHS = 3
_MOEX_WF_STEP_MONTHS = 2
_MOEX_PURGE_GAP = 21
_MOEX_MIN_SIGNALS = 15
_US_MIN_SIGNALS = 50
```

`generate_folds()` now accepts keyword-only fold parameters with US defaults, replacing all hardcoded constant references internally. The `_prepare_data()` main loop builds `fold_kwargs` branching on `is_moex`. `_run_fold()` accepts `min_signals: int = _US_MIN_SIGNALS` and passes it to `evaluate_fold(fold_metrics, min_signals=min_signals)`. `run_experiment()` selects `min_signals` before the fold loop using `_is_moex_segment()`.

Commits: `0a869f4`

### Task 2: Test MOEX fold generation on 730-day data (TDD)

Created `tests/unit/test_auto_ml_research_folds.py` with 5 tests:

| Test | Assertion |
|------|-----------|
| `test_moex_folds_730_days` | 730-day MOEX data + MOEX constants → ≥3 folds |
| `test_us_constants_on_moex_data_few_folds` | 730-day data + US constants → ≤1 folds (0 actual) |
| `test_us_folds_1825_days` | 1000-day US data + US defaults → ≥3 folds |
| `test_default_kwargs_backward_compatible` | no-kwargs == explicit US defaults |
| `test_each_moex_fold_has_data` | every MOEX fold has non-empty train and test sets |

Commits: `f00bf63`

## Verification

```
uv run pytest tests/unit/test_auto_ml_research_folds.py --no-cov   → 5 passed
uv run pytest tests/unit/test_quality_gates.py --no-cov             → 34 passed
uv run ruff check scripts/auto_ml_research.py                        → All checks passed
uv run ruff format --check scripts/auto_ml_research.py               → 1 file already formatted
```

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Test helper used weekday filtering, breaking fold count assertion**
- **Found during:** Task 2 (test execution)
- **Issue:** `_make_daily_timestamps()` generated weekday-only timestamps, so 730 "days" spanned ~1050 calendar days — enough for 4 US folds, contradicting `<= 1` assertion
- **Fix:** Switched to calendar-day timestamps (no weekday filter); also set `_US_DATASET_DAYS = 1000` to ensure ≥3 US folds on calendar-day basis. US fold window = 740 calendar days; 730-day MOEX dataset yields 0 US folds exactly as intended.
- **Files modified:** `tests/unit/test_auto_ml_research_folds.py`
- **Commit:** `f00bf63`

## Known Stubs

None.

## Threat Flags

None — no new network endpoints or trust boundaries introduced. Constants are hardcoded in source; CLI only selects segment, not fold parameters directly (T-41-03 accepted). T-41-04 (min_signals relaxation) is mitigated by degenerate predictor guard from Plan 01.

## Self-Check: PASSED

- `scripts/auto_ml_research.py` — exists, contains `_MOEX_WF_TRAIN_MONTHS = 8`
- `tests/unit/test_auto_ml_research_folds.py` — exists, contains `test_moex_folds_730_days`
- Commits `0a869f4` and `f00bf63` — confirmed in `git log --oneline -5`
