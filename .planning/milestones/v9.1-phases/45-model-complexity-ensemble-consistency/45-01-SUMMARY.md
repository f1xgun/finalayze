---
phase: 45-model-complexity-ensemble-consistency
plan: "01"
subsystem: ml-autoresearch
tags: [ml, moex, hyperparameters, overfitting, xgboost, lightgbm, catboost]
dependency_graph:
  requires: []
  provides: [MOEX-reduced-complexity-hparams, _get_hparams-helper, _log_complexity_profile-helper]
  affects: [scripts/auto_ml_research.py, tests/unit/test_auto_ml_research_moex.py]
tech_stack:
  added: []
  patterns: [segment-aware-hyperparameter-routing, TDD-red-green]
key_files:
  created:
    - tests/unit/test_auto_ml_research_moex.py (TestMoexHparams class — 12 new tests)
  modified:
    - scripts/auto_ml_research.py (_MOEX_HPARAMS, _get_hparams, _log_complexity_profile, _run_fold model constructors, run_research_loop baseline_config)
decisions:
  - "Use _get_hparams() helper with same _is_moex_segment() routing pattern as _get_lookback_days and _get_max_features"
  - "Extract _log_complexity_profile() helper to keep run_research_loop under 50-statement ruff PLR0915 limit"
  - "Pass hparams to baseline_config only (not to individual experiment configs) — experiments override hparams via their own config"
metrics:
  duration: ~8 minutes
  completed: "2026-04-14"
  tasks_completed: 1
  files_modified: 2
  tests_added: 12
---

# Phase 45 Plan 01: MOEX Hyperparameter Complexity Reduction Summary

**One-liner:** MOEX segments now use reduced-complexity XGBoost/LightGBM/CatBoost hyperparameters (depth=3, n_estimators=100, min_child_weight=20) via a segment-aware routing helper to prevent overfitting on ~850-sample datasets.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| RED | Add failing hparam tests | 8669689 | tests/unit/test_auto_ml_research_moex.py |
| GREEN | Implement MOEX hparams and routing | 8f70744 | scripts/auto_ml_research.py |

## What Was Built

### `_MOEX_HPARAMS` dict (scripts/auto_ml_research.py)

A module-level constant alongside `_DEFAULT_HPARAMS` with reduced complexity for MOEX segments (~850 training samples):

```python
_MOEX_HPARAMS = {
    "xgb_max_depth": 3,        # was 5 — prevents leaves with <5 samples
    "xgb_n_estimators": 100,   # was 200 — faster, less overfitting
    "xgb_min_child_weight": 20, # new — minimum sum of instance weights per leaf
    "lgbm_n_estimators": 100,
    "lgbm_num_leaves": 15,
    "cat_depth": 3,
    "cat_iterations": 100,
    ...learning_rates unchanged (0.05)...
}
```

### `_get_hparams(segment_id)` helper

Routes to MOEX or US profile using existing `_is_moex_segment()` check, returning a copy to prevent mutation of constants.

### `_run_fold` model constructor updates

XGBoostModel now receives `n_estimators` and `min_child_weight` from hparams. LightGBMModel receives `n_estimators` and `num_leaves`. CatBoostModel receives `iterations`. All with sane US defaults if keys missing.

### `_log_complexity_profile()` helper + baseline_config wiring

Emits a structured log line confirming which complexity profile is active at run start. `run_research_loop` passes `_get_hparams(segment_id)` to `baseline_config.hparams`.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing functionality] Extracted _log_complexity_profile helper to resolve PLR0915**
- **Found during:** GREEN phase lint check
- **Issue:** Adding 7 lines to `run_research_loop` pushed it to 51 statements (ruff PLR0915 limit is 50)
- **Fix:** Extracted the MOEX complexity log into `_log_complexity_profile()` helper, then inlined `_get_hparams()` call directly into `baseline_config` construction — net zero new statements in `run_research_loop`
- **Files modified:** scripts/auto_ml_research.py
- **Commit:** 8f70744

## Known Stubs

None — `_MOEX_HPARAMS` flows through to model constructors in `_run_fold`. All wiring is complete.

## Self-Check: PASSED

- [x] `_MOEX_HPARAMS` exists in scripts/auto_ml_research.py with `xgb_max_depth: 3`
- [x] `_get_hparams` helper exists and routes correctly
- [x] `xgb_n_estimators` passed to XGBoostModel in `_run_fold`
- [x] `MOEX complexity profile` log line exists
- [x] `TestMoexHparams` test methods exist in test file
- [x] `_DEFAULT_HPARAMS["xgb_max_depth"]` == 5 (US unchanged)
- [x] All 12 hparam tests pass
- [x] Commits 8669689 and 8f70744 exist
