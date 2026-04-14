---
phase: 45-model-complexity-ensemble-consistency
plan: "02"
subsystem: ml
tags: [ml, ensemble, xgboost, catboost, class-rebalancing, bug-fix]
dependency_graph:
  requires: []
  provides: [ENSM-01, ENSM-02]
  affects: [src/finalayze/ml/models/xgboost_model.py, src/finalayze/ml/models/catboost_model.py]
tech_stack:
  added: []
  patterns: [conditional-pos-weight, tdd-red-green]
key_files:
  created:
    - tests/unit/test_ensemble_consistency.py
  modified:
    - src/finalayze/ml/models/xgboost_model.py
    - src/finalayze/ml/models/catboost_model.py
decisions:
  - "Set scale_pos_weight=1.0 when sample_weight provided in XGBoost, matching existing LightGBM pattern"
  - "Set auto_class_weights=None when sample_weight provided in CatBoost"
  - "All 3 ensemble members now use sample_weight as sole class-rebalancing mechanism when provided"
metrics:
  duration: "~10 minutes"
  completed: "2026-04-14T10:48:21Z"
  tasks_completed: 2
  tasks_total: 2
  files_changed: 3
---

# Phase 45 Plan 02: Ensemble Consistency Fix Summary

**One-liner:** Fixed XGBoost and CatBoost double-rebalancing by making scale_pos_weight and auto_class_weights conditional on sample_weight presence, matching existing LightGBM behavior.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Fix XGBoost scale_pos_weight double-rebalancing | ce21d29 | xgboost_model.py, test_ensemble_consistency.py |
| 2 | Fix CatBoost auto_class_weights double-rebalancing | 2e5d69e | catboost_model.py, test_ensemble_consistency.py |

## What Was Done

### Task 1: XGBoost Fix

In `src/finalayze/ml/models/xgboost_model.py`, the `fit()` method always computed `spw = n_neg / n_pos` from class ratio, even when `sample_weight` was provided. This caused double-counting: sample_weight already encodes class rebalancing (from sequential bootstrapping), and scale_pos_weight multiplied it again.

Fix applied (matching existing LightGBM pattern):
```python
# When sample_weight is provided, it already handles class balance;
# applying scale_pos_weight on top would double-count the reweighting.
spw = 1.0 if sample_weight is not None else (n_neg / n_pos if n_pos > 0 else 1.0)
```

### Task 2: CatBoost Fix

In `src/finalayze/ml/models/catboost_model.py`, `auto_class_weights="Balanced"` was always set, even when `sample_weight` was provided.

Fix applied:
```python
# When sample_weight is provided, it already handles class balance;
# applying auto_class_weights on top would double-count the reweighting.
acw = None if sample_weight is not None else "Balanced"
```

### Test Coverage

Created `tests/unit/test_ensemble_consistency.py` with 6 tests:
- `TestXGBoostScalePosWeight::test_xgb_spw_with_sample_weight` — spw=1.0 when sw provided
- `TestXGBoostScalePosWeight::test_xgb_spw_without_sample_weight` — spw=class_ratio when no sw
- `TestCatBoostAutoClassWeights::test_catboost_acw_with_sample_weight` — acw=None when sw provided
- `TestCatBoostAutoClassWeights::test_catboost_acw_without_sample_weight` — acw="Balanced" when no sw
- `TestLightGBMScalePosWeight::test_lgbm_spw_with_sample_weight` — confirms already correct
- `TestLightGBMScalePosWeight::test_lgbm_spw_without_sample_weight` — confirms class_ratio behavior

All 6 tests pass.

## Deviations from Plan

None — plan executed exactly as written.

## Known Stubs

None.

## Threat Flags

None. Changes address T-45-03 and T-45-04 from the plan's threat model:
- T-45-03 (scale_pos_weight tampering): mitigated — conditional logic + unit test
- T-45-04 (auto_class_weights tampering): mitigated — conditional logic + unit test

## Self-Check: PASSED

- [x] `src/finalayze/ml/models/xgboost_model.py` — exists, contains fix
- [x] `src/finalayze/ml/models/catboost_model.py` — exists, contains fix
- [x] `tests/unit/test_ensemble_consistency.py` — exists, 6 tests pass
- [x] Commit ce21d29 — Task 1 (XGBoost + test file)
- [x] Commit 2e5d69e — Task 2 (CatBoost + updated tests)
- [x] Lint: ruff check passes on both modified model files
- [x] No regressions in existing ML tests (pre-existing failure in test_auto_ml_research_moex.py confirmed pre-existing via git stash verification)
