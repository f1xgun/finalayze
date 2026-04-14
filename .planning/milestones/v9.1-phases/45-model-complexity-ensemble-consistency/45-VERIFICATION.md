---
phase: 45-model-complexity-ensemble-consistency
verified: 2026-04-14T12:00:00Z
status: passed
score: 4/4
overrides_applied: 0
---

# Phase 45: Model Complexity & Ensemble Consistency Verification Report

**Phase Goal:** MOEX models are trained with reduced complexity to prevent overfitting on small datasets, and all three ensemble members use a consistent class rebalancing strategy
**Verified:** 2026-04-14T12:00:00Z
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths (from Roadmap Success Criteria)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | MOEX segments use max_depth=3, n_estimators=100, min_child_weight=20; log confirms MOEX complexity profile | VERIFIED | `_MOEX_HPARAMS` at line 199 has xgb_max_depth=3, xgb_n_estimators=100, xgb_min_child_weight=20. `_log_complexity_profile()` at line 1383 emits "Using MOEX complexity profile" for MOEX segments. Wired at line 1533. |
| 2 | MOEX hparams are a separate constant from US defaults — changing US depth does not affect MOEX | VERIFIED | `_DEFAULT_HPARAMS` (line 184) and `_MOEX_HPARAMS` (line 199) are separate module-level dicts. `_DEFAULT_HPARAMS["xgb_max_depth"]` is 5; `_MOEX_HPARAMS["xgb_max_depth"]` is 3. 12 tests in `TestMoexHparams` confirm independence. |
| 3 | XGBoost sets scale_pos_weight=1.0 when sample_weight is provided (matching LightGBM); unit tests confirm XGB and LGBM receive identical effective class weights | VERIFIED | `xgboost_model.py:97`: `spw = 1.0 if sample_weight is not None else (n_neg / n_pos ...)`. Tests `test_xgb_spw_with_sample_weight` and `test_lgbm_spw_with_sample_weight` pass. |
| 4 | All three ensemble members (XGB, LGBM, CatBoost) apply class rebalancing exclusively through sample_weight — no double-counting | VERIFIED | XGBoost: conditional spw (line 97). LightGBM: already correct (pre-existing pattern). CatBoost: `acw = None if sample_weight is not None else "Balanced"` at `catboost_model.py:100`. All 6 tests in `test_ensemble_consistency.py` pass. |

**Score:** 4/4 truths verified

### Plan-Level Must-Haves

**Plan 01 must-haves:**

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | MOEX segments use max_depth=3, n_estimators=100, min_child_weight=20 in autoresearch | VERIFIED | `_MOEX_HPARAMS` values confirmed in source |
| 2 | US segments still use original defaults (max_depth=5, n_estimators=200) | VERIFIED | `_DEFAULT_HPARAMS["xgb_max_depth"]` = 5, `["xgb_n_estimators"]` = 200 |
| 3 | MOEX hyperparameter dict is separate from US default dict | VERIFIED | Two distinct module-level constants |
| 4 | A log line confirms MOEX complexity profile is active at run start | VERIFIED | `_log_complexity_profile()` function at line 1383, called at line 1533 |

**Plan 02 must-haves:**

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | XGBoost sets scale_pos_weight=1.0 when sample_weight is provided | VERIFIED | `xgboost_model.py:97` |
| 2 | CatBoost disables auto_class_weights when sample_weight is provided | VERIFIED | `catboost_model.py:100`: `acw = None if sample_weight is not None else "Balanced"` |
| 3 | LightGBM already correct (scale_pos_weight=1.0 when sample_weight provided) — no change needed | VERIFIED | Confirmed by `test_lgbm_spw_with_sample_weight` passing |
| 4 | All 3 models use sample_weight exclusively for class rebalancing, never double-counting | VERIFIED | All 6 tests in `test_ensemble_consistency.py` pass |

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `scripts/auto_ml_research.py` | `_MOEX_HPARAMS` dict and routing in `_run_fold` | VERIFIED | `_MOEX_HPARAMS` at line 199, `_get_hparams` at line 233, wired into `baseline_config` at line 1531, model constructors at lines 682-696 |
| `tests/unit/test_auto_ml_research_moex.py` | Tests for MOEX hparams routing (`TestMoexHparams`) | VERIFIED | 12 tests in `TestMoexHparams` class (lines 443-522), all pass |
| `src/finalayze/ml/models/xgboost_model.py` | XGBoost `fit()` with conditional `scale_pos_weight` | VERIFIED | Fix at line 97: `spw = 1.0 if sample_weight is not None else (n_neg / n_pos ...)` with double-count comment |
| `src/finalayze/ml/models/catboost_model.py` | CatBoost `fit()` with conditional `auto_class_weights` | VERIFIED | Fix at line 100: `acw = None if sample_weight is not None else "Balanced"` with double-count comment |
| `tests/unit/test_ensemble_consistency.py` | Tests proving consistent rebalancing across all 3 models | VERIFIED | 6 tests across 3 classes (XGB, CatBoost, LightGBM), all pass |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `auto_ml_research.py:_get_hparams` | `_MOEX_HPARAMS` | `_is_moex_segment` check at line 235 | WIRED | `return dict(_MOEX_HPARAMS) if _is_moex_segment(segment_id) else dict(_DEFAULT_HPARAMS)` |
| `auto_ml_research.py:run_research_loop` | `_get_hparams` | `baseline_config.hparams=_get_hparams(segment_id)` at line 1531 | WIRED | Confirmed at line 1531 |
| `auto_ml_research.py:_run_fold` | `config.hparams` | `hp = config.hparams` → model constructors at lines 682-696 | WIRED | XGBoostModel, LightGBMModel, CatBoostModel all receive hparams values |
| `xgboost_model.py:fit` | `scale_pos_weight` | conditional on `sample_weight is not None` at line 97 | WIRED | Pattern `1.0 if sample_weight is not None else (n_neg / n_pos ...)` confirmed |
| `catboost_model.py:fit` | `auto_class_weights` | conditional on `sample_weight is not None` at line 100 | WIRED | `acw = None if sample_weight is not None else "Balanced"` → `auto_class_weights=acw` at line 110 |

### Data-Flow Trace (Level 4)

Not applicable — this phase modifies training scripts and model constructors (not components rendering dynamic data). The data flow is configuration values into model hyperparameters, which is verified by the key link checks above.

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| MOEX hparam values correct | `pytest tests/unit/test_auto_ml_research_moex.py -k "Hparam"` | 12 passed | PASS |
| XGBoost spw conditional | `pytest tests/unit/test_ensemble_consistency.py -k "xgb"` | 2 passed | PASS |
| CatBoost acw conditional | `pytest tests/unit/test_ensemble_consistency.py -k "catboost"` | 2 passed | PASS |
| LightGBM unchanged behavior | `pytest tests/unit/test_ensemble_consistency.py -k "lgbm"` | 2 passed | PASS |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|---------|
| MCPX-01 | 45-01-PLAN.md | Autoresearch uses reduced model complexity for MOEX (max_depth=3, n_estimators=100, min_child_weight=20) | SATISFIED | `_MOEX_HPARAMS` dict + `_run_fold` wiring confirmed |
| MCPX-02 | 45-01-PLAN.md | MOEX-specific hyperparameter defaults are separate from US defaults in autoresearch config | SATISFIED | Two distinct module-level constants (`_DEFAULT_HPARAMS` vs `_MOEX_HPARAMS`) |
| ENSM-01 | 45-02-PLAN.md | XGBoost sets scale_pos_weight=1.0 when sample_weight is provided (matching LightGBM behavior) | SATISFIED | `xgboost_model.py:97` fix + 2 passing tests |
| ENSM-02 | 45-02-PLAN.md | All 3 ensemble members (XGB, LGBM, CatBoost) use consistent class rebalancing strategy | SATISFIED | All conditional patterns in place; 6 tests across 3 models pass |

### Anti-Patterns Found

None. Scanned `scripts/auto_ml_research.py`, `src/finalayze/ml/models/xgboost_model.py`, `src/finalayze/ml/models/catboost_model.py`, `tests/unit/test_ensemble_consistency.py`, and `tests/unit/test_auto_ml_research_moex.py`. No TODO/FIXME, no placeholder returns, no empty implementations, no hardcoded empty data structures.

### Human Verification Required

None. All must-haves are verifiable programmatically.

### Notes on Pre-Existing Test Failures

Two tests in `TestGetLookbackDays` (`test_ru_blue_chips_returns_730`, `test_ru_energy_returns_730`) fail because they expect `_MOEX_LOOKBACK_DAYS` = 730 but the constant is 1095. This is a pre-existing mismatch from an earlier phase (lookback days were extended from 730 to 1095 before Phase 45). The SUMMARY.md for Plan 01 confirms this via git stash verification. These failures are not within Phase 45 scope and do not affect the phase goal.

### Commit Verification

All 4 documented commits exist in git history:
- `8669689` — test(45-01): add failing tests for MOEX hparams routing
- `8f70744` — feat(45-01): add MOEX hyperparameters and routing in autoresearch
- `ce21d29` — feat(45-02): fix XGBoost scale_pos_weight double-rebalancing
- `2e5d69e` — feat(45-02): fix CatBoost auto_class_weights double-rebalancing

### Gaps Summary

No gaps. All 4 roadmap success criteria are fully implemented, wired, and tested.

---

_Verified: 2026-04-14T12:00:00Z_
_Verifier: Claude (gsd-verifier)_
