---
phase: 44-new-search-strategies
plan: "01"
subsystem: ml-research
tags: [ml, feature-engineering, cross-segment, permutation-importance, auto-research]
dependency_graph:
  requires: []
  provides: [cross_segment_transfer_strategy, feature_engineering_strategy]
  affects: [scripts/auto_ml_research.py]
tech_stack:
  added: [sklearn.inspection.permutation_importance, xgboost.XGBClassifier (direct)]
  patterns: [TDD, generator-dispatcher pattern, pandas feature generation]
key_files:
  created:
    - tests/unit/test_auto_ml_research_transfer.py
    - tests/unit/test_auto_ml_research_feature_eng.py
  modified:
    - scripts/auto_ml_research.py
decisions:
  - "Used xgboost.XGBClassifier directly in _filter_by_permutation_importance instead of project XGBoostModel wrapper to avoid segment_id coupling"
  - "Split _generate_feature_candidates into 3 private helper functions to satisfy ruff PLR0912 (too many branches)"
  - "ARG001 suppressed for labels parameter in _generate_feature_candidates — reserved for future importance-aware generation ordering"
  - "_generate_experiments signature extended with optional all_features/labels/n_samples kwargs; default values preserve backward compat for existing callers"
metrics:
  duration: ~25m
  completed: "2026-04-13"
  tasks: 2
  files: 3
---

# Phase 44 Plan 01: New Search Strategies Summary

Two new autonomous ML search strategies added to `scripts/auto_ml_research.py`: cross-segment transfer (US-to-MOEX feature validation) and domain-motivated feature engineering with permutation importance filtering — both with overfitting guardrails and full CLI integration.

## What Was Built

### Task 1: Cross-Segment Transfer Strategy (STRAT-02)

**`generate_transfer_experiments(segment_id: str) -> list[ExperimentConfig]`**

- Reads `results/experiments/us_tech_experiment_log.jsonl`
- Selects the entry with highest `score` where `status == "keep"`
- Filters out market-specific features via `_MARKET_SPECIFIC_KEYWORDS = ("vix", "usdrub", "brent", "cbr", "imoex", "turnover")`
- Returns `ExperimentConfig(strategy="cross_segment_transfer", feature_subset=filtered_features)`
- Only applies to `ru_*` segments; US segments return `[]`
- Handles missing JSONL and empty "keep" entries gracefully with `logger.warning`

**Commits:** `43afb1b`

### Task 2: Feature Engineering Strategy with Permutation Filter (STRAT-03)

**Three helper functions:**
- `_add_lag_ratio_candidates`: lag ratios feat[t]/feat[t-lag] for close/volume features (lags 5, 10, 20)
- `_add_rolling_zscore_candidates`: rolling z-scores (windows 20, 60) for all base features
- `_add_interaction_candidates`: RSI × volume cross-feature interactions

**`_generate_feature_candidates(base_features, all_features, labels, cap)`**
- Orchestrates all three helpers; hard cap at `cap` total candidates
- Labels param reserved for future importance-aware ordering (noqa ARG001)

**`_filter_by_permutation_importance(features, labels, candidate_names, baseline_features)`**
- Trains `xgb.XGBClassifier` (lightweight, max_depth=4, n_estimators=50)
- Runs `sklearn.inspection.permutation_importance` (n_repeats=5, random_state=42)
- Keeps only candidates with mean importance > 0

**`generate_feature_engineering_experiments(baseline_features, all_features, labels, n_samples)`**
- Computes `cap = n_samples // 20` (36 for MOEX/730d, 91 for US/1825d)
- Returns single `ExperimentConfig(strategy="feature_engineering", feature_subset=baseline+survivors)`

**`_generate_experiments` updated:** New optional params `all_features`, `labels`, `n_samples` with defaults — backward compatible.

**Commits:** `9ad9e03`

## Test Coverage

| File | Tests | Scenarios |
|------|-------|-----------|
| test_auto_ml_research_transfer.py | 8 | T1-T8: configs, market-neutral filter, best-keep selection, missing JSONL, no-keep, CLI, dispatch |
| test_auto_ml_research_feature_eng.py | 12 | T1-T10 (+2 sub-cases): lag/zscore/interaction types, cap enforcement, cap math, filter remove, filter keep, return configs, baseline+survivors subset, CLI, dispatch routing, "all" strategy |

Total: 20 new tests. All 68 auto_ml_research tests pass (no regressions).

## Verification

```bash
uv run python scripts/auto_ml_research.py --help
# --strategy {ablation,efficiency,hyperparameter,random_subset,ensemble_weights,cross_segment_transfer,feature_engineering,all}

uv run pytest tests/unit/test_auto_ml_research_*.py --no-cov
# 68 passed, 2 warnings
```

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] XGBoostModel requires segment_id argument**
- **Found during:** Task 2 GREEN phase
- **Issue:** `XGBoostModel()` cannot be instantiated without `segment_id` — causes TypeError at runtime
- **Fix:** Used `xgb.XGBClassifier` directly (lightweight, no project coupling) instead of the wrapper
- **Files modified:** `scripts/auto_ml_research.py`
- **Commit:** `9ad9e03`

**2. [Rule 1 - Bug] `random_subset` generator fails with < 5 features in test fixtures**
- **Found during:** Task 1 T8 test (strategy="all")
- **Issue:** `random.randint(5, min(15, len(all_feature_names)))` raises ValueError when len < 5
- **Fix:** Updated T7/T8 test fixtures to use 10 synthetic feature names
- **Files modified:** `tests/unit/test_auto_ml_research_transfer.py`
- **Commit:** `43afb1b`

**3. [Rule 2 - Refactor] PLR0912 too many branches in `_generate_feature_candidates`**
- **Found during:** Task 2 ruff check
- **Issue:** 17 branches > 12 limit (ruff PLR0912)
- **Fix:** Extracted lag, zscore, and interaction generation into three private helpers
- **Files modified:** `scripts/auto_ml_research.py`
- **Commit:** `9ad9e03`

## Known Stubs

None. Both strategies return real `ExperimentConfig` objects with populated `feature_subset`. The permutation filter uses real XGBoost training.

## Threat Flags

None. No new network endpoints, auth paths, or schema changes. JSONL file reads are internal only (local file written by our own `_log_result`).

## Self-Check: PASSED

- `scripts/auto_ml_research.py` — exists and contains `generate_transfer_experiments`, `generate_feature_engineering_experiments`, `_generate_feature_candidates`, `_filter_by_permutation_importance`
- `tests/unit/test_auto_ml_research_transfer.py` — exists, 8 tests pass
- `tests/unit/test_auto_ml_research_feature_eng.py` — exists, 12 tests pass
- Commits `43afb1b` and `9ad9e03` exist on `main`
