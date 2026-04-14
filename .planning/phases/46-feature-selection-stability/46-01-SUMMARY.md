---
phase: 46-feature-selection-stability
plan: "01"
subsystem: ml/training
tags: [feature-selection, walk-forward, stability, tdd]
dependency_graph:
  requires: [45-01]
  provides: [stable-feature-selection-per-experiment]
  affects: [scripts/auto_ml_research.py, ml/training/feature_selection]
tech_stack:
  added: []
  patterns: [pre-fold-selection, config-injection-not-mutation]
key_files:
  created:
    - tests/unit/test_auto_ml_research_feature_selection_stability.py
  modified:
    - scripts/auto_ml_research.py
decisions:
  - "select_features_efficient called once on union of all training indices (excludes test, no look-ahead)"
  - "New ExperimentConfig created with feature_subset set — caller's config not mutated (T-46-02)"
  - "features_used in result still populated from fold 0's selected list (unchanged path)"
metrics:
  duration_seconds: 117
  completed_date: "2026-04-14"
  tasks_completed: 1
  tasks_total: 1
  files_modified: 2
---

# Phase 46 Plan 01: Feature Selection Stability Summary

**One-liner:** Pre-fold MI-based feature selection using union of all training indices, eliminating fold-to-fold feature churn in the autoresearch pipeline.

## What Was Built

Refactored `run_experiment()` in `scripts/auto_ml_research.py` to run `select_features_efficient` **once before the fold loop** rather than once per fold inside `_run_fold`. The selected feature list is injected into a new `ExperimentConfig` (not mutating the caller's object) so all folds train on the identical feature set.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Lift feature selection out of _run_fold into run_experiment | a397039 | scripts/auto_ml_research.py, tests/unit/test_auto_ml_research_feature_selection_stability.py |

## Implementation Details

**Change in `run_experiment()` (scripts/auto_ml_research.py):**

Before the `try` block and fold loop, when `config.feature_subset is None`:
1. Compute union of all `train_idx` across all folds (excluding test indices to prevent look-ahead)
2. Build DataFrame from those samples
3. Call `select_features_efficient(train_df, train_s, max_features=config.max_features)` — exactly once
4. Log `feature_selection_stable` with `selected_count` and first 5 feature names
5. Create a new `ExperimentConfig` with `feature_subset=selected_features` — does NOT mutate caller's object

**`_run_fold()` unchanged:** Already handles `config.feature_subset is not None` correctly — uses it directly, skips selection. No code removed from `_run_fold`.

## Tests Added (4 unit tests)

| Test | Assertion |
|------|-----------|
| `test_feature_selection_runs_once_before_folds` | `select_features_efficient` called exactly 1 time for 3-fold experiment |
| `test_explicit_feature_subset_skips_selection` | `select_features_efficient` never called when `config.feature_subset` provided |
| `test_selection_uses_all_train_data` | DataFrame passed to selection has `len(union_of_all_train_indices)` rows |
| `test_selected_features_logged_once` | Exactly 1 `feature_selection_stable` log event with correct `selected_count` |

## Deviations from Plan

**1. [Rule 1 - Bug] ruff format applied**
- Found during: post-implementation lint check
- Issue: `ruff format --check` reported the new block needed formatting
- Fix: Ran `uv run ruff format scripts/auto_ml_research.py`
- Files modified: scripts/auto_ml_research.py
- Commit: a397039 (included in same commit)

No other deviations. Plan executed as specified.

## Known Stubs

None.

## Threat Flags

None — changes are internal to the experiment pipeline with no new network endpoints, auth paths, or schema changes.

## Self-Check: PASSED

- [x] `tests/unit/test_auto_ml_research_feature_selection_stability.py` — FOUND
- [x] `scripts/auto_ml_research.py` — modified, FOUND
- [x] Commit a397039 — FOUND
- [x] All 4 new tests PASS
- [x] All 8 existing `test_auto_ml_research_experiment.py` tests PASS
- [x] `ruff check scripts/auto_ml_research.py` — clean
- [x] `ruff format --check scripts/auto_ml_research.py` — clean
