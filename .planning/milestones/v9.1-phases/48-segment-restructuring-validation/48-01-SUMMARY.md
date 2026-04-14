---
phase: 48-segment-restructuring-validation
plan: "01"
subsystem: config/ml-scripts
tags: [segments, ml-training, data-quality, moex]
dependency_graph:
  requires: []
  provides: [ru_finance_sberp_removed, min_history_gate_auto_ml_research, min_history_gate_train_models]
  affects: [scripts/auto_ml_research.py, scripts/train_models.py, config/segments.py]
tech_stack:
  added: []
  patterns: [tdd-red-green, history-gate-before-min-candles]
key_files:
  created: []
  modified:
    - config/segments.py
    - scripts/auto_ml_research.py
    - scripts/train_models.py
    - tests/unit/test_segments.py
    - tests/unit/test_auto_ml_research_moex.py
    - tests/unit/test_train_models_script.py
decisions:
  - "Gate order: _MIN_HISTORY_DAYS check before min_candles (semantic quality gate precedes technical window requirement)"
  - "Test fixture candle counts bumped from 200/300 to 500 to satisfy the new gate"
metrics:
  duration_seconds: 460
  completed_date: "2026-04-14"
  tasks_completed: 3
  tasks_total: 3
  files_modified: 6
---

# Phase 48 Plan 01: Segment Restructuring — SBERP Removal and 500-Day History Gate Summary

SBERP removed from ru_finance config and `_MIN_HISTORY_DAYS=500` gate added to both `auto_ml_research.py` and `train_models.py` using TDD.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Write failing tests (RED) | 13e5a47 | test_segments.py, test_auto_ml_research_moex.py, test_train_models_script.py |
| 2 | Remove SBERP + history gate in auto_ml_research (GREEN) | 7d8558c | config/segments.py, scripts/auto_ml_research.py |
| 3 | History gate in train_models + fix test fixtures | 715ec39 | scripts/train_models.py, test_auto_ml_research_moex.py, test_train_models_script.py |

## What Was Built

- `config/segments.py` ru_finance: `symbols=["SBER", "T", "CBOM", "BSPB", "MOEX"]` — SBERP removed (rho>0.95 with SBER, near-zero independent signal)
- `scripts/auto_ml_research.py`: `_MIN_HISTORY_DAYS = 500` constant; `build_full_dataset` loop changed from `values()` to `items()` with history gate before `min_candles` check; warning message on skip
- `scripts/train_models.py`: `_MIN_HISTORY_DAYS = 500` constant; `_build_dataset_triple_barrier` per-symbol loop gains history gate before existing `min_candles_tb` check; log message follows `[segment_id]` prefix convention
- 4 new test methods: `TestRuFinance::test_sberp_not_in_ru_finance`, `TestMinHistoryGate::test_constant_value`, `TestMinHistoryGate::test_sberp_not_in_ru_finance_symbols`, `TestMinHistoryGate::test_min_history_days_constant`

## Verification Results

- `grep -c SBERP config/segments.py` → 0 (SBERP fully absent)
- `grep _MIN_HISTORY_DAYS scripts/auto_ml_research.py scripts/train_models.py` → both files contain constant and gate
- 71 tests pass across 3 test files
- `ruff check config/segments.py scripts/auto_ml_research.py` → clean (pre-existing PLC0415 in train_models.py at line 1080 is out of scope)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Existing test fixtures used 200/300 candles — below new 500-day gate**

- **Found during:** Task 3
- **Issue:** `_CANDLE_COUNT=200` in `test_auto_ml_research_moex.py` and `n=300` in two `test_train_models_script.py` tests caused those tests to skip all symbols and produce empty datasets, breaking assertions like `assert features`
- **Fix:** Bumped `_CANDLE_COUNT` and `_MACRO_COUNT` from 200 to 500 in `test_auto_ml_research_moex.py`; bumped `n=300` to `n=500` in `test_script_creates_output_files_triple_barrier_mode` and `test_build_dataset_triple_barrier_returns_weights`
- **Files modified:** `tests/unit/test_auto_ml_research_moex.py`, `tests/unit/test_train_models_script.py`
- **Commit:** 715ec39

### Deferred Items (Out of Scope)

- `TestWalkForwardUsesLastFold::test_last_fold_models_saved_not_best_accuracy` — pre-existing failure (`mock_evaluate_fold_metrics got unexpected keyword argument 'calibrator'`); confirmed failing before this plan's changes; logged to deferred-items
- `PLC0415 import numpy as np` at `scripts/train_models.py:1080` — pre-existing ruff violation; not introduced by this plan

## Known Stubs

None — no stub patterns found in modified files.

## Threat Flags

None — changes are internal config and script-only; no new network endpoints, auth paths, or trust boundaries introduced.

## Self-Check: PASSED

- config/segments.py modified: FOUND
- scripts/auto_ml_research.py modified: FOUND
- scripts/train_models.py modified: FOUND
- Commits 13e5a47, 7d8558c, 715ec39: FOUND
