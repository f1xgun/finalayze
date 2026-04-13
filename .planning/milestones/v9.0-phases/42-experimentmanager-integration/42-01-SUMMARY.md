---
phase: 42-experimentmanager-integration
plan: "01"
subsystem: ml-research
tags: [experiment-manager, auto-ml, cli, tdd]
dependency_graph:
  requires:
    - src/finalayze/core/experiment_manager.py
    - src/finalayze/core/schemas.py (SuccessCriteria, ExperimentResult, ExperimentStatus)
  provides:
    - scripts/auto_ml_research.py --experiment-id flag
    - ExperimentManager lifecycle wiring (create/link/verdict)
  affects:
    - results/experiments/*.jsonl (unchanged — JSONL always written)
    - .planning/experiments/*.md (new files when --experiment-id used)
tech_stack:
  added: []
  patterns:
    - lazy import pattern for optional ExperimentManager dependency
    - try/except non-crashing wrapper for all ExperimentManager calls
    - helper extraction to stay within PLR0915 statement limit
key_files:
  modified:
    - scripts/auto_ml_research.py
  created:
    - tests/unit/test_auto_ml_research_experiment.py
decisions:
  - ExperimentManager init extracted to _init_experiment_manager() to reduce statement count in run_research_loop() below PLR0915 limit (59 > 50)
  - JSONL audit trail preserved unconditionally regardless of --experiment-id per plan CONTEXT.md decision
  - All ExperimentManager calls wrapped in try/except so a filesystem or schema error never aborts the research loop
  - Lazy import pattern: ExperimentManager only imported when --experiment-id is provided, so no-flag invocations have zero overhead and no new dependency
  - argparse type= function handles ID validation inline — invalid IDs rejected before run_research_loop() is called
metrics:
  duration: "~14 minutes"
  completed: "2026-04-13T06:41:21Z"
  tasks_completed: 2
  files_changed: 2
---

# Phase 42 Plan 01: ExperimentManager Integration into auto_ml_research.py Summary

**One-liner:** Opt-in `--experiment-id` flag wires ExperimentManager lifecycle (create/link/verdict) into auto_ml_research.py with JSONL audit trail always preserved and non-crashing error isolation.

## What Was Built

`scripts/auto_ml_research.py` now accepts an optional `--experiment-id ID` flag. When provided:

1. `_init_experiment_manager()` lazily imports `ExperimentManager`, calls `create_experiment()` with an auto-generated hypothesis (`"AutoML research: {strategy} on {segment_id}"`), a `SuccessCriteria(metric="composite_score", threshold=0.0, operator=">=")`, and sets status to `"running"`.
2. `_link_to_experiment_manager()` is called after each `_log_result()` invocation (baseline + all loop experiments), building an `ExperimentResultSchema` with score, accuracy, brier, profit_factor, feature_count, status.
3. After `_print_summary()`, `record_verdict(experiment_id, best_score)` records `ACCEPT/REJECT/INCONCLUSIVE` based on the success criteria.

Without `--experiment-id`, behavior is identical to before: JSONL only, no ExperimentManager import.

## Tasks Completed

| Task | Description | Commit |
|------|-------------|--------|
| 1 | TDD: failing tests (RED) → implementation (GREEN) — `--experiment-id` flag + ExperimentManager wiring | 1abc513 |
| 2 | Lint (ruff check), format (ruff format), regression tests | 1abc513 |

## Tests

8 new unit tests in `tests/unit/test_auto_ml_research_experiment.py`:

| Test | Behavior Verified |
|------|-------------------|
| `test_no_experiment_id` | No ExperimentManager import when experiment_id=None; JSONL written |
| `test_experiment_id_creates_entry` | create_experiment() called with correct experiment_id and non-empty hypothesis |
| `test_result_linking` | link_result() called at least once when experiment_id set |
| `test_verdict_recorded` | record_verdict() called with float metric_value after loop |
| `test_experiment_id_validation_invalid` | Spaces/slashes/@ in ID raise SystemExit |
| `test_experiment_id_validation_valid` | Letters/digits/hyphens/underscores accepted |
| `test_error_resilience` | RuntimeError from create_experiment doesn't abort loop; JSONL still written |
| `test_concurrent_isolation` | Two ExperimentManager instances with different dirs produce independent files |

All 8 pass. ExperimentManager regression suite (19 tests) also passes.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Refactor] Extracted _init_experiment_manager() helper**
- **Found during:** Task 2 lint — PLR0915 "Too many statements (59 > 50)" in run_research_loop()
- **Fix:** Extracted the ExperimentManager init block (lazy import + create_experiment + update_status + try/except) into a dedicated `_init_experiment_manager()` helper function
- **Files modified:** scripts/auto_ml_research.py
- **Commit:** 1abc513

**2. [Rule 1 - Fix] Module registration fix for test import**
- **Found during:** Task 1 RED phase — `AttributeError: 'NoneType' object has no attribute '__dict__'` when loading the script via `importlib.util` without registering in `sys.modules`
- **Fix:** Added `sys.modules[_MODULE_NAME] = mod` before `exec_module()` in `_import_module()` helper so dataclass string annotations resolve correctly
- **Files modified:** tests/unit/test_auto_ml_research_experiment.py
- **Commit:** 1abc513

## Known Stubs

None — all wiring is live (ExperimentManager calls are real, not mocked in production code).

## Threat Flags

T-42-01 (Tampering — path traversal via --experiment-id) mitigated: argparse `type=` function validates ID against `^[a-zA-Z0-9_-]+$` regex before any filesystem operation.

## Self-Check: PASSED

- scripts/auto_ml_research.py: FOUND
- tests/unit/test_auto_ml_research_experiment.py: FOUND
- .planning/phases/42-experimentmanager-integration/42-01-SUMMARY.md: FOUND
- commit 1abc513: FOUND
