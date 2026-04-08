---
phase: 34-experiment-registry-runner
plan: 02
subsystem: backtest
tags: [experiment-runner, hypothesis-testing, interaction-test, subprocess, tdd]

# Dependency graph
requires:
  - phase: 34-experiment-registry-runner
    provides: ExperimentManager, ExperimentState, ExperimentResult schemas, experiment YAML registry
provides:
  - --hypothesis and --run-name flags on run_iteration.py for experiment-linked backtests
  - _deep_merge() utility for recursive preset override merging
  - run_interaction_test.py A/B/AB comparison orchestrator with markdown table output
affects: [experiment-verdict-pipeline, preset-tuner, weekly-deep-dive]

# Tech tracking
tech-stack:
  added: []
  patterns: [hypothesis-linked-backtest, interaction-test-ab-comparison, preset-override-deep-merge]

key-files:
  created:
    - scripts/run_interaction_test.py
    - tests/unit/core/test_experiment_runner.py
  modified:
    - scripts/run_iteration.py

key-decisions:
  - "Preset overrides merged AFTER _load_preset() loop to avoid race conditions with per-segment loading"
  - "Combined A+B experiment created as real experiment file for ExperimentManager tracking continuity"
  - "subprocess.run(check=True) for child process error propagation in interaction tests"

patterns-established:
  - "Experiment result JSONs saved to results/experiments/{id}/{run_name}.json"
  - "Interaction tests produce markdown comparison tables with Delta(A)/Delta(B) columns"
  - "_deep_merge() for recursive dict merge of strategy preset overrides"

requirements-completed: [EXP-02, EXP-03]

# Metrics
duration: 5min
completed: 2026-04-08
---

# Phase 34 Plan 02: Experiment Runner Summary

**Hypothesis-linked backtest runner with --hypothesis/--run-name flags and A/B/AB interaction test comparison orchestrator**

## Performance

- **Duration:** 5 min
- **Started:** 2026-04-08T06:37:42Z
- **Completed:** 2026-04-08T06:42:16Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- Extended run_iteration.py with --hypothesis and --run-name CLI flags that load experiments, merge preset_overrides after _load_preset(), save result JSONs, and link results via ExperimentManager
- Created run_interaction_test.py that orchestrates A-only, B-only, A+B backtest runs via subprocess and produces markdown comparison tables with WF Sharpe, PF, Max DD, Trade Count metrics and delta columns
- 11 TDD tests covering deep merge, CLI arg parsing, comparison table formatting, and interaction test args

## Task Commits

Each task was committed atomically:

1. **Task 1: Extend run_iteration.py with --hypothesis flag** - `600271b` (feat)
2. **Task 2: Interaction test runner with A/B/AB comparison** - `160382f` (feat)

_Both tasks followed TDD: RED (failing tests) -> GREEN (implementation) -> verify_

## Files Created/Modified
- `scripts/run_iteration.py` - Added _deep_merge(), --hypothesis/--run-name args, experiment loading/linking/result saving
- `scripts/run_interaction_test.py` - New A/B/AB interaction test orchestrator with subprocess calls and comparison table
- `tests/unit/core/test_experiment_runner.py` - 11 tests for deep merge, CLI args, comparison table, interaction args

## Decisions Made
- Preset overrides merged AFTER _load_preset() loop completes (not during) to avoid Pitfall 2 from plan research
- Combined A+B experiment created as a real experiment file so ExperimentManager can track it normally (not ephemeral)
- Used subprocess.run(check=True) to propagate child process failures immediately
- Imported _deep_merge from scripts.run_iteration in interaction test to avoid DRY violation

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
- Ruff I001 (isort) rule required careful noqa placement for scripts.* imports after sys.path manipulation
- S603 (subprocess security) warning suppressed with noqa since all subprocess args come from argparse-validated values (per threat model T-34-05)

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Experiment-linked backtesting fully operational via --hypothesis flag
- Interaction test comparison ready for manual and automated experiment evaluation
- ExperimentManager verdict pipeline complete end-to-end (create -> run -> compare -> verdict)

---
*Phase: 34-experiment-registry-runner*
*Completed: 2026-04-08*
