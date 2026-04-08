---
phase: 34-experiment-registry-runner
verified: 2026-04-08T07:00:00Z
status: passed
score: 4/4
overrides_applied: 0
---

# Phase 34: Experiment Registry & Runner Verification Report

**Phase Goal:** Hypotheses are defined with success criteria before execution, backtest experiments test proposals in isolation and combination, and results are structured for comparison
**Verified:** 2026-04-08T07:00:00Z
**Status:** passed
**Re-verification:** No -- initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Experiment registry stores hypothesis, success criteria (metric + threshold), status, and linked backtest results -- every experiment has a pre-registered definition | VERIFIED | `ExperimentState` in `schemas.py` (lines 731-767): frozen Pydantic model with experiment_id, hypothesis, success_criteria (SuccessCriteria with metric/threshold/operator), status (6-value ExperimentStatus enum), results (list[ExperimentResult]). `ExperimentManager` persists as YAML frontmatter files in `.planning/experiments/`. 19 CRUD tests pass. |
| 2 | `run_iteration.py --hypothesis <id>` runs a parameterized backtest and links results to the hypothesis -- experiment results are automatically associated with their hypothesis | VERIFIED | `run_iteration.py` lines 1000-1003: `--hypothesis` and `--run-name` args. Lines 1069-1080: loads experiment, sets RUNNING status, merges preset_overrides AFTER `_load_preset()` loop. Lines 1321-1346: saves result JSON to `results/experiments/{id}/{run_name}.json` and calls `experiment_mgr.link_result()`. `tracker.save()` at line 1318 preserves history.jsonl recording. |
| 3 | Interaction testing: given hypotheses A and B, the runner executes A-only, B-only, and A+B runs and compares all three -- combination effects are measured, not assumed | VERIFIED | `scripts/run_interaction_test.py` (192 lines): accepts `--experiment-a` and `--experiment-b`, creates combined A+B experiment with deep-merged overrides, runs 3 subprocess calls to `run_iteration.py` with `--hypothesis`/`--run-name`, loads result JSONs, produces markdown comparison table with WF Sharpe/PF/Max DD/Trade Count and Delta(A)/Delta(B) columns. Saves comparison to `results/experiments/comparison-{a}-{b}.md`. |
| 4 | Experiment verdicts (ACCEPT/REJECT/INCONCLUSIVE) are recorded with reasoning and linked to the debate that triggered them | VERIFIED | `ExperimentManager.record_verdict()` (lines 202-232): reads experiment, calls `_compute_verdict()` which uses operator dispatch to determine ACCEPTED/REJECTED/INCONCLUSIVE with 10% relative band, writes status+verdict+reasoning to frontmatter. `create_experiment(debate_id=...)` calls `DebateManager.escalate_debate()` for bidirectional link (lines 134-138). Tests verify all 3 verdict outcomes and debate linkage with real DebateManager. |

**Score:** 4/4 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/finalayze/core/schemas.py` | ExperimentStatus, SuccessCriteria, ExperimentResult, ExperimentState models | VERIFIED | Lines 685-767: 4 frozen Pydantic models with operator whitelist, path-safe ID validation, terminal-requires-verdict constraint |
| `src/finalayze/core/experiment_manager.py` | ExperimentManager CRUD + verdict + debate linkage | VERIFIED | 296 lines, complete ExperimentManager class with create/read/update/list/record_verdict/link_result/get_by_debate + _compute_verdict pure function |
| `scripts/run_iteration.py` | --hypothesis and --run-name flags, preset override merge, experiment result saving | VERIFIED | Lines 950-955 (_deep_merge), 1000-1003 (args), 1069-1080 (experiment load + merge), 1321-1346 (result save + link) |
| `scripts/run_interaction_test.py` | Interaction test orchestrator for A/B/AB runs with comparison table | VERIFIED | 192 lines, _run_hypothesis subprocess caller, _format_comparison_table with Delta columns, main() orchestrating 3 runs + verdict |
| `tests/unit/core/test_experiment_schemas.py` | Schema validation tests | VERIFIED | 144 lines, 13 tests covering all enum values, operator validation, frozen models, terminal-requires-verdict, path-safe ID |
| `tests/unit/core/test_experiment_manager.py` | CRUD + verdict + debate linkage tests | VERIFIED | 235 lines, 19 tests covering create/read/list/update/link_result/record_verdict (3 outcomes)/get_by_debate/FileNotFoundError |
| `tests/unit/core/test_experiment_runner.py` | Tests for --hypothesis integration and interaction test comparison | VERIFIED | 138 lines, 11 tests covering _deep_merge (6 cases), hypothesis arg parsing (2 cases), comparison table (2 cases), interaction args (1 case) |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `experiment_manager.py` | `schemas.py` | TYPE_CHECKING import for ExperimentState, ExperimentResult | WIRED | Lines 19-20: `if TYPE_CHECKING: from finalayze.core.schemas import ExperimentResult, ExperimentState, SuccessCriteria`. Runtime lazy imports in read_experiment() (lines 154-156) and record_verdict() (lines 212-213). |
| `experiment_manager.py` | `debate_manager.py` | DebateManager.escalate_debate() call in create_experiment() | WIRED | Line 135-138: lazy import + `dm.escalate_debate(debate_id, experiment_id)`. Verified end-to-end by test_create_with_debate_link which uses real DebateManager. |
| `run_iteration.py` | `experiment_manager.py` | ExperimentManager.read_experiment() and link_result() calls | WIRED | Line 1071: imports ExperimentManager. Line 1075: calls read_experiment(). Line 1076: calls update_status(). Lines 1338-1345: calls link_result(). |
| `run_interaction_test.py` | `run_iteration.py` | subprocess.run() calls with --hypothesis and --run-name flags | WIRED | Line 47-60: _run_hypothesis() constructs subprocess command with --hypothesis and --run-name flags. Line 121: imports _deep_merge from scripts.run_iteration. |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| `experiment_manager.py` | ExperimentState | YAML frontmatter via yaml.safe_load() | Yes -- creates/reads real .md files | FLOWING |
| `run_iteration.py` (experiment path) | experiment, metrics_dict | ExperimentManager.read_experiment() + backtest metrics | Yes -- reads experiment file, runs real backtest, writes JSON | FLOWING |
| `run_interaction_test.py` | result_a, result_b, result_ab | JSON files from results/experiments/ | Yes -- reads real result JSONs produced by subprocess runs | FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| All 43 experiment tests pass | `uv run pytest tests/unit/core/test_experiment_schemas.py tests/unit/core/test_experiment_manager.py tests/unit/core/test_experiment_runner.py -x -v` | 43 passed in 3.57s | PASS |
| Commits exist | `git log --oneline 41de22e..160382f` | 4 commits: schemas, manager, --hypothesis, interaction test | PASS |
| No TODOs/stubs in new files | grep for TODO/FIXME/PLACEHOLDER in all 4 new files | 0 matches | PASS |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| EXP-01 | 34-01-PLAN | Experiment registry stores hypothesis, criteria, status, results | SATISFIED | ExperimentState schema + ExperimentManager CRUD |
| EXP-02 | 34-02-PLAN | run_iteration.py --hypothesis links results to hypothesis | SATISFIED | --hypothesis/--run-name flags, preset merge, result JSON + link_result |
| EXP-03 | 34-02-PLAN | Interaction testing A-only, B-only, A+B with comparison | SATISFIED | run_interaction_test.py orchestrates 3 runs + comparison table |
| EXP-04 | 34-01-PLAN | Verdicts recorded with reasoning, linked to debate | SATISFIED | record_verdict() with _compute_verdict() + debate escalation |

Note: EXP-01 through EXP-04 are referenced in ROADMAP.md but not defined in REQUIREMENTS.md. They are fully covered by the roadmap success criteria.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| -- | -- | No anti-patterns found | -- | -- |

No TODOs, FIXMEs, placeholders, empty returns, or stub implementations found in any phase 34 files.

### Human Verification Required

None required. All truths are verifiable through code inspection and test execution.

### Gaps Summary

No gaps found. All 4 roadmap success criteria are met, all 7 artifacts are substantive and wired, all 4 key links are connected, all 4 requirements are satisfied, and 43 tests pass.

---

_Verified: 2026-04-08T07:00:00Z_
_Verifier: Claude (gsd-verifier)_
