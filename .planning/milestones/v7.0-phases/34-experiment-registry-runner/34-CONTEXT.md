# Phase 34: Experiment Registry & Runner - Context

**Gathered:** 2026-04-08
**Status:** Ready for planning

<domain>
## Phase Boundary

This phase delivers the experiment lifecycle infrastructure: a registry for hypothesis definitions, a runner that executes parameterized backtests with interaction testing (A-only, B-only, A+B), and automated verdict determination. Scope: experiment schemas, ExperimentManager CRUD, run_iteration.py --hypothesis extension, interaction testing, verdict logic, debate linkage. Does NOT include UI (Phase 35).

</domain>

<decisions>
## Implementation Decisions

### Experiment Registry Design
- Experiment definitions stored as YAML files in `.planning/experiments/` — matches debate pattern from Phase 33
- Each experiment has: `hypothesis`, `success_criteria` (metric + threshold), `status`, `debate_id`, `results[]`
- Status enum: `PENDING`, `RUNNING`, `COMPLETED`, `ACCEPTED`, `REJECTED`, `INCONCLUSIVE` (matches SC-1 and SC-4)
- Registry implementation: Pydantic models in `core/schemas.py` + `ExperimentManager` in `core/experiment_manager.py` — mirrors Phase 33 DebateManager pattern

### Runner & Interaction Testing
- Extend `run_iteration.py` with `--hypothesis <id>` flag that loads experiment definition, runs backtest, links results
- Interaction testing: runner executes A-only, B-only, A+B as three separate `run_iteration.py` calls with isolated preset overrides
- Results stored in `results/experiments/{experiment_id}/` directory with A-only.json, B-only.json, AB.json
- Comparison output: structured markdown table showing metric deltas across A/B/AB runs

### Verdicts & Debate Linkage
- Automated verdict based on success_criteria thresholds — if metric meets threshold → ACCEPT, below → REJECT, ambiguous → INCONCLUSIVE
- Bidirectional debate link: `debate_id` field in experiment; DebateManager updates debate's `experiment_id` when experiment created
- `reasoning` field in verdict section of experiment file — auto-generated from metric comparison
- Experiment creation triggered manually via command + when debate has `status: escalated`

### Claude's Discretion
- Internal implementation details of preset override mechanics for interaction testing
- Specific ambiguity thresholds for INCONCLUSIVE verdict
- Comparison table column selection and formatting

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/finalayze/core/debate_manager.py` — DebateManager CRUD pattern to mirror for ExperimentManager
- `src/finalayze/core/schemas.py` — Debate protocol Pydantic models (Phase 33) as template
- `scripts/run_iteration.py` — Existing backtest runner to extend with --hypothesis flag
- `results/iterations/history.jsonl` — Existing iteration result storage format
- `src/finalayze/strategies/presets/*.yaml` — Strategy preset YAML files for override mechanics

### Established Patterns
- YAML frontmatter + markdown for `.planning/` state files (debates, phases)
- Pydantic v2 models with validators in `core/schemas.py` (Layer 0)
- Manager classes with file I/O for CRUD operations
- `run_iteration.py` accepts `--name`, `--description`, `--segments` flags

### Integration Points
- `.planning/experiments/` — new directory for experiment definitions
- `results/experiments/{id}/` — new directory for experiment results
- `scripts/run_iteration.py` — extended with `--hypothesis` flag
- `src/finalayze/core/debate_manager.py` — updated to support `experiment_id` linkage
- `src/finalayze/core/schemas.py` — new experiment Pydantic models

</code_context>

<specifics>
## Specific Ideas

No specific requirements — open to standard approaches matching project conventions.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>
