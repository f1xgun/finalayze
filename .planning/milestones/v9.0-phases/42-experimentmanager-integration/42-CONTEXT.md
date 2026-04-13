# Phase 42: ExperimentManager Integration - Context

**Gathered:** 2026-04-13
**Status:** Ready for planning

<domain>
## Phase Boundary

Add opt-in `--experiment-id` flag to auto_ml_research.py that creates ExperimentManager entries with hypothesis lifecycle and verdict recording. JSONL audit trail preserved. Backward compatible.

</domain>

<decisions>
## Implementation Decisions

### ExperimentManager API Usage
- One ExperimentManager entry per `run_research_loop()` call — all sub-experiment configs logged in JSONL under one umbrella experiment
- Auto-generated hypothesis: "AutoML research: {strategy} on {segment}" — descriptive and queryable
- Verdict logic: best score > baseline score → ACCEPT; all sub-experiments discard → REJECT; mixed results → INCONCLUSIVE
- Lazy import of ExperimentManager inside `--experiment-id` code path — no import overhead when flag not used

### Backward Compatibility
- JSONL log always written (both with and without --experiment-id) — JSONL is audit trail, ExperimentManager is lifecycle
- File-based ExperimentManager with unique IDs — no shared state between concurrent runs
- Validate --experiment-id is path-safe (alphanumeric, underscore, dash only) — ExperimentManager uses ID as filename

### Claude's Discretion
- ExperimentManager method calls (create, link_result, record_verdict) — follow existing API
- Error handling if ExperimentManager operations fail (should not crash the research loop)
- Test structure for concurrent ID isolation

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `ExperimentManager` in `src/finalayze/core/experiment_manager.py` — create(), get(), record_verdict(), compute_verdict()
- `ExperimentState`, `SuccessCriteria` in `src/finalayze/core/schemas.py`
- `_log_result()` in `auto_ml_research.py` — JSONL logging (already works)

### Integration Points
- `auto_ml_research.py:main()` — add `--experiment-id` argparse argument
- `auto_ml_research.py:run_research_loop()` — add experiment_id parameter, create/link/verdict calls
- `auto_ml_research.py:_print_summary()` — report ExperimentManager verdict if experiment_id set

</code_context>

<specifics>
## Specific Ideas

- ExperimentManager.create() takes hypothesis, success_criteria, status — match existing schema
- compute_verdict() already returns ACCEPT/REJECT/INCONCLUSIVE based on criteria
- Research confirmed: one entry per loop run, not per internal config — avoids namespace pollution

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>
