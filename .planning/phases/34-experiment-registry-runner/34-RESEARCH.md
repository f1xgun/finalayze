# Phase 34: Experiment Registry & Runner - Research

**Researched:** 2026-04-08
**Domain:** Experiment lifecycle infrastructure — registry, parameterized backtest runner, interaction testing, verdict determination, debate linkage
**Confidence:** HIGH

## Summary

Phase 34 delivers the experiment lifecycle infrastructure needed to translate structured debates (Phase 33) into falsifiable backtest experiments. The system must store hypothesis definitions with pre-registered success criteria, execute parameterized backtests linked to those hypotheses, run A-only/B-only/A+B interaction tests, and record ACCEPT/REJECT/INCONCLUSIVE verdicts tied back to the debate that triggered them.

The codebase already has a complete, working template: Phase 33 delivered `DebateManager` + `DebateState` schemas using the YAML frontmatter + markdown pattern in `.planning/debates/`. Phase 34 replicates this pattern almost exactly for experiments in `.planning/experiments/`. The primary new complexity is the interaction testing mechanic — three isolated `run_iteration.py` invocations with preset overrides — and the verdict determination logic.

`run_iteration.py` accepts `--segments`, `--start-date`, `--end-date`, `--name`, `--description` and loads strategy config from YAML presets in `src/finalayze/strategies/presets/`. Extending it with `--hypothesis <id>` means: load the experiment file, apply `preset_overrides` from the experiment definition, run the backtest, save results to `results/experiments/{experiment_id}/`, update experiment status and link the result in the frontmatter.

**Primary recommendation:** Mirror DebateManager/DebateState verbatim for ExperimentManager/ExperimentState. The interaction test runner is a thin orchestrator that calls `run_iteration.py` three times with different override flags and compares the three result JSON files.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Experiment definitions stored as YAML files in `.planning/experiments/` — matches debate pattern from Phase 33
- Each experiment has: `hypothesis`, `success_criteria` (metric + threshold), `status`, `debate_id`, `results[]`
- Status enum: `PENDING`, `RUNNING`, `COMPLETED`, `ACCEPTED`, `REJECTED`, `INCONCLUSIVE` (matches SC-1 and SC-4)
- Registry implementation: Pydantic models in `core/schemas.py` + `ExperimentManager` in `core/experiment_manager.py` — mirrors Phase 33 DebateManager pattern
- Extend `run_iteration.py` with `--hypothesis <id>` flag that loads experiment definition, runs backtest, links results
- Interaction testing: runner executes A-only, B-only, A+B as three separate `run_iteration.py` calls with isolated preset overrides
- Results stored in `results/experiments/{experiment_id}/` directory with A-only.json, B-only.json, AB.json
- Comparison output: structured markdown table showing metric deltas across A/B/AB runs
- Automated verdict based on success_criteria thresholds — if metric meets threshold → ACCEPT, below → REJECT, ambiguous → INCONCLUSIVE
- Bidirectional debate link: `debate_id` field in experiment; DebateManager updates debate's `experiment_id` when experiment created
- `reasoning` field in verdict section of experiment file — auto-generated from metric comparison
- Experiment creation triggered manually via command + when debate has `status: escalated`

### Claude's Discretion
- Internal implementation details of preset override mechanics for interaction testing
- Specific ambiguity thresholds for INCONCLUSIVE verdict
- Comparison table column selection and formatting

### Deferred Ideas (OUT OF SCOPE)
- None — discussion stayed within phase scope.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| EXP-01 | Experiment registry stores hypothesis, success criteria (metric + threshold), status, and linked backtest results | ExperimentState schema mirrors DebateState; YAML frontmatter file per experiment in `.planning/experiments/`. ExperimentManager CRUD mirrors DebateManager. |
| EXP-02 | `run_iteration.py --hypothesis <id>` runs a parameterized backtest and links results to the hypothesis | `_parse_args()` extended with `--hypothesis`; loader reads experiment file, merges `preset_overrides` into `_load_preset()` output, saves JSON to `results/experiments/{id}/`, calls `ExperimentManager.link_result()`. |
| EXP-03 | Interaction testing: given hypotheses A and B, runner executes A-only, B-only, and A+B runs and compares all three | New `scripts/run_interaction_test.py` orchestrates three `run_iteration.py` subprocess calls (or direct function calls with override injection). Reads three result JSONs, produces comparison markdown table. |
| EXP-04 | Experiment verdicts (ACCEPT/REJECT/INCONCLUSIVE) are recorded with reasoning and linked to the debate that triggered them | `ExperimentManager.record_verdict()` computes verdict from `success_criteria`, writes to frontmatter, appends reasoning section to body. `DebateManager.escalate_debate()` already accepts `experiment_id` — EXP-04 requires the reverse link to also be updated on verdict. |
</phase_requirements>

## Standard Stack

### Core (all already installed — no new dependencies)
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| pydantic v2 | 2.x (project-wide) | ExperimentState, SuccessCriteria schemas | Mandatory per CLAUDE.md; frozen models for Layer 0 |
| pyyaml | project-wide | YAML frontmatter read/write | Same as DebateManager |
| pathlib | stdlib | File I/O for `.planning/experiments/` | Same as DebateManager |
| argparse | stdlib | `--hypothesis` flag in run_iteration.py | Same as existing flags |
| subprocess / direct call | stdlib | Orchestrating three backtest runs for interaction test | No external dep needed |

[VERIFIED: codebase grep — all packages present in pyproject.toml/uv.lock]

### No New Dependencies Required
The entire phase can be implemented with libraries already in the project. The ExperimentManager pattern is a direct structural copy of DebateManager.

**Installation:** None needed.

## Architecture Patterns

### Recommended Project Structure
```
src/finalayze/core/
├── schemas.py                    # Add ExperimentStatus, SuccessCriteria, ExperimentState
├── experiment_manager.py         # NEW — mirrors debate_manager.py exactly
└── debate_manager.py             # UPDATE — add update_experiment_id() if needed (already has escalate_debate())

.planning/experiments/            # NEW directory
├── {experiment_id}.md            # YAML frontmatter + markdown body per experiment

results/experiments/              # NEW directory
├── {experiment_id}/
│   ├── A-only.json               # run_iteration output for hypothesis A alone
│   ├── B-only.json               # run_iteration output for hypothesis B alone
│   └── AB.json                   # run_iteration output for A+B combined

scripts/
├── run_iteration.py              # EXTEND with --hypothesis flag
└── run_interaction_test.py       # NEW — orchestrates A/B/AB runs, prints comparison table
```

### Pattern 1: ExperimentState Schema (mirrors DebateState)

**What:** Pydantic model with YAML-serializable fields. Frozen. Layer 0.
**When to use:** All experiment state read/write operations.

```python
# Source: mirrors src/finalayze/core/schemas.py DebateState pattern [VERIFIED: codebase]
from __future__ import annotations
from enum import auto
from typing import Any
from pydantic import BaseModel, ConfigDict, model_validator
from finalayze.core.schemas import StrEnum  # or re-use local StrEnum


class ExperimentStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    INCONCLUSIVE = "inconclusive"


class SuccessCriteria(BaseModel):
    model_config = ConfigDict(frozen=True)
    metric: str          # e.g. "wf_sharpe", "profit_factor"
    threshold: float     # e.g. 0.1 (must be >= this to ACCEPT)
    operator: str = ">=" # ">=" | "<=" | ">" | "<"


class ExperimentResult(BaseModel):
    model_config = ConfigDict(frozen=True)
    run_name: str        # e.g. "A-only", "B-only", "AB"
    iteration_name: str  # key into results/iterations/ or results/experiments/
    metrics: dict[str, Any]  # snapshot of key metrics at verdict time


class ExperimentState(BaseModel):
    model_config = ConfigDict(frozen=True)
    experiment_id: str
    hypothesis: str
    success_criteria: SuccessCriteria
    status: ExperimentStatus
    created: str         # ISO date "YYYY-MM-DD"
    debate_id: str | None = None
    results: list[ExperimentResult] = []
    verdict: str | None = None
    reasoning: str | None = None

    @model_validator(mode="after")
    def terminal_status_requires_verdict(self) -> ExperimentState:
        terminal = {ExperimentStatus.ACCEPTED, ExperimentStatus.REJECTED, ExperimentStatus.INCONCLUSIVE}
        if self.status in terminal and self.verdict is None:
            msg = "verdict is required when status is terminal (ACCEPTED/REJECTED/INCONCLUSIVE)"
            raise ValueError(msg)
        return self
```

[VERIFIED: matches DebateState validation pattern in schemas.py line 672-678]

### Pattern 2: ExperimentManager (mirrors DebateManager)

**What:** File I/O manager for `.planning/experiments/` directory.
**When to use:** All CRUD operations on experiment definitions.

Key methods to implement (parallel to DebateManager):
- `create_experiment(experiment_id, hypothesis, success_criteria, debate_id)` → Path
- `read_experiment(experiment_id)` → ExperimentState
- `update_status(experiment_id, status)` → None
- `link_result(experiment_id, result: ExperimentResult)` → None
- `record_verdict(experiment_id, verdict, reasoning)` → None
- `list_experiments()` → list[str]
- `get_by_debate(debate_id)` → ExperimentState | None (for reverse lookup)

The internal `_read_file()` / `_write_file()` helpers are identical to DebateManager — split on `---\n`, parse YAML frontmatter, write back.

[VERIFIED: DebateManager._read_file/_write_file pattern in debate_manager.py lines 46-80]

### Pattern 3: Preset Override Mechanics for Interaction Testing

**What:** The interaction test needs to run a backtest with hypothesis A's changes, B's changes, or both. The cleanest approach is `preset_overrides` stored in each experiment YAML: a nested dict matching the preset structure. The runner deep-merges overrides into the loaded preset before building strategies.

**Experiment file example:**
```yaml
experiment_id: "2026-04-08-dual-momentum-ru"
hypothesis: "Enabling dual_momentum on ru_blue_chips will improve PF above 1.3"
success_criteria:
  metric: profit_factor
  threshold: 1.3
  operator: ">="
status: pending
created: "2026-04-08"
debate_id: "2026-04-07-dual-momentum-debate"
preset_overrides:
  ru_blue_chips:
    strategies:
      dual_momentum:
        enabled: true
        weight: 0.25
results: []
verdict: null
reasoning: null
```

**Deep merge approach** (simpler than subprocess, avoids re-parsing):
```python
def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base, returning new dict."""
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result
```

[ASSUMED: specific deep merge implementation detail — no prior art in codebase for this]

### Pattern 4: --hypothesis Extension in run_iteration.py

**What:** New `--hypothesis` argument that, when provided:
1. Loads experiment file via ExperimentManager
2. Merges `preset_overrides` into each segment's strategy config
3. Runs backtest normally
4. Saves condensed result JSON to `results/experiments/{experiment_id}/{run_name}.json`
5. Calls `ExperimentManager.link_result()` to update frontmatter

**Extension point:**
```python
# In _parse_args():
parser.add_argument("--hypothesis", default=None, help="Experiment ID to link backtest results")
parser.add_argument("--run-name", default=None, help="Label for interaction test run (A-only, B-only, AB)")

# In main(), after loading strategy_configs:
if args.hypothesis:
    mgr = ExperimentManager()
    exp = mgr.read_experiment(args.hypothesis)
    mgr.update_status(args.hypothesis, ExperimentStatus.RUNNING)
    for seg, overrides in exp.preset_overrides.items():
        if seg in strategy_configs:
            strategy_configs[seg] = _deep_merge(strategy_configs[seg], overrides)
```

[VERIFIED: run_iteration.py _parse_args() at line 950, strategy_configs loading at lines 1046-1049]

### Pattern 5: Interaction Test Runner

**What:** New script `scripts/run_interaction_test.py` that:
1. Takes two experiment IDs (A and B)
2. Calls `run_iteration.py` three times via subprocess (or direct `main()` import)
3. Reads results from `results/experiments/{id}/`
4. Computes deltas across A-only/B-only/AB
5. Prints comparison markdown table

**Subprocess approach** (recommended — avoids coupling to run_iteration.py internals):
```python
import subprocess, sys

def _run(hypothesis: str, run_name: str, extra_args: list[str]) -> None:
    cmd = [sys.executable, "scripts/run_iteration.py",
           "--name", f"{hypothesis}-{run_name}",
           "--description", f"Interaction test: {run_name}",
           "--hypothesis", hypothesis,
           "--run-name", run_name,
           *extra_args]
    subprocess.run(cmd, check=True)
```

**Result JSON format** (`results/experiments/{experiment_id}/{run_name}.json`):
```json
{
  "experiment_id": "2026-04-08-dual-momentum-ru",
  "run_name": "A-only",
  "iteration_name": "2026-04-08-dual-momentum-ru-A-only",
  "wf_sharpe": 0.12,
  "profit_factor": 1.35,
  "wf_max_drawdown": 0.18,
  "trade_count": 47,
  "verdict": "ACCEPT"
}
```

[VERIFIED: results/iterations/history.jsonl format — name, wf_sharpe, wf_max_drawdown, trade_count, verdict fields]

### Pattern 6: Verdict Determination

**What:** Automated verdict computed from `success_criteria` after results are linked.
**Logic:**
- Evaluate `metric` from results against `threshold` using `operator`
- If meets threshold → ACCEPT
- If clearly misses threshold by >10% relative → REJECT
- Within 10% band → INCONCLUSIVE (Claude's discretion applies here for exact threshold)

```python
def _compute_verdict(criteria: SuccessCriteria, metric_value: float) -> tuple[str, str]:
    """Returns (verdict, reasoning)."""
    ops = {">=": operator.ge, "<=": operator.le, ">": operator.gt, "<": operator.lt}
    op_fn = ops[criteria.operator]
    
    if op_fn(metric_value, criteria.threshold):
        return "ACCEPTED", f"{criteria.metric}={metric_value:.4f} meets threshold {criteria.operator} {criteria.threshold}"
    
    # Compute relative distance from threshold
    relative_miss = abs(metric_value - criteria.threshold) / max(abs(criteria.threshold), 1e-9)
    if relative_miss <= 0.10:  # Within 10% — INCONCLUSIVE (threshold is Claude's discretion)
        return "INCONCLUSIVE", f"{criteria.metric}={metric_value:.4f} within 10% of threshold {criteria.threshold}"
    
    return "REJECTED", f"{criteria.metric}={metric_value:.4f} misses threshold {criteria.operator} {criteria.threshold} by {relative_miss:.1%}"
```

[ASSUMED: 10% INCONCLUSIVE band — reasonable default but explicitly Claude's discretion per CONTEXT.md]

### Anti-Patterns to Avoid

- **Direct file mutation without read-first:** Always `_read_file()` before `_write_file()` to avoid overwriting concurrent changes.
- **Importing ExperimentManager in Layer 0:** ExperimentManager has file I/O; it lives at Layer 0 only because DebateManager does, but must have zero project-layer imports (only yaml, pathlib, datetime, TYPE_CHECKING for schemas).
- **Hardcoding metric names:** Store metric name as a string field in SuccessCriteria; read from `IterationMetrics` via `getattr()` to avoid coupling.
- **Blocking interaction test runner:** Three sequential backtest runs can take 30+ minutes each. The comparison script should emit progress markers and allow segment filtering via `--segments` pass-through.
- **Mutable Pydantic models:** All schemas use `ConfigDict(frozen=True)`. When updating experiment state, read frontmatter dict, mutate dict, write back — never mutate the Pydantic model.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| YAML parsing | Custom parser | `yaml.safe_load()` / `yaml.dump()` | Already used in DebateManager |
| Deep dict merge | Recursive reimplementation | Established `_deep_merge()` helper | Simple 10-line function, no deps |
| Metric comparison | Custom eval() | `operator` module (stdlib) | Safe, readable, no injection risk |
| Progress tracking | Custom progress bar | `print()` statements matching existing run_iteration.py style | Consistency with existing output |

**Key insight:** The entire phase is wiring, not algorithm design. The backtest engine, metrics computation, and iteration tracking already exist. This phase adds a thin bookkeeping layer on top.

## Common Pitfalls

### Pitfall 1: Frozen Pydantic Model Update Pattern
**What goes wrong:** Trying to call `experiment.status = "running"` on a frozen Pydantic model.
**Why it happens:** All schemas use `ConfigDict(frozen=True)`.
**How to avoid:** Read frontmatter dict, update the dict, write back. Never try to mutate ExperimentState directly. ExperimentManager always works at the dict level internally.
**Warning signs:** `ValidationError: Instance is frozen` at runtime.

### Pitfall 2: Preset Override Order-of-Operations
**What goes wrong:** Overrides applied before `_load_preset()` returns, so the base preset is loaded fresh and overrides are discarded.
**Why it happens:** `strategy_configs[seg] = _load_preset(seg)` happens in a loop at line 1046-1049 of run_iteration.py. The override merge must happen AFTER this loop.
**How to avoid:** Apply `_deep_merge(strategy_configs[seg], overrides[seg])` in a second pass after the preset load loop.
**Warning signs:** Interaction test produces identical results for A-only and B-only runs.

### Pitfall 3: Result JSON Path Confusion
**What goes wrong:** Run results saved to `results/iterations/{name}/` instead of `results/experiments/{experiment_id}/{run_name}.json`.
**Why it happens:** `run_iteration.py` uses `output_root = Path(args.output)` which defaults to `results/iterations/`. The `--hypothesis` flow needs a separate output path.
**How to avoid:** When `--hypothesis` is set, override the output path to `results/experiments/{experiment_id}/`. The iteration tracker should still record to history.jsonl for tracking, but the main result JSON goes to the experiment directory.
**Warning signs:** `ExperimentManager.link_result()` can't find the result file.

### Pitfall 4: Interaction Test Race Between Status Updates
**What goes wrong:** Running A-only sets status to RUNNING; B-only run tries to set status to RUNNING on a different experiment and overwrites A's RUNNING status with its own COMPLETED status.
**Why it happens:** Both experiments share the same ExperimentManager and each run updates status independently.
**How to avoid:** The interaction test script manages status for the *interaction test* as a whole, not per-hypothesis. Each hypothesis experiment's status is updated only at the end of its own run.
**Warning signs:** One experiment shows COMPLETED while its results are missing.

### Pitfall 5: Missing `experiment_id` in DebateState on Escalation
**What goes wrong:** Debate `status=escalated` but `experiment_id=None`, causing `escalated_requires_experiment_id` validator to fail on read.
**Why it happens:** DebateManager.escalate_debate() already sets `experiment_id` correctly (line 142-152 of debate_manager.py). But creating an experiment manually (without coming from a debate) would leave the debate unlinked.
**How to avoid:** ExperimentManager.create_experiment() must call DebateManager.escalate_debate() if `debate_id` is provided. This is the bidirectional link requirement in EXP-04.
**Warning signs:** `ValidationError` on `read_debate()` after manual experiment creation.

## Code Examples

Verified patterns from official sources:

### ExperimentManager._write_file (mirrors DebateManager._write_file)
```python
# Source: src/finalayze/core/debate_manager.py lines 70-80 [VERIFIED: codebase]
def _write_file(self, experiment_id: str, frontmatter: dict, body: str) -> None:
    path = self._experiment_path(experiment_id)
    yaml_text = yaml.dump(
        frontmatter,
        default_flow_style=False,
        allow_unicode=True,
        sort_keys=False,
    )
    content = f"---\n{yaml_text}---\n{body}"
    path.write_text(content, encoding="utf-8")
```

### Linking result after backtest
```python
# After run_iteration.py completes its run loop, call:
if args.hypothesis:
    result_path = Path("results/experiments") / args.hypothesis / f"{run_name}.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps({
        "experiment_id": args.hypothesis,
        "run_name": run_name,
        "iteration_name": args.name,
        "wf_sharpe": float(metrics.wf_sharpe),
        "profit_factor": float(metrics.profit_factor),
        "wf_max_drawdown": float(metrics.wf_max_drawdown),
        "trade_count": metrics.trade_count,
    }, indent=2))
    mgr = ExperimentManager()
    mgr.link_result(args.hypothesis, ExperimentResult(
        run_name=run_name,
        iteration_name=args.name,
        metrics={...},
    ))
```

### Comparison table format (markdown, following existing style)
```
| Metric           | A-only | B-only |   A+B | Delta(A) | Delta(B) |
|------------------|--------|--------|-------|----------|----------|
| WF Sharpe        |  0.082 |  0.091 | 0.112 |   +0.030 |   +0.021 |
| Profit Factor    |  1.28  |  1.31  |  1.41 |   +0.130 |   +0.100 |
| Max Drawdown (%) |  0.19  |  0.17  |  0.16 |   -0.030 |   -0.010 |
| Trade Count      |     42 |     39 |    51 |       +9 |      +12 |
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Ad-hoc iteration naming | Structured IterationTracker with history.jsonl | Phase 5 (ML overhaul) | Gives us query-able baseline for experiment result comparison |
| No debate linkage | DebateState.experiment_id + DebateManager.escalate_debate() | Phase 33 | The reverse link (debate→experiment) is already built; Phase 34 adds experiment→debate |

**Available (already built):**
- `IterationTracker` — saves metrics to `results/iterations/history.jsonl`. Experiment results should ALSO be appended here for unified history.
- `DebateManager.escalate_debate(debate_id, experiment_id)` — already sets `status=escalated` and `experiment_id` on the debate. Phase 34 just needs to call this from ExperimentManager.create_experiment().

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | 10% relative miss from threshold → INCONCLUSIVE band | Architecture Patterns: Verdict Determination | Low — CONTEXT.md explicitly marks this as Claude's discretion; any reasonable value works |
| A2 | Deep merge is the right preset override mechanic (vs. full preset file per experiment) | Architecture Patterns: Preset Override Mechanics | Low — simpler than alternatives; easily changed |
| A3 | Subprocess approach for interaction test runner | Architecture Patterns: Interaction Test Runner | Medium — subprocess avoids coupling but adds process overhead; direct function import would be faster |

**Three assumptions, all low-to-medium risk.** All are Claude's discretion areas per CONTEXT.md.

## Open Questions

1. **run_name for single-hypothesis runs**
   - What we know: `--run-name` is only meaningful for interaction tests (A-only/B-only/AB)
   - What's unclear: Should `--hypothesis` without `--run-name` use "main" or the iteration name?
   - Recommendation: Default `run_name = "main"` when `--run-name` not specified.

2. **Which metric drives the verdict for interaction tests?**
   - What we know: Each experiment has one `success_criteria` metric. A+B run is the primary result.
   - What's unclear: Do we evaluate the criterion on A+B only, or on whichever run scores best?
   - Recommendation: Evaluate on A+B run as primary, record A-only and B-only for comparison only.

3. **IterationTracker integration for experiment runs**
   - What we know: `run_iteration.py` appends to `results/iterations/history.jsonl` via `IterationTracker`.
   - What's unclear: Should experiment runs also appear in history.jsonl (for `backtest-iteration` skill) or be experiment-only?
   - Recommendation: Yes — append to history.jsonl with a `tags: [experiment, {experiment_id}]` field so iteration history skill can filter.

## Environment Availability

Step 2.6: SKIPPED (no external dependencies — this phase is purely code/config/file-system changes within the existing Python project)

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (uv run pytest) |
| Config file | pyproject.toml [tool.pytest] |
| Quick run command | `uv run pytest tests/unit/core/test_experiment_schemas.py tests/unit/core/test_experiment_manager.py -x` |
| Full suite command | `uv run pytest tests/unit/core/ -x` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| EXP-01 | ExperimentState Pydantic model validates all fields, rejects terminal status without verdict | unit | `pytest tests/unit/core/test_experiment_schemas.py -x` | Wave 0 |
| EXP-01 | ExperimentManager creates, reads, lists, updates experiment files | unit | `pytest tests/unit/core/test_experiment_manager.py -x` | Wave 0 |
| EXP-02 | `--hypothesis` flag loads experiment, merges overrides, saves result JSON, links result | integration | `pytest tests/integration/test_run_hypothesis.py -x` | Wave 0 |
| EXP-03 | Interaction test produces three result JSONs and comparison table | integration | `pytest tests/integration/test_interaction_test.py -x` | Wave 0 |
| EXP-04 | record_verdict() sets status + reasoning; DebateManager.escalate_debate() called on create | unit | `pytest tests/unit/core/test_experiment_manager.py::TestVerdictRecording -x` | Wave 0 |

### Sampling Rate
- **Per task commit:** `uv run pytest tests/unit/core/test_experiment_schemas.py tests/unit/core/test_experiment_manager.py -x`
- **Per wave merge:** `uv run pytest tests/unit/core/ -x`
- **Phase gate:** Full suite green before `/gsd-verify-work`

### Wave 0 Gaps
- [ ] `tests/unit/core/test_experiment_schemas.py` — covers EXP-01 (schemas)
- [ ] `tests/unit/core/test_experiment_manager.py` — covers EXP-01 (CRUD), EXP-04 (verdict + debate linkage)
- [ ] `tests/integration/test_run_hypothesis.py` — covers EXP-02 (--hypothesis flag, dry-run mode)
- [ ] `tests/integration/test_interaction_test.py` — covers EXP-03 (three runs + comparison)

*(Integration tests should use `--dry-run` mode or mock the backtest engine to avoid network calls)*

## Security Domain

### Applicable ASVS Categories
| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | N/A — local file system only |
| V3 Session Management | no | N/A |
| V4 Access Control | no | N/A — local CLI tool |
| V5 Input Validation | yes | Pydantic v2 validators on ExperimentState; operator whitelist for verdict logic |
| V6 Cryptography | no | N/A |

### Known Threat Patterns for Stack
| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| YAML injection via experiment file content | Tampering | `yaml.safe_load()` (not `yaml.load()`); already used by DebateManager |
| operator injection in verdict logic | Tampering | Whitelist `operator` values: `[">=", "<=", ">", "<"]`; reject anything else at Pydantic validation time |
| Path traversal via experiment_id | Tampering | Validate `experiment_id` contains only `[a-zA-Z0-9_-]` characters; reject `../` prefixes |

## Sources

### Primary (HIGH confidence)
- `src/finalayze/core/debate_manager.py` — DebateManager CRUD pattern (full read, verified line numbers)
- `src/finalayze/core/schemas.py` lines 587-679 — Debate protocol schemas (DebateState, FactCheckReport, AgentOutput)
- `scripts/run_iteration.py` lines 950-1220 — CLI argument parsing, strategy config loading, result output
- `results/iterations/history.jsonl` — Existing result format (verified 5 records)
- `tests/unit/core/test_debate_manager.py` — Test structure pattern for ExperimentManager tests
- `tests/unit/core/test_debate_schemas.py` — Test structure pattern for ExperimentState tests

### Secondary (MEDIUM confidence)
- `.planning/ROADMAP.md` lines 191-200 — Phase 34 success criteria and EXP-01..04 descriptions

### Tertiary (LOW confidence)
- None

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all libraries already in project, verified
- Architecture: HIGH — direct mirror of Phase 33 pattern, verified in codebase
- Pitfalls: HIGH — identified from actual code reading (frozen models, preset load order, path conventions)
- Integration: MEDIUM — subprocess interaction test and --hypothesis integration not yet coded, but pattern is clear

**Research date:** 2026-04-08
**Valid until:** 2026-05-08 (stable codebase, no external dependencies)
