---
phase: "37"
plan: "01"
subsystem: orchestration
tags: [agent-orchestrator, conflict-detection, debate, experiment, schemas]
dependency_graph:
  requires:
    - "36-01: ConflictDetector"
    - "36-02: DebateManager, ExperimentManager"
    - "core/schemas.py: FileLineSource, AgentOutput, ConflictReport, FactCheckReport"
  provides:
    - "AgentOrchestrator: full conflict-to-debate pipeline coordinator"
    - "snapshot_sha field on FileLineSource: file integrity check for arbiter"
    - "compute_file_sha(): helper for agent claim creation"
  affects:
    - "src/finalayze/orchestration/ (new module)"
    - "src/finalayze/core/schemas.py (FileLineSource extended)"
tech_stack:
  added: []
  patterns:
    - "TDD: RED → GREEN for both tasks"
    - "Dependency injection via constructor (DebateManager, ExperimentManager)"
    - "Fresh ConflictDetector per run() call — no stale dedup"
    - "Frozenset grouping for multi-agent conflict clustering"
key_files:
  created:
    - src/finalayze/orchestration/agent_orchestrator.py
    - tests/unit/core/test_agent_orchestrator.py
  modified:
    - src/finalayze/core/schemas.py
    - tests/unit/core/test_debate_schemas.py
decisions:
  - "snapshot_sha added as optional field with None default for backward compatibility"
  - "compute_file_sha placed in schemas.py (not a separate module) to keep Layer 0 self-contained"
  - "AgentOrchestrator groups conflicts by frozenset(agent_names) — independent pairs get separate debates"
  - "debate_id uses SHA-256 of sorted(agents) + ISO-minute timestamp — idempotent within a minute"
  - "finalize_debate() hypothesis uses first 3 contradicted claim statements concatenated"
metrics:
  duration_seconds: 253
  completed_date: "2026-04-12"
  tasks_completed: 2
  tasks_total: 2
  files_created: 2
  files_modified: 2
---

# Phase 37 Plan 01: AgentOrchestrator + snapshot_sha Summary

AgentOrchestrator pipeline coordinator with fresh-per-run ConflictDetector, debate grouping by agent pair, and experiment escalation on contradictions; snapshot_sha field on FileLineSource for arbiter integrity checking.

## Tasks Completed

| Task | Description | Commit | Status |
|------|-------------|--------|--------|
| 1 | Add snapshot_sha to FileLineSource + compute_file_sha helper | 0edf3f3 | Done |
| 2 | Build AgentOrchestrator with TDD tests | ba95816 | Done |

## What Was Built

### Task 1: snapshot_sha on FileLineSource (ORCH-03)

Added `snapshot_sha: str | None = None` to `FileLineSource` in `src/finalayze/core/schemas.py`. The field is optional with `None` default so all existing claims without a SHA remain valid (zero breaking change).

Also added `compute_file_sha(path: str) -> str` helper at the end of `schemas.py` that computes SHA-256 of a file's content. Agent definitions use this when creating `FileLineSource` claims to capture the file's integrity at claim-creation time.

4 new tests in `test_debate_schemas.py`:
- Backward compatibility (no sha → None default)
- Construction with SHA
- JSON serialization with None sha
- Claim wrapping FileLineSource with sha (regression)

### Task 2: AgentOrchestrator (ORCH-01)

Created `src/finalayze/orchestration/agent_orchestrator.py` with class `AgentOrchestrator`:

**`run(outputs: list[AgentOutput]) -> list[str]`:**
1. Instantiates a fresh `ConflictDetector()` per call (avoids stale dedup pitfall)
2. Calls `detector.detect(outputs)`
3. Short-circuits and returns `[]` if no conflicts
4. Groups conflicts by `frozenset(agent_names)` — independent agent pairs get separate debates
5. Generates debate_id via SHA-256(sorted_agents + ISO-minute timestamp)
6. Creates debate via `DebateManager.create_debate()`
7. Adds each agent's position via `DebateManager.add_agent_position()`
8. Returns list of debate_ids

**`finalize_debate(debate_id, report) -> str | None`:**
1. Calls `DebateManager.add_arbiter_report()`
2. If contradictions: creates experiment via `ExperimentManager.create_experiment()`, returns experiment_id
3. If no contradictions: calls `DebateManager.resolve_debate()`, returns None

Constructor accepts `debate_manager`, `experiment_manager`, `debates_dir`, `experiments_dir` for dependency injection.

7 unit tests cover all branches including the fresh-detector invariant.

## Deviations from Plan

None — plan executed exactly as written.

## Verification Results

```
tests/unit/core/test_agent_orchestrator.py  7 passed
tests/unit/core/test_debate_schemas.py     31 passed (4 new snapshot_sha tests)
Total: 38 passed
```

Acceptance criteria satisfied:
- `grep -n "snapshot_sha" src/finalayze/core/schemas.py` → 2 matches (field + docstring)
- `grep -n "snapshot_sha" tests/unit/core/test_debate_schemas.py` → 8 matches
- `grep -n "class AgentOrchestrator"` → 1 match at line 46
- `grep -n "def run"` → 1 match at line 82
- `grep -n "def finalize_debate"` → 1 match at line 146
- All orchestrator tests pass (7 PASSED)

## Self-Check: PASSED

- `src/finalayze/orchestration/agent_orchestrator.py` exists
- `tests/unit/core/test_agent_orchestrator.py` exists (113+ lines)
- Commit 0edf3f3 exists (snapshot_sha task)
- Commit ba95816 exists (AgentOrchestrator task)
- 38 tests pass with zero regressions in new code (pre-existing `test_bond_threshold_stale` failure is unrelated)
