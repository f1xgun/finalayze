---
phase: 33-structured-debate-protocol
plan: "02"
subsystem: core
tags: [debate, arbiter, fact-checking, agent-coordination]
dependency_graph:
  requires: ["33-01"]
  provides: ["DEBATE-02", "DEBATE-03"]
  affects: ["src/finalayze/core/", ".claude/agents/"]
tech_stack:
  added: []
  patterns: ["YAML frontmatter markdown files", "TYPE_CHECKING lazy imports", "ast-index verification protocol"]
key_files:
  created:
    - src/finalayze/core/debate_manager.py
    - tests/unit/core/test_debate_manager.py
    - .planning/debates/.gitkeep
    - .claude/agents/arbiter-agent.md
  modified: []
decisions:
  - "Use yaml.safe_load (not yaml.load) for YAML parsing — prevents arbitrary Python object deserialization (T-33-05)"
  - "Local import inside read_debate for DebateState/FactCheckReport to avoid circular dependency at module init time"
  - "Use claude-sonnet-4-20250514 for arbiter (not opus) — mechanical verification, not creative reasoning"
  - "Float tolerance 0.01 for metric comparisons — consistent with scientific float comparison norms"
  - "Path scope enforcement in arbiter (src/, tests/, config/, docs/) — mitigates T-33-04 path traversal threat"
  - "ESCALATE if any single contradiction — zero-tolerance policy ensures disputes get resolved experimentally"
metrics:
  duration_minutes: 9
  completed_date: "2026-04-08"
  tasks_completed: 2
  files_created: 4
  files_modified: 0
---

# Phase 33 Plan 02: Arbiter Agent and Debate File Manager Summary

**One-liner:** DebateManager with YAML-frontmatter CRUD and arbiter-agent with ast-index code-claim and history.jsonl metric-claim verification paths.

## What Was Built

### Task 1: DebateManager + Tests

`src/finalayze/core/debate_manager.py` — Layer 0 class providing full CRUD for debate files:

- `create_debate(debate_id, topic, agents)` — writes `{debate_id}.md` with YAML frontmatter (status=open)
- `read_debate(debate_id)` — parses YAML frontmatter, deserializes `FactCheckReport` from nested dict
- `resolve_debate(debate_id, resolution)` — sets status=resolved, writes resolution field
- `escalate_debate(debate_id, experiment_id)` — sets status=escalated, writes experiment_id field
- `list_debates()` — sorted list of debate IDs from directory glob
- `add_agent_position(debate_id, agent_name, agent_output)` — appends `## {agent} Position` section to body
- `add_arbiter_report(debate_id, report)` — serializes FactCheckReport to frontmatter dict, appends `## Arbiter Fact-Check` section

Internal helpers `_read_file` / `_write_file` use `yaml.safe_load` and `yaml.dump` for safe YAML parsing. `.planning/debates/.gitkeep` establishes the debates directory.

10 tests in `tests/unit/core/test_debate_manager.py` — all green. Roundtrip test confirms create → add positions → add arbiter report → read back with all data intact.

### Task 2: Arbiter Agent Definition

`.claude/agents/arbiter-agent.md` — Claude sub-agent definition with YAML frontmatter (`model: claude-sonnet-4-20250514`) and structured verification protocol:

**Path A (code claims, `source.kind == "file"`):**
1. Validate path scope (`src/`, `tests/`, `config/`, `docs/` only — rejects traversal)
2. `ast-index rebuild` if stale (>1 hour since last rebuild)
3. `ast-index outline {path}` to confirm file indexed
4. Read file at claimed line, compare against excerpt
5. VERIFIED (exact or shifted line) / CONTRADICTED (excerpt absent) / UNTESTABLE (file missing)

**Path B (metric claims, `source.kind == "metric"`):**
1. Check `results/iterations/history.jsonl` exists and non-empty
2. Scan line-by-line for record where `name == source.iteration`
3. Extract `metrics.{source.metric_name}`
4. VERIFIED if `abs(actual - claimed) <= 0.01` / CONTRADICTED if delta > 0.01 / UNTESTABLE if iteration or metric missing

Output format: `## Verified`, `## Contradicted`, `## Untestable`, `## Summary` with RESOLVE/ESCALATE recommendation.

## Deviations from Plan

None — plan executed exactly as written. The `PLC0415` ruff warning for the local import inside `read_debate` was suppressed with `# noqa: PLC0415` because the import is intentionally deferred (Layer 0 module importing its own schemas to avoid circular initialization).

## Threat Mitigations Applied

| Threat | Mitigation |
|--------|-----------|
| T-33-04 path traversal | Arbiter Section 5, Rule 2: path must start with `src/`, `tests/`, `config/`, or `docs/` — reject before any tool call |
| T-33-05 YAML deserialization | `yaml.safe_load()` used throughout DebateManager (not `yaml.load()`) |
| T-33-06 large history.jsonl | Accepted — linear scan on bounded internal file |

## Self-Check

- `/Users/f1xgun/finalayze/src/finalayze/core/debate_manager.py` — FOUND
- `/Users/f1xgun/finalayze/tests/unit/core/test_debate_manager.py` — FOUND
- `/Users/f1xgun/finalayze/.planning/debates/.gitkeep` — FOUND
- `/Users/f1xgun/finalayze/.claude/agents/arbiter-agent.md` — FOUND
- Commit `3131dcc` (Task 1) — FOUND
- Commit `65176aa` (Task 2) — FOUND
- 10 tests passing — CONFIRMED
- 94 core tests passing — CONFIRMED
- ruff clean — CONFIRMED

## Self-Check: PASSED
