---
phase: 36-conflict-detection-foundation
plan: "02"
subsystem: orchestration
tags: [conflict-detection, agent-output, tdd, deterministic]
dependency_graph:
  requires: [36-01]
  provides: [ConflictDetector, agent-output-format]
  affects: [orchestration/conflict_detector.py, .claude/agents/]
tech_stack:
  added: []
  patterns: [difflib.SequenceMatcher, hashlib.sha256, itertools.combinations, Pydantic frozen models]
key_files:
  created:
    - src/finalayze/orchestration/conflict_detector.py
    - tests/unit/core/test_conflict_detector.py
  modified:
    - .claude/agents/quant-analyst.md
    - .claude/agents/risk-officer.md
    - .claude/agents/ml-engineer.md
    - .claude/agents/strategies-agent.md
    - .claude/agents/portfolio-strategist.md
    - .claude/agents/systems-architect.md
decisions:
  - "Confidence delta filter (<=0.15) applied per agent pair (max claim confidence each side) -- prevents noise escalation"
  - "SHA-256 dedup key uses sorted(agents) + sorted(topics) + conflict_type -- deterministic, collision-proof"
  - "Test confidence constants _CONF_HIGH=0.90, _CONF_LOW=0.65 (delta=0.25) to clear filter threshold"
metrics:
  duration_seconds: 383
  completed_date: "2026-04-12"
  tasks_completed: 2
  files_created: 2
  files_modified: 6
  tests_added: 15
---

# Phase 36 Plan 02: ConflictDetector and Agent Output Format Summary

ConflictDetector with deterministic rule-based detection (direction/metric/statement conflicts, SHA-256 dedup, confidence delta filter) plus Output Format sections added to all 6 domain agent definitions.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Build ConflictDetector with TDD tests | 1a34c41 | conflict_detector.py, test_conflict_detector.py |
| 2 | Add Output Format to 6 domain agent definitions | 6419884 | 6 .claude/agents/*.md files |

## What Was Built

### Task 1: ConflictDetector (TDD)

`src/finalayze/orchestration/conflict_detector.py` -- 230-line deterministic detector, no LLM imports.

**Three conflict types:**
- `_check_direction()`: uppercase word extraction on `recommendation` field, BUY vs SELL keyword set intersection → CRITICAL severity
- `_check_metrics()`: matches `(metric_name, iteration)` pairs across outputs, relative divergence `abs(va-vb)/max(|va|,|vb|)` > 15% → METRIC conflict (HIGH >30%, LOW 15-30%)
- `_check_statements()`: `difflib.SequenceMatcher.ratio() > 0.85` on claim statements combined with divergent recommendations → STATEMENT/LOW

**Cross-cutting:**
- Confidence delta filter: `abs(max_conf_a - max_conf_b) <= 0.15` suppresses conflict (both sides must meaningfully disagree)
- Session dedup: SHA-256 of `(sorted agents, sorted topics, conflict_type)` stored in `_seen_conflicts: set[str]`
- `reset()` clears dedup store for new orchestrator cycles

**TDD cycle:**
- RED: 15 failing tests committed (58da99a)
- GREEN: Implementation passes all 15 tests (1a34c41)
- No refactor phase needed

### Task 2: Agent Output Format Sections

All 6 domain agent `.md` definitions updated with `## Output Format` section containing:
- Agent-specific `AgentOutput` JSON example
- Source reference guide (file vs metric kinds)
- "No unsourced assertions allowed" rule with fallback pattern

Agents updated: `quant-analyst`, `risk-officer`, `ml-engineer`, `strategies-agent`, `portfolio-strategist`, `systems-architect`

## Decisions Made

1. **Confidence delta filter uses max-claim confidence per agent** — takes `max(c.confidence for c in agent.claims)` not average. This means a single high-confidence claim anchors the agent's position.

2. **Test fixture design: _CONF_HIGH=0.90, _CONF_LOW=0.65** — delta=0.25 reliably clears the 0.15 threshold. Tests that expect conflicts use (HIGH, LOW), Test 5 uses (0.80, 0.70) delta=0.10 for filter verification.

3. **Metric dedup key includes both metric_name and iteration** — prevents cross-iteration dedup collisions when same metric appears in different experiment runs.

4. **Statement conflict requires BOTH similarity AND recommendation divergence** — similarity alone (same-direction agents citing same facts) should not trigger a conflict.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Test confidence values caused unintended filter suppression**
- **Found during:** Task 1 GREEN phase, first test run
- **Issue:** Original tests used confidences 0.85 vs 0.90 (delta=0.05), which the confidence delta filter (<=0.15) suppressed — tests expecting conflicts got empty results
- **Fix:** Rewrote test file with `_CONF_HIGH=0.90`, `_CONF_LOW=0.65` (delta=0.25) for all conflict-expecting tests; kept Test 5 at (0.80, 0.70) delta=0.10 for filter verification
- **Files modified:** tests/unit/core/test_conflict_detector.py
- **Commit:** 1a34c41

## Known Stubs

None. ConflictDetector is fully functional. All 3 detection algorithms are wired.

## Threat Flags

None. ConflictDetector is internal-only (not API-exposed). Inputs are frozen Pydantic models already validated upstream. No new network endpoints or trust boundaries introduced.

## Self-Check: PASSED

- `src/finalayze/orchestration/conflict_detector.py` -- FOUND
- `tests/unit/core/test_conflict_detector.py` -- FOUND
- Commit 58da99a (RED tests) -- FOUND
- Commit 1a34c41 (GREEN implementation) -- FOUND
- Commit 6419884 (agent Output Format) -- FOUND
- All 6 agent .md files contain `## Output Format` -- VERIFIED (grep count=6)
- 15/15 tests pass -- VERIFIED
- ruff check passes -- VERIFIED
- No LLM imports in conflict_detector.py -- VERIFIED
