---
phase: 33-structured-debate-protocol
plan: "01"
subsystem: core/schemas
tags: [debate-protocol, pydantic, schemas, tdd]
dependency_graph:
  requires: []
  provides: [debate-protocol-schemas]
  affects: [core/schemas.py]
tech_stack:
  added: []
  patterns: [discriminated-union, model_validator, field_validator, frozen-pydantic]
key_files:
  created:
    - tests/unit/core/test_debate_schemas.py
  modified:
    - src/finalayze/core/schemas.py
decisions:
  - "Debate schemas placed in Layer 0 (core/schemas.py) per locked CONTEXT.md decision"
  - "ClaimSource implemented as Annotated discriminated union on 'kind' field"
  - "DebateState escalation constraint implemented with model_validator(mode='after')"
  - "FactCheckReport.has_contradictions is a @property (not a field) for computed access"
metrics:
  duration: "< 5 minutes"
  completed: "2026-04-07"
  tasks_completed: 2
  files_modified: 2
---

# Phase 33 Plan 01: Debate Protocol Schemas Summary

Pydantic schemas for the structured debate protocol: typed claim models with discriminated union sources, fact-check report with markdown rendering, and debate state with escalation validation.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | RED — failing tests for debate protocol schemas | 3e87ef2 | tests/unit/core/test_debate_schemas.py |
| 2 | GREEN — implement all debate protocol schemas | f260dda | src/finalayze/core/schemas.py |

## What Was Built

10 new models added to `src/finalayze/core/schemas.py` under a `# ── Debate Protocol Schemas` section:

- **DebateStatus** — StrEnum: OPEN / RESOLVED / ESCALATED
- **FileLineSource** — frozen model: kind="file", path, line, excerpt
- **MetricSource** — frozen model: kind="metric", metric_name, value, iteration
- **ClaimSource** — Annotated discriminated union on `kind` field (FileLineSource | MetricSource)
- **Claim** — frozen model with `confidence` validator enforcing [0.0, 1.0]
- **AgentOutput** — frozen model with `claims: list[Claim] = Field(min_length=1)`
- **ClaimVerdict** — StrEnum: VERIFIED / CONTRADICTED / UNTESTABLE
- **ClaimCheckResult** — frozen model: claim + verdict + evidence
- **FactCheckReport** — frozen model with `has_contradictions` property and `to_markdown()` method
- **DebateState** — frozen model with `model_validator` requiring `experiment_id` when status is ESCALATED

## Test Coverage

19 test functions in `tests/unit/core/test_debate_schemas.py` — all pass.

Covers:
- Valid construction of all 8 model types
- Boundary validation: confidence 0.0 and 1.0 succeed; -0.1 and 1.1 raise ValidationError
- Discriminated union deserialization from raw dicts
- AgentOutput empty claims raises ValidationError (min_length=1)
- FactCheckReport.has_contradictions True/False
- FactCheckReport.to_markdown() section headers
- DebateState escalation: experiment_id=None raises ValidationError, non-None succeeds
- All 84 existing core tests remain green

## Deviations from Plan

None — plan executed exactly as written.

## Threat Surface Scan

No new network endpoints, auth paths, file access patterns, or schema changes at trust boundaries. FileLineSource.path stores a string only — no path validation or file access occurs in schemas (deferred to Plan 02 arbiter per T-33-01 threat register).

## Self-Check

- [x] `tests/unit/core/test_debate_schemas.py` exists
- [x] `src/finalayze/core/schemas.py` modified
- [x] Commit 3e87ef2 exists (RED)
- [x] Commit f260dda exists (GREEN)
- [x] All 19 debate schema tests pass
- [x] All 84 core unit tests pass
- [x] Ruff clean on schemas.py

## Self-Check: PASSED
