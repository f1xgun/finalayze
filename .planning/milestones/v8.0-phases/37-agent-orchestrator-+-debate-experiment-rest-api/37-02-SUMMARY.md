---
phase: 37-agent-orchestrator-+-debate-experiment-rest-api
plan: "02"
subsystem: api
tags: [api, debates, experiments, agent-orchestrator, rest, tdd]
dependency_graph:
  requires: ["37-01"]
  provides: ["debates-rest-api", "experiments-rest-api", "agent-orchestrator-agent"]
  affects: ["src/finalayze/api/v1/router.py"]
tech_stack:
  added: []
  patterns: ["FastAPI router with dependency injection", "JSONResponse for dynamic status codes", "TestClient with dependency_overrides for auth testing"]
key_files:
  created:
    - src/finalayze/api/v1/debates.py
    - src/finalayze/api/v1/experiments.py
    - tests/unit/test_api_debates.py
    - tests/unit/test_api_experiments.py
    - .claude/agents/agent-orchestrator.md
  modified:
    - src/finalayze/api/v1/router.py
decisions:
  - "Used JSONResponse with dynamic status_code to return 201 vs 200 from POST /debates depending on conflict detection result"
  - "Instantiate AgentOrchestrator and DebateManager/ExperimentManager per request (not at module level) to avoid stale state and CWD-relative path issues"
  - "Experiments router is read-only (GET only) per CONTEXT.md deferred decisions — write operations deferred to Phase 38"
  - "Tests use app.dependency_overrides[api_key_auth] = lambda: None pattern for auth bypass in test isolation"
metrics:
  duration: "4 minutes"
  completed_date: "2026-04-12"
  tasks_completed: 3
  files_created: 5
  files_modified: 1
---

# Phase 37 Plan 02: Debates + Experiments REST API + Agent Orchestrator Definition Summary

**One-liner:** Debates and experiments REST endpoints (POST/GET debates, GET experiments) wired into FastAPI router with X-API-Key auth, plus agent-orchestrator.md 8-step pipeline coordination protocol.

## What Was Built

### Task 1: Debates REST Router (TDD)

Created `src/finalayze/api/v1/debates.py` with three endpoints:
- `POST /api/v1/debates` — runs `AgentOrchestrator().run(outputs)`, returns 201 with `debate_id` if conflicts found, 200 if none
- `GET /api/v1/debates` — lists all debate IDs via `DebateManager().list_debates()`
- `GET /api/v1/debates/{id}` — returns `DebateDetailResponse` with `has_arbiter_report` flag, or 404

All endpoints protected by `Depends(api_key_auth)` on the router prefix.

9 tests in `tests/unit/test_api_debates.py` covering: conflict/no-conflict POST, list, detail, 404, and auth (401 without header).

### Task 2: Experiments REST Router (TDD, Read-Only)

Created `src/finalayze/api/v1/experiments.py` with two read-only endpoints:
- `GET /api/v1/experiments` — lists all experiment IDs
- `GET /api/v1/experiments/{id}` — returns `ExperimentDetailResponse` with success_criteria, results, verdict, or 404

No write endpoints (deferred to Phase 38). 7 tests covering: list, detail with verdict, 404, auth (401), and read-only assertion (POST returns 404/405).

Updated `src/finalayze/api/v1/router.py` to include both `debates_router` and `experiments_router`.

### Task 3: agent-orchestrator.md

Created `.claude/agents/agent-orchestrator.md` as a Claude Code sub-agent definition with:
- YAML frontmatter: `name: agent-orchestrator`, `model: claude-sonnet-4-20250514`
- 8-step protocol: topic → spawn domain agents → collect AgentOutput JSON → write/run Python conflict detection script → branch on conflicts → invoke arbiter-agent → finalize_debate() script → structured report
- References `quant-analyst`, `risk-officer`, `ml-engineer` as domain agents
- References `arbiter-agent` for fact-checking step
- References `AgentOrchestrator` Python class for pipeline execution

## Decisions Made

1. **Dynamic status code via JSONResponse** — POST /debates uses `JSONResponse(content=..., status_code=201 if debate_ids else 200)` since FastAPI route decorators set a fixed default status. This cleanly separates conflict-found (201 Created) from no-conflict (200 OK).

2. **Per-request instantiation** — `AgentOrchestrator()`, `DebateManager()`, and `ExperimentManager()` are instantiated fresh per request to avoid (a) stale `ConflictDetector` dedup state and (b) CWD-relative path issues that arise with module-level singletons.

3. **Experiments read-only** — Per CONTEXT.md deferred decisions, write operations on experiments are deferred to Phase 38. The router has no POST/PUT/PATCH/DELETE routes.

4. **Test auth override pattern** — Tests use `app.dependency_overrides[api_key_auth] = lambda: None` for happy-path tests and a separate `_make_app_no_auth()` factory (no override) with a mocked `get_settings()` for auth failure tests.

## Deviations from Plan

None — plan executed exactly as written, with one minor auto-fix:

**[Rule 1 - Bug] Fixed dynamic 201/200 status on POST /debates**
- **Found during:** Task 1 GREEN phase
- **Issue:** FastAPI route decorator `@router.post("")` defaults to 200; returning a `CreateDebateResponse` model always returned 200 regardless of conflict detection result
- **Fix:** Changed return type to `JSONResponse` and used `status_code = 201 if debate_ids else 200`
- **Files modified:** `src/finalayze/api/v1/debates.py`
- **Commit:** 7df9460

## Commits

| Task | Description | Hash |
|------|-------------|------|
| 1 | feat(37-02): debates REST router with TDD tests | 7df9460 |
| 2 | feat(37-02): experiments REST router with TDD tests (read-only) | e248d65 |
| 3 | feat(37-02): create agent-orchestrator.md Claude Code sub-agent definition | 727634c |

## Known Stubs

None. All endpoints call real manager/orchestrator instances (mocked only in tests).

## Threat Flags

No new security surface beyond what was declared in the plan's threat model. T-37-04 (spoofing) mitigated by `api_key_auth` on all router prefixes. T-37-05 (tampering) mitigated by Pydantic v2 AgentOutput validation on POST /debates.

## Self-Check

See below.
