---
phase: 37-agent-orchestrator-+-debate-experiment-rest-api
verified: 2026-04-12T00:00:00Z
status: gaps_found
score: 4/5
overrides_applied: 0
gaps:
  - truth: "FileLineSource carries a snapshot_sha field; when the referenced file has changed since the claim was recorded, the arbiter marks that claim UNTESTABLE instead of CONTRADICTED — stale source references do not trigger false conflict escalations"
    status: partial
    reason: "snapshot_sha field exists on FileLineSource with correct docstring, but arbiter-agent.md Protocol (Section 3, Path A) has no step to compute the current file SHA and compare it to snapshot_sha. The arbiter proceeds directly to reading the file and checking excerpt presence — no UNTESTABLE short-circuit when the SHA differs. The safety mechanism is schema-level only; the enforcement is missing from the arbiter's verification protocol."
    artifacts:
      - path: "src/finalayze/core/schemas.py"
        issue: "Field and docstring exist correctly (line 551). Field is not the problem."
      - path: ".claude/agents/arbiter-agent.md"
        issue: "Section 3 Path A has 5 steps. None of them check snapshot_sha. After validating path scope, the arbiter reads the file and checks the excerpt — no SHA comparison step exists."
    missing:
      - "Add a snapshot_sha check step to arbiter-agent.md Section 3 Path A (after step 3 — file confirmed in index, before step 4 — read the file): 'If source.snapshot_sha is not None, compute SHA-256 of the file content and compare. If they differ, mark UNTESTABLE with evidence: File {path} has changed since claim was recorded (snapshot_sha mismatch). Do not proceed to excerpt check.'"
---

# Phase 37: Agent Orchestrator + Debate/Experiment REST API Verification Report

**Phase Goal:** The full conflict→debate→arbiter→experiment→verdict pipeline runs end-to-end, manually triggerable via REST, with snapshot safety preventing false contradiction verdicts after code changes
**Verified:** 2026-04-12T00:00:00Z
**Status:** gaps_found
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | `AgentOrchestrator.run(outputs)` executes the full pipeline — detected conflict triggers a DebateManager entry; entire flow completes without manual intervention | VERIFIED | `agent_orchestrator.py` lines 82-144: fresh `ConflictDetector()` per call, groups by frozenset(agent_names), calls `create_debate()` and `add_agent_position()`. 7 unit tests pass including `test_run_fresh_conflict_detector_per_call` |
| 2 | `GET /api/v1/debates` and `GET /api/v1/debates/{id}` return debate list and detail; `POST /api/v1/debates` creates a debate manually — the pipeline is invocable without writing Python | VERIFIED | `debates.py` has POST/GET list/GET detail endpoints, all behind `Depends(api_key_auth)` on router prefix. Wired to `router.py` lines 25-26. 9 tests pass including 401 auth test |
| 3 | `GET /api/v1/experiments` and `GET /api/v1/experiments/{id}` return experiment state and linked backtest results — all experiment data is accessible via REST without filesystem access | VERIFIED | `experiments.py` has GET list and GET detail endpoints (read-only per CONTEXT.md deferred). Wired to `router.py`. 7 tests pass including auth test and read-only assertion |
| 4 | `FileLineSource` carries a `snapshot_sha` field; when the referenced file has changed since the claim was recorded, the arbiter marks that claim `UNTESTABLE` instead of `CONTRADICTED` — stale source references do not trigger false conflict escalations | FAILED | `snapshot_sha: str | None = None` field exists in `schemas.py:551` with correct docstring. `compute_file_sha()` helper exists at `schemas.py:812`. BUT `arbiter-agent.md` Section 3 Path A has no SHA comparison step. The arbiter never computes or compares file SHAs — it reads the file and checks excerpt presence regardless of whether the file changed. The prevention mechanism does not function. |
| 5 | The `.claude/agents/agent-orchestrator.md` definition exists and can be invoked as a Claude Code sub-agent to run a full orchestration cycle autonomously | VERIFIED | File exists with `name: agent-orchestrator`, `model: claude-sonnet-4-20250514`. 8-step protocol references `quant-analyst`, `risk-officer`, `ml-engineer`, `arbiter-agent`, and `AgentOrchestrator` Python class (3 references each for arbiter and class) |

**Score:** 4/5 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/finalayze/orchestration/agent_orchestrator.py` | AgentOrchestrator pipeline coordinator | VERIFIED | 271 lines. `class AgentOrchestrator` at line 46. `run()` at 82, `finalize_debate()` at 146. Imports ConflictDetector, DebateManager, ExperimentManager. |
| `src/finalayze/core/schemas.py` | snapshot_sha field on FileLineSource | VERIFIED | `snapshot_sha: str | None = None` at line 551. `compute_file_sha()` at 812. Field is substantive with docstring. |
| `tests/unit/core/test_agent_orchestrator.py` | Unit tests for orchestrator pipeline (min 80 lines) | VERIFIED | 323 lines. 7 tests in 2 test classes covering all branches including fresh-detector invariant. |
| `src/finalayze/api/v1/debates.py` | Debates REST router | VERIFIED | POST/GET list/GET detail with `api_key_auth` dependency and `AgentOrchestrator` usage |
| `src/finalayze/api/v1/experiments.py` | Experiments REST router | VERIFIED | GET list/GET detail, read-only (no POST/PUT), `api_key_auth` dependency |
| `.claude/agents/agent-orchestrator.md` | Claude Code agent definition | VERIFIED | Valid YAML frontmatter, 8-step protocol, all required references present |
| `tests/unit/test_api_debates.py` | Debates API tests (min 60 lines) | VERIFIED | 263 lines, 9 tests |
| `tests/unit/test_api_experiments.py` | Experiments API tests (min 40 lines) | VERIFIED | 189 lines, 7 tests |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `agent_orchestrator.py` | `conflict_detector.py` | `ConflictDetector().detect(outputs)` | WIRED | `ConflictDetector()` instantiated fresh at line 101, `.detect(outputs)` called at 102 |
| `agent_orchestrator.py` | `debate_manager.py` | `create_debate()` + `add_agent_position()` | WIRED | `self._dm.create_debate()` at line 128, `self._dm.add_agent_position()` at line 138 |
| `router.py` | `debates.py` | `include_router(debates_router)` | WIRED | `from finalayze.api.v1.debates import router as debates_router` + `api_router.include_router(debates_router)` at lines 5, 25 |
| `router.py` | `experiments.py` | `include_router(experiments_router)` | WIRED | `from finalayze.api.v1.experiments import router as experiments_router` + `api_router.include_router(experiments_router)` at lines 6, 26 |
| `debates.py` | `agent_orchestrator.py` | `AgentOrchestrator().run()` | WIRED | `from finalayze.orchestration.agent_orchestrator import AgentOrchestrator` at line 17, `orch = AgentOrchestrator()` + `orch.run()` at lines 80-81 |
| `arbiter-agent.md` | `schemas.py FileLineSource.snapshot_sha` | Check SHA on file claims | NOT WIRED | No SHA comparison step in arbiter protocol Section 3 Path A. The field is consumed by no agent logic. |

### Data-Flow Trace (Level 4)

This phase produces API endpoints (not components rendering DB data). The data flows through manager classes to/from `.planning/` filesystem — not a DB layer. No Level 4 gap: `DebateManager.list_debates()` and `ExperimentManager.list_experiments()` read real files from disk; `AgentOrchestrator.run()` reads actual `ConflictDetector` output. No static returns.

### Behavioral Spot-Checks

| Behavior | Result | Status |
|----------|--------|--------|
| `test_agent_orchestrator.py` — 7 tests | All 7 PASSED (`uv run pytest tests/unit/core/test_agent_orchestrator.py --no-cov`) | PASS |
| `test_api_debates.py` — 9 tests | All 9 PASSED | PASS |
| `test_api_experiments.py` — 7 tests | All 7 PASSED | PASS |
| `test_debate_schemas.py` — 31 tests (includes 4 snapshot_sha tests) | All 31 PASSED | PASS |
| Total phase 37 new tests | 54 passed in 0.42s | PASS |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| ORCH-01 | 37-01-PLAN | AgentOrchestrator coordinates full pipeline: conflict → debate → arbiter → experiment → backtest → verdict | SATISFIED | `AgentOrchestrator.run()` and `finalize_debate()` implement full flow. 7 tests verify all branches. |
| ORCH-02 | 37-02-PLAN | REST API endpoints for debates (list, detail, create) and experiments (list, detail) — manual pipeline invocation | SATISFIED | POST/GET debates, GET experiments — all wired, tested, auth-gated. |
| ORCH-03 | 37-01-PLAN | `snapshot_sha` field on `FileLineSource` prevents false CONTRADICTED verdicts after code changes | PARTIAL | Field exists but arbiter does not check it. The prevention mechanism is documented in the schema docstring but not implemented in the arbiter's verification protocol. |
| ORCH-04 | 37-02-PLAN | Claude Code `agent-orchestrator.md` definition enables autonomous pipeline runs | SATISFIED | File exists with complete 8-step protocol, all required references. |

### Anti-Patterns Found

| File | Pattern | Severity | Impact |
|------|---------|----------|--------|
| `arbiter-agent.md` | No `snapshot_sha` check in Path A verification steps | Blocker | The roadmap success criterion requires the arbiter to mark claims UNTESTABLE when the file SHA has changed. Without this check, a modified file still triggers the excerpt-comparison and will produce CONTRADICTED instead of UNTESTABLE — exactly the false verdict the field was meant to prevent. |

### Human Verification Required

None. All gaps are code/protocol deficiencies verifiable programmatically.

### Gaps Summary

One gap blocking full goal achievement:

**ORCH-03 — snapshot_sha enforcement missing from arbiter protocol**

The `snapshot_sha` field was added to `FileLineSource` correctly. The field's docstring accurately describes the intended behavior. However, `arbiter-agent.md` Section 3, Path A (Code Claims verification) contains no step to:
1. Compute the current SHA-256 of the claimed file
2. Compare it against `source.snapshot_sha`
3. Short-circuit with UNTESTABLE if they differ

Without this step, the arbiter reads the file and compares the excerpt regardless of whether the file has been modified since the claim was created. A stale claim on a refactored file will produce CONTRADICTED (excerpt not found) rather than UNTESTABLE — which is the false verdict the phase goal explicitly required to prevent.

**Fix:** Add a new step 4 to arbiter-agent.md Section 3 Path A (between the current "Confirm file is indexed" step and "Read the file at the claimed line" step):

> **4. Check file integrity (snapshot_sha)**: If `source.snapshot_sha` is not None, compute `hashlib.sha256(Path(source.path).read_bytes()).hexdigest()`. If the result differs from `source.snapshot_sha`, mark **UNTESTABLE** with evidence: "File `{path}` has changed since claim was recorded — snapshot SHA mismatch. Cannot reliably verify line {line}." Do not proceed to excerpt check.

This is a single-file, low-risk change to `.claude/agents/arbiter-agent.md`.

---

_Verified: 2026-04-12T00:00:00Z_
_Verifier: Claude (gsd-verifier)_
