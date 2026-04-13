# Phase 37: Agent Orchestrator + Debate/Experiment REST API - Context

**Gathered:** 2026-04-12
**Status:** Ready for planning

<domain>
## Phase Boundary

This phase delivers the full conflict→debate→arbiter→experiment→verdict pipeline as `AgentOrchestrator` (Layer 5), REST API endpoints for debates and experiments (Layer 6), `snapshot_sha` safety on `FileLineSource` (Layer 0), and the `agent-orchestrator.md` Claude Code agent definition.

</domain>

<decisions>
## Implementation Decisions

### Orchestrator Pipeline Design
- `AgentOrchestrator` lives in `orchestration/agent_orchestrator.py` (Layer 5) — imports ConflictDetector, DebateManager, ExperimentManager
- Receives pre-collected `list[AgentOutput]` as input — does NOT invoke agents directly (Claude Code `agent-orchestrator.md` handles spawning externally)
- Stateless per invocation — reads/writes through DebateManager and ExperimentManager which handle file-based persistence
- Reuses existing `ExperimentState.debate_id` bidirectional linkage from v7.0

### REST API Design
- Two new router files: `api/v1/debates.py` + `api/v1/experiments.py` — follows existing pattern (portfolio.py, risk.py)
- Auth: X-API-Key header — same as existing `/kill` endpoint
- `POST /api/v1/debates` accepts `list[AgentOutput]` JSON → runs ConflictDetector → creates debate if conflicts found → returns debate ID
- Experiments endpoints are read-only for now (list, detail) — write operations (apply) deferred to Phase 38

### snapshot_sha + Agent Orchestrator Definition
- Add optional `snapshot_sha: str | None = None` to `FileLineSource` in `core/schemas.py` — SHA-256 of file content at claim creation time
- Arbiter checks: compare current file SHA against claim's snapshot_sha. If different → mark claim `UNTESTABLE` not `CONTRADICTED`
- `agent-orchestrator.md` defines: spawn 3+ domain agents → collect AgentOutput → call Python pipeline via Bash tool
- Orchestrator trigger: on-demand (invoked by user or scheduled skill) — not per-trading-cycle

### Claude's Discretion
- Internal pipeline flow control (e.g., whether to short-circuit when no conflicts found)
- REST response schemas (Pydantic response models for API)
- Error handling patterns in orchestrator (retry vs fail-fast)
- agent-orchestrator.md prompt structure and agent selection logic

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `core/schemas.py:569-626` — Claim, AgentOutput, ClaimVerdict, ClaimCheckResult, FactCheckReport, DebateState, ExperimentState
- `core/schemas.py:599-626` — ConflictType, ConflictSeverity, ConflictReport (from Phase 36)
- `core/debate_manager.py` — DebateManager CRUD (file-based, .planning/debates/)
- `core/experiment_manager.py` — ExperimentManager with verdict logic (file-based, .planning/experiments/)
- `orchestration/conflict_detector.py` — ConflictDetector.detect() (from Phase 36)
- `api/v1/router.py` — existing API router with include_router pattern
- `api/v1/auth.py` — X-API-Key auth dependency
- `.claude/agents/arbiter-agent.md` — arbiter agent definition

### Established Patterns
- FastAPI routers in `api/v1/` with `APIRouter(prefix="/debates", tags=["debates"])` pattern
- Pydantic response models for all API endpoints
- `from finalayze.api.v1.auth import require_api_key` for protected endpoints
- File-based persistence: JSON read/write with Pydantic `model_validate_json()` / `model_dump_json()`

### Integration Points
- `api/v1/router.py` needs new router includes for debates and experiments
- `orchestration/agent_orchestrator.py` imports from Layer 0 (schemas, debate_manager, experiment_manager) and Layer 5 (conflict_detector)
- `FileLineSource` in `core/schemas.py` needs `snapshot_sha` field added
- `.claude/agents/` needs new `agent-orchestrator.md` definition

</code_context>

<specifics>
## Specific Ideas

- The orchestrator should be a thin coordinator — most logic is already in DebateManager, ExperimentManager, and ConflictDetector
- REST endpoints should return Pydantic models serialized as JSON, not raw file content
- snapshot_sha should be optional (None = legacy claims without snapshot tracking)

</specifics>

<deferred>
## Deferred Ideas

- `POST /experiments/{id}/apply` endpoint — deferred to Phase 38 (auto-apply)
- Experiment write operations via REST — deferred to Phase 38
- Scheduled orchestrator runs — deferred to v8.x (need validation first)

</deferred>
