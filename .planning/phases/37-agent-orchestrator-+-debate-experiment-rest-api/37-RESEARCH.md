# Phase 37: Agent Orchestrator + Debate/Experiment REST API - Research

**Researched:** 2026-04-12
**Domain:** Python orchestration pipeline + FastAPI REST API + Pydantic schema extension
**Confidence:** HIGH

## Summary

Phase 37 wires together the conflict detection foundation (Phase 36) into a full end-to-end pipeline via a thin `AgentOrchestrator` class, two new REST API routers, a `snapshot_sha` safety field on `FileLineSource`, and an `agent-orchestrator.md` Claude Code agent definition.

The codebase already contains all building blocks: `ConflictDetector.detect()` in `orchestration/conflict_detector.py`, `DebateManager` and `ExperimentManager` in `core/`, and `DebateState`/`ExperimentState`/`AgentOutput` schemas in `core/schemas.py`. The orchestrator is a thin coordinator that sequences calls to these existing components. The REST API follows the well-established pattern from `api/v1/risk.py` and `api/v1/portfolio.py`.

The `snapshot_sha` extension is minimal: add an optional `str | None` field to `FileLineSource` in `core/schemas.py` (Layer 0), then update the arbiter path in `arbiter-agent.md` to compare the file's current SHA-256 against the stored value and emit `UNTESTABLE` on mismatch.

**Primary recommendation:** Implement `AgentOrchestrator` as a plain Python class (not async), since all DebateManager/ExperimentManager I/O is synchronous file-based YAML. Mirror the `api/v1/risk.py` pattern exactly for both new routers.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- `AgentOrchestrator` lives in `orchestration/agent_orchestrator.py` (Layer 5) — imports ConflictDetector, DebateManager, ExperimentManager
- Receives pre-collected `list[AgentOutput]` as input — does NOT invoke agents directly
- Stateless per invocation — reads/writes through DebateManager and ExperimentManager which handle file-based persistence
- Reuses existing `ExperimentState.debate_id` bidirectional linkage from v7.0
- Two new router files: `api/v1/debates.py` + `api/v1/experiments.py`
- Auth: X-API-Key header — same as existing `/kill` endpoint via `api_key_auth` dependency
- `POST /api/v1/debates` accepts `list[AgentOutput]` JSON → runs ConflictDetector → creates debate if conflicts found → returns debate ID
- Experiments endpoints are read-only for now (list, detail) — write operations deferred to Phase 38
- Add optional `snapshot_sha: str | None = None` to `FileLineSource` in `core/schemas.py`
- Arbiter checks: compare current file SHA against claim's `snapshot_sha`. If different → mark `UNTESTABLE` not `CONTRADICTED`
- `agent-orchestrator.md` defines: spawn 3+ domain agents → collect AgentOutput → call Python pipeline via Bash tool
- Orchestrator trigger: on-demand (invoked by user or scheduled skill) — not per-trading-cycle

### Claude's Discretion

- Internal pipeline flow control (e.g., whether to short-circuit when no conflicts found)
- REST response schemas (Pydantic response models for API)
- Error handling patterns in orchestrator (retry vs fail-fast)
- `agent-orchestrator.md` prompt structure and agent selection logic

### Deferred Ideas (OUT OF SCOPE)

- `POST /experiments/{id}/apply` endpoint — deferred to Phase 38 (auto-apply)
- Experiment write operations via REST — deferred to Phase 38
- Scheduled orchestrator runs — deferred to v8.x
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| ORCH-01 | `AgentOrchestrator` coordinates full pipeline: conflict → debate → arbiter → experiment → backtest → verdict | ConflictDetector.detect(), DebateManager.create_debate()/add_agent_position()/add_arbiter_report(), ExperimentManager.create_experiment() all exist and are verified |
| ORCH-02 | REST API endpoints for debates (list, detail, create) and experiments (list, detail) — manual pipeline invocation | api/v1/ pattern verified from risk.py, portfolio.py, sandbox.py; api_key_auth dependency works exactly as needed |
| ORCH-03 | `snapshot_sha` field on `FileLineSource` prevents false CONTRADICTED verdicts after code changes | FileLineSource at core/schemas.py:541-549 — adding optional `snapshot_sha: str | None = None` is a non-breaking Pydantic v2 change |
| ORCH-04 | Claude Code `agent-orchestrator.md` definition enables autonomous pipeline runs | arbiter-agent.md pattern verified — same frontmatter structure applies |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| FastAPI | existing | REST API routing | Already used for all 20+ endpoints |
| Pydantic v2 | existing | Request/response models + schema extension | Project standard, all schemas are Pydantic |
| structlog | existing | Structured logging in orchestrator | Project standard |
| hashlib (stdlib) | stdlib | SHA-256 for snapshot_sha computation | No dependency needed |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| yaml (PyYAML) | existing | DebateManager/ExperimentManager file I/O | Already used by both managers |
| pathlib (stdlib) | stdlib | File path handling | Already used everywhere |

**No new packages needed.** All required functionality exists in the current dependency set.

## Architecture Patterns

### Recommended Project Structure
```
src/finalayze/
├── orchestration/
│   └── agent_orchestrator.py    # NEW — Layer 5, thin coordinator
├── api/v1/
│   ├── debates.py               # NEW — Layer 6, debates router
│   ├── experiments.py           # NEW — Layer 6, experiments router
│   └── router.py                # MODIFIED — include new routers
├── core/
│   └── schemas.py               # MODIFIED — snapshot_sha on FileLineSource
└── .claude/agents/
    └── agent-orchestrator.md    # NEW — Claude Code agent definition
```

### Pattern 1: AgentOrchestrator Pipeline Flow

**What:** Thin coordinator that sequences: ConflictDetector.detect() → DebateManager.create_debate() + add_agent_position() for each agent → [arbiter invoked externally] → ExperimentManager.create_experiment() if escalation needed.

**When to use:** Called from REST endpoint or CLI when a batch of `AgentOutput` objects needs to be processed.

**Pipeline steps (Claude's discretion for short-circuit):**
1. `detector.detect(outputs)` → `list[ConflictReport]`
2. If no conflicts → return early (short-circuit is valid per discretion)
3. For each conflict group: `debate_manager.create_debate(debate_id, topic, agents)`
4. For each agent output: `debate_manager.add_agent_position(debate_id, agent_name, agent_output)`
5. Return debate IDs (arbiter is invoked separately — it is a Claude Code sub-agent, not Python)
6. After arbiter completes and writes `FactCheckReport`: `debate_manager.add_arbiter_report(debate_id, report)` and optionally `experiment_manager.create_experiment(...)` if report has contradictions

**Note:** The `AgentOrchestrator.run()` method only handles steps 1-4 synchronously. Steps 5-6 (arbiter invocation and experiment creation) are handled by the `agent-orchestrator.md` agent calling the pipeline via Bash tool after arbiter completes.

**Example:**
```python
# Source: orchestration/conflict_detector.py + core/debate_manager.py (VERIFIED)
class AgentOrchestrator:
    def __init__(
        self,
        debate_manager: DebateManager | None = None,
        experiment_manager: ExperimentManager | None = None,
    ) -> None:
        self._detector = ConflictDetector()
        self._dm = debate_manager or DebateManager()
        self._em = experiment_manager or ExperimentManager()

    def run(self, outputs: list[AgentOutput]) -> list[str]:
        """Run conflict detection and create debates. Returns debate IDs."""
        conflicts = self._detector.detect(outputs)
        if not conflicts:
            return []
        # Group conflicts → create one debate per conflict cluster
        debate_id = _generate_debate_id(conflicts)
        topic = _summarize_conflict_topic(conflicts)
        agent_names = list({n for c in conflicts for n in c.agent_names})
        self._dm.create_debate(debate_id, topic, agent_names)
        for output in outputs:
            if output.agent_name in agent_names:
                self._dm.add_agent_position(debate_id, output.agent_name, output)
        return [debate_id]
```

### Pattern 2: FastAPI Router — debates.py

**What:** Two routers in `api/v1/debates.py` — GET list + detail, POST create. Mirrors `api/v1/risk.py` structure exactly.

**Key implementation notes:**
- `APIRouter(prefix="/debates", tags=["debates"], dependencies=[Depends(api_key_auth)])`
- `POST /debates` body: `list[AgentOutput]` — Pydantic parses directly from JSON body
- Response for POST: `{"debate_id": str, "conflicts_found": int}` or 204 if no conflicts
- `GET /debates` → list debate IDs from `DebateManager().list_debates()`
- `GET /debates/{debate_id}` → `DebateManager().read_debate(debate_id)` wrapped in response model

**Example:**
```python
# Source: api/v1/risk.py pattern (VERIFIED)
from __future__ import annotations
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict
from finalayze.api.v1.auth import api_key_auth
from finalayze.core.debate_manager import DebateManager
from finalayze.core.schemas import AgentOutput, DebateState

router = APIRouter(
    prefix="/debates",
    tags=["debates"],
    dependencies=[Depends(api_key_auth)],
)

class CreateDebateRequest(BaseModel):
    outputs: list[AgentOutput]

class CreateDebateResponse(BaseModel):
    model_config = ConfigDict(frozen=True)
    debate_id: str | None  # None if no conflicts detected
    conflicts_found: int

class DebateListResponse(BaseModel):
    model_config = ConfigDict(frozen=True)
    debate_ids: list[str]
```

### Pattern 3: snapshot_sha on FileLineSource

**What:** Add optional `snapshot_sha: str | None = None` to `FileLineSource`. The arbiter-agent checks this at verification time. Non-breaking — existing claims without `snapshot_sha` remain valid (None = legacy).

**Where:** `core/schemas.py` line ~541-549 (VERIFIED: FileLineSource definition at lines 541-549)

```python
# Source: core/schemas.py:541-549 (VERIFIED)
class FileLineSource(BaseModel):
    model_config = ConfigDict(frozen=True)
    kind: Literal["file"] = "file"
    path: str
    line: int
    excerpt: str
    snapshot_sha: str | None = None  # ADD THIS FIELD
```

**SHA-256 computation pattern:**
```python
import hashlib
from pathlib import Path

def compute_file_sha(path: str) -> str:
    content = Path(path).read_bytes()
    return hashlib.sha256(content).hexdigest()
```

**Arbiter logic update:** In `arbiter-agent.md` Path A verification, add step after confirming file exists:
- If `source.snapshot_sha is not None`: compute current SHA-256 of `source.path`. If current SHA != `source.snapshot_sha` → mark `UNTESTABLE` with evidence: "File {path} has changed since claim was created (snapshot SHA mismatch). Claim cannot be verified against current code."
- If `source.snapshot_sha is None`: proceed with existing excerpt-matching logic (legacy behavior preserved).

### Pattern 4: agent-orchestrator.md Agent Definition

**What:** Claude Code sub-agent that spawns domain experts, collects `AgentOutput`, then invokes the Python pipeline.

**Follows arbiter-agent.md frontmatter structure (VERIFIED at .claude/agents/arbiter-agent.md:1-5):**
```yaml
---
name: agent-orchestrator
description: Use when you want to run the full conflict→debate→arbiter→experiment pipeline. Spawns domain agents (quant-analyst, risk-officer, ml-engineer), collects their AgentOutput JSON, runs ConflictDetector via Python, coordinates debates and arbiter, and escalates to experiments if contradictions found.
model: claude-sonnet-4-20250514
---
```

**Key agent-orchestrator.md sections:**
1. Role description (pipeline coordinator, not a domain expert)
2. Step-by-step protocol: spawn agents → collect outputs → call Python orchestrator via Bash → coordinate arbiter invocation → handle experiment creation
3. Input format (which agents to spawn, what topic/question to investigate)
4. Output format (structured summary of debates created, verdicts, escalations)

### Anti-Patterns to Avoid

- **Async orchestrator:** DebateManager and ExperimentManager are synchronous file I/O — no async needed. Don't add `async def run()` unnecessarily.
- **LLM in conflict detection path:** ConflictDetector is intentionally rule-based. Do not add LLM calls to AgentOrchestrator.
- **Mutable shared state in orchestrator:** ConflictDetector has session-scoped dedup via `_seen_conflicts: set`. AgentOrchestrator should call `detector.reset()` between invocations if reusing the same instance.
- **Importing AgentOrchestrator from Layer 6:** Router files must instantiate DebateManager/ExperimentManager directly or accept them via dependency injection — do not import from Layer 5 (orchestration) into the router functions unless carefully scoped.
- **Breaking FileLineSource frozen model:** `ConfigDict(frozen=True)` means instances are immutable. The `snapshot_sha` field must be set at construction time, not mutated later.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Debate ID generation | Custom UUID or timestamp scheme | `hashlib.sha256` over conflict data | ConflictDetector already uses this pattern for `conflict_id` |
| File-based state storage | Custom JSON/pickle store | DebateManager + ExperimentManager | Already handles YAML frontmatter, listing, CRUD |
| Auth middleware | Custom token check | `api_key_auth` dependency from `api/v1/auth.py` | Constant-time comparison, already tested |
| SHA-256 computation | Custom hash | `hashlib.sha256(content).hexdigest()` | Stdlib, no deps |
| Pydantic response model serialization | `json.dumps(obj.dict())` | `response_model=` on FastAPI endpoint + `.model_dump_json()` | FastAPI handles serialization automatically |

**Key insight:** The orchestrator is a coordinator, not an implementer. All complex logic lives in ConflictDetector, DebateManager, and ExperimentManager.

## Common Pitfalls

### Pitfall 1: DebateManager instantiated at module import time in router
**What goes wrong:** `_dm = DebateManager()` at module level in `debates.py` creates the `.planning/debates/` directory relative to wherever the process starts — can be wrong in test environments.
**Why it happens:** Module-level singletons use CWD at import time.
**How to avoid:** Instantiate `DebateManager()` inside endpoint functions, or use a dependency-injected factory. See sandbox.py's `_go_no_go_reporter` pattern for module-level singleton with explicit setter.
**Warning signs:** Tests fail with FileNotFoundError or wrong directory paths.

### Pitfall 2: `list[AgentOutput]` as request body
**What goes wrong:** FastAPI cannot directly parse a top-level JSON array as a request body without wrapping it in a Pydantic model.
**Why it happens:** FastAPI uses request body as a single JSON object by default.
**How to avoid:** Wrap in a request model: `class CreateDebateRequest(BaseModel): outputs: list[AgentOutput]` — then endpoint signature is `async def create_debate(req: CreateDebateRequest)`.
**Warning signs:** 422 Unprocessable Entity on POST with JSON array body.

### Pitfall 3: snapshot_sha breaks frozen model validation
**What goes wrong:** If `snapshot_sha` is computed after construction and someone tries to assign it, Pydantic v2 raises `ValidationError` because the model is frozen.
**Why it happens:** `ConfigDict(frozen=True)` makes all fields immutable after `__init__`.
**How to avoid:** Compute SHA-256 before constructing `FileLineSource` and pass it as a constructor argument.
**Warning signs:** `ValidationError: Instance is frozen` at runtime.

### Pitfall 4: ConflictDetector.reset() not called between pipeline runs
**What goes wrong:** The second REST call to `POST /debates` with the same conflicts produces no output (dedup suppresses them).
**Why it happens:** `ConflictDetector._seen_conflicts` is an instance-level set that persists across calls.
**How to avoid:** Either instantiate a fresh `ConflictDetector()` per request (cleanest), or call `self._detector.reset()` at the start of `AgentOrchestrator.run()`.
**Warning signs:** `POST /debates` returns `conflicts_found: 0` on the second call with identical inputs.

### Pitfall 5: DebateState validator rejects ESCALATED without experiment_id
**What goes wrong:** `DebateManager.read_debate()` fails with `ValidationError` when the debate file has `status: escalated` but `experiment_id: null`.
**Why it happens:** `DebateState` has a `model_validator` enforcing `experiment_id is not None` when status is ESCALATED (schemas.py:703-709, VERIFIED).
**How to avoid:** Always call `debate_manager.escalate_debate(debate_id, experiment_id)` before writing, never manually set status without the experiment_id.
**Warning signs:** `ValidationError: experiment_id is required when status is 'escalated'`.

## Code Examples

### POST /debates — create debate via REST
```python
# Source: api/v1/risk.py and core/debate_manager.py patterns (VERIFIED)
import hashlib
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict

from finalayze.api.v1.auth import api_key_auth
from finalayze.core.debate_manager import DebateManager
from finalayze.core.schemas import AgentOutput
from finalayze.orchestration.agent_orchestrator import AgentOrchestrator

router = APIRouter(
    prefix="/debates",
    tags=["debates"],
    dependencies=[Depends(api_key_auth)],
)

class CreateDebateRequest(BaseModel):
    outputs: list[AgentOutput]

class CreateDebateResponse(BaseModel):
    model_config = ConfigDict(frozen=True)
    debate_id: str | None
    conflicts_found: int

@router.post("", response_model=CreateDebateResponse, status_code=201)
async def create_debate(req: CreateDebateRequest) -> CreateDebateResponse:
    orch = AgentOrchestrator()
    debate_ids = orch.run(req.outputs)
    return CreateDebateResponse(
        debate_id=debate_ids[0] if debate_ids else None,
        conflicts_found=len(debate_ids),
    )
```

### GET /experiments/{experiment_id} — read experiment detail
```python
# Source: core/experiment_manager.py:142-172 (VERIFIED)
@router.get("/{experiment_id}", response_model=ExperimentDetailResponse)
async def get_experiment(experiment_id: str) -> ExperimentDetailResponse:
    em = ExperimentManager()
    try:
        state = em.read_experiment(experiment_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return ExperimentDetailResponse(
        experiment_id=state.experiment_id,
        hypothesis=state.hypothesis,
        status=str(state.status),
        verdict=state.verdict,
        reasoning=state.reasoning,
        debate_id=state.debate_id,
        results=[...],
    )
```

### snapshot_sha computation at claim creation time
```python
# Standard library pattern (ASSUMED — no existing usage in codebase)
import hashlib
from pathlib import Path

from finalayze.core.schemas import Claim, FileLineSource

def make_file_claim_with_snapshot(
    statement: str,
    path: str,
    line: int,
    excerpt: str,
    confidence: float,
) -> Claim:
    sha = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    source = FileLineSource(
        path=path,
        line=line,
        excerpt=excerpt,
        snapshot_sha=sha,
    )
    return Claim(statement=statement, source=source, confidence=confidence)
```

### router.py — adding new routers
```python
# Source: api/v1/router.py:1-26 (VERIFIED)
from finalayze.api.v1.debates import router as debates_router
from finalayze.api.v1.experiments import router as experiments_router

api_router.include_router(debates_router)
api_router.include_router(experiments_router)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Manual debate creation via file editing | DebateManager CRUD + REST POST /debates | Phase 33 + Phase 37 | Automated pipeline |
| arbiter invoked via CLI only | agent-orchestrator.md Claude Code agent | Phase 37 | On-demand autonomous runs |
| FileLineSource with no staleness detection | snapshot_sha optional field | Phase 37 | Prevents false CONTRADICTED after refactors |

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `AgentOrchestrator.run()` should be synchronous (not async) since all I/O is file-based | Architecture Patterns | If DebateManager or ExperimentManager gain async I/O in future, needs refactor |
| A2 | One debate per `run()` call (all conflicts from one batch → one debate) | Architecture Patterns | May need multi-debate creation if conflicts are unrelated |
| A3 | Short-circuiting on zero conflicts is the right behavior for `AgentOrchestrator.run()` | Architecture Patterns | Some callers may want a response even with no conflicts; add flag if needed |

## Open Questions

1. **Debate ID generation strategy**
   - What we know: ConflictDetector uses SHA-256 of agents+topics+type for `conflict_id`
   - What's unclear: Should debate_id be derived from conflict_ids, or a separate timestamp-based ID?
   - Recommendation: Use SHA-256 of sorted agent names + timestamp truncated to minute — ensures reproducibility within a session but uniqueness across sessions. Claude's discretion.

2. **How does the REST POST /debates trigger arbiter invocation?**
   - What we know: arbiter-agent is a Claude Code sub-agent — cannot be invoked from Python REST handler
   - What's unclear: Does the REST endpoint only create the debate structure (steps 1-4), leaving arbiter invocation to the human/agent-orchestrator?
   - Recommendation: YES — `POST /debates` returns the debate_id, caller (agent-orchestrator.md or human) then invokes arbiter-agent manually with that debate_id. This matches the CONTEXT.md decision that REST is for manual triggering.

3. **agent-orchestrator.md: which domain agents to spawn?**
   - What we know: CONTEXT.md says "spawn 3+ domain agents" — quant-analyst, risk-officer, ml-engineer exist
   - What's unclear: Fixed list vs configurable
   - Recommendation: Fixed list of 3 in agent-orchestrator.md (quant-analyst, risk-officer, ml-engineer). Claude's discretion on prompt structure.

## Environment Availability

Step 2.6: SKIPPED (no external dependencies — all I/O is file-based, uses existing FastAPI/Pydantic/hashlib)

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (existing) |
| Config file | pyproject.toml [tool.pytest.ini_options] |
| Quick run command | `uv run pytest tests/unit/core/test_agent_orchestrator.py tests/unit/test_api_debates.py tests/unit/test_api_experiments.py -x` |
| Full suite command | `uv run pytest tests/unit/ -x --timeout=60` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| ORCH-01 | AgentOrchestrator.run() creates debates from conflicts | unit | `uv run pytest tests/unit/core/test_agent_orchestrator.py -x` | ❌ Wave 0 |
| ORCH-01 | AgentOrchestrator.run() short-circuits on zero conflicts | unit | `uv run pytest tests/unit/core/test_agent_orchestrator.py::test_no_conflicts_returns_empty -x` | ❌ Wave 0 |
| ORCH-02 | POST /api/v1/debates returns 201 with debate_id | unit | `uv run pytest tests/unit/test_api_debates.py -x` | ❌ Wave 0 |
| ORCH-02 | GET /api/v1/debates returns list | unit | `uv run pytest tests/unit/test_api_debates.py::test_list_debates -x` | ❌ Wave 0 |
| ORCH-02 | GET /api/v1/experiments/{id} returns detail | unit | `uv run pytest tests/unit/test_api_experiments.py -x` | ❌ Wave 0 |
| ORCH-02 | Endpoints return 401 without X-API-Key | unit | `uv run pytest tests/unit/test_api_debates.py::test_auth_required -x` | ❌ Wave 0 |
| ORCH-03 | FileLineSource accepts snapshot_sha field | unit | `uv run pytest tests/unit/core/test_debate_schemas.py -x` | ✅ (extend) |
| ORCH-03 | Arbiter marks claim UNTESTABLE on SHA mismatch | unit | `uv run pytest tests/unit/core/test_agent_orchestrator.py::test_snapshot_sha_mismatch -x` | ❌ Wave 0 |
| ORCH-04 | agent-orchestrator.md has valid YAML frontmatter | manual | manual review | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `uv run pytest tests/unit/core/test_agent_orchestrator.py tests/unit/test_api_debates.py tests/unit/test_api_experiments.py -x`
- **Per wave merge:** `uv run pytest tests/unit/ -x --timeout=60`
- **Phase gate:** Full suite green before `/gsd-verify-work`

### Wave 0 Gaps
- [ ] `tests/unit/core/test_agent_orchestrator.py` — covers ORCH-01 and snapshot_sha UNTESTABLE logic
- [ ] `tests/unit/test_api_debates.py` — covers ORCH-02 debates endpoints (POST, GET list, GET detail, 401)
- [ ] `tests/unit/test_api_experiments.py` — covers ORCH-02 experiments endpoints (GET list, GET detail, 401)

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | yes | X-API-Key via `api_key_auth` dependency (existing, hmac.compare_digest) |
| V3 Session Management | no | Stateless REST endpoints |
| V4 Access Control | no | Single API key gates all endpoints |
| V5 Input Validation | yes | Pydantic v2 with field validators on AgentOutput, FileLineSource |
| V6 Cryptography | yes | hashlib.sha256 for snapshot_sha (stdlib — never hand-roll) |

### Known Threat Patterns for FastAPI + file-based persistence

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Path traversal via FileLineSource.path | Tampering | Arbiter-agent already validates path must start with src/, tests/, config/, docs/ |
| Debate ID collision (SHA-256 birthday attack) | Tampering | SHA-256 provides 2^128 collision resistance — acceptable |
| JSON injection via agent_name/recommendation | Tampering | Pydantic v2 validates types; YAML serialization escapes special chars |

## Sources

### Primary (HIGH confidence)
- `src/finalayze/core/schemas.py:530-795` — All relevant schemas verified by direct read
- `src/finalayze/core/debate_manager.py:1-202` — Full DebateManager public API verified
- `src/finalayze/core/experiment_manager.py:1-295` — Full ExperimentManager public API verified
- `src/finalayze/orchestration/conflict_detector.py:1-328` — ConflictDetector.detect() signature verified
- `src/finalayze/api/v1/router.py` — include_router pattern verified
- `src/finalayze/api/v1/risk.py` — Router/auth/response model pattern verified
- `src/finalayze/api/v1/auth.py` — api_key_auth dependency verified
- `.claude/agents/arbiter-agent.md` — Agent definition format and FileLineSource verification protocol verified

### Secondary (MEDIUM confidence)
- `src/finalayze/api/v1/portfolio.py` — Additional router pattern confirmation
- `src/finalayze/api/v1/sandbox.py` — Module-level singleton pattern for optional dependencies
- `.planning/phases/37-agent-orchestrator-+-debate-experiment-rest-api/37-CONTEXT.md` — All locked decisions

### Tertiary (LOW confidence)
- None — all claims in this research are VERIFIED from direct code reads

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all libraries already in use, verified from existing imports
- Architecture: HIGH — patterns verified from existing code, no new patterns introduced
- Pitfalls: HIGH — derived from direct schema/code inspection, especially the frozen model and DebateState validator issues

**Research date:** 2026-04-12
**Valid until:** 2026-05-12 (stable codebase, no fast-moving external dependencies)
