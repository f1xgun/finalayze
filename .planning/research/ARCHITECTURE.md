# Architecture Research

**Domain:** Agent integration & autonomous decision loop for a live trading system
**Researched:** 2026-04-12
**Confidence:** HIGH — based on direct codebase inspection

---

## Standard Architecture

### System Overview — v8.0 Target State

```
┌─────────────────────────────────────────────────────────────────────┐
│  Layer 6: API / Dashboard                                           │
│  ┌──────────────────┐  ┌───────────────────┐  ┌─────────────────┐  │
│  │ api/v1/debates   │  │ api/v1/experiments│  │ dashboard/pages │  │
│  │  (NEW REST)      │  │  (NEW REST)       │  │ experiments_list│  │
│  └──────────┬───────┘  └────────┬──────────┘  └────────┬────────┘  │
├─────────────┼───────────────────┼─────────────────────┼────────────┤
│  Layer 5: Orchestration                                             │
│  ┌──────────┴───────────────────┴─────────────────────┴─────────┐  │
│  │  orchestration/agent_orchestrator.py  (NEW)                  │  │
│  │  - Collects AgentOutputs from domain agents                  │  │
│  │  - Runs ConflictDetector                                     │  │
│  │  - Routes conflicts → DebateManager → arbiter-agent          │  │
│  │  - Routes escalations → ExperimentManager → backtest runner  │  │
│  │  - On ACCEPTED verdict → PresetApplicator                    │  │
│  └──────────────────────────────────────────────────────────────┘  │
│  ┌─────────────────────────┐   ┌──────────────────────────────┐    │
│  │ orchestration/conflict_ │   │ orchestration/preset_        │    │
│  │ detector.py  (NEW)      │   │ applicator.py  (NEW)         │    │
│  └─────────────────────────┘   └──────────────────────────────┘    │
├─────────────────────────────────────────────────────────────────────│
│  Layer 4: Strategy / Risk                                           │
│  strategies/combiner.py      risk/position_sizing_pipeline.py       │
│  (UNCHANGED — receives applied presets)                             │
├─────────────────────────────────────────────────────────────────────│
│  Layer 3: Analysis / ML                                             │
│  analysis/*, ml/*            (domain agents invoke these)           │
├─────────────────────────────────────────────────────────────────────│
│  Layer 0: Types & Schemas                                           │
│  ┌────────────────┐  ┌────────────────────┐  ┌──────────────────┐  │
│  │ core/schemas.py│  │ core/debate_manager│  │ core/experiment_ │  │
│  │ AgentOutput    │  │ .py  (EXISTS)      │  │ manager.py       │  │
│  │ Claim          │  │                    │  │ (EXISTS)         │  │
│  │ DebateState    │  └────────────────────┘  └──────────────────┘  │
│  │ ExperimentState│                                                 │
│  └────────────────┘                                                 │
└─────────────────────────────────────────────────────────────────────┘

File-based stores (.planning/debates/, .planning/experiments/):
  ← DebateManager reads/writes →  ← ExperimentManager reads/writes →

Script-based runner (scripts/run_iteration.py --hypothesis <id>):
  ← invoked by orchestrator or manually →
```

---

## Component Responsibilities

### Existing Components — Unchanged or Lightly Modified

| Component | Layer | Responsibility | v8.0 Change |
|-----------|-------|----------------|-------------|
| `core/schemas.py` | 0 | Pydantic types: `AgentOutput`, `Claim`, `DebateState`, `ExperimentState` | Add `ConflictReport` schema; no structural changes |
| `core/debate_manager.py` | 0 | CRUD for `.planning/debates/*.md` YAML frontmatter files | No changes needed |
| `core/experiment_manager.py` | 0 | CRUD for `.planning/experiments/*.md`, verdict logic | No changes needed |
| `scripts/run_iteration.py` | Script | Backtest runner; `--hypothesis` flag merges `preset_overrides` into strategy configs | No changes needed — already fully wired |
| `scripts/run_interaction_test.py` | Script | A/B/AB comparison via subprocess calls to `run_iteration.py` | No changes needed |
| `.claude/agents/arbiter-agent.md` | Agent | Fact-checks claims from `AgentOutput`, produces `FactCheckReport` | No changes needed |
| `strategies/presets/*.yaml` | Config | Strategy parameter files per segment | MODIFIED by `PresetApplicator` on ACCEPT |
| `dashboard/pages/experiments_list.py` | 6 | Experiment Lab UI with list/detail/history tabs | Minor: add debate link display in detail view |

### New Components Required

| Component | Layer | Responsibility |
|-----------|-------|----------------|
| `orchestration/conflict_detector.py` | 5 | Compares multiple `AgentOutput` objects; detects contradictions in recommendations or overlapping metric claims |
| `orchestration/preset_applicator.py` | 5 | Applies `ExperimentState.preset_overrides` to `strategies/presets/*.yaml` on ACCEPT verdict with backup + diff logging |
| `orchestration/agent_orchestrator.py` | 5 | Full pipeline coordinator: collect outputs → detect conflicts → trigger debate → arbiter → experiment → verdict → apply |
| `api/v1/debates.py` | 6 | REST endpoints: `GET /debates`, `GET /debates/{id}`, `POST /debates` |
| `api/v1/experiments.py` | 6 | REST endpoints: `GET /experiments`, `GET /experiments/{id}`, `POST /experiments/{id}/apply` |

---

## Recommended Project Structure

```
src/finalayze/
├── core/
│   ├── schemas.py              # EXISTS — add ConflictReport schema only
│   ├── debate_manager.py       # EXISTS — no changes
│   └── experiment_manager.py   # EXISTS — no changes
├── orchestration/
│   ├── trading_loop.py         # EXISTS
│   ├── bond_cycle.py           # EXISTS
│   ├── conflict_detector.py    # NEW
│   ├── preset_applicator.py    # NEW
│   └── agent_orchestrator.py   # NEW
├── api/
│   └── v1/
│       ├── router.py           # EXISTS — register new routers
│       ├── debates.py          # NEW
│       └── experiments.py      # NEW (experiments apply endpoint)
└── dashboard/
    └── pages/
        └── experiments_list.py # EXISTS — minor: show debate link in detail

scripts/
├── run_iteration.py            # EXISTS — already wired with --hypothesis
└── run_interaction_test.py     # EXISTS — already wired

.planning/
├── debates/                    # EXISTS (empty at start) — debate files written here
└── experiments/                # EXISTS (empty at start) — experiment files written here

strategies/presets/
├── ru_blue_chips.yaml          # EXISTS — modified by PresetApplicator on ACCEPT
├── ru_blue_chips.yaml.bak.{experiment_id}  # CREATED by PresetApplicator as backup
└── ...
```

---

## Architectural Patterns

### Pattern 1: Structured Output Emission

**What:** Each domain agent (quant-analyst, risk-officer, ml-engineer) must return an `AgentOutput` Pydantic model when asked for a recommendation. The schema is already defined in `core/schemas.py`.

**When to use:** Any time two or more domain agents are invoked on the same question (e.g., "should we increase momentum weight in ru_blue_chips?").

**Critical constraint:** `AgentOutput.claims` requires at least one `Claim` with a typed `source` (`FileLineSource` or `MetricSource`). Agents cannot emit opinions without evidence-backed assertions. This is enforced by `min_length=1` in the field definition.

**How agents connect today:** Agent definitions in `.claude/agents/*.md` are Claude Code sub-agents. They do not have a Python calling interface. The orchestrator (itself a Claude Code agent) invokes them via the sub-agent protocol, then parses the returned text to extract the `AgentOutput` JSON structure. The orchestrator prompt must demand JSON-formatted `AgentOutput`.

### Pattern 2: Conflict Detection

**What:** `ConflictDetector` compares a list of `AgentOutput` objects and identifies contradictions.

**Implementation approach:** Rule-based detection only — no LLM semantic comparison. Two outputs conflict when:
1. Their `recommendation` fields contain opposing keywords (enable vs disable, increase vs decrease, raise vs lower)
2. Their `MetricSource` claims reference the same `metric_name` from the same `iteration` but with values differing by more than a tolerance (e.g., 0.01)

**Output:** A new `ConflictReport` schema in `core/schemas.py` (Layer 0) containing the conflicting agent pair and a human-readable conflict description.

**Why rule-based:** Deterministic, fast (no LLM call in the hot path), unit-testable with pure fixtures.

### Pattern 3: Debate → Arbiter → Escalation Pipeline

**What:** When `ConflictDetector` returns a non-empty `ConflictReport`, the orchestrator executes:
1. `DebateManager.create_debate(topic, agents)` — creates `.planning/debates/{id}.md`
2. `DebateManager.add_agent_position(debate_id, agent_name, output)` for each conflicting agent
3. Invokes `arbiter-agent` sub-agent with debate ID and both `AgentOutput` JSON objects
4. Arbiter produces `FactCheckReport`; orchestrator calls `DebateManager.add_arbiter_report(debate_id, report)`
5. If `FactCheckReport.has_contradictions`: calls `ExperimentManager.create_experiment()` which internally calls `DebateManager.escalate_debate()` (bidirectional link)
6. If no contradictions: calls `DebateManager.resolve_debate(debate_id, resolution)`

**The bidirectional link is already implemented:** `ExperimentManager.create_experiment(debate_id=...)` calls `DebateManager.escalate_debate()` at line 135 of `experiment_manager.py`. No new wiring needed here.

### Pattern 4: Experiment → Backtest → Verdict → Auto-Apply

**What:** After experiment creation (status=PENDING), the orchestrator triggers the backtest pipeline:
1. Invokes `scripts/run_iteration.py --hypothesis <experiment_id>` (or `run_interaction_test.py` for A/B comparison)
2. The script reads `ExperimentState.preset_overrides`, deep-merges into strategy configs (lines 1068-1080 of `run_iteration.py`), runs backtest, calls `ExperimentManager.link_result()`
3. The script calls `ExperimentManager.record_verdict(metric_value)` — computes ACCEPTED/REJECTED/INCONCLUSIVE
4. Orchestrator reads final `ExperimentState`; if ACCEPTED → `PresetApplicator.apply(experiment)`
5. If REJECTED/INCONCLUSIVE → no file changes; orchestrator logs and Telegram-alerts

**The backtest wiring is already complete:** Steps 1-3 exist in `run_iteration.py`. The only missing piece is step 4 (PresetApplicator) and the orchestrator reading the final verdict.

### Pattern 5: PresetApplicator Safety Protocol

**What:** A single-responsibility module writing `preset_overrides` to `strategies/presets/*.yaml` on ACCEPT.

**Where:** `orchestration/preset_applicator.py` (Layer 5). Imports `core/schemas.py` (Layer 0). Writes to `strategies/presets/` (filesystem, not a Python import — no layer violation).

**Safety requirements:**
1. Always back up before writing: `{segment}.yaml.bak.{experiment_id}`
2. Use the same `_deep_merge()` function pattern as `run_iteration.py` (lines 950-959) for consistency
3. Log the full YAML diff at INFO level before writing
4. On write failure, restore from backup and raise
5. Write a `{segment}.yaml.applied.{experiment_id}` marker file after successful apply — idempotency guard

**Scope limitation:** PresetApplicator writes YAML only. The live `TradingLoop` reads presets at startup. A live-reload signal or restart is required for changes to affect live trading. This is explicitly out of scope for v8.0 — presets apply to the next backtest or restart.

---

## Data Flow

### Full Orchestration Flow

```
Domain agents invoked (quant-analyst, risk-officer, ml-engineer)
    ↓ each returns AgentOutput JSON (recommendation + claims with sources)
ConflictDetector.detect(outputs: list[AgentOutput]) → ConflictReport
    ↓ if no conflicts
    → debate skipped; winning recommendation logged
    ↓ if conflicts found
DebateManager.create_debate(topic, agents) → debate_id
DebateManager.add_agent_position(debate_id, agent, output) × N
    ↓
arbiter-agent invoked with debate_id + AgentOutputs JSON
    ↓ returns FactCheckReport
DebateManager.add_arbiter_report(debate_id, report)
    ↓ if report.has_contradictions == False
DebateManager.resolve_debate(debate_id, resolution)
    ↓ if report.has_contradictions == True
ExperimentManager.create_experiment(
    experiment_id, hypothesis, success_criteria, debate_id, preset_overrides)
    → internally calls DebateManager.escalate_debate(debate_id, experiment_id)
    ↓
scripts/run_iteration.py --hypothesis <experiment_id>
    → ExperimentManager.update_status(RUNNING)
    → reads preset_overrides, deep-merges into strategy_configs
    → runs backtest engine
    → ExperimentManager.link_result(ExperimentResult)
    → ExperimentManager.record_verdict(primary_metric_value)
      → status = ACCEPTED | REJECTED | INCONCLUSIVE
    ↓ if status == ACCEPTED
PresetApplicator.apply(experiment_state)
    → backs up strategies/presets/{segment}.yaml → .yaml.bak.{experiment_id}
    → deep-merges preset_overrides into YAML
    → writes new YAML
    → logs diff
    → writes .yaml.applied.{experiment_id} marker
    ↓ if status == REJECTED or INCONCLUSIVE
    → log + Telegram alert; no file changes
```

### State Persistence

All state is file-based. No new database tables are needed for v8.0.

```
.planning/debates/{debate_id}.md         — YAML frontmatter + markdown body
.planning/experiments/{exp_id}.md        — YAML frontmatter + markdown body
results/experiments/{exp_id}/            — per-run backtest JSON files
results/experiments/{exp_id}/control.json
results/experiments/{exp_id}/A-only.json (interaction test)
results/experiments/{exp_id}/B-only.json
results/experiments/{exp_id}/AB.json
strategies/presets/{segment}.yaml        — modified by PresetApplicator on ACCEPT
strategies/presets/{segment}.yaml.bak.{exp_id}  — backup before modification
strategies/presets/{segment}.yaml.applied.{exp_id}  — idempotency marker
```

### Agent Invocation Model

Domain agents (quant-analyst, risk-officer, ml-engineer) are Claude Code sub-agents in `.claude/agents/*.md`. They have no Python callable interface and require the Claude Code runtime for MCP tools (`Read`, `Bash`, `ast-index`).

For v8.0, the orchestration pipeline is triggered in two ways:

1. **Manually via Claude Code agent** — human invokes `agent-orchestrator` sub-agent which spawns domain agents, collects outputs, and calls the Python pipeline modules via `Bash` tool
2. **Via REST API** — `POST /debates` endpoint accepts pre-collected `AgentOutput` JSON bodies and runs the Python pipeline directly (no Claude Code sub-agent invocation from Python)

A fully autonomous trigger from `TradingLoop` (e.g., schedule-based agent invocations) is out of scope for v8.0 because agents require the Claude Code runtime environment, which is not available within the Python process.

---

## Integration Points — New vs Modified

| Component | Status | Integration Boundary |
|-----------|--------|---------------------|
| `core/schemas.py` | MODIFIED (minimal) | Add `ConflictReport` schema; imported by `conflict_detector.py` (Layer 5) |
| `orchestration/conflict_detector.py` | NEW | Imports `AgentOutput`, `ConflictReport` from Layer 0 |
| `orchestration/preset_applicator.py` | NEW | Imports `ExperimentState` from Layer 0; writes `strategies/presets/*.yaml` via filesystem |
| `orchestration/agent_orchestrator.py` | NEW | Imports `ConflictDetector`, `PresetApplicator` (Layer 5); calls `DebateManager`, `ExperimentManager` (Layer 0) |
| `api/v1/debates.py` | NEW | Registers on `api/v1/router.py`; reads/writes via `DebateManager` |
| `api/v1/experiments.py` | NEW | Registers on `api/v1/router.py`; reads via `ExperimentManager`; POST /apply triggers `PresetApplicator` |
| `api/v1/router.py` | MODIFIED | Include `debates_router` and `experiments_router` |
| `dashboard/pages/experiments_list.py` | MODIFIED (minor) | Add debate link display in experiment detail tab |
| `scripts/run_iteration.py` | NOT MODIFIED | Already fully wired with `--hypothesis` and `preset_overrides` merge |
| `scripts/run_interaction_test.py` | NOT MODIFIED | Already wired for A/B/AB comparison |
| `core/debate_manager.py` | NOT MODIFIED | All required methods exist: `create_debate`, `add_agent_position`, `add_arbiter_report`, `resolve_debate`, `escalate_debate` |
| `core/experiment_manager.py` | NOT MODIFIED | All required methods exist: `create_experiment`, `link_result`, `record_verdict`, `get_by_debate` |

---

## Suggested Build Order

Dependencies drive the order. Lower-layer components must exist before higher-layer consumers.

### Phase 36: Conflict Detection

1. Add `ConflictReport` schema to `core/schemas.py` (Layer 0 — no deps)
2. Implement `orchestration/conflict_detector.py` with `detect(outputs: list[AgentOutput]) -> ConflictReport`
3. Update domain agent `.md` definitions (quant-analyst, risk-officer, ml-engineer) to include instructions for emitting `AgentOutput` JSON
4. Unit tests for `ConflictDetector` with synthetic `AgentOutput` fixtures

**Rationale:** No upstream dependencies. Can be built and validated in isolation. ConflictDetector is the entry point of the whole pipeline — everything else gates on it.

### Phase 37: Agent Orchestrator + Debate API

1. Implement `orchestration/agent_orchestrator.py` — full pipeline from conflict detection through debate creation and arbiter invocation
2. Add `api/v1/debates.py` REST endpoints (`GET /debates`, `GET /debates/{id}`, `POST /debates`)
3. Add read-only endpoints to `api/v1/experiments.py` (`GET /experiments`, `GET /experiments/{id}`)
4. Register both routers in `api/v1/router.py`
5. Integration test: supply two conflicting `AgentOutput` JSON objects via `POST /debates`, verify debate file created and arbiter-agent invoked correctly

**Rationale:** Orchestrator depends on ConflictDetector (Phase 36) and existing DebateManager/ExperimentManager. REST endpoints depend on orchestrator. PresetApplicator is not needed yet — keep scope tight.

### Phase 38: PresetApplicator + Auto-Apply

1. Implement `orchestration/preset_applicator.py` with YAML backup + deep merge + diff logging + idempotency marker
2. Wire `POST /experiments/{id}/apply` endpoint to `PresetApplicator`
3. Add "Apply to Presets" button to `dashboard/pages/experiments_list.py` detail view (calls API endpoint)
4. End-to-end integration test: create experiment → run backtest → ACCEPT verdict → verify YAML written, backup exists, marker file created

**Rationale:** PresetApplicator is last because it has the highest blast radius — it writes production config files. Validate the full pipeline (Phases 36-37) before enabling write-back.

---

## Anti-Patterns

### Anti-Pattern 1: Moving DebateManager / ExperimentManager to Layer 5

**What people do:** Notice that these managers do file I/O (not pure data types) and move them to `orchestration/`.

**Why it's wrong:** `core/schemas.py` depends on `DebateState` and `ExperimentState`. If managers move to Layer 5, the UI (Layer 6) and API (Layer 6) that directly import managers would create an import path from Layer 6 through Layer 5 — which is legal. But then tests at Layer 0 could not import managers. More importantly, the Layer 0 classification is based on zero project imports (only stdlib + pydantic + yaml), which DebateManager and ExperimentManager satisfy. Moving them would be a cosmetic change with no structural benefit.

**Do this instead:** Keep managers in `core/`. The file I/O is an implementation detail. The Layer 0 rule is about import dependencies, not about "pure data only".

### Anti-Pattern 2: Invoking Claude Code Agents from Python via Subprocess

**What people do:** Implement `agent_orchestrator.py` as a Python script that calls `claude --agent quant-analyst` via subprocess and parses stdout.

**Why it's wrong:** Claude Code sub-agents require the full Claude Code runtime (MCP tools, file access, `ast-index`). Subprocess invocation does not provide this context and will fail for any agent that uses `Read`, `Bash`, or `ast-index` — which is all of them.

**Do this instead:** The orchestrator is itself a Claude Code agent (`.claude/agents/agent-orchestrator.md`). It spawns domain agents via the Claude Code sub-agent protocol. The Python `orchestration/agent_orchestrator.py` module provides the data pipeline (conflict detection, DebateManager calls, ExperimentManager calls, PresetApplicator) and is called via the `Bash` tool by the Claude Code orchestrator agent after collecting outputs.

### Anti-Pattern 3: ConflictDetector Using LLM Semantic Comparison

**What people do:** Send both `AgentOutput.recommendation` strings to an LLM and ask it to determine if they conflict.

**Why it's wrong:** Nondeterministic output, adds LLM latency to every orchestration run, costly, and impossible to unit-test reliably.

**Do this instead:** Rule-based detection on structured fields. `recommendation` fields contain binary-opposition keywords. `MetricSource` claims with the same `metric_name` + `iteration` but different `value` are objective contradictions. This is deterministic and covered by pure Python unit tests.

### Anti-Pattern 4: PresetApplicator Writing Without Backup

**What people do:** Write directly to `strategies/presets/ru_blue_chips.yaml` on ACCEPT verdict.

**Why it's wrong:** A bug in `preset_overrides` (wrong key path, wrong data type) silently corrupts the preset and breaks all subsequent backtests and live trading. No recovery path.

**Do this instead:** Always write backup first (`{segment}.yaml.bak.{experiment_id}`), then overwrite. On any write error, restore from backup. Log the full diff of old vs new YAML at INFO level. Write idempotency marker file last.

### Anti-Pattern 5: Arbiter Calling `record_verdict()` Directly

**What people do:** Have the arbiter sub-agent call `ExperimentManager.record_verdict()` as part of its fact-check run.

**Why it's wrong:** The arbiter's role is fact-checking only — it verifies claims but does not decide on experiment outcomes. Verdict determination is a separate step based on backtest metrics, not claim verification. Mixing these responsibilities makes the arbiter a decision-making agent, which violates its defined constraints (see `arbiter-agent.md` section 5, rule 1).

**Do this instead:** Arbiter produces `FactCheckReport`. Orchestrator reads the report, decides whether to escalate (create experiment) or resolve. Verdict is computed by `ExperimentManager.record_verdict()` after backtest results are linked.

---

## Scaling Considerations

| Scale | Architecture Note |
|-------|------------------|
| Current (file-based, <100 experiments) | `.planning/debates/` and `.planning/experiments/` work fine. `list_experiments()` scans directory — O(n), negligible at this volume. |
| Future (>500 experiments) | Replace directory scan with a SQLite index or append-only JSONL index file. `ExperimentManager` API does not need to change — only persistence backend. |
| Future (autonomous conflict detection) | If `TradingLoop` needs to trigger debates without human initiation, add a Redis Streams event (`debate.conflict_detected`) that the orchestrator listens to. Infrastructure exists in `core/events.py`. |
| Future (parallel agent execution) | Domain agents currently invoked sequentially by the orchestrator. Claude Code sub-agent protocol can invoke them in parallel via tool batching. No code change needed — orchestrator agent prompt handles this. |

---

## Sources

- Direct inspection of `src/finalayze/core/schemas.py` — `AgentOutput`, `Claim`, `DebateState`, `ExperimentState` schemas (HIGH confidence)
- Direct inspection of `src/finalayze/core/debate_manager.py` — full CRUD API including `add_arbiter_report`, `escalate_debate`, `add_agent_position` (HIGH confidence)
- Direct inspection of `src/finalayze/core/experiment_manager.py` lines 92-255 — `create_experiment` with bidirectional debate link at line 135, `record_verdict`, `get_by_debate` (HIGH confidence)
- Direct inspection of `scripts/run_iteration.py` lines 1068-1345 — `--hypothesis` flag, `preset_overrides` deep merge, `ExperimentManager.link_result()` call (HIGH confidence)
- Direct inspection of `scripts/run_interaction_test.py` — A/B/AB subprocess pattern via `_run_hypothesis()` (HIGH confidence)
- Direct inspection of `.claude/agents/arbiter-agent.md` — arbiter fact-checking protocol, input/output format, path scope constraints (HIGH confidence)
- Direct inspection of `src/finalayze/core/CLAUDE.md` — Layer 0 constraint: zero project imports (HIGH confidence)
- Direct inspection of `.planning/phases/33-structured-debate-protocol/33-CONTEXT.md` — Phase 33 design decisions (HIGH confidence)
- Direct inspection of `.planning/phases/34-experiment-registry-runner/34-CONTEXT.md` — Phase 34 design decisions (HIGH confidence)

---

*Architecture research for: v8.0 Agent Integration & Autonomous Decision Loop*
*Researched: 2026-04-12*
