# Project Research Summary

**Project:** Finalayze v8.0 — Agent Integration & Autonomous Decision Loop
**Domain:** Multi-agent orchestration with conflict detection and autonomous trading parameter management
**Researched:** 2026-04-12
**Confidence:** HIGH

## Executive Summary

Finalayze v8.0 is not a greenfield build — it is a wiring problem. v7.0 shipped fully functional `DebateManager`, `ExperimentManager`, `AgentOutput`/`Claim` schemas, a backtest engine with `--hypothesis` support, and an Experiment Lab UI. The gap is entirely in wiring: agents emit unstructured markdown, conflict detection does not exist, debates are opened manually, and experiment verdicts have no downstream effect on strategy parameters. v8.0 closes this gap by building three new Python modules (`ConflictDetector`, `AgentOrchestrator`, `PresetApplicator`) and two REST API routers, wiring existing components into an end-to-end autonomous loop.

The recommended architecture follows a strict separation of concerns: rule-based deterministic conflict detection (no LLM semantic comparison in the hot path), file-based state persistence for debates and experiments (no new DB tables), and a two-phase staging-then-apply pattern for preset YAML writes. Domain agents emit structured `AgentOutput` via `anthropic==0.83.0`'s native `client.messages.parse()`, which guarantees schema compliance. The orchestrator itself is a Claude Code sub-agent (not a Python subprocess invoker), because domain agents require the full Claude Code MCP runtime.

The highest-risk part of v8.0 is auto-apply: writing experiment verdicts to live strategy YAML presets that govern real MOEX capital (500K–2.5M RUB). Four distinct failure modes threaten this path — bypassing the pre-trade pipeline, racing the live strategy cycle, strategy toggles leaving open positions unmanaged, and circular arbiter contradictions after code changes. All four are preventable with a circuit-breaker gate, atomic staging-file rename, position ownership tracking, and a `snapshot_sha` field on `FileLineSource` claims.

## Key Findings

### Recommended Stack

Zero new packages required. The installed environment (`anthropic==0.83.0`, `pydantic>=2.10`, `pyyaml>=6.0.2`, `redis>=5.2.0`, Python 3.12 stdlib) covers all capability gaps.

**Core technologies:**
- `anthropic` SDK 0.83.0: `client.messages.parse(output_format=AgentOutput)` for structured emission — GA, no beta header, auto-derives JSON Schema from existing Pydantic models
- `difflib.SequenceMatcher` (stdlib): deterministic claim similarity scoring — threshold 0.85 + divergent source values = contradiction candidate; no LLM in the hot path
- `asyncio.gather()` (stdlib): parallel domain agent output collection
- `structlog` (installed): audit logging — bind `debate_id`, `experiment_id`, `conflict_type` to every event
- `PyYAML` (installed): preset YAML read/write with atomic `os.replace()` rename

Explicitly rejected: `langchain`/`langgraph`/`crewai`, `instructor`, `celery`, new DB tables, `numpy`/`scipy` for conflict scoring.

### Expected Features

**Must have (table stakes — v8.0 blockers):**
- Agents emit `AgentOutput` with structured `Claim` objects and mandatory source references
- Conflict detector comparing multi-agent outputs for contradictions (deterministic, rule-based)
- Arbiter auto-triggers on detected conflict → `FactCheckReport` produced
- Full orchestration pipeline: conflict → debate → arbiter → experiment → backtest → verdict
- Auto-apply on ACCEPT: `preset_overrides` written to strategy YAML with backup snapshot
- INCONCLUSIVE routing: Telegram alert, no auto-apply

**Should have (v8.x):**
- Conflict severity scoring to reduce noise
- Rollback safety gate: auto-revert on post-apply metric degradation
- Debate→experiment UI link in Experiment Lab

**Defer (v9+):**
- Automated daily-review conflict detection (weekly cycle must be validated first)
- Agent performance tracking (VERIFIED vs CONTRADICTED rates per agent)

### Architecture Approach

Three new modules in Layer 5 (`orchestration/`): `conflict_detector.py`, `preset_applicator.py`, `agent_orchestrator.py`. Existing Layer 0 components need minimal changes. Two new Layer 6 API routers. The orchestrator is a Claude Code agent definition that spawns domain agents and collects structured outputs.

**Major components:**
1. `orchestration/conflict_detector.py` — compares `list[AgentOutput]`, returns `ConflictReport`; pure logic, no I/O
2. `orchestration/agent_orchestrator.py` — full pipeline coordinator: collect outputs → detect conflicts → trigger debate → arbiter → experiment → verdict → apply
3. `orchestration/preset_applicator.py` — writes preset overrides to strategy YAML with backup, diff logging, idempotency, atomic rename
4. `api/v1/debates.py` + `api/v1/experiments.py` — REST interface for pipeline interaction
5. `.claude/agents/agent-orchestrator.md` — Claude Code agent definition for autonomous pipeline

### Critical Pitfalls

1. **Auto-apply bypasses pre-trade pipeline** — circuit-breaker gate must be FIRST check in `apply_verdict()`; run dry-run pre-trade check with synthetic signal on new parameters before writing YAML
2. **Debate storm from oversensitive conflict detection** — topic-level deduplication (slug hash); require 2+ consecutive cycle contradictions before escalation; minimum confidence delta > 0.15
3. **File-write race with live strategy cycle** — write to `{segment}.yaml.pending`, rename atomically at cycle start via `os.replace()`; call `combiner.invalidate_segment_cache()` after rename
4. **Strategy toggle leaves open positions unmanaged** — add `_entry_strategy` dict to `TradingLoop`; block disable auto-apply if positions exist
5. **Arbiter false CONTRADICTED after code change** — add `snapshot_sha` to `FileLineSource`; changed files → `UNTESTABLE` not `CONTRADICTED`
6. **Backtest ACCEPT ≠ live performance** — mandatory sandbox validation gate (≥3 trading days) before live apply

## Implications for Roadmap

### Phase 36: Conflict Detection Foundation

**Rationale:** Lowest-layer component with no upstream dependencies. Entry point of the entire v8.0 pipeline.
**Delivers:** `ConflictReport` schema; `orchestration/conflict_detector.py` with unit tests; updated agent definitions requiring `AgentOutput` JSON emission; `AnthropicClient.parse_structured()` method
**Addresses:** "Agents emit AgentOutput" + "Conflict detector"
**Avoids:** Debate storm pitfall — debouncing built in from day one

### Phase 37: Agent Orchestrator + Debate/Experiment REST API

**Rationale:** Depends on Phase 36. Wires the full debate lifecycle through verdict. Excludes write-back — validate pipeline first.
**Delivers:** `orchestration/agent_orchestrator.py` (pipeline through verdict, not apply); `api/v1/debates.py`; `api/v1/experiments.py` (read-only); `.claude/agents/agent-orchestrator.md`; `snapshot_sha` on `FileLineSource`
**Uses:** Existing `DebateManager`, `ExperimentManager`, `arbiter-agent.md` unchanged

### Phase 38: PresetApplicator + Auto-Apply Loop

**Rationale:** Highest blast radius last. All safety gates are preconditions, not afterthoughts.
**Delivers:** `orchestration/preset_applicator.py` (backup + atomic rename + diff logging); `POST /experiments/{id}/apply` endpoint; INCONCLUSIVE Telegram routing; `_entry_strategy` tracking in `TradingLoop`; `combiner.invalidate_segment_cache()` method
**Implements:** All auto-apply safety protocol

### Phase Ordering Rationale

- Dependency chain is strict: structured output → detection → orchestration → write-back
- Blast radius increases: Phase 36 read-only, Phase 37 file I/O (recoverable), Phase 38 live YAML writes (highest risk)
- Matches build order from ARCHITECTURE.md

### Research Flags

Phases needing deeper research during planning:
- **Phase 37:** Claude Code sub-agent orchestrator protocol — how the orchestrator spawns domain agents and collects structured JSON outputs
- **Phase 38:** Sandbox gate scoring criteria — pass/fail thresholds for sandbox validation

Standard patterns (skip research-phase):
- **Phase 36:** Pure Python with verified patterns; `difflib`, Pydantic, `client.messages.parse()` all confirmed
- **Phase 38 YAML mechanics:** Identical to existing `DebateManager` code pattern

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | All packages version-verified; zero new packages required |
| Features | HIGH | Grounded in direct codebase inspection with line citations |
| Architecture | HIGH | Based on inspection of 9 source files; layer assignments verified |
| Pitfalls | HIGH | Six critical pitfalls grounded in specific code paths |

**Overall confidence:** HIGH

### Gaps to Address

- **Claude Code orchestrator invocation protocol:** How the orchestrator agent spawns domain agents and receives `AgentOutput` JSON — validate in Phase 37 planning
- **`combiner.invalidate_segment_cache()` method:** Does not yet exist — must be added in Phase 38 as a prerequisite
- **Sandbox gate scoring criteria:** Pass/fail thresholds undefined — define in Phase 38 planning

## Sources

### Primary (HIGH confidence)
- `src/finalayze/core/schemas.py` — AgentOutput, Claim, DebateState, ExperimentState
- `src/finalayze/core/debate_manager.py` — full CRUD API
- `src/finalayze/core/experiment_manager.py` — create_experiment, record_verdict, _compute_verdict
- `scripts/run_iteration.py` — --hypothesis flag, preset_overrides deep merge
- `src/finalayze/risk/circuit_breaker.py`, `pre_trade_check.py` — 11-check pipeline
- `src/finalayze/orchestration/trading_loop.py` — _stop_states, _entry_prices, APScheduler cycle
- `src/finalayze/strategies/combiner.py` — preset cache loading
- `pyproject.toml` + `uv pip show anthropic` — anthropic==0.83.0 confirmed

### Secondary (MEDIUM confidence)
- Anthropic Structured Outputs docs — `client.messages.parse()` GA status
- TradingAgents framework (arxiv:2412.20138 v7, 2025) — hybrid orchestration pattern

---
*Research completed: 2026-04-12*
*Ready for roadmap: yes*
