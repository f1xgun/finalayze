# Phase 36: Conflict Detection Foundation - Context

**Gathered:** 2026-04-12
**Status:** Ready for planning

<domain>
## Phase Boundary

This phase delivers structured agent output emission across all domain/analysis agents, a `parse_structured()` method on all LLM clients, a deterministic `ConflictDetector` with `ConflictReport` schema, and debouncing/deduplication logic to prevent debate storms.

</domain>

<decisions>
## Implementation Decisions

### Conflict Detection Algorithm
- Direction contradictions detected via keyword match on `recommendation` field (BUY vs SELL) — fast, deterministic
- Metric value contradictions detected via `MetricSource.value` comparison: >15% relative divergence = contradiction
- Statement contradictions detected via `difflib.SequenceMatcher.ratio() > 0.85` + divergent conclusions
- Severity scoring: 3 levels — CRITICAL (direction), HIGH (metric >30% divergence), LOW (statement similarity)

### Agent Output Integration
- All domain/analysis agents must emit structured `AgentOutput`: quant-analyst, risk-officer, ml-engineer, strategies-agent, portfolio-strategist, systems-architect
- `parse_structured()` added to `_CachingLLMClient` base class — all 3 clients (Anthropic, OpenAI, OpenRouter) inherit it
- Agent `.md` definitions get `## Output Format` section requiring `AgentOutput` JSON block
- Fallback: wrap free-text in single Claim with `MetricSource(metric_name="unstructured", value=0.0)` and confidence=0.0

### Debouncing & Deduplication
- Dedup key: SHA-256 hash of `(sorted agent_names, sorted claim topics, conflict_type)` — deterministic, collision-proof
- Dedup window: per-session (cleared on orchestrator restart) — aligns with weekly/daily agent cycles
- Minimum confidence delta: 0.15 — agents must disagree by >15% confidence to trigger escalation
- Skip consecutive cycle requirement for v8.0 — orchestrator runs on-demand (weekly), not per-cycle

### Claude's Discretion
- Internal data structures for ConflictDetector (e.g., topic extraction method)
- Test fixture design for synthetic AgentOutput objects

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `core/schemas.py:569` — `Claim` model with `ClaimSource` discriminated union (`FileLineSource | MetricSource`)
- `core/schemas.py:588` — `AgentOutput` model with `agent_name`, `recommendation`, `claims`, `timestamp`
- `core/schemas.py:541` — `FileLineSource` (path, line, excerpt) — needs `snapshot_sha` addition (Phase 37)
- `core/schemas.py:551` — `MetricSource` (metric_name, value, iteration)
- `analysis/llm_client.py:246` — `AnthropicClient` extending `_CachingLLMClient` with retry and caching
- `analysis/llm_client.py` — `_CachingLLMClient` base class, `OpenRouterClient`, `OpenAIClient`
- `core/debate_manager.py` — `DebateManager` CRUD for debate persistence (`.planning/debates/`)

### Established Patterns
- Pydantic v2 with `ConfigDict(frozen=True)` for all schemas
- `StrEnum` for enum types (ruff UP042)
- `_CachingLLMClient` uses SHA-256 LRU cache and exponential backoff retry
- Layer 0 constraint: `core/schemas.py` has zero project imports (only Pydantic, stdlib)
- `from __future__ import annotations` in every file

### Integration Points
- `ConflictReport` schema goes in `core/schemas.py` (Layer 0)
- `ConflictDetector` goes in `orchestration/conflict_detector.py` (Layer 5 — imports from Layer 0)
- `parse_structured()` goes on `_CachingLLMClient` in `analysis/llm_client.py` (Layer 3)
- Agent `.md` files in `.claude/agents/` — 6 agents to update

</code_context>

<specifics>
## Specific Ideas

- User wants ALL analysis/domain agents to produce structured output, not just the initial 3
- `parse_structured()` must be on the base class `_CachingLLMClient`, not just `AnthropicClient`
- Each LLM provider (Anthropic, OpenAI, OpenRouter) needs its own implementation of parse_structured() since the SDKs differ

</specifics>

<deferred>
## Deferred Ideas

- `snapshot_sha` on `FileLineSource` — deferred to Phase 37 (orchestrator needs it for arbiter safety)
- Consecutive cycle requirement for debouncing — only needed if orchestrator runs per-cycle (v8.x)

</deferred>
