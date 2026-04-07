# Phase 33: Structured Debate Protocol - Context

**Gathered:** 2026-04-07
**Status:** Ready for planning

<domain>
## Phase Boundary

This phase delivers structured evidence requirements for agent outputs, an arbiter agent for conflict resolution, and persistent debate state tracking. Scope: agent output schemas, fact-checking infrastructure, and `.planning/debates/` audit trail. Does NOT include experiment execution (Phase 34) or UI (Phase 35).

</domain>

<decisions>
## Implementation Decisions

### Claim Schema Design
- Pydantic models with `Claim(statement, source, confidence)` — type-safe, matches project conventions
- Source references use two explicit types: `file:line` for code, `metric:value` for data
- Schema lives in `src/finalayze/core/schemas.py` (Layer 0) — agents across all layers can import
- Strict validation at schema level — claims without source raise ValueError

### Arbiter Agent Design
- Arbiter uses `ast-index` + grep to verify code claims (file:line references exist and match)
- Metric claims verified by running the metric computation and comparing against stated value
- Fact-check reports use structured markdown with verified/contradicted/untestable sections (matches SC-2)
- Arbiter implemented as Claude Code sub-agent (`.claude/agents/arbiter-agent.md`) — can use all tools for verification

### Debate Persistence & Escalation
- Debate state stored as markdown with YAML frontmatter (matches existing `.planning/` patterns)
- Naming convention: `{date}-{topic-slug}.md` — chronological + descriptive
- Conflicts auto-escalate to experiments when arbiter marks ≥1 claim as "contradicted" and both agents maintain their position
- `experiment_id` field in debate frontmatter provides forward reference for Phase 34 to consume

### Claude's Discretion
- Internal implementation details of claim validation logic
- Arbiter agent prompt engineering and fact-check heuristics
- Debate file template structure beyond required fields

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- 21 existing Claude Code sub-agents in `.claude/agents/` — established agent definition patterns
- `src/finalayze/core/schemas.py` — existing Pydantic schema location (Layer 0)
- `src/finalayze/core/exceptions.py` — FinalayzeError base class for domain exceptions
- `ast-index` tool available for codebase navigation (used by agents)

### Established Patterns
- Pydantic v2 for all schemas with strict typing
- YAML frontmatter in `.planning/` markdown files for structured metadata
- Agent definitions in `.claude/agents/*.md` with structured prompts
- `from __future__ import annotations` in all Python files
- StrEnum for enums, ruff formatting, mypy strict

### Integration Points
- `.planning/debates/` — new directory for debate state (consumed by Phase 34 experiment registry)
- `.claude/agents/arbiter-agent.md` — new agent definition
- `src/finalayze/core/schemas.py` — new Claim/DebateState models added to existing file
- Agent output validation hooks — agents reference claim schema in their output

</code_context>

<specifics>
## Specific Ideas

No specific requirements — open to standard approaches matching project conventions.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>
