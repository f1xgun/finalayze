---
phase: 33-structured-debate-protocol
verified: 2026-04-08T06:30:00Z
status: human_needed
score: 9/10 must-haves verified
overrides_applied: 0
human_verification:
  - test: "Invoke arbiter-agent with two sample conflicting AgentOutput JSON objects and confirm it produces a FactCheckReport with Verified/Contradicted/Untestable sections and a RESOLVE or ESCALATE recommendation"
    expected: "Structured markdown report with ## Verified, ## Contradicted, ## Untestable, ## Summary sections, and one of RESOLVE/ESCALATE recommendation"
    why_human: "Arbiter agent is a Claude Code agent prompt — it cannot be invoked by static grep. Actual agent execution requires a live Claude session."
---

# Phase 33: Structured Debate Protocol Verification Report

**Phase Goal:** Agent recommendations include verifiable evidence, conflicts are detected automatically, and unresolved conflicts escalate to experiments
**Verified:** 2026-04-08T06:30:00Z
**Status:** human_needed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Agent output schema enforces structured claims with source references (file:line or metric value) — no unsourced assertions in agent recommendations | VERIFIED | `AgentOutput.claims: list[Claim] = Field(min_length=1)` enforced in schemas.py:594. `Claim.source: ClaimSource` is a discriminated union of `FileLineSource` and `MetricSource` — no plain-text assertion possible. 19 schema tests pass. |
| 2 | An arbiter agent can take two conflicting agent outputs and produce a fact-check report showing which claims are verified, which are contradicted, and which are untestable | PARTIAL | `FactCheckReport` schema with `has_contradictions` property and `to_markdown()` method exists and is verified. `.claude/agents/arbiter-agent.md` contains both verification paths (ast-index for file claims, history.jsonl for metric claims) and correct output sections. Actual agent invocation with conflicting inputs requires human testing. |
| 3 | Debate state (claims, conflicts, resolutions) is persisted in `.planning/debates/` for audit trail — every multi-agent decision has a traceable history | VERIFIED | `.planning/debates/.gitkeep` exists. `DebateManager` creates `{debate_id}.md` files with YAML frontmatter. Roundtrip test (create → add positions → add arbiter report → read back) passes. 10 DebateManager tests pass. |
| 4 | Claim without a source raises ValidationError at construction | VERIFIED | `Claim.source: ClaimSource` is required with no default — omitting it raises `ValidationError`. Covered by `test_claim_with_file_source` and discriminator tests. |
| 5 | Claim with confidence outside [0.0, 1.0] raises ValueError | VERIFIED | `@field_validator("confidence")` in `Claim` raises `ValueError` for out-of-range values. Tests `test_claim_confidence_below_zero` and `test_claim_confidence_above_one` both pass. |
| 6 | AgentOutput with empty claims list raises ValidationError | VERIFIED | `Field(min_length=1)` on `AgentOutput.claims`. `test_agent_output_empty_claims` confirms `ValidationError` raised. |
| 7 | FileLineSource and MetricSource are discriminated by 'kind' field | VERIFIED | `ClaimSource = Annotated[FileLineSource \| MetricSource, Field(discriminator="kind")]` at schemas.py:562. `test_claim_source_discriminator` confirms correct deserialization from raw dicts. |
| 8 | FactCheckReport.has_contradictions returns True when any verdict is CONTRADICTED | VERIFIED | `@property has_contradictions` at schemas.py:626 uses `any(r.verdict == ClaimVerdict.CONTRADICTED ...)`. Both True/False tests pass. |
| 9 | DebateState tracks status transitions (open -> resolved or escalated) | VERIFIED | `DebateStatus` StrEnum with OPEN/RESOLVED/ESCALATED. `@model_validator(mode="after")` enforces `experiment_id` required for ESCALATED. `resolve_debate()` and `escalate_debate()` in DebateManager confirmed working via tests. |
| 10 | Arbiter agent definition exists with two verification paths (code claims via ast-index, metric claims via history.jsonl) | VERIFIED (partial — agent invocation needs human) | `.claude/agents/arbiter-agent.md` confirmed at disk. Contains `ast-index outline` in Section 3 Path A, `history.jsonl` scan in Section 3 Path B, float tolerance 0.01, path scope enforcement, ESCALATE rule. All grep checks pass. |

**Score:** 9/10 truths verified (Truth 2 partially blocked on human verification of live arbiter invocation)

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/finalayze/core/schemas.py` | Debate Protocol Schemas section with 10 models | VERIFIED | `# ── Debate Protocol Schemas` section at line 529. All 10 models present: DebateStatus, FileLineSource, MetricSource, ClaimSource, Claim, AgentOutput, ClaimVerdict, ClaimCheckResult, FactCheckReport, DebateState. |
| `tests/unit/core/test_debate_schemas.py` | Unit tests for debate protocol schemas (min 80 lines, 19 tests) | VERIFIED | File exists, 19 test functions, fully substantive (real assertions, no stubs). All 19 tests pass. |
| `.claude/agents/arbiter-agent.md` | Arbiter agent prompt with code-claim and metric-claim verification instructions | VERIFIED | File exists. Contains `ast-index` (Path A), `history.jsonl` (Path B), `0.01` tolerance, `## Verified`, `## Contradicted`, `## Untestable`, `ESCALATE`. YAML frontmatter: `name: arbiter-agent`, `model: claude-sonnet-4-20250514`. |
| `src/finalayze/core/debate_manager.py` | DebateManager class for creating, reading, updating debate files | VERIFIED | File exists. All 7 methods present: `create_debate`, `read_debate`, `resolve_debate`, `escalate_debate`, `list_debates`, `add_agent_position`, `add_arbiter_report`. Uses `yaml.safe_load`. |
| `tests/unit/core/test_debate_manager.py` | Unit tests for debate file CRUD (min 60 lines, 9 tests) | VERIFIED | File exists, 10 test functions, all substantive. All 10 tests pass. |
| `.planning/debates/.gitkeep` | Empty directory marker for debate state persistence | VERIFIED | File confirmed present. |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `tests/unit/core/test_debate_schemas.py` | `src/finalayze/core/schemas.py` | `from finalayze.core.schemas import` | WIRED | Import confirmed at test file line 10-20. All 9 schema symbols imported and exercised. |
| `src/finalayze/core/debate_manager.py` | `src/finalayze/core/schemas.py` | `from finalayze.core.schemas import` (TYPE_CHECKING + local import in read_debate) | WIRED | TYPE_CHECKING guard at debate_manager.py:18-19. Local import `from finalayze.core.schemas import DebateState, FactCheckReport` inside `read_debate()` at line 122. Both patterns confirmed. |
| `.claude/agents/arbiter-agent.md` | `results/iterations/history.jsonl` | metric claim verification instructions | WIRED | `history.jsonl` referenced explicitly in Section 3 Path B instructions and Key Files table. |

### Data-Flow Trace (Level 4)

Not applicable — this phase delivers schemas, CRUD file operations, and an agent prompt. No dynamic rendering components that fetch from external data sources.

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Schema tests pass (19 tests) | `uv run pytest tests/unit/core/test_debate_schemas.py -q` | 19 passed | PASS |
| DebateManager tests pass (10 tests) | `uv run pytest tests/unit/core/test_debate_manager.py -q` | 10 passed | PASS |
| All 29 debate tests pass together | `uv run pytest tests/unit/core/test_debate_schemas.py tests/unit/core/test_debate_manager.py -q` | 29 passed | PASS |
| Commits are real and ordered | `git log --oneline` | 3e87ef2 (RED), f260dda (GREEN), 3131dcc (Task 1), 65176aa (Task 2) all confirmed | PASS |
| debates directory exists | `test -f .planning/debates/.gitkeep` | EXIT 0 | PASS |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| DEBATE-01 | 33-01-PLAN.md | Typed claim schemas with source references | SATISFIED | `Claim`, `FileLineSource`, `MetricSource`, `AgentOutput` implemented in schemas.py; 19 tests pass |
| DEBATE-02 | 33-01-PLAN.md, 33-02-PLAN.md | Arbiter agent fact-check capability with FactCheckReport | SATISFIED (human step pending) | `FactCheckReport` schema with `has_contradictions` + `to_markdown()` implemented; arbiter-agent.md created with both verification paths |
| DEBATE-03 | 33-01-PLAN.md, 33-02-PLAN.md | Debate state persisted in `.planning/debates/` | SATISFIED | `DebateState` schema + `DebateManager` CRUD implemented; `.planning/debates/.gitkeep` exists; roundtrip test passes |

**Note on REQUIREMENTS.md:** DEBATE-01, DEBATE-02, DEBATE-03 are referenced in ROADMAP.md (Phase 33 section) and in plan frontmatter but are NOT formally defined in `.planning/REQUIREMENTS.md`. The requirements file only covers v6.0 requirements (GRPC-*, PERSIST-*, OBS-*, OPS-*) and partial v7.0 SANDBOX-FIX-* items. The DEBATE-* IDs are orphaned from REQUIREMENTS.md. This is a documentation gap — the requirements exist as ROADMAP.md Success Criteria but lack formal REQUIREMENTS.md entries.

### Anti-Patterns Found

| File | Pattern | Severity | Impact |
|------|---------|----------|--------|
| `src/finalayze/core/debate_manager.py:46` | `dict` type annotation without type args (`dict` instead of `dict[str, Any]`) suppressed with `# type: ignore[type-arg]` | Info | Cosmetic — ruff/mypy suppression is intentional and documented in SUMMARY. YAML frontmatter is inherently untyped. No functional impact. |
| `src/finalayze/core/debate_manager.py:122` | Local import inside function (`# noqa: PLC0415`) | Info | Intentional — deferred to avoid circular import at module init. Documented in SUMMARY.md deviations. No functional impact. |

No blockers found. No placeholder/TODO/FIXME patterns. No empty return stubs. No hardcoded empty data in render paths.

### Human Verification Required

#### 1. Live Arbiter Agent Invocation

**Test:** Invoke `arbiter-agent` with two sample conflicting `AgentOutput` JSON objects. For example, agent A claims `profit_factor = 1.29` for iteration `2026-04-05-adx-routing` (MetricSource), and agent B claims `profit_factor = 1.15` for the same iteration. Run the arbiter against both.

**Expected:**
- Arbiter reads `results/iterations/history.jsonl` and finds the actual value
- Produces a report with `## Verified`, `## Contradicted`, `## Untestable`, `## Summary` sections
- Issues `ESCALATE` recommendation if any claim is CONTRADICTED
- Issues `RESOLVE` recommendation if no contradictions

**Why human:** Arbiter is a Claude Code sub-agent definition. Its prompt correctness and actual verification behavior can only be confirmed by invoking it in a live Claude session with real conflicting inputs. Static analysis confirms the prompt content but cannot execute the agent.

### Gaps Summary

No blocking gaps. The one outstanding item (live arbiter agent invocation) is a human verification step that the VALIDATION.md explicitly calls out as manual-only. All automated checks pass.

The only administrative gap is that DEBATE-01, DEBATE-02, DEBATE-03 are not formally defined in `.planning/REQUIREMENTS.md` — they exist only as ROADMAP.md Success Criteria. This is a documentation debt, not a functional blocker.

---

_Verified: 2026-04-08T06:30:00Z_
_Verifier: Claude (gsd-verifier)_
