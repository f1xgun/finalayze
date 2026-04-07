# Phase 33: Structured Debate Protocol - Research

**Researched:** 2026-04-07
**Domain:** Agent output schemas, fact-checking infrastructure, structured evidence enforcement
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Claim Schema Design**
- Pydantic models with `Claim(statement, source, confidence)` — type-safe, matches project conventions
- Source references use two explicit types: `file:line` for code, `metric:value` for data
- Schema lives in `src/finalayze/core/schemas.py` (Layer 0) — agents across all layers can import
- Strict validation at schema level — claims without source raise ValueError

**Arbiter Agent Design**
- Arbiter uses `ast-index` + grep to verify code claims (file:line references exist and match)
- Metric claims verified by running the metric computation and comparing against stated value
- Fact-check reports use structured markdown with verified/contradicted/untestable sections (matches SC-2)
- Arbiter implemented as Claude Code sub-agent (`.claude/agents/arbiter-agent.md`) — can use all tools for verification

**Debate Persistence & Escalation**
- Debate state stored as markdown with YAML frontmatter (matches existing `.planning/` patterns)
- Naming convention: `{date}-{topic-slug}.md` — chronological + descriptive
- Conflicts auto-escalate to experiments when arbiter marks ≥1 claim as "contradicted" and both agents maintain their position
- `experiment_id` field in debate frontmatter provides forward reference for Phase 34 to consume

### Claude's Discretion
- Internal implementation details of claim validation logic
- Arbiter agent prompt engineering and fact-check heuristics
- Debate file template structure beyond required fields

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope.
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| DEBATE-01 | Agent output schema enforces structured claims with source references (file:line or metric:value) | Pydantic v2 `Claim` + `AgentOutput` models in `core/schemas.py`; field_validator raises ValueError on missing source |
| DEBATE-02 | Arbiter agent takes two conflicting agent outputs and produces a structured fact-check report (verified / contradicted / untestable sections) | `.claude/agents/arbiter-agent.md`; uses `ast-index` for code claim verification, metric runner for data claims |
| DEBATE-03 | Debate state persisted in `.planning/debates/` for audit trail | Markdown + YAML frontmatter files; naming `{date}-{topic-slug}.md`; `experiment_id` forward-ref field for Phase 34 |
</phase_requirements>

---

## Summary

Phase 33 introduces a structured evidence layer on top of the existing 21-agent system. The core deliverable is three tightly coupled pieces: (1) Pydantic schemas that make unsourced assertions impossible at the type level, (2) an arbiter sub-agent that verifies claims against the actual codebase and metric history, and (3) a `.planning/debates/` directory holding the audit trail of every contested recommendation.

The domain is 100% within the existing stack — no new libraries required. The work touches Layer 0 (`core/schemas.py`), the `.claude/agents/` agent registry, and the `.planning/` file system convention already established across 32 prior phases. The biggest design decision (locked by CONTEXT.md) is the two-type source discriminator: `file:line` for code claims and `metric:value` for data claims. This distinction drives the arbiter's verification logic — code claims are checked with `ast-index` / grep; metric claims are checked by comparing against `results/iterations/history.jsonl`.

The only implementation risk is that metric claim verification requires a defined format for citing metric values that the arbiter can parse and re-compute. A clear `MetricSource(metric_name, value, iteration)` sub-model should be used to avoid ambiguity in the arbiter prompt.

**Primary recommendation:** Add `Claim`, `ClaimSource`, `AgentOutput`, `DebateState`, and `FactCheckReport` Pydantic models to `src/finalayze/core/schemas.py`. Implement `arbiter-agent.md` with two verification paths: ast-index for code claims, history.jsonl lookup for metric claims. Create `.planning/debates/` directory with YAML-frontmatter markdown template.

---

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Pydantic v2 | Already in pyproject.toml | Schema definition and strict validation for Claim models | Locked project convention — all schemas use Pydantic v2 |
| Python stdlib `pathlib` | 3.12 | Creating/reading `.planning/debates/` files | No extra dependency, already used project-wide |
| Python stdlib `datetime` | 3.12 | Debate file timestamps | Already used in all schemas |
| `ast-index` CLI | v3.27.0 (installed) | Code claim verification in arbiter agent | Project-installed tool; confirmed available `[VERIFIED: CLAUDE.md + memory]` |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `structlog` | Already in stack | Structured logging in claim validator | When claim validation failures should be observable |
| `yaml` (PyYAML) | Already in stack | Parsing YAML frontmatter in debate files | Reading/writing `.planning/debates/` frontmatter |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| YAML frontmatter in markdown | Pure YAML or JSON files | Markdown keeps human-readable narrative alongside structured metadata — matches all 32 existing `.planning/` files |
| Pydantic discriminated union for source types | Plain `str` with regex | Discriminated union gives type-safe branching in arbiter; plain str is easier to misuse |

**Installation:**
No new packages required — all dependencies are already in the project stack. `[VERIFIED: pyproject.toml]`

---

## Architecture Patterns

### Recommended Project Structure
```
src/finalayze/core/
└── schemas.py              # Add: Claim, ClaimSource, AgentOutput, DebateState, FactCheckReport

.claude/agents/
└── arbiter-agent.md        # New: arbiter sub-agent definition

.planning/debates/
└── {YYYY-MM-DD}-{topic-slug}.md   # New: debate state files
```

### Pattern 1: Pydantic Discriminated Union for Claim Sources

**What:** Use a Pydantic discriminated union (`Literal` tag field) to distinguish `FileLineSource` from `MetricSource`. The `ClaimSource` type is the union.

**When to use:** Any time a Claim is constructed — at schema construction time, not at verification time.

**Example:**
```python
# Source: established Pydantic v2 discriminated union pattern [ASSUMED - standard Pydantic docs pattern]
from __future__ import annotations
from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator

class FileLineSource(BaseModel):
    """A source reference pointing to a specific file and line."""
    model_config = ConfigDict(frozen=True)
    kind: Literal["file"] = "file"
    path: str            # e.g. "src/finalayze/strategies/combiner.py"
    line: int            # e.g. 142
    excerpt: str         # the actual text at that line (for arbiter comparison)

class MetricSource(BaseModel):
    """A source reference citing a metric value from iteration history."""
    model_config = ConfigDict(frozen=True)
    kind: Literal["metric"] = "metric"
    metric_name: str     # e.g. "profit_factor"
    value: float         # e.g. 1.29
    iteration: str       # e.g. "2026-04-05-adx-routing" — must exist in history.jsonl

ClaimSource = Annotated[
    FileLineSource | MetricSource,
    Field(discriminator="kind")
]

class Claim(BaseModel):
    """A verifiable assertion made by an agent."""
    model_config = ConfigDict(frozen=True)
    statement: str
    source: ClaimSource
    confidence: float    # 0.0 – 1.0

    @field_validator("confidence")
    @classmethod
    def confidence_must_be_probability(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            msg = f"confidence must be in [0.0, 1.0], got {v}"
            raise ValueError(msg)
        return v
```

### Pattern 2: AgentOutput Wrapping Existing Outputs

**What:** A thin wrapper that requires agents to supply `claims` alongside their existing recommendation text. Does not replace existing agent outputs — adds evidence layer on top.

**Example:**
```python
# [ASSUMED - designed to match project patterns]
class AgentOutput(BaseModel):
    """Structured agent recommendation with verifiable evidence."""
    model_config = ConfigDict(frozen=True)
    agent_name: str
    recommendation: str
    claims: list[Claim] = Field(min_length=1)  # At least one claim required
    timestamp: datetime
```

### Pattern 3: FactCheckReport

**What:** The arbiter agent's output — a structured verdict on each claim.

**Example:**
```python
# [ASSUMED - designed to match project conventions]
from enum import StrEnum

class ClaimVerdict(StrEnum):
    VERIFIED = "verified"
    CONTRADICTED = "contradicted"
    UNTESTABLE = "untestable"

class ClaimCheckResult(BaseModel):
    model_config = ConfigDict(frozen=True)
    claim: Claim
    verdict: ClaimVerdict
    evidence: str        # What the arbiter found (ast-index output, metric value, etc.)

class FactCheckReport(BaseModel):
    model_config = ConfigDict(frozen=True)
    debate_id: str
    arbiter_timestamp: datetime
    results: list[ClaimCheckResult]

    @property
    def has_contradictions(self) -> bool:
        return any(r.verdict == ClaimVerdict.CONTRADICTED for r in self.results)
```

### Pattern 4: Debate File YAML Frontmatter

**What:** Each `.planning/debates/` file uses YAML frontmatter (matching all existing `.planning/` files) plus a narrative markdown body.

**Example:**
```yaml
---
debate_id: 2026-04-07-adx-routing-threshold
topic: "ADX threshold for trend regime — 25 vs 30"
status: "open"         # open | resolved | escalated
created: "2026-04-07"
agents:
  - quant-analyst
  - risk-officer
arbiter_report: null   # populated after arbiter runs
resolution: null       # populated when resolved
experiment_id: null    # populated on escalation (for Phase 34)
---

# Debate: ADX threshold for trend regime

## quant-analyst Position
...claims with sources...

## risk-officer Position
...claims with sources...

## Arbiter Fact-Check
...structured report...
```

### Pattern 5: Arbiter Verification Logic (Two Paths)

**What:** The arbiter agent follows two verification paths depending on claim source kind.

**Code-claim path (`file:line`):**
```bash
# 1. Verify the file and line exist
ast-index outline src/finalayze/strategies/combiner.py   # confirms file structure
# 2. Grep for the exact excerpt at the stated line
grep -n "excerpt_text" src/finalayze/strategies/combiner.py
# 3. If line+excerpt match → VERIFIED. If line exists but excerpt differs → CONTRADICTED.
#    If file not found → UNTESTABLE (file may have been moved).
```

**Metric-claim path (`metric:value`):**
```bash
# 1. Look up the iteration in results/iterations/history.jsonl
grep "\"name\": \"iteration-name\"" results/iterations/history.jsonl
# 2. Extract the stated metric field
# 3. Compare against claimed value (allow ±0.01 tolerance for float)
# 4. Match → VERIFIED. Mismatch → CONTRADICTED. Iteration not found → UNTESTABLE.
```

### Anti-Patterns to Avoid
- **Free-text source strings:** `source: "somewhere in the combiner"` — not verifiable. Schema must enforce typed `ClaimSource`.
- **Orphan debates:** Debate files created but arbiter never run — leave `arbiter_report: null` in frontmatter. The planner should include a task to run the arbiter after creating the debate file.
- **Claims on future state:** A claim citing a line that doesn't exist yet (e.g., code not yet written) — arbiter returns UNTESTABLE, which is correct; debate escalates.
- **Putting debate schemas in a new file:** CONTEXT.md is locked — all schemas go in `core/schemas.py` (Layer 0), not in a new `core/debate.py`. This keeps import paths consistent.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| YAML frontmatter parsing | Custom regex parser | Python `yaml.safe_load()` after stripping `---` delimiters | Edge cases in multiline strings, nested keys |
| Code claim verification | Custom Python AST parser | `ast-index` CLI + `grep -n` | ast-index already installed, handles renames, hierarchy |
| Metric lookup | Custom JSON parser | `json.loads()` line-by-line on `history.jsonl` | history.jsonl is newline-delimited JSON; stdlib handles it cleanly |
| Debate ID generation | UUID | `{date}-{topic-slug}` naming from CONTEXT.md | Human-readable, chronologically sortable, audit-friendly |

**Key insight:** The arbiter does not need to understand strategy logic. It only needs to verify that (a) a file+line exists with the stated text, or (b) a metric value matches history. These are purely mechanical checks — no domain knowledge required.

---

## Common Pitfalls

### Pitfall 1: Line Number Drift
**What goes wrong:** Agent writes a `file:line` claim. Code is refactored. Arbiter runs later and finds different content at that line, marking the claim CONTRADICTED when the code actually supports the claim — it just moved.
**Why it happens:** Line numbers are unstable references.
**How to avoid:** Require `excerpt` field alongside `line`. Arbiter checks the excerpt text exists in the file (anywhere), and only uses line as a hint for locating it. If excerpt found at a different line → VERIFIED with a note. If excerpt not found at all → CONTRADICTED.
**Warning signs:** All code claims return CONTRADICTED after a refactor.

### Pitfall 2: Metric Tolerance
**What goes wrong:** Agent claims `profit_factor: 1.29`. history.jsonl stores `1.290001` (float precision). Arbiter marks CONTRADICTED.
**Why it happens:** Float representation differences between storage and display.
**How to avoid:** Use a tolerance of `abs(claimed - actual) <= 0.01` for float metric comparisons. Document this tolerance in the arbiter agent definition.
**Warning signs:** All metric claims return CONTRADICTED despite correct values.

### Pitfall 3: Schema Location Confusion
**What goes wrong:** Developer creates `src/finalayze/core/debate.py` for the new schemas instead of adding to `schemas.py`. Agents importing from both files, circular imports possible.
**Why it happens:** `schemas.py` is already large; developer creates a new file for cleanliness.
**How to avoid:** CONTEXT.md locks schema location to `src/finalayze/core/schemas.py`. Add a comment block `# ── Debate Protocol Schemas ──` to group the new models, matching the `# ── Iteration Tracking Schemas ──` pattern already in the file.
**Warning signs:** Import from `core.debate` anywhere in the codebase.

### Pitfall 4: Empty Claims List
**What goes wrong:** Existing agents produce `AgentOutput(claims=[])` — technically valid but defeats the purpose of DEBATE-01.
**Why it happens:** `claims` field defaults to empty list, agents forget to populate it.
**How to avoid:** Use `Field(min_length=1)` on the `claims` list. This is Pydantic v2 standard — raises `ValidationError` on empty list at construction time.
**Warning signs:** Agents constructing `AgentOutput` with no claims.

### Pitfall 5: Debate Files Without Arbiter Run
**What goes wrong:** A debate file is created with both agent positions, but no one triggers the arbiter. `arbiter_report: null` stays indefinitely.
**Why it happens:** The arbiter is a manual trigger — it's a sub-agent, not an automated pipeline.
**How to avoid:** Plan tasks must explicitly include an "arbiter run" step after each debate file creation. The planner should treat debate creation and arbiter execution as paired tasks in the same wave.
**Warning signs:** `.planning/debates/` files with `status: open` and `arbiter_report: null` older than 1 day.

---

## Code Examples

Verified patterns from existing codebase:

### Adding to schemas.py (matching existing section pattern)
```python
# Source: src/finalayze/core/schemas.py lines 221-297 (IterationTracking section pattern) [VERIFIED: file read]

# ── Debate Protocol Schemas ─────────────────────────────────────────────────

class FileLineSource(BaseModel):
    model_config = ConfigDict(frozen=True)
    kind: Literal["file"] = "file"
    path: str
    line: int
    excerpt: str

class MetricSource(BaseModel):
    model_config = ConfigDict(frozen=True)
    kind: Literal["metric"] = "metric"
    metric_name: str
    value: float
    iteration: str

ClaimSource = Annotated[
    FileLineSource | MetricSource,
    Field(discriminator="kind")
]

class Claim(BaseModel):
    model_config = ConfigDict(frozen=True)
    statement: str
    source: ClaimSource
    confidence: float

    @field_validator("confidence")
    @classmethod
    def confidence_must_be_probability(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            msg = f"confidence must be in [0.0, 1.0], got {v}"
            raise ValueError(msg)
        return v
```

### Existing StrEnum convention (from schemas.py and exceptions.py) [VERIFIED: file read]
```python
class ClaimVerdict(StrEnum):   # StrEnum not str,Enum — ruff UP042
    VERIFIED = "verified"
    CONTRADICTED = "contradicted"
    UNTESTABLE = "untestable"
```

### Existing agent definition structure (from evaluation-agent.md) [VERIFIED: file read]
```markdown
---
name: arbiter-agent
description: Use when two agent outputs conflict and you need a fact-check report...
model: claude-opus-4-6
---

You are the Arbiter Agent...
```

### Debate file YAML frontmatter (matching STATE.md pattern) [VERIFIED: file read]
```yaml
---
debate_id: 2026-04-07-example-topic
topic: "Short description of the contested claim"
status: open
created: "2026-04-07"
agents: [quant-analyst, risk-officer]
arbiter_report: null
resolution: null
experiment_id: null
---
```

### Reading history.jsonl for metric verification [ASSUMED - standard pattern for JSONL]
```python
import json
from pathlib import Path

def lookup_metric(iteration_name: str, metric_name: str) -> float | None:
    history = Path("results/iterations/history.jsonl")
    for line in history.read_text().splitlines():
        record = json.loads(line)
        if record.get("name") == iteration_name:
            return record.get("metrics", {}).get(metric_name)
    return None
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Agent produces freeform markdown recommendation | Agent produces `AgentOutput` with typed `claims` list | Phase 33 | Claims become verifiable, not just readable |
| Conflict detected by human reading two reports | Arbiter agent mechanically checks both outputs | Phase 33 | Reproducible, auditable conflict detection |
| Debate outcome stored in chat history only | Debate persisted in `.planning/debates/` | Phase 33 | Audit trail consumable by Phase 34 experiment registry |

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Pydantic v2 `Field(discriminator="kind")` with `Annotated` union works for `ClaimSource` type alias | Architecture Patterns / Standard Stack | Minor — alternative is a `RootModel`; same end result |
| A2 | `Field(min_length=1)` on a `list[Claim]` field raises `ValidationError` in Pydantic v2 | Architecture Patterns | Low — if wrong, use `@model_validator` instead |
| A3 | `history.jsonl` uses `name` field as the iteration identifier | Code Examples | Medium — wrong field name means metric lookup silently returns `None`; arbiter returns UNTESTABLE instead of VERIFIED |
| A4 | `ast-index outline <file>` confirms a file exists and is indexed | Architecture Patterns / arbiter verification path | Low — arbiter can fall back to `pathlib.Path.exists()` for file existence check |
| A5 | Existing `.planning/` YAML frontmatter files do not use a strict schema enforcer (pure convention) | Architecture Patterns | None — debate files follow the same convention, no additional tooling needed |

---

## Open Questions

1. **What triggers the arbiter agent?**
   - What we know: The arbiter is a Claude Code sub-agent invoked manually or by another agent.
   - What's unclear: Is the trigger a human running `/gsd:arbiter <debate-file>`, or does the planner include an explicit "run arbiter" task in each plan that produces a debate?
   - Recommendation: Include an explicit "arbiter task" in each plan that creates a debate file. Avoids orphaned debates with `status: open`.

2. **How are existing agents updated to emit `AgentOutput`?**
   - What we know: 21 existing agents produce freeform markdown. Retrofitting all of them is large scope.
   - What's unclear: Does Phase 33 require all agents to adopt the schema, or just the agents that participate in debates?
   - Recommendation: Phase 33 scope should be the schema + arbiter + debate persistence infrastructure. Agent adoption is per-debate-scenario, not a bulk migration. Planner should reflect this.

3. **What is "both agents maintain their position" for auto-escalation?**
   - What we know: CONTEXT.md says escalation happens when arbiter marks ≥1 claim as "contradicted" AND both agents maintain position.
   - What's unclear: There is no automated mechanism for agents to respond to the arbiter report. Who determines "maintained position"?
   - Recommendation: In Phase 33, implement the schema and arbiter. Escalation rule becomes: if `arbiter_report.has_contradictions` is True at debate close time, set `status: escalated` and assign `experiment_id`. The "both agents maintain position" check is deferred to a human or Phase 34 orchestration.

---

## Environment Availability

Step 2.6: No new external dependencies. All required tools already confirmed available.

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| ast-index | Arbiter agent code-claim verification | ✓ | 3.27.0 | `grep -n` (less precise) |
| Pydantic v2 | Claim/AgentOutput schemas | ✓ | Already in pyproject.toml | None needed |
| PyYAML | Debate file frontmatter parsing | ✓ | Already in stack | `tomllib` or manual split |
| Python 3.12 | All new code | ✓ | 3.12 | — |

**Missing dependencies with no fallback:** None.

---

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (already configured) |
| Config file | `pyproject.toml` |
| Quick run command | `uv run pytest tests/unit/core/test_debate_schemas.py -x` |
| Full suite command | `uv run pytest tests/unit/core/ -v` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| DEBATE-01 | `Claim` with no source raises `ValueError` | unit | `uv run pytest tests/unit/core/test_debate_schemas.py::test_claim_requires_source -x` | ❌ Wave 0 |
| DEBATE-01 | `FileLineSource` validates path/line/excerpt fields | unit | `uv run pytest tests/unit/core/test_debate_schemas.py::test_file_line_source_fields -x` | ❌ Wave 0 |
| DEBATE-01 | `MetricSource` validates metric_name/value/iteration fields | unit | `uv run pytest tests/unit/core/test_debate_schemas.py::test_metric_source_fields -x` | ❌ Wave 0 |
| DEBATE-01 | `AgentOutput` with empty claims list raises `ValidationError` | unit | `uv run pytest tests/unit/core/test_debate_schemas.py::test_agent_output_requires_claims -x` | ❌ Wave 0 |
| DEBATE-01 | `confidence` outside [0.0, 1.0] raises `ValueError` | unit | `uv run pytest tests/unit/core/test_debate_schemas.py::test_claim_confidence_bounds -x` | ❌ Wave 0 |
| DEBATE-02 | `FactCheckReport.has_contradictions` returns True when any verdict is CONTRADICTED | unit | `uv run pytest tests/unit/core/test_debate_schemas.py::test_fact_check_report_has_contradictions -x` | ❌ Wave 0 |
| DEBATE-02 | `FactCheckReport` serializes to structured markdown (verified/contradicted/untestable sections) | unit | `uv run pytest tests/unit/core/test_debate_schemas.py::test_fact_check_report_markdown -x` | ❌ Wave 0 |
| DEBATE-03 | Debate file with valid YAML frontmatter can be created and read back | unit | `uv run pytest tests/unit/core/test_debate_schemas.py::test_debate_file_roundtrip -x` | ❌ Wave 0 |
| DEBATE-03 | `DebateState` escalation sets `status=escalated` and populates `experiment_id` | unit | `uv run pytest tests/unit/core/test_debate_schemas.py::test_debate_escalation -x` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `uv run pytest tests/unit/core/test_debate_schemas.py -x`
- **Per wave merge:** `uv run pytest tests/unit/core/ -v`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/unit/core/test_debate_schemas.py` — covers all DEBATE-01, DEBATE-02, DEBATE-03 schema unit tests

---

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | n/a — internal tool, no auth surface |
| V3 Session Management | no | n/a — stateless schemas and files |
| V4 Access Control | no | n/a — `.planning/debates/` is local filesystem |
| V5 Input Validation | yes | Pydantic v2 strict field validators — claim statement and source fields |
| V6 Cryptography | no | n/a |

### Known Threat Patterns for This Stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Prompt injection via `statement` field | Tampering | Pydantic `str` with no eval — the arbiter reads the string, never executes it |
| Path traversal in `FileLineSource.path` | Tampering | Arbiter should validate path is within project root before running `ast-index` or `grep` |
| Malformed YAML frontmatter in debate files | Tampering | Use `yaml.safe_load()` (not `yaml.load()`); catch `yaml.YAMLError` |

---

## Sources

### Primary (HIGH confidence)
- `src/finalayze/core/schemas.py` — verified existing model patterns, section structure, validator conventions
- `src/finalayze/core/exceptions.py` — verified FinalayzeError hierarchy and naming conventions
- `src/finalayze/core/CLAUDE.md` — verified Layer 0 constraints (zero project imports)
- `.claude/agents/evaluation-agent.md` — verified agent definition format and structure
- `.planning/phases/32-critical-sandbox-fixes/32-CONTEXT.md` — verified YAML frontmatter pattern in planning files
- `.planning/ROADMAP.md` — verified phase 33 success criteria and DEBATE requirement IDs
- `CLAUDE.md` — verified coding conventions, ast-index availability, Pydantic v2 requirement

### Secondary (MEDIUM confidence)
- `results/iterations/history.jsonl` — inferred JSONL format for metric lookup (file exists per MEMORY.md)

### Tertiary (LOW confidence)
- Pydantic v2 `Field(discriminator="kind")` with `Annotated` type alias — standard documented pattern, not verified against installed version in this session

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all libraries already in project, no new dependencies
- Architecture: HIGH — patterns verified against existing codebase (schemas.py, agent definitions, planning files)
- Pitfalls: HIGH — derived from direct codebase inspection and locked CONTEXT.md decisions
- Test map: MEDIUM — test file names are designed, not yet verified to compile

**Research date:** 2026-04-07
**Valid until:** 2026-05-07 (stable domain — pure internal tooling, no external APIs)
