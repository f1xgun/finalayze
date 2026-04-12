# Phase 36: Conflict Detection Foundation - Research

**Researched:** 2026-04-12
**Domain:** Structured agent output emission, LLM structured parsing, deterministic conflict detection
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Conflict Detection Algorithm**
- Direction contradictions detected via keyword match on `recommendation` field (BUY vs SELL) — fast, deterministic
- Metric value contradictions detected via `MetricSource.value` comparison: >15% relative divergence = contradiction
- Statement contradictions detected via `difflib.SequenceMatcher.ratio() > 0.85` + divergent conclusions
- Severity scoring: 3 levels — CRITICAL (direction), HIGH (metric >30% divergence), LOW (statement similarity)

**Agent Output Integration**
- All domain/analysis agents must emit structured `AgentOutput`: quant-analyst, risk-officer, ml-engineer, strategies-agent, portfolio-strategist, systems-architect
- `parse_structured()` added to `_CachingLLMClient` base class — all 3 clients (Anthropic, OpenAI, OpenRouter) inherit it
- Agent `.md` definitions get `## Output Format` section requiring `AgentOutput` JSON block
- Fallback: wrap free-text in single Claim with `MetricSource(metric_name="unstructured", value=0.0)` and confidence=0.0

**Debouncing & Deduplication**
- Dedup key: SHA-256 hash of `(sorted agent_names, sorted claim topics, conflict_type)` — deterministic, collision-proof
- Dedup window: per-session (cleared on orchestrator restart) — aligns with weekly/daily agent cycles
- Minimum confidence delta: 0.15 — agents must disagree by >15% confidence to trigger escalation
- Skip consecutive cycle requirement for v8.0 — orchestrator runs on-demand (weekly), not per-cycle

### Claude's Discretion
- Internal data structures for ConflictDetector (e.g., topic extraction method)
- Test fixture design for synthetic AgentOutput objects

### Deferred Ideas (OUT OF SCOPE)
- `snapshot_sha` on `FileLineSource` — deferred to Phase 37 (orchestrator needs it for arbiter safety)
- Consecutive cycle requirement for debouncing — only needed if orchestrator runs per-cycle (v8.x)
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| AGOUT-01 | Domain agents emit `AgentOutput` with structured `Claim` objects and mandatory source references | Existing `AgentOutput`/`Claim` schema at `core/schemas.py:569-596`; 6 agent `.md` files need `## Output Format` section |
| AGOUT-02 | `AnthropicClient.parse_structured()` wraps `client.messages.parse()` with Pydantic model schema derivation | `messages.parse()` verified in Anthropic SDK 0.83.0; `output_format` param accepts Pydantic type; `ParsedMessage.parsed_output` returns typed instance |
| CONF-01 | `ConflictDetector` compares `list[AgentOutput]` and returns `ConflictReport` — no LLM in hot path | difflib.SequenceMatcher benchmarked at 0.08ms/call; keyword matching is O(1); stays well under 50ms per pair |
| CONF-02 | `ConflictReport` schema added to `core/schemas.py` with conflict type, severity, and involved claims | Layer 0 placement confirmed correct; pattern follows existing `FactCheckReport` model |
| CONF-03 | Debouncing: topic-level deduplication and minimum confidence delta (>0.15) before escalation | SHA-256 dedup key verified at 0.001ms/call; per-session `set[str]` is sufficient for weekly cadence |
| CONF-04 | Conflict severity scoring ranks contradictions by impact | 3-level severity (CRITICAL/HIGH/LOW) maps directly to the 3 detection rules |
</phase_requirements>

---

## Summary

Phase 36 builds the conflict detection pipeline on top of the v7.0 debate schema foundation already shipped in Phase 33. All foundational types (`AgentOutput`, `Claim`, `ClaimSource`, `FileLineSource`, `MetricSource`) exist and are tested at `core/schemas.py`. No schema migration is required — only additions.

The two implementation surfaces are: (1) structured output via `parse_structured()` on the LLM client base class, and (2) the `ConflictDetector` at Layer 5 in `orchestration/conflict_detector.py`. Each provider SDK has a native structured-output method verified available at runtime: Anthropic uses `messages.parse(output_format=PydanticModel)` returning `ParsedMessage.parsed_output`; OpenAI and its-compatible clients use `beta.chat.completions.parse(response_format=PydanticModel)` returning `ParsedChatCompletion.choices[0].message.parsed`. OpenRouter is OpenAI-compatible, so it uses the same path.

The six domain agent `.md` files (quant-analyst, risk-officer, ml-engineer, strategies-agent, portfolio-strategist, systems-architect) require a new `## Output Format` section instructing the agent to emit its final recommendation as a JSON `AgentOutput` block. This is a documentation-only change — no Python code change to the agent definitions.

**Primary recommendation:** Implement in two vertical slices — (1) schema + `parse_structured()` + agent `.md` updates as Wave 1, (2) `ConflictDetector` + `ConflictReport` schema + dedup logic as Wave 2. No dependency on external services.

---

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `difflib` (stdlib) | stdlib | Statement-level similarity scoring via `SequenceMatcher` | Zero dependencies, deterministic, benchmarked at 0.08ms per pair |
| `hashlib` (stdlib) | stdlib | SHA-256 dedup key generation | Already used in `_CachingLLMClient`; consistent pattern |
| `anthropic` | 0.83.0 | `messages.parse()` for structured Anthropic output | `ParsedMessage.parsed_output` returns typed Pydantic instance |
| `openai` | 2.21.0 | `beta.chat.completions.parse()` for OpenAI/OpenRouter structured output | `ParsedChatCompletion` with `response_format=PydanticModel` |
| `pydantic` v2 | existing | `ConflictReport` schema and `AgentOutput` model schema derivation | Already the project standard; `model_json_schema()` used for prompt injection |

[VERIFIED: runtime inspection of installed SDKs]

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `structlog` | existing | Structured conflict detection logs | Every conflict detection event |
| `datetime` (stdlib) | stdlib | `ConflictReport.detected_at` timestamp | Standard UTC timestamps |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `difflib.SequenceMatcher` | `rapidfuzz` / `fuzzywuzzy` | External deps, overkill at 0.08ms/call; stdlib is sufficient |
| Per-session `set[str]` dedup store | Redis-backed dedup | Redis adds infra complexity for weekly cadence; in-memory is fine |
| Anthropic `messages.parse()` | JSON mode + manual `model_validate_json()` | SDK parse guarantees type safety with `ParsedMessage.parsed_output` — prefer native |

---

## Architecture Patterns

### File Locations

```
src/finalayze/
├── core/schemas.py                    # Add ConflictReport, ConflictType, ConflictSeverity (Layer 0)
├── orchestration/conflict_detector.py # New file: ConflictDetector (Layer 5)
├── analysis/llm_client.py             # Add parse_structured() to _CachingLLMClient base + each provider
└── .claude/agents/
    ├── quant-analyst.md               # Add ## Output Format section
    ├── risk-officer.md                # Add ## Output Format section
    ├── ml-engineer.md                 # Add ## Output Format section
    ├── strategies-agent.md            # Add ## Output Format section
    ├── portfolio-strategist.md        # Add ## Output Format section
    └── systems-architect.md           # Add ## Output Format section
```

### Pattern 1: ConflictReport Schema in core/schemas.py

**What:** New Pydantic models at Layer 0 — `ConflictType` (StrEnum), `ConflictSeverity` (StrEnum), `ConflictReport` (frozen BaseModel).

**When to use:** Layer 0 placement required because `ConflictDetector` at Layer 5 imports it; all layers can import Layer 0.

```python
# Source: Existing patterns in core/schemas.py (ClaimVerdict, FactCheckReport)
from __future__ import annotations

class ConflictType(StrEnum):
    DIRECTION = "direction"      # BUY vs SELL recommendation
    METRIC = "metric"            # >15% relative divergence in MetricSource values
    STATEMENT = "statement"      # SequenceMatcher ratio > 0.85 + divergent conclusions

class ConflictSeverity(StrEnum):
    CRITICAL = "critical"        # DIRECTION conflicts
    HIGH = "high"                # METRIC conflicts with >30% divergence
    LOW = "low"                  # STATEMENT similarity conflicts

class ConflictReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    conflict_id: str             # SHA-256 dedup key (hex digest)
    conflict_type: ConflictType
    severity: ConflictSeverity
    involved_claims: list[Claim]
    agent_names: list[str]
    detected_at: datetime
    confidence_delta: float | None = None  # Only for METRIC/STATEMENT types
```

### Pattern 2: parse_structured() on _CachingLLMClient

**What:** Abstract method on `_CachingLLMClient` with provider-specific implementations in `AnthropicClient`, `OpenAIClient`, `OpenRouterClient`.

**When to use:** Any agent invocation that must return typed `AgentOutput` — bypasses the raw `complete()` string path.

```python
# Source: Anthropic SDK 0.83.0 runtime inspection
# AnthropicClient implementation
async def parse_structured(
    self,
    prompt: str,
    system: str,
    response_model: type[T],
    *,
    max_tokens: int | None = None,
) -> T:
    message = await self._client.messages.parse(
        model=self._model,
        max_tokens=max_tokens or 1024,
        system=system,
        messages=[{"role": "user", "content": prompt}],
        output_format=response_model,   # SDK derives JSON schema from Pydantic type
    )
    return message.parsed_output  # type: T guaranteed by SDK

# OpenAI/OpenRouter implementation
async def parse_structured(
    self,
    prompt: str,
    system: str,
    response_model: type[T],
    *,
    max_tokens: int | None = None,
) -> T:
    completion = await self._client.beta.chat.completions.parse(
        model=self._model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_tokens or 1024,
        response_format=response_model,
    )
    parsed = completion.choices[0].message.parsed
    if parsed is None:
        msg = "Structured output returned None — model refused"
        raise LLMError(msg)
    return parsed
```

**Fallback for Groq/OpenRouter models that don't support structured output:**
If `parse_structured()` raises (e.g., model doesn't support `response_format`), catch the error and fall back to `complete()` with JSON mode, then `response_model.model_validate_json()`. This keeps the method callable on all providers.

### Pattern 3: ConflictDetector

**What:** Stateful (per-session dedup store) detector at Layer 5. Accepts `list[AgentOutput]`, returns `list[ConflictReport]`.

**When to use:** After collecting domain agent outputs during a debate or orchestrator cycle.

```python
# Source: CONTEXT.md locked decisions + difflib stdlib
from __future__ import annotations

import difflib
import hashlib
from datetime import UTC, datetime
from itertools import combinations

from finalayze.core.schemas import (
    AgentOutput, Claim, ConflictReport, ConflictSeverity, ConflictType, MetricSource
)

_DIRECTION_KEYWORDS_BUY = frozenset({"BUY", "LONG", "ENABLE", "INCREASE"})
_DIRECTION_KEYWORDS_SELL = frozenset({"SELL", "SHORT", "DISABLE", "DECREASE"})
_MIN_CONFIDENCE_DELTA = 0.15
_METRIC_CONTRADICTION_THRESHOLD = 0.15   # >15% relative divergence
_METRIC_HIGH_SEVERITY_THRESHOLD = 0.30   # >30% = HIGH, else LOW if METRIC type
_STATEMENT_SIMILARITY_THRESHOLD = 0.85   # SequenceMatcher ratio

class ConflictDetector:
    """Deterministic conflict detector for pairs of AgentOutput objects."""

    def __init__(self) -> None:
        self._seen_conflicts: set[str] = set()  # Per-session dedup store

    def detect(self, outputs: list[AgentOutput]) -> list[ConflictReport]:
        """Compare all pairs and return deduplicated ConflictReports."""
        reports: list[ConflictReport] = []
        for a, b in combinations(outputs, 2):
            reports.extend(self._compare_pair(a, b))
        return reports

    def _dedup_key(
        self, agents: list[str], topics: list[str], conflict_type: ConflictType
    ) -> str:
        payload = str(sorted(agents)) + str(sorted(topics)) + str(conflict_type)
        return hashlib.sha256(payload.encode()).hexdigest()

    def _compare_pair(
        self, a: AgentOutput, b: AgentOutput
    ) -> list[ConflictReport]:
        conflicts: list[ConflictReport] = []
        # 1. Direction contradiction
        dir_conflict = self._check_direction(a, b)
        if dir_conflict:
            conflicts.append(dir_conflict)
        # 2. Metric contradictions (per claim pair)
        conflicts.extend(self._check_metrics(a, b))
        # 3. Statement contradictions
        conflicts.extend(self._check_statements(a, b))
        return conflicts
```

### Pattern 4: Topic Extraction (Claude's Discretion)

**Recommendation:** Extract topics from claim statements using simple word-set intersection. A "topic" is the first capitalized word sequence (ticker, strategy name) found in the statement. This avoids NLP dependencies while being sufficient for weekly agent debates.

```python
import re

def _extract_topic(statement: str) -> str:
    """Extract primary topic from claim statement (first ALL-CAPS word or capitalized proper noun)."""
    # Matches tickers like SBER, GAZP, or strategy names like dual_momentum
    match = re.search(r'\b([A-Z]{2,}|[a-z_]+_[a-z_]+)\b', statement)
    return match.group(1).lower() if match else statement[:20].lower()
```

### Anti-Patterns to Avoid

- **Putting `ConflictDetector` at Layer 3 or Layer 4:** It imports from Layer 0 (`ConflictReport`, `AgentOutput`) — placing it at Layer 5 is correct since it's orchestration-level, not strategy/risk logic.
- **Making `parse_structured()` bypass the LRU cache:** The structured output path should NOT cache (responses include structured data tied to specific Pydantic schemas — cache invalidation would be complex). Cache only the `complete()` text path.
- **Raising on `confidence_delta < 0.15` instead of returning empty list:** The detector should silently drop low-delta conflicts. Raising would be unexpected behavior for callers.
- **Mutable dedup store shared across tests:** Each test must use a fresh `ConflictDetector()` instance to avoid state leak.
- **OpenRouter structured output assumption:** OpenRouter passes `response_format` to the underlying model but not all models support it. Always wrap in try/except and fall back to JSON mode.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Pydantic schema for `messages.parse()` | Custom JSON schema extractor | `output_format=PydanticModel` in Anthropic SDK | SDK derives schema from Pydantic type automatically |
| OpenAI structured output | Manual JSON parsing loop | `beta.chat.completions.parse(response_format=Model)` | SDK handles schema injection and type coercion |
| String similarity | Custom edit distance | `difflib.SequenceMatcher` | Stdlib, deterministic, 0.08ms per call |
| SHA-256 hashing | Custom hash | `hashlib.sha256()` | Already pattern in `_CachingLLMClient._cache_key()` |

**Key insight:** Both Anthropic and OpenAI SDKs provide native structured output that derives the JSON schema from the Pydantic model — there is no need to call `model_json_schema()` manually or inject a system prompt with schema text.

---

## Common Pitfalls

### Pitfall 1: Anthropic `messages.parse()` vs `messages.create()`

**What goes wrong:** Calling `messages.create()` with `json_mode=True` and then trying to parse `AgentOutput` from the text response fails when the model adds commentary around the JSON block.

**Why it happens:** `messages.create()` returns free text even in JSON mode; `messages.parse()` enforces structured output at the protocol level.

**How to avoid:** Always use `messages.parse(output_format=AgentOutput)` in `parse_structured()`. The return type is `ParsedMessage[AgentOutput]` and `parsed_output` is guaranteed to be an `AgentOutput` instance.

**Warning signs:** `json.JSONDecodeError` in production logs from `complete()` calls when `json_mode=True`.

### Pitfall 2: OpenRouter models don't universally support `response_format`

**What goes wrong:** `beta.chat.completions.parse()` raises `openai.BadRequestError` ("model does not support response_format") for some OpenRouter-proxied models.

**Why it happens:** OpenRouter forwards the request to the underlying model provider; models like Claude-2 or older GPT-3.5 don't support the `response_format` structured output feature.

**How to avoid:** Wrap the `beta.chat.completions.parse()` call in a try/except catching `openai.BadRequestError`. Fall back to `complete()` with JSON mode + `model_validate_json()`. Log a warning with the model name.

**Warning signs:** `BadRequestError` in OpenRouterClient logs.

### Pitfall 3: Dedup key collision across different sessions

**What goes wrong:** The per-session dedup store (`set[str]`) is never cleared if the `ConflictDetector` instance is long-lived across multiple orchestrator runs.

**Why it happens:** CONTEXT.md specifies "per-session" — if the same `ConflictDetector` is injected as a singleton and reused across weekly runs, old dedup keys suppress new (legitimate) conflicts.

**How to avoid:** Create a fresh `ConflictDetector()` per orchestrator invocation, or add a `reset()` method that clears `_seen_conflicts`. Phase 37 (AgentOrchestrator) must create it fresh per run.

**Warning signs:** ConflictDetector returning zero reports on second run despite new agent outputs.

### Pitfall 4: Metric contradiction symmetry

**What goes wrong:** Agent A claims `profit_factor=1.29`, Agent B claims `profit_factor=1.05`. Relative divergence from A's perspective: `|1.29-1.05|/1.29 = 18.6%` (contradiction). From B's perspective: `|1.05-1.29|/1.05 = 22.9%` (also contradiction). Both exceed 15% — but they reference the same metric pair.

**How to avoid:** When building claim pairs, match `MetricSource.metric_name` AND `MetricSource.iteration` — only flag as contradiction if both refer to the same iteration and metric but with divergent values. Use `max(va, vb)` as denominator to make the relative divergence calculation symmetric.

**Warning signs:** Duplicate `ConflictReport` objects for the same metric pair but different "perspectives."

### Pitfall 5: `AgentOutput.claims` minimum 1 enforcement on fallback

**What goes wrong:** The fallback path (wrap free-text in single `Claim` with `confidence=0.0`) must pass `AgentOutput.claims` validation which requires `min_length=1` [VERIFIED: `core/schemas.py:595`]. An empty list raises `ValidationError`.

**How to avoid:** The fallback must always construct exactly one `Claim` with a `MetricSource(metric_name="unstructured", value=0.0, iteration="fallback")`.

---

## Code Examples

### ConflictReport Schema (verified pattern from core/schemas.py)

```python
# Source: core/schemas.py existing patterns (FactCheckReport, DebateState)
from __future__ import annotations

from datetime import datetime
from enum import auto

from pydantic import BaseModel, ConfigDict, Field

class ConflictType(StrEnum):
    DIRECTION = auto()
    METRIC = auto()
    STATEMENT = auto()

class ConflictSeverity(StrEnum):
    CRITICAL = auto()
    HIGH = auto()
    LOW = auto()

class ConflictReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    conflict_id: str
    conflict_type: ConflictType
    severity: ConflictSeverity
    involved_claims: list[Claim] = Field(min_length=2)
    agent_names: list[str] = Field(min_length=2)
    detected_at: datetime
    confidence_delta: float | None = None
```

### Anthropic parse_structured() (verified against SDK 0.83.0)

```python
# Source: runtime inspection of anthropic 0.83.0
from typing import TypeVar
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

async def parse_structured(
    self,
    prompt: str,
    system: str,
    response_model: type[T],
    *,
    max_tokens: int | None = None,
) -> T:
    try:
        message = await self._client.messages.parse(
            model=self._model,
            max_tokens=max_tokens or 1024,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            output_format=response_model,
        )
    except anthropic.RateLimitError as exc:
        msg = f"Anthropic rate limit: {exc}"
        raise LLMRateLimitError(msg) from exc
    except anthropic.APIError as exc:
        msg = f"Anthropic API error: {exc}"
        raise LLMError(msg) from exc
    return message.parsed_output
```

### Agent .md Output Format section

```markdown
## Output Format

After your analysis, emit a final `AgentOutput` JSON block:

```json
{
  "agent_name": "quant-analyst",
  "recommendation": "ENABLE dual_momentum on ru_blue_chips with weight 0.25",
  "claims": [
    {
      "statement": "dual_momentum shows PF=1.29 in 2022-2025 us_tech backtest",
      "source": {
        "kind": "metric",
        "metric_name": "profit_factor",
        "value": 1.29,
        "iteration": "2026-04-05-adx-routing"
      },
      "confidence": 0.85
    }
  ],
  "timestamp": "2026-04-12T00:00:00Z"
}
```

Each claim MUST have a source (file:line or metric). No unsourced assertions.
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `messages.create()` + JSON mode | `messages.parse(output_format=Model)` | Anthropic SDK 0.42+ | SDK guarantees Pydantic compliance — no manual parse |
| `chat.completions.create()` + JSON | `beta.chat.completions.parse(response_format=Model)` | OpenAI SDK 1.40+ | Typed `ParsedChatCompletion` return |
| Ad-hoc agent recommendations | `AgentOutput` with `Claim` + `ClaimSource` | Phase 33 (v7.0) | Verifiable, structured debate protocol |

**Deprecated/outdated:**
- `json_mode=True` + manual `json.loads()`: superseded by native structured output in both Anthropic and OpenAI SDKs. Still used as fallback for models that don't support structured output.

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest (existing) |
| Config file | `pyproject.toml` `[tool.pytest.ini_options]` |
| Quick run command | `uv run pytest tests/unit/core/test_debate_schemas.py tests/unit/test_llm_client.py tests/unit/core/test_conflict_detector.py -x` |
| Full suite command | `uv run pytest tests/unit/ -x --ignore=tests/integration` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| AGOUT-01 | `AgentOutput` with claims emitted by agents | unit (schema) | `uv run pytest tests/unit/core/test_debate_schemas.py -x` | Yes (existing) |
| AGOUT-02 | `parse_structured()` returns typed Pydantic instance | unit (mock) | `uv run pytest tests/unit/test_llm_client.py -x -k parse_structured` | No — Wave 0 |
| CONF-01 | `ConflictDetector.detect()` returns `list[ConflictReport]` | unit | `uv run pytest tests/unit/core/test_conflict_detector.py -x` | No — Wave 0 |
| CONF-02 | `ConflictReport` schema validates correctly | unit (schema) | `uv run pytest tests/unit/core/test_conflict_detector.py -x -k schema` | No — Wave 0 |
| CONF-03 | Dedup key suppresses repeat conflicts | unit | `uv run pytest tests/unit/core/test_conflict_detector.py -x -k dedup` | No — Wave 0 |
| CONF-04 | Severity scoring maps correctly to conflict types | unit | `uv run pytest tests/unit/core/test_conflict_detector.py -x -k severity` | No — Wave 0 |

### Sampling Rate
- **Per task commit:** `uv run pytest tests/unit/core/test_conflict_detector.py tests/unit/test_llm_client.py -x`
- **Per wave merge:** `uv run pytest tests/unit/ -x --ignore=tests/integration`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps

- [ ] `tests/unit/core/test_conflict_detector.py` — covers CONF-01, CONF-02, CONF-03, CONF-04
- [ ] `tests/unit/test_llm_client.py` — extend with `test_parse_structured_*` tests covering AGOUT-02

---

## Project Constraints (from CLAUDE.md)

| Directive | Impact on Phase 36 |
|-----------|-------------------|
| `from __future__ import annotations` in every file | Must be first line in `conflict_detector.py`, any modified files |
| `StrEnum` not `str, Enum` (ruff UP042) | `ConflictType`, `ConflictSeverity` must use `StrEnum` |
| Exception names must end with `Error` (ruff N818) | No new exception classes needed; reuse `LLMError` |
| Pydantic v2, `ConfigDict(frozen=True)` for all schemas | `ConflictReport` must be frozen |
| TDD: write failing test FIRST, then implement | Test files created in Wave 0, before implementation |
| `ruff check .` and `mypy src/` must pass | `parse_structured()` needs proper TypeVar/Generic annotations |
| Async-first, `httpx` for HTTP | `parse_structured()` must be `async def` |
| Layer 0 constraint on `core/schemas.py` | `ConflictReport` in `core/schemas.py` has zero project imports — only Pydantic/stdlib |
| `ConflictDetector` in `orchestration/` (Layer 5) | Imports from Layer 0 only — no Layer 3/4 imports |

---

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| `anthropic` SDK | `parse_structured()` (Anthropic) | Yes | 0.83.0 | — |
| `openai` SDK | `parse_structured()` (OpenAI/OpenRouter) | Yes | 2.21.0 | — |
| `difflib` (stdlib) | `ConflictDetector` statement comparison | Yes | stdlib | — |
| `hashlib` (stdlib) | Dedup key generation | Yes | stdlib | — |

No missing dependencies. All tools available.

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `OpenRouterClient` can use `beta.chat.completions.parse()` since it wraps `openai.AsyncOpenAI` | Architecture Patterns | OpenRouter may not forward `response_format` to all models — fallback to JSON mode required |
| A2 | Groq-compatible models support `beta.chat.completions.parse()` | Architecture Patterns | If Groq doesn't support it, `GroqClient.parse_structured()` needs only the fallback path |

**Claims A1 and A2 are flagged ASSUMED** — they were not verified by invoking the API at runtime. The fallback (JSON mode + `model_validate_json()`) handles both failure cases, so the risk impact is LOW.

---

## Open Questions

1. **strategies-agent structured output**
   - What we know: `strategies-agent.md` exists but was not fully read — it's an implementer agent, not a domain expert
   - What's unclear: Does CONTEXT.md require `strategies-agent` specifically, or is it listed as one of the 6 domain/analysis agents?
   - Recommendation: Check CONTEXT.md — it lists `strategies-agent` explicitly. Add `## Output Format` section.

2. **Groq client parse_structured() fallback path**
   - What we know: `GroqClient` exists in `llm_client.py` (line 290) using OpenAI-compatible SDK
   - What's unclear: Whether `beta.chat.completions.parse()` works with Groq's API
   - Recommendation: Use same OpenAI-compatible implementation pattern; document that fallback to JSON mode activates if `BadRequestError` is raised

---

## Sources

### Primary (HIGH confidence)
- `src/finalayze/core/schemas.py:530-680` — Verified: existing `AgentOutput`, `Claim`, `ClaimSource`, `FileLineSource`, `MetricSource` schemas
- `src/finalayze/analysis/llm_client.py:106-330` — Verified: `_CachingLLMClient` base, `AnthropicClient._complete_once()`, `OpenRouterClient._complete_once()`, `OpenAIClient._complete_once()`
- `anthropic` SDK 0.83.0 — Verified by runtime inspection: `messages.parse(output_format=Model)` returns `ParsedMessage[T]` with `.parsed_output: T`
- `openai` SDK 2.21.0 — Verified by runtime inspection: `beta.chat.completions.parse(response_format=Model)` returns `ParsedChatCompletion[T]`
- `tests/unit/core/test_debate_schemas.py` — Verified: 19 tests passing, existing test infrastructure covers existing schemas
- difflib benchmark — Verified: 0.08ms per `SequenceMatcher.ratio()` call (1000 iterations)
- SHA-256 benchmark — Verified: 0.001ms per `hashlib.sha256()` call

### Secondary (MEDIUM confidence)
- `.planning/phases/36-conflict-detection-foundation/36-CONTEXT.md` — Authoritative user decisions for algorithm choices, severity levels, and dedup strategy
- `.claude/agents/quant-analyst.md`, `risk-officer.md`, `ml-engineer.md`, `portfolio-strategist.md`, `systems-architect.md` — Verified: all 5 agent `.md` files lack `## Output Format` section

### Tertiary (LOW confidence)
- None

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all library versions verified at runtime
- Architecture: HIGH — layer placement follows existing project conventions; all patterns verified against existing code
- Pitfalls: HIGH — derived from verified code behavior (SDK versions, schema constraints) with one ASSUMED item around OpenRouter compatibility

**Research date:** 2026-04-12
**Valid until:** 2026-05-12 (SDK versions stable; stdlib difflib/hashlib permanent)
