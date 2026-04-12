# Stack Research

**Domain:** Agent orchestration, conflict detection, auto-apply for AI trading system (v8.0)
**Researched:** 2026-04-12
**Confidence:** HIGH (anthropic SDK verified against live docs; all other libs already installed and confirmed)

---

## Premise: What Already Exists (Do Not Re-Research)

v7.0 shipped these capabilities. They are done. v8.0 builds on them without rewrites.

| Already Exists | Location |
|----------------|----------|
| `AgentOutput`, `Claim`, `ClaimSource`, `FactCheckReport`, `DebateState`, `ExperimentState` schemas | `src/finalayze/core/schemas.py` |
| `DebateManager` — CRUD for `.planning/debates/*.md` (YAML frontmatter + markdown body) | `src/finalayze/core/debate_manager.py` |
| `ExperimentManager` — CRUD + automated verdict (ACCEPT/REJECT/INCONCLUSIVE) | `src/finalayze/core/experiment_manager.py` |
| `arbiter-agent.md` — fact-checking sub-agent with `ast-index` + `history.jsonl` verification | `.claude/agents/arbiter-agent.md` |
| `LLMClient` ABC + `AnthropicClient` / `OpenRouterClient` / `OpenAIClient` | `src/finalayze/analysis/llm_client.py` |
| `EventBus` — Redis Streams pub/sub (`XADD`/`XREAD`/consumer groups) | `src/finalayze/core/events.py` |
| Backtest engine with walk-forward validation | `src/finalayze/backtest/` |
| Streamlit Experiment Lab UI | `src/finalayze/dashboard/pages/experiments_list.py` |
| `anthropic>=0.42.0` in pyproject.toml — installed version confirmed 0.83.0 | `pyproject.toml` |

---

## What v8.0 Needs: Gap Analysis

The five active requirements map to three capability gaps, all addressable with existing libraries.

| Requirement | Gap | Solution |
|-------------|-----|---------|
| Agents emit `AgentOutput` with structured `Claim` objects | No LLM call currently produces `AgentOutput` — agents return freetext | Use `client.messages.parse(output_format=AgentOutput)` natively in `AnthropicClient` |
| Conflict detector comparing multi-agent outputs for contradictions | No `ConflictDetector` class exists | New `core/conflict_detector.py` using stdlib `difflib.SequenceMatcher` + claim value comparison |
| Arbiter auto-triggers on detected conflicts | Arbiter sub-agent exists but is manually invoked | New `orchestration/agent_orchestrator.py` that calls `DebateManager.open()` and programmatically invokes arbiter |
| Full orchestration: disagreement → debate → arbiter → experiment → backtest → verdict | No orchestration pipeline exists | `AgentOrchestrator` wires: collect outputs → `ConflictDetector` → `DebateManager` → arbiter → `ExperimentManager` → backtest |
| Auto-apply: ACCEPT/REJECT → parameter changes or strategy toggles | `ExperimentState.preset_overrides` dict exists but nothing reads and applies it | New `strategies/presets_applier.py` that reads YAML presets and writes staged changes |

**Zero new packages required.** All gaps close with existing dependencies and stdlib.

---

## Recommended Stack

### Core Technologies (no new packages)

| Technology | Version (installed) | Purpose | Why Recommended |
|------------|--------------------|---------|----|
| `anthropic` Python SDK | 0.83.0 | Structured `AgentOutput` emission from LLM calls | `client.messages.parse()` + Pydantic model derives JSON Schema automatically, guarantees schema compliance via constrained decoding — no parsing errors, no retries for malformed JSON |
| `pydantic` v2 | >=2.10.0 (installed) | Schema for `AgentOutput`, `Claim`, `FactCheckReport` — already the project standard | `model.model_json_schema()` converts existing frozen Pydantic models to JSON Schema for `output_config`; zero new schema code needed |
| `PyYAML` | >=6.0.2 (installed) | Read/write strategy preset YAML on auto-apply | Already used by `DebateManager`/`ExperimentManager`; same `yaml.safe_load()` + `yaml.dump()` pattern |
| `structlog` | >=24.4.0 (installed) | Structured audit logging for conflict detection and auto-apply events | Critical for auditability of autonomous decisions — bind `debate_id`, `experiment_id`, `conflict_type` to every event |
| `difflib` | Python 3.12 stdlib | String-similarity scoring for claim conflict detection | `SequenceMatcher.ratio()` on claim statements; threshold 0.85 similarity + divergent source values = contradiction candidate. Deterministic, zero latency |
| `asyncio` | Python 3.12 stdlib | Parallel multi-agent output collection | `asyncio.gather()` — domain agents are independent Claude API calls and must run in parallel, not sequentially |

### Supporting Libraries (no new packages)

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `redis` | >=5.2.0 (installed) | Publish `conflict_detected` / `verdict_applied` events via existing `EventBus` | New event types only; use existing `EventBus.publish()` — no new Redis patterns |
| `pathlib` | Python 3.12 stdlib | Preset YAML path resolution on auto-apply | Same pattern as `DebateManager._debate_path()` |
| `operator` | Python 3.12 stdlib | Comparison operators for `SuccessCriteria` evaluation | Already used in `ExperimentManager` — same `op.ge`, `op.le` etc. |

### New Modules to Create (no new packages)

These are the v8.0 deliverables — pure Python using existing imports.

| Module | Layer | Responsibility |
|--------|-------|---------------|
| `core/conflict_detector.py` | Layer 0 | Takes `list[AgentOutput]`, returns `list[ConflictPair]` — pure logic, no I/O, no LLM |
| `orchestration/agent_orchestrator.py` | Layer 5 | Wire: collect parallel agent outputs → detect conflicts → open debate → invoke arbiter → create experiment → await verdict |
| `strategies/presets_applier.py` | Layer 4 | Read `ExperimentState.preset_overrides` → stage YAML changes → emit `EventBus` notification |

---

## Structured Output: Exact API (HIGH confidence)

The installed `anthropic==0.83.0` supports `client.messages.parse()` with Pydantic models. No beta header required — structured outputs are GA as of 2026 for Claude Sonnet 4.6 / Opus 4.6.

```python
from anthropic import Anthropic
from finalayze.core.schemas import AgentOutput

client = Anthropic()  # or reuse existing AnthropicClient._client

response = client.messages.parse(
    model="claude-sonnet-4-6",
    max_tokens=2048,
    messages=[{"role": "user", "content": prompt}],
    output_format=AgentOutput,   # Pydantic model -> JSON Schema auto-derived
)

agent_output: AgentOutput = response.parsed_output
```

**Integration point:** Extend existing `AnthropicClient` in `analysis/llm_client.py` with a `parse_structured()` method that wraps `client.messages.parse()`. Do not duplicate the `_CachingLLMClient` retry/cache logic — wire `parse_structured()` through the same retry wrapper.

**Key schema constraint:** `AgentOutput.claims` has `Field(min_length=1)` — structured output will refuse to emit zero-claim responses. No extra validation needed.

**Datetime handling:** `AgentOutput.timestamp` is a `datetime` field. Pydantic serializes as ISO string; Claude emits ISO string; `parse()` deserializes back to `datetime`. Round-trip works correctly with Pydantic v2.

---

## Conflict Detection: Design Recommendation (HIGH confidence)

Do NOT use an LLM for conflict detection. It adds 1-3 seconds of latency, API cost, and nondeterminism to what must be a fast deterministic gate.

Use deterministic heuristics in `ConflictDetector`:

```python
# Three conflict signals (implement in core/conflict_detector.py):
# 1. Direction conflict:   agent_a.recommendation == "BUY X", agent_b.recommendation == "SELL X"
#    -> Check via keyword extraction on recommendation strings
# 2. Metric value conflict: claim_a.source.kind == "metric" AND claim_b.source.kind == "metric"
#    AND claim_a.source.metric_name == claim_b.source.metric_name
#    AND abs(claim_a.source.value - claim_b.source.value) > 0.05  # 5% relative tolerance
# 3. Statement conflict:   SequenceMatcher(claim_a.statement, claim_b.statement).ratio() > 0.85
#    AND source values diverge (signals same topic, different assertions)
```

`ConflictPair` output schema:

```python
@dataclass(frozen=True)
class ConflictPair:
    agent_a: str          # agent_name
    agent_b: str          # agent_name
    claim_a: Claim
    claim_b: Claim
    conflict_type: str    # "direction" | "metric_value" | "statement"
    similarity_score: float
```

This belongs in Layer 0 (`core/`) — pure logic, no I/O, frozen dataclass output.

---

## Auto-Apply: Safety Design (MEDIUM confidence)

Auto-applying experiment verdicts to YAML presets is the highest-risk part of v8.0. A false-positive ACCEPT verdict must not silently corrupt live trading parameters.

Recommended two-phase apply pattern in `strategies/presets_applier.py`:

1. Write changes to `presets/<segment>.yaml.pending` — not the live YAML
2. Emit `preset_staged` event via `EventBus` with full diff logged via `structlog`
3. Only promote `*.pending` → live YAML after explicit `preset_promote` trigger (initially manual, can be automated in v9.0)

`ExperimentState.preset_overrides` carries the dict of changes — the applier maps dict keys to YAML paths using existing `PyYAML` + `pathlib`.

**Why staging?** The backtest-to-live gap means an experiment that passes walk-forward may still be wrong for current market conditions. The staging step creates an audit checkpoint with zero operational complexity added.

---

## Alternatives Considered

| Recommended | Alternative | Why Not |
|-------------|-------------|---------|
| `difflib.SequenceMatcher` for conflict detection | LLM call for semantic conflict detection | LLM adds 1-3s latency, API cost, nondeterminism — conflict detection is a gate that must be deterministic and fast |
| `client.messages.parse(output_format=AgentOutput)` | `instructor` library | `instructor` is a third-party wrapper; `messages.parse()` is native Anthropic SDK since 0.58.0, zero extra dependency |
| New `AgentOrchestrator` in `orchestration/` | Extending `TradingLoop` | `TradingLoop` is already complex; agent orchestration is an independent concern and should be independently testable |
| Two-phase YAML staging for auto-apply | Direct YAML overwrite | Direct write risks live parameter corruption from false-positive verdicts |
| `asyncio.gather()` for parallel agent collection | Sequential agent calls | Sequential defeats the purpose; domain agents (quant-analyst, risk-officer, ml-engineer) are independent and their outputs must be collected in parallel |
| Extend existing `AnthropicClient` with `parse_structured()` | New `StructuredClient` class | Existing client already has retry + cache; extending avoids duplicating that logic |

---

## What NOT to Add

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| `langchain`, `langgraph`, `crewai` | Heavy frameworks conflict with existing async stack and Pydantic v2; overkill for 3 domain agents + 1 arbiter | Anthropic SDK `messages.parse()` + custom `AgentOrchestrator` |
| `celery` for agent orchestration | Already installed for background tasks, but designed for distributed task queues — overkill for in-process LLM call coordination | `asyncio.gather()` — agents are Claude API calls lasting 1-5 seconds, not long-running workers |
| New DB tables for agent outputs | Adds migration + schema complexity; agent outputs are transient orchestration artifacts | Persist only final `DebateState` + `ExperimentState` to files (existing pattern in `.planning/debates/` and `.planning/experiments/`) |
| Redis pub/sub for conflict events | Redis Streams (`EventBus`) is the project's existing pattern — do not introduce a second Redis messaging model | Use existing `EventBus.publish()` with new stream keys: `conflict_detected`, `verdict_applied`, `preset_staged` |
| `opentelemetry` for agent tracing | Too heavy for current scale; 3 domain agents is not a distributed microservices system | `structlog` with `debate_id` / `experiment_id` bound to all log events covers traceability |
| `numpy` / `scipy` for conflict scoring | Cosine similarity on claim vectors is unnecessary complexity; text claims are short strings | `difflib.SequenceMatcher.ratio()` is sufficient and already in stdlib |

---

## Installation

No new packages needed. All required capabilities are in the environment.

Verify current state:

```bash
uv run python -c "import anthropic; print(anthropic.__version__)"
# Expected: 0.83.0

uv run python -c "import pydantic; print(pydantic.__version__)"
# Expected: 2.x.x

uv run python -c "import yaml; print(yaml.__version__)"
# Expected: 6.x.x
```

---

## Version Compatibility

| Package | Installed | Feature Used | Constraint |
|---------|-----------|-------------|------------|
| `anthropic` | 0.83.0 | `client.messages.parse(output_format=PydanticModel)` | Requires >=0.58.0; `output_config` is GA — no beta header needed in 0.83.0 |
| `pydantic` | >=2.10.0 | `model_json_schema()` for schema derivation into `output_config` | v2 only — Pydantic v1 schemas are not compatible with `messages.parse()` |
| `pyyaml` | >=6.0.2 | `yaml.safe_load()` / `yaml.dump()` for preset YAML read/write | No version concerns |
| `redis` | >=5.2.0 | `EventBus.publish()` for new `conflict_detected` / `preset_staged` events | No version concerns |

---

## Sources

- [Anthropic Structured Outputs docs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs) — `client.messages.parse()` API verified, GA status confirmed for Sonnet 4.6 / Opus 4.6, no beta header required (HIGH confidence)
- [Anthropic SDK Python changelog](https://github.com/anthropics/anthropic-sdk-python/blob/main/CHANGELOG.md) — `parse()` with `output_config` available since 0.58.0; installed 0.83.0 (MEDIUM confidence via WebSearch; HIGH confidence for installed version via `uv pip show`)
- `src/finalayze/core/schemas.py` — `AgentOutput`, `Claim`, `FactCheckReport`, `ExperimentState` schema details confirmed by direct read (HIGH confidence)
- `src/finalayze/core/debate_manager.py`, `experiment_manager.py` — existing CRUD patterns confirmed by direct read (HIGH confidence)
- `pyproject.toml` + `uv pip show anthropic` — installed version 0.83.0 confirmed (HIGH confidence)

---
*Stack research for: Finalayze v8.0 Agent Integration & Autonomous Decision Loop*
*Researched: 2026-04-12*
