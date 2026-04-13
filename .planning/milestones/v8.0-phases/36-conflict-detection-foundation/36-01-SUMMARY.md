---
phase: 36-conflict-detection-foundation
plan: 01
status: complete
started: 2026-04-12
completed: 2026-04-12
---

# Plan 36-01: ConflictReport Schema + parse_structured()

## Outcome

**Status:** Complete
**Tasks:** 2/2

## What Was Built

### Task 1: ConflictReport schema in core/schemas.py
Added `ConflictType` (StrEnum: DIRECTION, METRIC, STATEMENT), `ConflictSeverity` (StrEnum: CRITICAL, HIGH, LOW), and `ConflictReport` (frozen Pydantic model with conflict_id, conflict_type, severity, involved_claims, agent_names, detected_at, confidence_delta). 8 tests covering validation, field constraints, and frozen behavior.

### Task 2: parse_structured() on all LLM clients
Added abstract `parse_structured()` to `LLMClient` ABC, with rate-limited implementation on `_CachingLLMClient` base. Shared `_openai_parse_structured_once()` helper handles BadRequestError fallback to JSON mode. All 5 concrete clients (Anthropic, OpenAI, OpenRouter, Groq, DeepSeek) and `FallbackLLMClient` implement it. 8 tests covering all client paths.

## Key Files

### Created
- (none — all modifications to existing files)

### Modified
- `src/finalayze/core/schemas.py` — ConflictType, ConflictSeverity, ConflictReport models (+30 lines)
- `src/finalayze/analysis/llm_client.py` — parse_structured() on all clients (+445 lines)
- `tests/unit/core/test_debate_schemas.py` — 8 schema tests
- `tests/unit/test_llm_client.py` — 8 parse_structured tests

## Deviations

None — plan followed exactly.

## Self-Check

- [x] ConflictReport, ConflictType, ConflictSeverity importable from core.schemas
- [x] parse_structured() exists on all 5 LLM client classes
- [x] Anthropic uses messages.parse(), OpenAI-compatible clients use beta.chat.completions.parse()
- [x] 16 new tests, all passing
