---
phase: 39-rest-endpoint-hardening
plan: "01"
subsystem: api
tags: [rest-api, experiments, debates, telegram, circuit-breaker, tdd]
dependency_graph:
  requires: []
  provides: [real-alerter-in-apply, real-circuit-breaker-in-apply, multi-debate-response, finalize-debate-endpoint]
  affects: [src/finalayze/api/v1/experiments.py, src/finalayze/api/v1/debates.py]
tech_stack:
  added: []
  patterns: [factory-function-pattern, TDD-red-green, FastAPI-endpoint]
key_files:
  created: []
  modified:
    - src/finalayze/api/v1/experiments.py
    - src/finalayze/api/v1/debates.py
    - tests/unit/test_api_experiments.py
    - tests/unit/test_api_debates.py
decisions:
  - "REST circuit breaker starts at NORMAL level (independent of TradingLoop) — documented limitation, full integration needs Redis state sharing"
  - "TC001 noqa suppressed for AgentOutput+FactCheckReport: FastAPI needs runtime Pydantic model resolution, TYPE_CHECKING block would break request deserialization"
  - "TelegramAlerter is already no-op when bot_token is empty string — no separate no-op class needed"
metrics:
  duration_seconds: 267
  completed_date: "2026-04-12"
  tasks_completed: 2
  files_modified: 4
requirements: [ORCH-01, ORCH-02, APPLY-02, APPLY-05]
---

# Phase 39 Plan 01: REST Endpoint Hardening Summary

**One-liner:** Wired real TelegramAlerter from settings and CircuitBreaker instance into /apply, replaced truncated debate_id with debate_ids list, and added POST /debates/{id}/finalize endpoint calling AgentOrchestrator.finalize_debate().

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Wire real alerter and circuit breaker into /apply endpoint | f166b54 | experiments.py, test_api_experiments.py |
| 2 | Fix multi-debate response and add finalize endpoint | 52aee5a | debates.py, test_api_debates.py |

## What Was Built

### Task 1: /apply endpoint hardening (APPLY-05, APPLY-02)

- Replaced `_make_no_op_alerter()` (inner class, no Telegram calls) with `_make_alerter()` factory that reads `config.settings.get_settings()` and instantiates a real `TelegramAlerter(bot_token, chat_id)`. When `telegram_bot_token` is empty, `TelegramAlerter` already returns immediately without network calls.
- Replaced `circuit_breakers={}` (silently skipped the gate) with `_get_circuit_breakers()` that returns `{"moex": CircuitBreaker("moex")}`. The REST breaker starts at NORMAL level — documented limitation logged as WARNING on each call.
- Added `TelegramAlerter` and `CircuitBreaker` imports at module level.
- New tests: `test_apply_experiment_uses_real_alerter` (patches `TelegramAlerter` class, verifies called with credentials), `test_apply_experiment_circuit_breaker_real_instance` (patches `_get_circuit_breakers`, verifies it's called).

### Task 2: debates.py fixes (ORCH-02, ORCH-01)

- `CreateDebateResponse.debate_id: str | None` → `debate_ids: list[str]` — all IDs from `orch.run()` now returned.
- `create_debate` endpoint: sets `debate_ids=debate_ids` (full list), `conflicts_found=len(debate_ids)`.
- Added `FinalizeDebateRequest(report: FactCheckReport)` and `FinalizeDebateResponse(debate_id, experiment_id, resolved)` models.
- Added `POST /debates/{debate_id}/finalize` endpoint: creates `AgentOrchestrator()`, calls `finalize_debate(debate_id, req.report)`, returns `experiment_id` (str when contradictions, None when resolved), catches `FileNotFoundError` → 404.
- Updated existing `test_post_debates_with_conflicts_returns_201` and `test_post_debates_no_conflicts_returns_200` to assert on `debate_ids` list.
- Added `TestPostDebatesMultiDebate` class: 3-debate case and empty case.
- Added `TestFinalizeDebate` class: contradictions → experiment_id, no contradictions → resolved=True, nonexistent → 404.

## Test Results

```
26 passed in 0.57s (test_api_experiments.py + test_api_debates.py)
ruff check: All checks passed
mypy: Success: no issues found in 2 source files
```

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] TC001 ruff violation introduced by FactCheckReport import**
- **Found during:** Task 2 ruff check
- **Issue:** Adding `FactCheckReport` to the existing `from finalayze.core.schemas import AgentOutput` line introduced a new TC001 (flake8-type-checking) violation. `AgentOutput` already had TC001 pre-existing.
- **Fix:** Added `# noqa: TC001` to the import line. Both `AgentOutput` and `FactCheckReport` are needed at runtime for FastAPI/Pydantic request body deserialization — moving to `TYPE_CHECKING` would break the endpoint.
- **Files modified:** src/finalayze/api/v1/debates.py
- **Commit:** 52aee5a

**2. [Rule 1 - Bug] Test assertion used positional args but factory uses keyword args**
- **Found during:** Task 1 GREEN phase test run
- **Issue:** `mock_telegram_cls.assert_called_once_with("my-bot-token", "123456")` failed because `_make_alerter()` calls `TelegramAlerter(bot_token=..., chat_id=...)` with keyword arguments.
- **Fix:** Updated assertion to `assert_called_once_with(bot_token="my-bot-token", chat_id="123456")`.
- **Files modified:** tests/unit/test_api_experiments.py
- **Commit:** f166b54

## Known Stubs

None — all wiring uses real components. The NORMAL-level REST circuit breaker is a documented limitation (not a stub), clearly logged as a warning.

## Threat Flags

| Flag | File | Description |
|------|------|-------------|
| threat_flag: input-validation | src/finalayze/api/v1/debates.py | POST /finalize accepts FactCheckReport from untrusted client — mitigated by Pydantic validation (T-39-01 in plan's threat model, disposition: mitigate, confirmed covered) |

## Self-Check: PASSED

- `src/finalayze/api/v1/experiments.py` — exists, contains `TelegramAlerter`, `CircuitBreaker`, `_make_alerter`, `_get_circuit_breakers`
- `src/finalayze/api/v1/debates.py` — exists, contains `debate_ids`, `finalize`, `FinalizeDebateResponse`, `FinalizeDebateRequest`
- `tests/unit/test_api_experiments.py` — exists, contains `test_apply_experiment_uses_real_alerter`
- `tests/unit/test_api_debates.py` — exists, contains `test_finalize`
- Commits f166b54 and 52aee5a present in git log
