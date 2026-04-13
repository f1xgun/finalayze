---
phase: 38-presetapplicator-auto-apply-loop
plan: "01"
subsystem: orchestration
tags: [preset-applicator, safety-gates, atomic-write, circuit-breaker, sandbox-gate, REST, experiments]
dependency_graph:
  requires: []
  provides:
    - PresetApplicator with 7-gate safety pipeline
    - SandboxGate DB validator
    - POST /experiments/{id}/apply REST endpoint
  affects:
    - src/finalayze/orchestration/preset_applicator.py
    - src/finalayze/api/v1/experiments.py
tech_stack:
  added:
    - PresetApplicator (orchestration layer, atomic YAML writer)
    - SandboxGate (async DB query for trading day validation)
    - ApplyResult (frozen dataclass)
    - PresetApplyBlockedError / PresetValidationError (exception classes)
    - ApplyResultResponse / ApplyRequest (Pydantic models)
  patterns:
    - TDD RED->GREEN->REFACTOR (17 unit tests written before implementation)
    - os.replace() atomic write via .pending staging file
    - shutil.copy2() timestamped backup before any write
    - Deferred imports (PLC0415) for Layer 5->6 boundary crossing
    - TYPE_CHECKING guard for Callable (ruff TC003)
key_files:
  created:
    - src/finalayze/orchestration/preset_applicator.py
    - tests/unit/test_preset_applicator.py
  modified:
    - src/finalayze/api/v1/experiments.py
    - tests/unit/test_api_experiments.py
decisions:
  - "_check_position_ownership extracted to helper to stay under PLR0912 branch limit"
  - "REST endpoint uses deferred imports for PresetApplicator to avoid circular import at module level"
  - "Phase 38 limitation block comment in experiments.py documents no-op circuit breakers and skipped cache invalidation in REST context"
  - "SandboxGate uses date() comparison (not datetime) for distinct calendar day counting"
  - "original param removed from _atomic_write_yaml -- shutil.copy2 reads from disk directly"
metrics:
  duration_minutes: 6
  completed_date: "2026-04-12"
  tasks_completed: 2
  files_created: 2
  files_modified: 2
  tests_added: 20
---

# Phase 38 Plan 01: PresetApplicator + Auto-Apply Loop Summary

**One-liner:** Atomic YAML preset writer with 7-gate safety pipeline (circuit breaker, sandbox validation, INCONCLUSIVE Telegram routing, position ownership, key/type validation, backup+os.replace(), cache invalidation) and REST endpoint for triggering applies.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Create PresetApplicator + SandboxGate with TDD | c1e5467 | preset_applicator.py, test_preset_applicator.py |
| 2 | Add POST /experiments/{id}/apply REST endpoint | f445c93 | experiments.py, test_api_experiments.py |

## What Was Built

### Task 1: PresetApplicator + SandboxGate

`src/finalayze/orchestration/preset_applicator.py` exports:
- `PresetApplicator` — full `apply_verdict()` pipeline with 7 safety gates
- `SandboxGate` — async DB query requiring 3+ distinct calendar dates with `fill_rate > 0` and no `drawdown_pct >= 0.10`
- `ApplyResult` — frozen dataclass (experiment_id, applied, backup_path, verdict, reason)
- `PresetApplyBlockedError` — raised by safety gates
- `PresetValidationError` — raised by key/type validation

**Gate order in apply_verdict():**
1. Circuit breaker (FIRST — before any file I/O or DB query)
2. ExperimentManager.read_experiment() — raises FileNotFoundError if missing
3. INCONCLUSIVE routing — alerter.send_alert() called with AlertPriority.IMPORTANT, no YAML write
4. Non-ACCEPTED early return
5. SandboxGate.check() — DB query, blocks if < 3 days or high drawdown
6. Key/type validation — unknown keys or type mismatches raise PresetValidationError
7. Position ownership — disabling strategy with open positions raises PresetApplyBlockedError
8. Atomic write — shutil.copy2 backup + yaml.dump to .pending + os.replace()
9. Cache invalidation — combiner.invalidate_segment_cache(segment_id) when combiner provided

**Security mitigations (per threat model):**
- T-38-01: experiment_id safe via ExperimentState validator (upstream)
- T-38-02: _validate_keys() rejects unknown keys and type mismatches
- T-38-03: Path.resolve() + startswith check prevents segment_id path traversal

### Task 2: POST /experiments/{id}/apply

`src/finalayze/api/v1/experiments.py` additions:
- `ApplyResultResponse` and `ApplyRequest` Pydantic models
- `apply_experiment()` endpoint: POST /{experiment_id}/apply
  - 404 on FileNotFoundError
  - 409 on PresetApplyBlockedError
  - 422 on PresetValidationError
  - 200 with ApplyResultResponse on success or INCONCLUSIVE

**Phase 38 limitations documented in code:**
- `circuit_breakers={}` — circuit breaker gate is a no-op in REST context
- `entry_strategy_getter=lambda: {}` — position ownership check is skipped
- `combiner=None` — cache invalidation is skipped (combiner reads from disk anyway)

## Test Coverage

- `tests/unit/test_preset_applicator.py`: 17 tests (TDD RED->GREEN->REFACTOR)
  - SandboxGate: 4 tests (pass, insufficient days, zero fill rate, high drawdown)
  - Circuit breaker: 2 tests (CAUTION, HALTED)
  - INCONCLUSIVE routing: 1 test (Telegram alert called, no YAML write)
  - Missing experiment: 1 test
  - Sandbox gate integration: 1 test
  - Atomic write: 3 tests (backup created, no .pending left, deep merge)
  - Validation: 2 tests (unknown key, wrong type)
  - Position ownership: 1 test
  - Cache invalidation: 2 tests (with combiner, without combiner)
- `tests/unit/test_api_experiments.py`: +3 tests (404, success, inconclusive)
- **Total: 27 tests passing**

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Refactor] Extracted _check_position_ownership helper**
- **Found during:** Task 1 GREEN phase lint check
- **Issue:** apply_verdict() had 13 branches (ruff PLR0912 limit is 12)
- **Fix:** Extracted position ownership check into `_check_position_ownership()` private method
- **Files modified:** src/finalayze/orchestration/preset_applicator.py

**2. [Rule 1 - Bug] Removed unused `original` parameter from _atomic_write_yaml**
- **Found during:** Task 1 GREEN phase lint check (ruff ARG002)
- **Issue:** `_atomic_write_yaml` accepted `original: dict` but used `shutil.copy2` to copy from disk (not from the parsed dict), making the parameter dead code
- **Fix:** Removed parameter and updated call site
- **Files modified:** src/finalayze/orchestration/preset_applicator.py

**3. [Rule 1 - Import] Moved Callable to TYPE_CHECKING block**
- **Found during:** Task 1 GREEN phase lint check (ruff TC003/UP035)
- **Issue:** `Callable` used only in type annotations, should be in TYPE_CHECKING block
- **Fix:** Moved to TYPE_CHECKING guard (safe because `from __future__ import annotations` is present)
- **Files modified:** src/finalayze/orchestration/preset_applicator.py

**4. [Rule 1 - Patch target] Fixed deferred import mock patching in API tests**
- **Found during:** Task 2 test run
- **Issue:** `patch("finalayze.api.v1.experiments.PresetApplicator")` fails because PresetApplicator is imported inside the function body (deferred), not at module level
- **Fix:** Tests patch `finalayze.orchestration.preset_applicator.PresetApplicator` (the source module) and `finalayze.core.db.get_async_session_factory`
- **Files modified:** tests/unit/test_api_experiments.py

## Known Stubs

None — all functionality is wired. Phase 38 limitations (no circuit breakers, no position ownership in REST) are intentional design constraints documented in code comments, not stubs.

## Threat Flags

None beyond those already in the plan's threat model (T-38-01 through T-38-05). No new network endpoints or auth paths introduced beyond the documented POST endpoint.

## Self-Check: PASSED

| Item | Result |
|------|--------|
| preset_applicator.py exists | FOUND |
| test_preset_applicator.py exists | FOUND |
| experiments.py exists | FOUND |
| test_api_experiments.py exists | FOUND |
| Commit c1e5467 (Task 1) | FOUND |
| Commit f445c93 (Task 2) | FOUND |
| class PresetApplicator | OK |
| class SandboxGate | OK |
| class PresetApplyBlockedError | OK |
| CircuitLevel.NORMAL gate | OK |
| os.replace atomic write | OK |
| yaml.bak backup | OK |
| AlertPriority.IMPORTANT | OK |
| fill_rate SandboxGate check | OK |
| invalidate_segment_cache | OK |
| 27 tests passing | OK |
