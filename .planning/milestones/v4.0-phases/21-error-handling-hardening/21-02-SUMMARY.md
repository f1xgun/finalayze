---
phase: 21-error-handling-hardening
plan: 02
subsystem: error-handling
tags: [structlog, telegram-alerts, grpc, tinkoff, error-tracking]

# Dependency graph
requires:
  - phase: 21-error-handling-hardening
    provides: "Error handling foundation from plan 01"
provides:
  - "Structured error logging with error_type field in TinkoffFetcher"
  - "Consecutive cycle failure alerting in TradingLoop"
  - "Per-layer consecutive error tracking in BondCycleProcessor"
affects: [monitoring, operations, alerting]

# Tech tracking
tech-stack:
  added: []
  patterns: ["Consecutive error counter with threshold-based escalation", "Structured log fields for error classification"]

key-files:
  created: []
  modified:
    - src/finalayze/data/fetchers/tinkoff_data.py
    - src/finalayze/core/trading_loop.py
    - src/finalayze/core/bond_cycle.py
    - tests/unit/test_tinkoff_data.py
    - tests/unit/test_trading_loop.py
    - tests/unit/test_bond_cycle.py

key-decisions:
  - "Used AlertPriority.CRITICAL for consecutive failure alerts (highest priority)"
  - "Per-layer error tracking in BondCycleProcessor (independent counters for core/tactical/strategic/short)"
  - "Threshold of 3 consecutive failures before escalation (both TradingLoop and BondCycleProcessor)"

patterns-established:
  - "Consecutive error counter pattern: increment on failure, reset on success, escalate at threshold"
  - "Structured error_type field: type(exc).__name__ added to all exception logs"

requirements-completed: [ERR-03, ERR-04, ERR-05]

# Metrics
duration: 6min
completed: 2026-03-22
---

# Phase 21 Plan 02: Structured Error Logging and Consecutive Failure Alerting Summary

**Structured error_type logging in TinkoffFetcher, consecutive failure counters with Telegram alerting in TradingLoop, and per-layer error escalation in BondCycleProcessor**

## Performance

- **Duration:** 6 min
- **Started:** 2026-03-22T20:56:46Z
- **Completed:** 2026-03-22T21:02:20Z
- **Tasks:** 2
- **Files modified:** 6

## Accomplishments
- TinkoffFetcher logs include error_type, timeframe, and contextual fields in all 4 fetch method exception handlers
- TradingLoop sends Telegram CRITICAL alert after 3 consecutive equity or bond cycle failures, with counters resetting on success
- BondCycleProcessor tracks per-layer consecutive gRPC failures independently, escalating log level from exception to error after threshold

## Task Commits

Each task was committed atomically (TDD: test -> feat):

1. **Task 1: TinkoffFetcher structured error logging and TradingLoop consecutive error counter**
   - `7d5cd23` (test) - Failing tests for error_type logging and consecutive counters
   - `b6d6227` (feat) - Implementation of structured logging and error counters
2. **Task 2: BondCycleProcessor consecutive gRPC error counter**
   - `683a4c2` (test) - Failing tests for per-layer consecutive error tracking
   - `a6ad1a9` (feat) - Implementation of per-layer error counter with escalation

## Files Created/Modified
- `src/finalayze/data/fetchers/tinkoff_data.py` - Added error_type field to all 4 fetch method exception logs
- `src/finalayze/core/trading_loop.py` - Added consecutive equity/bond error counters with Telegram alerting
- `src/finalayze/core/bond_cycle.py` - Added per-layer consecutive error tracking with log escalation
- `tests/unit/test_tinkoff_data.py` - 4 new tests for error_type logging
- `tests/unit/test_trading_loop.py` - 7 new tests for consecutive error counters
- `tests/unit/test_bond_cycle.py` - 5 new tests for per-layer error tracking

## Decisions Made
- Used AlertPriority.CRITICAL (not IMPORTANT) for consecutive failure alerts -- systematic failures warrant highest priority
- Per-layer tracking in BondCycleProcessor uses layer.value as dict key (string) for simplicity
- Threshold of 3 consecutive failures chosen as default -- configurable via _MAX_CONSECUTIVE_ERRORS / _layer_error_threshold

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] AlertPriority.HIGH does not exist**
- **Found during:** Task 1 (TradingLoop consecutive error counter)
- **Issue:** Plan specified AlertPriority.HIGH but the enum only has CRITICAL, IMPORTANT, INFO
- **Fix:** Used AlertPriority.CRITICAL instead (appropriate for systematic failure alerts)
- **Files modified:** src/finalayze/core/trading_loop.py
- **Verification:** All tests pass
- **Committed in:** b6d6227

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Minor enum name correction. No scope creep.

## Issues Encountered
- Bond cycle tests required mocking _now() and _is_market_open() instead of datetime/is_moex_trading_day due to deferred imports -- resolved by using patch.object on instance methods

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Error handling hardening phase complete (both plans 01 and 02)
- All three modules now provide structured error context for operators
- Consecutive failure counters make systematic degradation visible via Telegram

---
*Phase: 21-error-handling-hardening*
*Completed: 2026-03-22*
